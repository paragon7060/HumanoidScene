"""ROS Noetic worker source, sent over SSH; importing this module does no I/O."""

WORKER = r'''
import json, math, queue, signal, sys, threading, time
from collections import deque
import rospy
from kuavo_msgs.msg import jointCmd, sensorsData
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Bool, Int8, Float64, Float64MultiArray

request = json.loads(sys.stdin.readline())
mode = request['mode']
lock = threading.RLock()
latest, arrivals, advanced, stamps, foreign = {}, {}, {}, {}, {}
sensor_history = deque(maxlen=2000)
stop = threading.Event()
stop_reason = [None]
heartbeat = [time.monotonic()]
heartbeat_state = {'received_count':0, 'last_sequence':None}
start = time.monotonic()
log_queue = queue.Queue(maxsize=4096)
log_error = [None]
log_closed = threading.Event()
logged_at = {}
graph_status = {'checked': None, 'error': None}
topics = {
 'sensors': ('/sensors_data_raw', sensorsData),
 'motor_command': ('/joint_cmd', jointCmd),
 'orientation': ('/imu', Imu),
 'body_target': ('/humanoid_wheel/leg_target_qpos_quick_mode', Float64MultiArray),
 'arm_target': ('/humanoid_wheel/arm_target_qpos_quick_mode', Float64MultiArray),
 'mpc_state': ('/humanoid_wheel/optimizedState_mrt', Float64MultiArray),
 'filtered_state': ('/humanoid_wheel/optimizedState_mrt_kinemicLimit', Float64MultiArray),
 'wbc_state': ('/humanoid_wheel/optimizedState_wbc_in', Float64MultiArray),
 'enabled': ('/enable_control_state', Bool),
 'lb_mode': ('/mobile_manipulator/lb_mpc_control_mode', Int8),
 'arm_transport': ('/ik_debug/arm_traj_receive/transport', Float64),
}

def enqueue_log(row):
    if log_closed.is_set():
        return
    try:
        log_queue.put_nowait(row)
    except queue.Full:
        log_error[0] = 'Telemetry queue full; recording cannot keep up'
        stop.set()

def log_guard():
    if log_error[0]:
        raise ValueError(log_error[0])

def log_writer():
    try:
        finished = False
        while not finished:
            row = log_queue.get()
            batch = []
            while row is not None:
                batch.append(json.dumps(row, allow_nan=row.get('kind') == 'message'))
                if len(batch) == 128:
                    break
                try:
                    row = log_queue.get_nowait()
                except queue.Empty:
                    break
            if row is None:
                finished = True
            if batch:
                sys.stdout.write('\n'.join(batch)+'\n')
                sys.stdout.flush()
    except Exception as exc:
        log_error[0] = f'Telemetry writer failed: {exc}'
        stop.set()

def finish_logging():
    log_closed.set()
    try:
        log_queue.put(None, timeout=2.)
    except queue.Full:
        return False
    log_thread.join(timeout=2.)
    return not log_thread.is_alive() and not log_error[0]

def emit(row):
    enqueue_log(dict(row, receipt_unix_s=time.time(),
                     receipt_monotonic_s=time.monotonic()))

def receive(msg, key):
    now = time.monotonic()
    row = {'kind': 'message', 'topic': topics[key][0],
           'publisher': msg._connection_header.get('callerid', '')}
    if key == 'sensors':
        row.update({n: list(getattr(msg.joint_data, n)) for n in
                    ('joint_q', 'joint_v', 'joint_vd', 'joint_torque')})
        row['imu_quat_xyzw'] = [msg.imu_data.quat.x, msg.imu_data.quat.y,
                                 msg.imu_data.quat.z, msg.imu_data.quat.w]
        row['imu_gyro'] = [msg.imu_data.gyro.x, msg.imu_data.gyro.y, msg.imu_data.gyro.z]
    elif key == 'orientation':
        row['imu_quat_xyzw'] = [msg.orientation.x, msg.orientation.y,
                                msg.orientation.z, msg.orientation.w]
        row['orientation_covariance'] = list(msg.orientation_covariance)
        row['frame_id'] = msg.header.frame_id
    elif key == 'motor_command':
        row.update({n: list(getattr(msg, n)) for n in
                    ('joint_q', 'joint_v', 'tau', 'tau_max', 'tau_ratio',
                     'joint_kp', 'joint_kd', 'control_modes')})
    else:
        row['data'] = list(msg.data) if key in ('body_target', 'arm_target', 'mpc_state', 'filtered_state', 'wbc_state') else msg.data
    should_log = False
    with lock:
        if hasattr(msg, 'header'):
            stamp = msg.header.stamp.to_sec()
            row['header_s'] = stamp
            row['header_seq'] = msg.header.seq
            if stamp > stamps.get(key, -1):
                advanced[key] = now
            stamps[key] = stamp
        latest[key], arrivals[key] = row, now
        if key == 'sensors':
            sensor_history.append((now, row['joint_q'][:18], row['joint_v'][:18]))
            while sensor_history and now-sensor_history[0][0] > 2.:
                sensor_history.popleft()
        recording = mode in ('record', 'send', 'prepare') or request.get('watch_s', 0) > 0
        if recording and now-logged_at.get(key, -100.) >= .01:
            logged_at[key] = now
            should_log = True
    if should_log:
        # Feedback/guards retain every callback. Only stored telemetry is sampled
        # at <=100 Hz/topic, with original receipt times. Never write under lock.
        enqueue_log(dict(row, receipt_unix_s=time.time(), receipt_monotonic_s=now))

def competing(msg, topic):
    caller = msg._connection_header.get('callerid', '')
    if caller != rospy.get_name():
        with lock:
            foreign[topic] = (time.monotonic(), caller)

def ros_payload(value):
    """Preserve ROS fields/units, including raw device times; no conversion."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, 'to_sec'):
        return value.to_sec()
    if isinstance(value, (list, tuple)):
        return [ros_payload(v) for v in value]
    if hasattr(value, '__slots__'):
        return {name: ros_payload(getattr(value, name)) for name in value.__slots__ if not name.startswith('_')}
    raise ValueError('Unsupported ROS input field type: '+type(value).__name__)

def receive_vr_input(msg, spec):
    topic, typ, period = spec
    now = time.monotonic()
    with lock:
        if now-vr_logged_at.get(topic, -100.) < period:
            return
        vr_logged_at[topic] = now
        vr_counts[topic] = vr_counts.get(topic, 0)+1
    try:
        payload=ros_payload(msg)
    except Exception as exc:
        log_error[0]='VR input serialization failed on '+topic+': '+str(exc)
        stop.set()
        return
    row = {'kind':'message', 'layer':'vr_input', 'topic':topic, 'ros_type':typ,
           'publisher':msg._connection_header.get('callerid',''),
           'receipt_unix_s':time.time(), 'receipt_monotonic_s':now, 'payload':payload}
    if hasattr(msg, 'header'):
        row.update(header_s=msg.header.stamp.to_sec(), header_seq=msg.header.seq)
    enqueue_log(row)

def request_stop(reason):
    if stop_reason[0] is None:
        stop_reason[0] = reason
    stop.set()

def stdin_watchdog():
    for line in sys.stdin:
        parts = line.strip().split()
        if parts and parts[0] == 'heartbeat' and (len(parts) == 1 or
                                                len(parts) == 2 and parts[1].isdigit()):
            received = time.monotonic()
            gap = received-heartbeat[0]
            heartbeat[0] = received
            heartbeat_state['received_count'] += 1
            heartbeat_state['last_sequence'] = int(parts[1]) if len(parts) == 2 else None
            emit({'kind':'heartbeat_rx', 'sequence':heartbeat_state['last_sequence'],
                  'interval_s':gap})
        else:
            request_stop('Unexpected stdin message')
            break
    request_stop('Local stdin EOF')

rospy.init_node('s63_body_arm_calibration', anonymous=True, disable_signals=True)
signal.signal(signal.SIGTERM, lambda *_: request_stop('Remote SIGTERM'))
signal.signal(signal.SIGINT, lambda *_: request_stop('Remote SIGINT'))
threading.Thread(target=stdin_watchdog, daemon=True).start()
config = rospy.get_param('/kuavo_configuration', {})
config = json.loads(config) if isinstance(config, str) else config
version = rospy.get_param('/robot_version', None)
log_thread = threading.Thread(target=log_writer, daemon=True)
log_thread.start()
subscriptions = [rospy.Subscriber(t, typ, receive, callback_args=k, queue_size=500)
                 for k, (t, typ) in topics.items()]
subscriptions += [rospy.Subscriber(t, JointState, competing, callback_args=t, queue_size=100)
                  for t in ('/lb_leg_traj', '/kuavo_arm_traj')]
vr_counts, vr_logged_at = {}, {}
vr_specs = []
if request.get('vr_inputs'):
    if mode != 'record':
        raise ValueError('VR inputs option is read-only record mode only')
    from roslib.message import get_message_class
    vr_specs = [
      ('/kuavo_arm_traj', 'sensor_msgs/JointState', .01),
      ('/lb_leg_traj', 'sensor_msgs/JointState', .01),
      ('/kuavo_arm_traj_origin', 'std_msgs/Float32MultiArray', .01),
      ('/kuavo_arm_traj_filtered', 'std_msgs/Float32MultiArray', .01),
      ('/vr_incremental/arm_traj_rad', 'sensor_msgs/JointState', .01),
      ('/vr_incremental/arm_traj_shadow_rad', 'sensor_msgs/JointState', .01),
      ('/cmd_torso_pose_vr', 'geometry_msgs/PoseStamped', .02),
      ('/cmd_lb_torso_pose', 'geometry_msgs/Twist', .02),
      ('/cmd_torso_vel', 'geometry_msgs/Twist', .02),
      ('/torso_open_loop_state', 'geometry_msgs/Twist', .02),
      ('/vr_whole_torso_ctrl', 'std_msgs/Bool', .02),
      ('/mm/two_arm_hand_pose_cmd', 'kuavo_msgs/twoArmHandPoseCmd', .02),
      ('/drake_ik/input_pos', 'kuavo_msgs/Float32MultiArrayStamped', .02),
      ('/leju_quest_bone_poses', 'noitom_hi5_hand_udp_python/PoseInfoList', .05),
      ('/quest_joystick_data', 'kuavo_msgs/JoySticks', .05),
      ('/humanoid/mpc/arm_control_mode', 'std_msgs/Float64MultiArray', .02),
    ]
    for spec in vr_specs:
        cls = get_message_class(spec[1])
        if cls is None:
            emit({'kind':'vr_topic_unavailable', 'topic':spec[0], 'ros_type':spec[1],
                  'reason':'message class not installed'})
            continue
        subscriptions.append(rospy.Subscriber(spec[0], cls, receive_vr_input, callback_args=spec, queue_size=100))
    emit({'kind':'vr_record_metadata', 'topics':[{'topic':t,'ros_type':typ,'max_hz':1/period} for t,typ,period in vr_specs],
          'semantics':'Raw ROS payloads; preserve source units, names and device/header/receipt times. No publishers or mode services.',
          'units_note':'Python ik_ros_uni.py arm_traj/origin/filtered are degrees; joint_cmd and measured q/v are radians. Confirm active publisher before mapping.',
          'shared_memory_note':'CPP may bypass arm_traj ROS publication; record controller inputs and report missing layers, never enable shadow publication automatically.'})
emit({'kind': 'metadata', 'robot_version': version,
      'configuration': config, 'mode': mode,
      'telemetry_max_hz_per_topic':100, 'graph_max_age_s':1.5,
      'torque_note': 'joint_cmd.tau is demanded; joint_torque source/units need separate verification'})

def snapshot():
    with lock:
        body = latest.get('body_target', {}).get('data')
        arms = latest.get('arm_target', {}).get('data')
        verified = (isinstance(body, list) and len(body) == 4 and
                    isinstance(arms, list) and len(arms) == 14 and
                    all(time.monotonic()-arrivals.get(k, 0) < .3 for k in ('body_target', 'arm_target')))
        measured = latest.get('sensors', {}).get('joint_q', [])
        baseline = body + arms if verified else measured[:18]
        return {'robot_version': version, 'joint_names': NAMES,
                'baseline_rad': baseline, 'baseline_source': 'live_quick_mode_targets' if verified else 'measured_only',
                'captured_unix_s': time.time(), 'observations': dict(latest),
                'configuration': config}

def graph_guard():
    pubs, subs, _ = rospy.get_master().getSystemState()[2]
    pubmap, submap = dict(pubs), dict(subs)
    main = set(pubmap.get('/joint_cmd', []))
    if len(main) != 1 or len(pubmap.get('/sensors_data_raw', [])) != 1 or len(pubmap.get('/imu', [])) != 1:
        raise ValueError('Need exactly one joint_cmd, sensors_data_raw and imu publisher')
    for topic in ('/kuavo_arm_traj', '/lb_leg_traj'):
        others = set(pubmap.get(topic, [])) - {rospy.get_name()}
        # The WBC advertises lb_leg_traj even when VR is idle. Actual foreign
        # messages are checked separately; all other registered publishers block.
        if topic == '/lb_leg_traj':
            others -= main
        if others:
            raise ValueError(f'Competing publishers on {topic}: {sorted(others)}')
        if not main.intersection(submap.get(topic, [])):
            raise ValueError(f'WBC subscriber missing on {topic}')
        if time.monotonic() - foreign.get(topic, (-100., ''))[0] < 2.:
            raise ValueError(f'Foreign command received on {topic}: {foreign[topic][1]}')

def graph_monitor():
    while not stop.is_set():
        began = time.monotonic()
        try:
            graph_guard()
            with lock:
                graph_status.update(checked=time.monotonic(), error=None)
            emit({'kind':'graph_audit', 'query_s':time.monotonic()-began})
        except Exception as exc:
            with lock:
                graph_status['error'] = str(exc)
            return
        stop.wait(.5)

def start_graph_monitor():
    # The synchronous preflight already succeeded. Subsequent XML-RPC queries
    # run on a daemon thread; a blocked query expires the cached result.
    graph_status.update(checked=time.monotonic(), error=None)
    threading.Thread(target=graph_monitor, daemon=True).start()

def graph_status_guard():
    log_guard()
    if graph_status['error']:
        raise ValueError(graph_status['error'])
    checked = graph_status['checked']
    if checked is None or time.monotonic()-checked > 1.5:
        raise ValueError('ROS graph audit stale >1.5 s')

def state_guard(plan, desired=None, require_quick=True, handover=False):
    stationary = None
    if version != 63 or config.get('NUM_JOINT') != 20 or config.get('NUM_ARM_JOINT') != 14 or config.get('NUM_HEAD_JOINT') != 2:
        raise ValueError('Expected S63 20 joints: body4, left7, right7, head2')
    now = time.monotonic()
    required = ('sensors', 'motor_command', 'orientation', 'filtered_state', 'wbc_state')
    if require_quick:
        required += ('body_target', 'arm_target')
    for key in required:
        if now - arrivals.get(key, 0) > .3:
            raise ValueError(f'Missing/stale {key}; quick body AND arm paths must already be active')
    for key in ('sensors', 'motor_command', 'orientation'):
        if stamps.get(key, 0) <= 0 or now - advanced.get(key, 0) > .3:
            raise ValueError(f'Frozen/invalid header time: {key}')
    if latest.get('enabled', {}).get('data') is not True:
        raise ValueError('enable_control_state must be true')
    if latest.get('lb_mode', {}).get('data') not in (1, 3):
        raise ValueError('lb MPC mode must be 1 or 3')
    if latest.get('arm_transport', {}).get('data') not in (0., 2.):
        raise ValueError('Arm transport must be ROS topic (0 or 2), not SHM')
    q = finite_vector(latest['sensors']['joint_q'], 20, 'measured q')[:18]
    v = finite_vector(latest['sensors']['joint_v'], 20, 'measured v')[:18]
    motor_q = finite_vector(latest['motor_command']['joint_q'], 20, 'motor q')[:18]
    downstream = {'measured': q, 'motor': motor_q}
    for key in ('filtered_state', 'wbc_state'):
        # Active S63 MPC state = base3 + body4 + arms14 (radians).
        downstream[key] = finite_vector(latest[key]['data'], 21, key)[-18:]
    if latest['motor_command']['control_modes'][:18] != [2]*18:
        raise ValueError('Expected motor control_modes=2 for body/arms')
    external = (finite_vector(latest['body_target']['data'], 4, 'body quick target') +
                finite_vector(latest['arm_target']['data'], 14, 'arm quick target')) if require_quick else q
    ref = plan['baseline_rad'] if desired is None else desired
    tracking_limit = min(request['tracking_deg'], .5) if handover else request['tracking_deg']
    for key, values in downstream.items():
        errors = [abs(a-b) for a,b in zip(values, ref)]
        i = max(range(18), key=errors.__getitem__)
        if errors[i] > math.radians(tracking_limit):
            raise ValueError(f'{key} joint tracking error: {NAMES[i]}, '
                             f'{math.degrees(errors[i]):.4f} deg (limit {tracking_limit})')
    if max(abs(x) for x in v) > math.radians(10):
        raise ValueError('Measured speed exceeds 10 deg/s')
    orientation = latest['orientation']
    captured_orientation = plan['capture']['observations']['orientation']
    quat = finite_vector(orientation['imu_quat_xyzw'],4,'IMU quaternion')
    reference_quat = finite_vector(captured_orientation['imu_quat_xyzw'],4,'capture IMU quaternion')
    if not orientation['frame_id'] or (orientation['frame_id'], orientation['publisher']) != (captured_orientation['frame_id'], captured_orientation['publisher']):
        raise ValueError('IMU frame/publisher changed; recapture')
    covariance = finite_vector(orientation['orientation_covariance'],9,'IMU covariance')
    if covariance[0] < 0:
        raise ValueError('IMU reports orientation unavailable')
    norm = math.sqrt(sum(x*x for x in quat))
    ref_norm = math.sqrt(sum(x*x for x in reference_quat))
    if not .9 <= norm <= 1.1 or not .9 <= ref_norm <= 1.1:
        raise ValueError('Invalid IMU quaternion norm')
    cosine = abs(sum(a*b for a,b in zip(quat,reference_quat))/(norm*ref_norm))
    if 2*math.acos(min(1.,cosine)) > math.radians(5):
        raise ValueError('IMU orientation changed >5 deg from capture')
    if desired is None:
        if max(abs(a-b) for a,b in zip(external, ref)) > math.radians(.25):
            raise ValueError('Baseline no longer matches live external targets; recapture')
        if not require_quick:
            if max(abs(a-b) for a,b in zip(motor_q, ref)) > math.radians(.5):
                raise ValueError('Current motor target differs >0.5 deg from prepare hold')
            stationary = stationarity_metrics(sensor_history, now)
        else:
            for key in ('motor', 'filtered_state', 'wbc_state'):
                if max(abs(a-b) for a,b in zip(downstream[key], ref)) > math.radians(.25):
                    raise ValueError(f'{key} has not reached quick baseline within 0.25 deg')
    # Intersect the local URDF range with the loaded hardware range (degrees).
    low = finite_vector(config.get('min_joint_position_limits'), 20, 'hardware lower')[:18]
    high = finite_vector(config.get('max_joint_position_limits'), 20, 'hardware upper')[:18]
    for phase in plan['phases']:
        for i, x in enumerate(phase['to_rad']):
            tol = BASELINE_LIMIT_TOLERANCE if x == plan['baseline_rad'][i] else 0.
            if not max(plan['lower_rad'][i]-tol, math.radians(low[i])) <= x <= min(plan['upper_rad'][i]+tol, math.radians(high[i])):
                raise ValueError(f'Hardware/URDF limit violation: {NAMES[i]}')
    return stationary

publishers = None
last_q = None
def publish(q, v, label):
    for topic, ids in (('/lb_leg_traj', range(4)), ('/kuavo_arm_traj', range(4,18))):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [NAMES[i] for i in ids]
        msg.position = [math.degrees(q[i]) for i in ids]
        msg.velocity = [math.degrees(v[i]) for i in ids]
        # Leave effort empty: no added torque, gain change, or second g(q).
        publishers[topic].publish(msg)
    emit({'kind':'target', 'phase':label, 'position_rad':q, 'velocity_rad_s':v,
          'publication_only':True})

try:
    summary_extra = {}
    # Gather status and detect actual competing body/VR commands before advertising.
    while time.monotonic()-start < 2.5 and not stop.is_set():
        log_guard()
        time.sleep(.025)
    log_guard()
    if stop.is_set():
        raise ValueError('Local connection stopped before preflight')
    if mode == 'capture':
        c = snapshot()
        finite_vector(c['baseline_rad'], 18, 'capture baseline')
        if version != 63 or config.get('NUM_JOINT') != 20 or config.get('NUM_ARM_JOINT') != 14 or config.get('NUM_HEAD_JOINT') != 2:
            raise ValueError('Expected S63 20-joint configuration')
        if time.monotonic()-arrivals.get('sensors',0) > .3:
            raise ValueError('Missing/stale measured state; no zero fallback')
        finite_vector(c['observations']['sensors']['joint_q'], 20, 'capture measured q')
        finite_vector(c['observations']['sensors']['joint_v'], 20, 'capture measured v')
        if time.monotonic()-arrivals.get('orientation',0) > .3:
            raise ValueError('Missing/stale /imu orientation')
        orientation = c['observations']['orientation']
        quat = finite_vector(orientation['imu_quat_xyzw'],4,'capture /imu quaternion')
        if not .9 <= math.sqrt(sum(x*x for x in quat)) <= 1.1 or orientation['orientation_covariance'][0] < 0 or not orientation['frame_id']:
            raise ValueError('Invalid /imu orientation')
        emit({'kind':'capture', 'capture':c})
    elif mode == 'record':
        while time.monotonic()-start < request['duration_s'] and not stop.is_set() and not rospy.is_shutdown():
            log_guard()
            if time.monotonic()-heartbeat[0] > 1.:
                raise ValueError('Local heartbeat lost')
            time.sleep(.025)
        if request.get('vr_inputs'):
            with lock:
                counts=dict(vr_counts)
                core={topics[k][0]:int(k in latest) for k in ('sensors','motor_command')}
            emit({'kind':'vr_record_coverage','sampled_input_counts':counts,
                  'core_topic_seen':core,'missing_input_topics':[t for t,_,_ in vr_specs if not counts.get(t)],
                  'note':'Coverage only, not verified motion, timing alignment or calibrated motor parameters'})
    else:
        plan = validate_plan(request['plan'])
        preparing = mode in ('prepare', 'prepare_check')
        if preparing and plan['kind'] != 'hold':
            raise ValueError('prepare accepts a hold plan only')
        if not preparing and plan['capture']['baseline_source'] != 'live_quick_mode_targets':
            raise ValueError('Measured-only capture is preview-only; prepare WBC then recapture')
        if time.time()-plan['capture']['captured_unix_s'] > 300 or time.time() < plan['capture']['captured_unix_s']-2:
            raise ValueError('Capture older than 5 min or clock invalid; recapture')
        graph_guard()
        with lock:
            stationary = state_guard(plan, require_quick=not preparing, handover=preparing)
        emit({'kind':'preflight', 'ok':True, 'duration_s':plan['duration_s'],
              'stationarity':stationary})
        if mode in ('check', 'prepare_check') and request.get('watch_s', 0) > 0:
            # Rehearse the same guard/sampling/logging load, without creating
            # command publishers or calling any mode service.
            start_graph_monitor()
            watch_start = time.monotonic()
            last_tick = watch_start
            ticks, max_gap, max_guard = 0, 0., 0.
            while time.monotonic()-watch_start < request['watch_s']:
                graph_status_guard()
                if stop.is_set() or time.monotonic()-heartbeat[0] > 1.:
                    raise ValueError('Interrupted during read-only timing check')
                with lock:
                    began = time.monotonic()
                    state_guard(plan, require_quick=not preparing, handover=preparing)
                    sampled_s = min(plan['duration_s'], (began-watch_start)/request['watch_s']*plan['duration_s'])
                    _,_,label = sample(plan, sampled_s)
                    checked = time.monotonic()
                    gap = checked-last_tick
                    if gap > .2:
                        raise ValueError('Read-only timing loop stalled >0.2 s')
                    max_gap = max(max_gap, gap)
                    max_guard = max(max_guard, checked-began)
                    emit({'kind':'timing_tick', 'phase':label, 'published':False,
                          'interval_s':gap, 'guard_s':checked-began,
                          'queue_depth':log_queue.qsize()})
                    last_tick = checked
                    ticks += 1
                time.sleep(.02)
            summary_extra = {'read_only':True, 'watch_ticks':ticks,
                             'max_control_gap_s':max_gap, 'max_guard_s':max_guard}
        if mode in ('send', 'prepare'):
            publishers = {t:rospy.Publisher(t, JointState, queue_size=1, latch=False)
                          for t in ('/lb_leg_traj','/kuavo_arm_traj')}
            deadline = time.monotonic()+2
            while any(p.get_num_connections() == 0 for p in publishers.values()):
                if time.monotonic() > deadline or stop.is_set():
                    raise ValueError('Command subscribers did not connect')
                time.sleep(.01)
            start_graph_monitor()
            if preparing:
                from kuavo_msgs.srv import changeLbQuickModeSrv
                # BOTH topics also reach MPC: these are live commands, not
                # passive buffer writes. Bound handover at 0.5 deg. Stationarity
                # applies before the first command; later ticks check tracking.
                emit({'kind':'handover', 'max_deviation_deg':min(request['tracking_deg'], .5),
                      'note':'Body/arm commands also update MPC references before quick3'})
                seed_start = time.monotonic()
                while time.monotonic()-seed_start < 1.:
                    graph_status_guard()
                    if stop.is_set() or time.monotonic()-heartbeat[0] > 1.:
                        raise ValueError('Interrupted before quick mode service')
                    with lock:
                        state_guard(plan, last_q, require_quick=False, handover=True)
                        for t in ('/lb_leg_traj','/kuavo_arm_traj'):
                            if time.monotonic()-foreign.get(t,(-100.,''))[0] < 2:
                                raise ValueError('Foreign command during prepare')
                        publish(plan['baseline_rad'],[0.]*18,'prepare_seed_both_buffers')
                        last_q = plan['baseline_rad']
                    time.sleep(.02)
                rospy.wait_for_service('/enable_lb_arm_quick_mode', timeout=2.)
                response = rospy.ServiceProxy('/enable_lb_arm_quick_mode',changeLbQuickModeSrv)(3)
                emit({'kind':'mode_service', 'quick_mode':3, 'success':response.success,
                      'message':response.message})
                if not response.success:
                    raise ValueError('WBC rejected quick3')
                # No gain or interpolation switch. If the existing interpolator
                # bypasses arm quick targets, this verification fails visibly.
                deadline = time.monotonic()+1.
                while time.monotonic() < deadline:
                    graph_status_guard()
                    if stop.is_set() or time.monotonic()-heartbeat[0] > 1.:
                        raise ValueError('Interrupted during prepare verification')
                    with lock:
                        state_guard(plan, last_q, require_quick=False, handover=True)
                        for t in ('/lb_leg_traj','/kuavo_arm_traj'):
                            if time.monotonic()-foreign.get(t,(-100.,''))[0] < 2:
                                raise ValueError('Foreign command during prepare verification')
                        publish(plan['baseline_rad'],[0.]*18,'prepare_verify')
                    time.sleep(.02)
                graph_status_guard()
                with lock:
                    state_guard(plan, handover=True)
            motion_start = time.monotonic()
            last_tick = motion_start
            while not rospy.is_shutdown():
                graph_status_guard()
                if stop.is_set() or time.monotonic()-heartbeat[0] > 1.:
                    raise ValueError('Interrupted/local connection lost; sequence aborted')
                elapsed = time.monotonic()-motion_start
                if time.monotonic()-last_tick > .2:
                    raise ValueError('Control loop stalled >0.2 s')
                with lock:
                    state_guard(plan, last_q, handover=preparing)
                    for t in ('/lb_leg_traj','/kuavo_arm_traj'):
                        if time.monotonic()-foreign.get(t,(-100.,''))[0] < 2:
                            raise ValueError('Foreign command during sequence')
                    # Check again after acquiring the state lock and running
                    # guards: never publish a new trajectory point after a stall.
                    if time.monotonic()-last_tick > .2:
                        raise ValueError('Control loop stalled >0.2 s before publication')
                    elapsed = time.monotonic()-motion_start
                    q,v,label = sample(plan, elapsed)
                    publish(q,v,label)
                    last_q = q
                    last_tick = time.monotonic()
                if elapsed >= plan['duration_s']:
                    break
                time.sleep(.02)
    log_guard()
    emit({'kind':'summary', 'ok':True, 'elapsed_s':time.monotonic()-start, **summary_extra})
except Exception as exc:
    if publishers and last_q is not None:
        # Hold the last sent target with zero target velocity; no return sweep.
        # This is not a hardware E-stop and cannot guarantee hold after a failure.
        try:
            publish(last_q, [0.]*18, 'abort_last_target')
            time.sleep(.05)
        except Exception:
            pass
    emit({'kind':'error', 'ok':False, 'error':str(exc),
          'stop_requested':stop.is_set(), 'stop_reason':stop_reason[0],
          'heartbeat_age_s':time.monotonic()-heartbeat[0],
          'heartbeat_state':dict(heartbeat_state),
          'logging_error':log_error[0]})
    sys.exit(1)
finally:
    stop.set()
    rospy.signal_shutdown('calibration worker finished')
    if not finish_logging():
        sys.exit(1)
'''
