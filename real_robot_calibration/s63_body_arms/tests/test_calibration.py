"""No ROS/SSH/motor I/O: verify bounds, ownership, stale feedback and publication."""
import ast
import contextlib
import copy
import io
import json
import math
import queue
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
from collections import deque
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import calibrate
import trajectory as t
from remote import WORKER


def capture():
    q = [.2, -.4, .2, 0.] + [.2, .1, 0., -.5, 0., 0., 0.] + [.2, -.1, 0., -.5, 0., 0., 0.]
    return {"robot_version":63, "joint_names":t.NAMES, "baseline_rad":q,
            "baseline_source":"live_quick_mode_targets", "captured_unix_s":100.,
            "observations":{"sensors":{"imu_quat_xyzw":[0.,0.,0.,0.]},
                            "orientation":{"imu_quat_xyzw":[0.,0.,0.,1.],"frame_id":"imu_link",
                                           "publisher":"/imu_node","orientation_covariance":[0.]*9}}}


def functions_from_worker(*names, **environment):
    tree = ast.parse(WORKER)
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(environment)
    exec(compile(ast.Module(body=defs, type_ignores=[]), "worker_functions", "exec"), scope)
    return scope


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.c = capture()
        self.limits = calibrate.load_limits(calibrate.DEFAULT_URDF)

    def plan(self, **kwargs):
        return t.make_plan(self.c, self.limits, **kwargs)

    def test_hold_is_exactly_current_all_body_arms(self):
        p = self.plan()
        for elapsed in (0., 2.5, 5., 10.):
            q,v,_ = t.sample(p, elapsed)
            self.assertEqual(q,self.c['baseline_rad'])
            self.assertEqual(v,[0.]*18)

    def test_small_motion_one_joint_at_time_and_returns(self):
        p = self.plan(kind='motion',joints=['waist_pitch_joint','zarm_l4_joint','zarm_r4_joint'])
        self.assertEqual(p['max_velocity_deg_s'], .46875)
        self.assertEqual(p['duration_s'],257.)
        for i in range(int(p['duration_s']*50)+1):
            q,v,_ = t.sample(p,i/50)
            delta = [abs(a-b) for a,b in zip(q,self.c['baseline_rad'])]
            self.assertLessEqual(sum(x>1e-12 for x in delta),1)
            self.assertLessEqual(max(delta),math.radians(.5)+1e-12)
            self.assertLessEqual(max(abs(x) for x in v),math.radians(.46875)+1e-12)
        self.assertEqual(t.sample(p,p['duration_s'])[0],self.c['baseline_rad'])
        self.assertEqual(t.sample(p,p['duration_s'])[1],[0.]*18)

    def test_ramp_endpoints_zero_velocity(self):
        p = self.plan(kind='static',joints=['zarm_l4_joint'])
        self.assertEqual(t.sample(p,5)[1],[0.]*18)
        self.assertEqual(t.sample(p,7)[1],[0.]*18)
        self.assertAlmostEqual(t.sample(p,7)[0][7],self.c['baseline_rad'][7]+math.radians(.5))

    def test_sampling_independent_of_control_tick_rate(self):
        p = self.plan(kind='motion',joints=['waist_yaw_joint'])
        expected = t.sample(p,6.)
        for i in range(600):
            t.sample(p,i/100)
        self.assertEqual(t.sample(p,6.),expected)

    def test_reject_nonfinite_or_unbounded_motion(self):
        for args in ({'amplitude':math.nan},{'amplitude':1.01},{'amplitude':0.},
                     {'ramp':1.},{'hold':math.inf},{'cycles':6},
                     {'kind':'motion'}, {'kind':'motion','joints':['waist_pitch_joint'],'amplitude':.6},
                     {'kind':'motion','joints':['unknown']},{'kind':'hold','joints':['waist_yaw_joint']}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                self.plan(**args)

    def test_reject_bad_baseline_and_limits_without_clipping(self):
        for value in ([],[0.]*17,[math.nan]*18,[True]*18,[100.]*18):
            self.c['baseline_rad']=value
            with self.assertRaises(ValueError):self.plan()
        self.c=capture();self.c['baseline_rad'][7]=-.001
        with self.assertRaises(ValueError):self.plan(kind='motion',joints=['zarm_l4_joint'])

    def test_plan_tampering_rejected(self):
        p=self.plan(kind='static',joints=['waist_yaw_joint'])
        self.assertEqual(t.validate_plan(p),p)
        p['phases'][1]['to_rad'][0]=100.
        with self.assertRaises(ValueError):t.validate_plan(p)

    def test_tiny_resting_limit_noise_preserved_but_moving_limits_stay_strict(self):
        self.c['baseline_rad'][0]=math.radians(-.0006)
        p=self.plan(kind='static',joints=['zarm_l4_joint'],amplitude=.25)
        self.assertEqual(t.sample(p,8)[0][0],self.c['baseline_rad'][0])
        with self.assertRaises(ValueError):self.plan(kind='static',joints=['knee_joint'],amplitude=.25)
        self.c['baseline_rad'][0]=math.radians(-.011)
        with self.assertRaises(ValueError):self.plan()

    def test_preview_and_prepare_preview_never_connect(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hold.json';p.write_text(json.dumps(self.plan()))
            for command in ('run','prepare'):
                with patch.object(calibrate.subprocess,'Popen') as remote,contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(calibrate.main([command,'--plan',str(p)]),0)
                    remote.assert_not_called()

    def test_send_requires_log_and_prepare_requires_hold(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'moving.json';p.write_text(json.dumps(self.plan(kind='static',joints=['waist_yaw_joint'])))
            for args in (['run','--plan',str(p),'--send'],['prepare','--plan',str(p),'--send','--log',str(Path(d)/'log')]):
                with patch.object(calibrate.subprocess,'Popen') as remote,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(calibrate.main(args),1)
                    remote.assert_not_called()

    def test_preparation_check_remains_read_only_request(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hold.json';p.write_text(json.dumps(self.plan()))
            with patch.object(calibrate,'run_remote') as remote,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(['check','--for-prepare','--plan',str(p),
                                                 '--log',str(Path(d)/'check.jsonl')]),0)
                self.assertEqual(remote.call_args.args[1]['mode'],'prepare_check')

    def test_watch_check_is_read_only_and_duration_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hold.json';p.write_text(json.dumps(self.plan()))
            with patch.object(calibrate,'run_remote') as remote,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(['check','--plan',str(p),'--watch-s','15',
                                                 '--log',str(Path(d)/'watch.jsonl')]),0)
                self.assertEqual(remote.call_args.args[1]['mode'],'check')
                self.assertEqual(remote.call_args.args[1]['watch_s'],15.)
            for duration in ('nan','0.5','31'):
                with patch.object(calibrate,'run_remote') as remote,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(calibrate.main(['check','--plan',str(p),'--watch-s',duration,
                                                     '--log',str(Path(d)/'watch.jsonl')]),1)
                    remote.assert_not_called()

    def test_capture_replaces_existing_json_and_capture_log(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'ready.json';output.write_text('old capture')
            log=output.with_suffix('.jsonl');log.write_text('old log')
            worker_rows=[{'kind':'capture','capture':capture()},{'kind':'summary','ok':True}]
            process=Mock()
            process.stdin=io.StringIO()
            process.stdout=io.StringIO(''.join(json.dumps(r)+'\n' for r in worker_rows))
            process.wait.return_value=0
            with patch.object(calibrate.subprocess,'Popen',return_value=process),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(['capture','--output',str(output)]),0)
            self.assertEqual(json.loads(output.read_text()),capture())
            self.assertNotIn('old log',log.read_text())
            self.assertEqual(list(Path(d).glob('*.tmp')),[])

    def test_failed_recapture_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'ready.json';output.write_text('old capture')
            with (patch.object(calibrate,'run_remote',side_effect=RuntimeError('SSH failed')),
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(calibrate.main(['capture','--output',str(output)]),1)
            self.assertEqual(output.read_text(),'old capture')

    def test_initial_pipe_failure_is_logged_cleaned_up_and_never_retried(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(workspace='/workspace',ros_master='http://localhost:11311',host='lab@robot')
            process=Mock()
            process.stdin.write.side_effect=BrokenPipeError('SSH disconnected')
            with patch.object(calibrate.subprocess,'Popen',return_value=process) as popen:
                with self.assertRaises(BrokenPipeError):
                    calibrate.run_remote(args,{'mode':'capture'},Path(d)/'attempt.jsonl')
                self.assertEqual(popen.call_count,1)
                process.stdin.close.assert_called_once()
                process.stdout.close.assert_called_once()
                process.wait.assert_called()
            self.assertIn('local_attempt',(Path(d)/'attempt.jsonl').read_text())

    def test_raw_telemetry_is_preserved_without_local_json_decoding(self):
        raw=json.dumps({'kind':'message','joint_q':[0.]*20})+'\n'
        lines=raw*20+json.dumps({'kind':'summary','ok':True})+'\n'
        process=Mock();process.stdin=io.StringIO();process.stdout=io.StringIO(lines)
        process.wait.return_value=0
        args=SimpleNamespace(workspace='/workspace',ros_master='http://localhost:11311',host='lab@robot')
        original=json.loads
        def decode(value):
            self.assertFalse(value.startswith('{"kind": "message",'))
            return original(value)
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'record.jsonl'
            with patch.object(calibrate.subprocess,'Popen',return_value=process),patch.object(calibrate.json,'loads',side_effect=decode),contextlib.redirect_stdout(io.StringIO()):
                calibrate.run_remote(args,{'mode':'record'},log)
            data=log.read_text()
            self.assertEqual(data.count(raw),20)
            summary=original(data.splitlines()[-1])
            self.assertEqual(summary['kind'],'local_heartbeat_summary')
            self.assertGreaterEqual(summary['sent_count'],1)


class WorkerGuardTests(unittest.TestCase):
    def setUp(self):
        self.plan=t.make_plan(capture(),calibrate.load_limits(calibrate.DEFAULT_URDF))
        q=self.plan['baseline_rad']+[0.,0.]
        self.latest={'sensors':{'joint_q':q,'joint_v':[0.]*20,'imu_quat_xyzw':[0.,0.,0.,1.]},
                     'orientation':copy.deepcopy(capture()['observations']['orientation']),
                     'motor_command':{'joint_q':q[:],'control_modes':[2]*20},
                     'filtered_state':{'data':[0.]*3+q[:18]},
                     'wbc_state':{'data':[0.]*3+q[:18]},
                     'enabled':{'data':True},'lb_mode':{'data':3},'arm_transport':{'data':2.},
                     'body_target':{'data':q[:4]},'arm_target':{'data':q[4:18]}}
        self.arrivals={k:100. for k in self.latest}
        config={'NUM_JOINT':20,'NUM_ARM_JOINT':14,'NUM_HEAD_JOINT':2,
                'min_joint_position_limits':[-180.]*20,'max_joint_position_limits':[180.]*20}
        self.scope=functions_from_worker('state_guard',math=math,NAMES=t.NAMES,
                                         BASELINE_LIMIT_TOLERANCE=t.BASELINE_LIMIT_TOLERANCE,
                                         stationarity_metrics=t.stationarity_metrics,
                                         sensor_history=[(98.+i/50,q[:18],[0.]*18) for i in range(101)],
                                         finite_vector=t.finite_vector,version=63,config=config,
                                         time=SimpleNamespace(monotonic=lambda:100.),latest=self.latest,
                                         arrivals=self.arrivals,stamps={'sensors':100.,'motor_command':100.,'orientation':100.},
                                         advanced={'sensors':100.,'motor_command':100.,'orientation':100.},request={'tracking_deg':3.})

    def test_valid_state_and_stale_feedback(self):
        self.scope['state_guard'](self.plan)
        self.arrivals['sensors']=99.
        with self.assertRaisesRegex(ValueError,'stale'):self.scope['state_guard'](self.plan)

    def test_frozen_header_and_shm_mode_rejected(self):
        self.scope['advanced']['motor_command']=99.
        with self.assertRaisesRegex(ValueError,'Frozen'):self.scope['state_guard'](self.plan)
        self.scope['advanced']['motor_command']=100.
        self.latest['arm_transport']['data']=1.
        with self.assertRaisesRegex(ValueError,'SHM'):self.scope['state_guard'](self.plan)

    def test_pose_changed_or_tracking_error_rejected(self):
        self.latest['body_target']['data'][0]+=.01
        with self.assertRaisesRegex(ValueError,'recapture'):self.scope['state_guard'](self.plan)
        self.latest['body_target']['data'][0]-=.01
        self.latest['sensors']['joint_q'][4]+=.1
        with self.assertRaisesRegex(ValueError,'tracking'):self.scope['state_guard'](self.plan)

    def test_preparation_does_not_require_quick_but_requires_stationary(self):
        self.arrivals.pop('body_target');self.arrivals.pop('arm_target')
        self.scope['state_guard'](self.plan,require_quick=False)
        self.scope['sensor_history']=[(98.+i/50,self.latest['sensors']['joint_q'][:18],
                                       [math.radians(1.)]+[0.]*17) for i in range(101)]
        with self.assertRaisesRegex(ValueError,'stationary'):self.scope['state_guard'](self.plan,require_quick=False)

    def test_imu_orientation_changed_rejected(self):
        self.latest['orientation']['imu_quat_xyzw']=[0.,math.sin(math.radians(3)),0.,math.cos(math.radians(3))]
        with self.assertRaisesRegex(ValueError,'IMU orientation'):self.scope['state_guard'](self.plan)

    def test_uses_separate_imu_and_rejects_missing_or_changed_source(self):
        self.latest['sensors']['imu_quat_xyzw']=[0.]*4
        self.scope['state_guard'](self.plan)
        self.arrivals['orientation']=99.
        with self.assertRaisesRegex(ValueError,'stale'):self.scope['state_guard'](self.plan)
        self.arrivals['orientation']=100.
        self.latest['orientation']['publisher']='/other_imu'
        with self.assertRaisesRegex(ValueError,'publisher changed'):self.scope['state_guard'](self.plan)

    def test_hardware_limits_intersect_urdf(self):
        self.scope['config']['max_joint_position_limits'][0]=1.
        with self.assertRaisesRegex(ValueError,'limit violation'):self.scope['state_guard'](self.plan)

    def test_competing_publishers_and_actual_wbc_vr_message(self):
        state=[['/joint_cmd',['/wbc']],['/sensors_data_raw',['/hw']],['/lb_leg_traj',['/wbc']],['/imu',['/imu_node']]]
        master=SimpleNamespace(getSystemState=lambda:(1,'',[state,[['/lb_leg_traj',['/wbc']],['/kuavo_arm_traj',['/wbc']]],[]]))
        foreign={}
        scope=functions_from_worker('graph_guard',rospy=SimpleNamespace(get_master=lambda:master,get_name=lambda:'/cal'),
                                    time=SimpleNamespace(monotonic=lambda:100.),foreign=foreign)
        scope['graph_guard']()
        state.append(['/kuavo_arm_traj',['/ik']])
        with self.assertRaisesRegex(ValueError,'Competing'):scope['graph_guard']()
        state.pop();foreign['/lb_leg_traj']=(99.,'/wbc')
        with self.assertRaisesRegex(ValueError,'Foreign'):scope['graph_guard']()

    def test_publication_degrees_and_no_effort_gain_or_base_command(self):
        class JointState:
            def __init__(self):self.header=SimpleNamespace(stamp=None);self.effort=[]
        pubs={k:Mock() for k in ('/lb_leg_traj','/kuavo_arm_traj')}
        scope=functions_from_worker('publish',JointState=JointState,NAMES=t.NAMES,math=math,
                                    rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:100)),publishers=pubs,emit=Mock())
        scope['publish']([math.pi/180]*18,[0.]*18,'hold')
        for topic,count in (('/lb_leg_traj',4),('/kuavo_arm_traj',14)):
            msg=pubs[topic].publish.call_args.args[0]
            self.assertEqual(msg.position,[1.]*count)
            self.assertEqual(msg.velocity,[0.]*count)
            self.assertEqual(msg.effort,[])
            self.assertEqual(len(msg.name),count)

    def test_prepare_lifecycle_seeds_before_switch_and_tolerates_fresh_callback_delay(self):
        rows, service_calls = self.prepare_lifecycle()
        self.assertEqual(service_calls,[3])
        self.assertEqual(rows[-1]['kind'],'summary')
        self.assertTrue(rows[-1]['ok'])

    def test_prepare_settling_after_first_command_is_not_idle_motion(self):
        # Failure log: q was stationary before seed, then knee moved 0.14 deg.
        # MPC also changed motor q by 0.16 deg before the quick3 service.
        rows, service_calls = self.prepare_lifecycle(settling=.14)
        self.assertEqual(service_calls,[3])
        self.assertEqual(rows[-1]['kind'],'summary')
        self.assertTrue(rows[-1]['ok'])

    def test_prepare_transition_drift_aborts_before_mode_service(self):
        rows, service_calls = self.prepare_lifecycle(settling=.6, expect_failure=True)
        self.assertEqual(service_calls,[])
        self.assertEqual(rows[-1]['kind'],'error')
        self.assertIn('knee_joint',rows[-1]['error'])
        self.assertIn('limit 0.5',rows[-1]['error'])

    def test_quick_diagnostic_alone_does_not_prove_wbc_received_baseline(self):
        rows, service_calls = self.prepare_lifecycle(bypass=.3, expect_failure=True)
        self.assertEqual(service_calls,[3])
        self.assertEqual(rows[-1]['kind'],'error')
        self.assertIn('quick baseline',rows[-1]['error'])

    def test_timing_rehearsal_never_creates_publishers_or_calls_services(self):
        rows, services = self.prepare_lifecycle(mode='check',watch_s=3.)
        self.assertEqual(services,[])
        self.assertFalse(any(r['kind']=='target' for r in rows))
        ticks=[r for r in rows if r['kind']=='timing_tick']
        self.assertGreater(len(ticks),100)
        self.assertTrue(all(r['published'] is False for r in ticks))
        self.assertTrue(rows[-1]['read_only'])

    def test_stall_during_guards_aborts_before_another_trajectory_point(self):
        rows, services = self.prepare_lifecycle(mode='send',stall_guard_s=.21,expect_failure=True)
        self.assertEqual(services,[])
        self.assertIn('before publication',rows[-1]['error'])
        targets=[r for r in rows if r['kind']=='target']
        self.assertEqual([r['phase'] for r in targets],['baseline_hold','abort_last_target'])

    def test_full_fast_batch_lifecycle_queries_master_only_before_publication(self):
        joints=[n for n in t.NAMES[4:] if n!='zarm_l4_joint']
        rows, services = self.prepare_lifecycle(mode='send',joints=joints)
        self.assertEqual(services,[])
        self.assertTrue(rows[-1]['ok'])
        self.assertTrue(any(r.get('phase')=='zarm_r7_joint/1/-1/hold' for r in rows))

    def prepare_lifecycle(self, settling=0., bypass=0., expect_failure=False,
                          mode='prepare',watch_s=0.,stall_guard_s=0.,joints=None):
        # Execute the real worker lifecycle with fake ROS only. Master calls
        # delay callbacks by 60 ms, reproducing the post-preflight failure.
        clock=SimpleNamespace(now=100.)
        pubs={};switched=[mode!='prepare'];service_calls=[];rows=[];master_calls=[]
        if joints:
            self.plan=t.make_plan(capture(),calibrate.load_limits(calibrate.DEFAULT_URDF),
                                  kind='static',joints=joints,amplitude=.25,ramp=2,hold=2)
        scope=dict(self.scope)
        scope.update(mode=mode,request={'mode':mode,'plan':self.plan,'tracking_deg':3.,'watch_s':watch_s},
                     validate_plan=t.validate_plan,sample=t.sample,
                     start=100.,heartbeat=[100.],stop=SimpleNamespace(is_set=lambda:False,set=lambda:None),
                     lock=threading.RLock(),sensor_history=deque(self.scope['sensor_history']),
                     publishers=None,last_q=None,json=json,sys=sys,
                     log_guard=lambda:None,finish_logging=lambda:True,
                     enqueue_log=lambda row:rows.append(row),graph_status={'checked':100.,'error':None},
                     log_queue=SimpleNamespace(qsize=lambda:0),stop_reason=[None],log_error=[None],
                     heartbeat_state={'received_count':1,'last_sequence':1})
        if mode=='prepare':self.plan['capture']['baseline_source']='measured_only'
        command_start=[None]

        def sleep(delay):
            clock.now+=delay
            scope['heartbeat'][0]=clock.now
            scope['graph_status']['checked']=clock.now
            keys=('sensors','motor_command','orientation','filtered_state','wbc_state')
            if switched[0]:keys+=('body_target','arm_target')
            for key in keys:
                scope['arrivals'][key]=clock.now
                scope['advanced'][key]=clock.now
                scope['stamps'][key]=clock.now
            if command_start[0] is not None:
                offset=settling*min(1.,(clock.now-command_start[0])/.3)
                measured=self.plan['baseline_rad'][:]+[0.,0.]
                measured[0]+=math.radians(offset)
                scope['latest']['sensors']['joint_q']=measured
                goal=self.plan['baseline_rad'][:]+[0.,0.]
                goal[0]+=math.radians(bypass if switched[0] else offset*8/7)
                scope['latest']['motor_command']['joint_q']=goal
                for key in ('filtered_state','wbc_state'):
                    scope['latest'][key]={'data':[0.]*3+goal[:18]}
            scope['sensor_history'].append((clock.now,self.latest['sensors']['joint_q'][:18],[0.]*18))
            while scope['sensor_history'] and clock.now-scope['sensor_history'][0][0]>2.:
                scope['sensor_history'].popleft()
            if switched[0] and pubs and all(p.last is not None for p in pubs.values()):
                for key,topic in (('body_target','/lb_leg_traj'),('arm_target','/kuavo_arm_traj')):
                    scope['latest'][key]={'data':[math.radians(x) for x in pubs[topic].last.position]}
                    scope['arrivals'][key]=clock.now

        class Publisher:
            def __init__(self,topic,*args,**kwargs):self.last=None;pubs[topic]=self
            def get_num_connections(self):sleep(.01);return 1
            def publish(self,msg):
                self.last=msg
                if command_start[0] is None:command_start[0]=clock.now

        class JointState:
            def __init__(self):self.header=SimpleNamespace(stamp=None);self.effort=[]

        def system_state():
            master_calls.append(clock.now)
            clock.now+=.06  # no callback during this short scheduling gap
            published=[['/joint_cmd',['/wbc']],['/sensors_data_raw',['/hw']],['/imu',['/imu_node']],
                       ['/lb_leg_traj',['/wbc']+(['/cal'] if '/lb_leg_traj' in pubs else [])]]
            if '/kuavo_arm_traj' in pubs:published.append(['/kuavo_arm_traj',['/cal']])
            return (1,'',[published,[['/lb_leg_traj',['/wbc']],['/kuavo_arm_traj',['/wbc']]],[]])

        def switch_mode(value):
            self.assertEqual(value,3)
            # Both buffers must have been seeded with the actual baseline.
            seeded=pubs['/lb_leg_traj'].last.position+pubs['/kuavo_arm_traj'].last.position
            for actual,expected in zip(seeded,self.plan['baseline_rad']):
                self.assertAlmostEqual(math.radians(actual),expected)
            service_calls.append(value);switched[0]=True
            return SimpleNamespace(success=True,message='mock quick3')

        def service_proxy(topic,service_type):
            self.assertEqual(topic,'/enable_lb_arm_quick_mode')
            return switch_mode

        scope.update(time=SimpleNamespace(monotonic=lambda:clock.now,time=lambda:clock.now,sleep=sleep),
                     foreign={},JointState=JointState,
                     start_graph_monitor=lambda:scope['graph_status'].update(checked=clock.now,error=None),
                     print=lambda value,**kwargs:rows.append(json.loads(value)),
                     rospy=SimpleNamespace(get_master=lambda:SimpleNamespace(getSystemState=system_state),
                                           get_name=lambda:'/cal',Publisher=Publisher,
                                           Time=SimpleNamespace(now=lambda:clock.now),
                                           is_shutdown=lambda:False,signal_shutdown=lambda why:None,
                                           wait_for_service=lambda *args,**kwargs:None,ServiceProxy=service_proxy))
        worker=functions_from_worker('state_guard','graph_guard','graph_status_guard','publish','emit',**scope)
        if stall_guard_s:
            real_guard=worker['state_guard'];stalled=[False]
            def delayed_guard(*args,**kwargs):
                result=real_guard(*args,**kwargs)
                if command_start[0] is not None and not stalled[0]:
                    clock.now+=stall_guard_s;stalled[0]=True
                return result
            worker['state_guard']=delayed_guard
        package=ModuleType('kuavo_msgs');package.__path__=[]
        srv=ModuleType('kuavo_msgs.srv');srv.changeLbQuickModeSrv=object();package.srv=srv
        main=next(n for n in ast.parse(WORKER).body if isinstance(n,ast.Try))
        with patch.dict(sys.modules,{'kuavo_msgs':package,'kuavo_msgs.srv':srv}):
            if expect_failure:
                with self.assertRaises(SystemExit):
                    exec(compile(ast.Module(body=[main],type_ignores=[]),'mock_prepare_worker','exec'),worker)
            else:
                exec(compile(ast.Module(body=[main],type_ignores=[]),'mock_prepare_worker','exec'),worker)
        self.assertEqual(len(master_calls),1,'Synchronous graph query ran after publication began')
        if mode=='check':self.assertEqual(pubs,{})
        return rows,service_calls

    def test_missing_wrong_size_and_divergent_downstream_targets_rejected(self):
        self.arrivals['wbc_state']=99.
        with self.assertRaisesRegex(ValueError,'stale wbc_state'):
            self.scope['state_guard'](self.plan)
        self.arrivals['wbc_state']=100.
        self.latest['wbc_state']['data']=[0.]*18
        with self.assertRaisesRegex(ValueError,'wbc_state'):
            self.scope['state_guard'](self.plan)
        self.latest['wbc_state']['data']=[0.]*3+self.plan['baseline_rad'][:]
        self.latest['motor_command']['joint_q'][0]+=math.radians(.6)
        with self.assertRaisesRegex(ValueError,'motor joint tracking error.*knee_joint'):
            self.scope['state_guard'](self.plan,self.plan['baseline_rad'],require_quick=False,handover=True)


class WorkerConcurrencyTests(unittest.TestCase):
    def test_numbered_heartbeat_receipts_are_recorded(self):
        state={'received_count':0,'last_sequence':None};emit=Mock()
        scope=functions_from_worker('request_stop','stdin_watchdog',stop=threading.Event(),
            stop_reason=[None],heartbeat=[99.],heartbeat_state=state,emit=emit,
            sys=SimpleNamespace(stdin=io.StringIO('heartbeat 7\nheartbeat 8\n')),
            time=SimpleNamespace(monotonic=lambda:100.))
        scope['stdin_watchdog']()
        self.assertEqual(state,{'received_count':2,'last_sequence':8})
        self.assertEqual([c.args[0]['sequence'] for c in emit.call_args_list],[7,8])

    def test_stdin_disconnect_reason_distinguished_from_heartbeat_age(self):
        stop=threading.Event();reason=[None];heartbeat=[99.]
        scope=functions_from_worker('request_stop','stdin_watchdog',
            stop=stop,stop_reason=reason,heartbeat=heartbeat,
            heartbeat_state={'received_count':0,'last_sequence':None},emit=Mock(),
            sys=SimpleNamespace(stdin=io.StringIO('heartbeat\n')),
            time=SimpleNamespace(monotonic=lambda:100.))
        scope['stdin_watchdog']()
        self.assertTrue(stop.is_set())
        self.assertEqual(reason[0],'Local stdin EOF')
        self.assertEqual(heartbeat[0],100.)
        scope['request_stop']('Remote SIGTERM')
        self.assertEqual(reason[0],'Local stdin EOF')

    def logging_scope(self, stdout=None, clock=None, size=4096):
        env=dict(queue=queue,json=json,time=clock or time,threading=threading,
                 sys=SimpleNamespace(stdout=stdout or io.StringIO()),
                 log_queue=queue.Queue(maxsize=size),log_error=[None],
                 log_closed=threading.Event(),stop=threading.Event(),
                 lock=threading.RLock(),latest={},arrivals={},advanced={},stamps={},
                 sensor_history=deque(maxlen=2000),logged_at={},mode='send',request={},
                 topics={'sensors':('/sensors_data_raw',None)})
        return functions_from_worker('enqueue_log','log_guard','log_writer',
                                     'finish_logging','emit','receive',**env)

    def sensor(self, position=0.):
        return SimpleNamespace(_connection_header={'callerid':'/hw'},
            joint_data=SimpleNamespace(joint_q=[position]*20,joint_v=[0.]*20,
                                       joint_vd=[0.]*20,joint_torque=[0.]*20),
            imu_data=SimpleNamespace(quat=SimpleNamespace(x=0.,y=0.,z=0.,w=1.),
                                     gyro=SimpleNamespace(x=0.,y=0.,z=0.)))

    def test_blocked_stdout_does_not_block_feedback_or_state_lock(self):
        entered=threading.Event();release=threading.Event();received=threading.Event()
        output=io.StringIO()
        class SlowSink:
            def write(self,value):
                entered.set();release.wait(2.);output.write(value)
            def flush(self):pass
        scope=self.logging_scope(stdout=SlowSink())
        scope['log_thread']=threading.Thread(target=scope['log_writer'],daemon=True)
        scope['log_thread'].start();scope['emit']({'kind':'test'})
        self.assertTrue(entered.wait(1.))
        def feedback():
            for i in range(100):scope['receive'](self.sensor(i/10000),'sensors')
            received.set()
        callback=threading.Thread(target=feedback,daemon=True);callback.start()
        try:
            self.assertTrue(received.wait(1.),'Feedback waited for stdout writer')
            self.assertFalse(release.is_set())
            self.assertEqual(len(scope['sensor_history']),100)
            self.assertEqual(scope['latest']['sensors']['joint_q'][0],.0099)
            self.assertTrue(scope['lock'].acquire(timeout=.1))
            scope['lock'].release()
        finally:
            release.set();callback.join(1.)
            self.assertTrue(scope['finish_logging']())
        self.assertTrue(output.getvalue().endswith('\n'))

    def test_full_rate_feedback_retained_while_telemetry_is_sampled(self):
        clock=SimpleNamespace(now=0.)
        clock.monotonic=lambda:clock.now;clock.time=lambda:clock.now
        scope=self.logging_scope(clock=clock)
        for i in range(1001):
            clock.now=i*.002;scope['receive'](self.sensor(i/10000),'sensors')
        self.assertEqual(len(scope['sensor_history']),1001)
        self.assertEqual(scope['latest']['sensors']['joint_q'][0],.1)
        self.assertLessEqual(scope['log_queue'].qsize(),201)
        self.assertGreater(scope['log_queue'].qsize(),150)

    def test_queue_overflow_aborts_instead_of_silent_data_loss(self):
        scope=self.logging_scope(size=1)
        scope['emit']({'kind':'first'});scope['emit']({'kind':'overflow'})
        self.assertTrue(scope['stop'].is_set())
        with self.assertRaisesRegex(ValueError,'queue full'):scope['log_guard']()

    def test_batch_writer_preserves_order_and_final_summary(self):
        scope=self.logging_scope()
        for i in range(257):scope['enqueue_log']({'kind':'target','index':i})
        scope['enqueue_log']({'kind':'message','raw':math.nan})
        scope['enqueue_log']({'kind':'summary','ok':True})
        scope['log_queue'].put(None);scope['log_writer']()
        rows=[json.loads(s) for s in scope['sys'].stdout.getvalue().splitlines()]
        self.assertEqual([r['index'] for r in rows[:257]],list(range(257)))
        self.assertTrue(math.isnan(rows[-2]['raw']))
        self.assertEqual(rows[-1],{'kind':'summary','ok':True})
        self.assertIsNone(scope['log_error'][0])

    def test_broken_writer_aborts(self):
        class BrokenSink:
            def write(self,value):raise BrokenPipeError('closed sink')
        scope=self.logging_scope(stdout=BrokenSink())
        scope['emit']({'kind':'target'});scope['log_writer']()
        self.assertTrue(scope['stop'].is_set())
        with self.assertRaisesRegex(ValueError,'writer failed'):scope['log_guard']()

    def test_blocked_graph_query_does_not_block_control_but_cache_expires(self):
        entered=threading.Event();release=threading.Event();stop=threading.Event()
        clock=SimpleNamespace(now=100.,monotonic=lambda:clock.now)
        def query():entered.set();release.wait(2.)
        scope=functions_from_worker('graph_monitor','graph_status_guard',
            graph_guard=query,graph_status={'checked':100.,'error':None},
            time=clock,lock=threading.RLock(),stop=stop,log_guard=lambda:None,emit=Mock())
        monitor=threading.Thread(target=scope['graph_monitor'],daemon=True);monitor.start()
        try:
            self.assertTrue(entered.wait(1.))
            clock.now=100.5;scope['graph_status_guard']()
            self.assertFalse(release.is_set())
            clock.now=101.51
            with self.assertRaisesRegex(ValueError,'graph audit stale'):scope['graph_status_guard']()
        finally:stop.set();release.set();monitor.join(1.)

    def test_background_graph_failure_still_blocks_control(self):
        def query():raise ValueError('Competing publishers on /kuavo_arm_traj')
        scope=functions_from_worker('graph_monitor','graph_status_guard',
            graph_guard=query,graph_status={'checked':100.,'error':None},
            time=SimpleNamespace(monotonic=lambda:100.),lock=threading.RLock(),
            stop=threading.Event(),log_guard=lambda:None,emit=Mock())
        scope['graph_monitor']()
        with self.assertRaisesRegex(ValueError,'Competing publishers'):scope['graph_status_guard']()


class StationarityTests(unittest.TestCase):
    def history(self, position=None, velocity=None):
        q=capture()['baseline_rad']
        return [(i/100, [q[j]+(position(i/100) if j==4 and position else 0.) for j in range(18)],
                 [(velocity(i) if j==4 and velocity else 0.) for j in range(18)]) for i in range(201)]

    def test_stationary_velocity_spikes_do_not_cause_false_positive(self):
        h=self.history(position=lambda s:math.radians(.005*math.sin(12*s)),
                       velocity=lambda i:math.radians(4. if i%4==0 else -4. if i%4==2 else 0.))
        metrics=t.stationarity_metrics(h,2.)
        self.assertLess(metrics['position_span_deg']['zarm_l1_joint'],.05)

    def test_real_movement_rejected_even_if_velocity_reports_zero(self):
        with self.assertRaisesRegex(ValueError,'zarm_l1_joint.*position_span'):
            t.stationarity_metrics(self.history(position=lambda s:math.radians(.06*s)),2.)

    def test_oscillation_not_hidden_by_opposite_velocities(self):
        with self.assertRaisesRegex(ValueError,'not stationary'):
            t.stationarity_metrics(self.history(position=lambda s:math.radians(.1*math.sin(2*math.pi*s)),
                                                 velocity=lambda i:math.radians(1. if i%2 else -1.)),2.)

    def test_sustained_velocity_and_large_spike_rejected(self):
        for velocity in (lambda i:math.radians(.6),lambda i:math.radians(11. if i==50 else 0.)):
            with self.assertRaisesRegex(ValueError,'not stationary'):
                t.stationarity_metrics(self.history(velocity=velocity),2.)

    def test_missing_stale_and_nonfinite_history_rejected(self):
        for h,now in ((self.history()[:10],2.),(self.history(),2.3),
                      (self.history(position=lambda s:math.nan),2.)):
            with self.assertRaises(ValueError):t.stationarity_metrics(h,now)

    def test_fresh_delayed_callback_still_has_full_observed_window(self):
        metrics=t.stationarity_metrics(self.history(),2.08)
        self.assertEqual(metrics['window_s'],2.)
        with self.assertRaisesRegex(ValueError,'stale'):
            t.stationarity_metrics(self.history(),2.11)


if __name__=='__main__':unittest.main()
