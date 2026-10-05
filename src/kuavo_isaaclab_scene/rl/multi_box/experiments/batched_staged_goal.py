"""Independent physical base stages sharing one SAC; no live motion teacher."""
import json
import math
from types import SimpleNamespace

import torch

from .staged_base_hold import StagedBaseHoldDiagnostic


WAVE_RESET_CONTROLLER_CONTRACT = 'neutral_scene_reset_fresh_base_wrench_and_robot_FK_v2'


def reset_wave_controller_state(env, assets, ids):
    """Clear physical controller history before installing another neutral scene.

    Resetting managers alone does not reset an articulation's permanent wrench
    or joint effort targets. The floating base stores a body-frame support
    wrench there; replaying it after a different root/arm teleport is unsafe.
    Scene reset also clears contact sensors, while explicit zero targets remove
    effort/velocity commands that Articulation.reset deliberately preserves.
    Positions are installed by the caller before controllers are re-captured.
    """
    robot=env.scene['robot']
    composer=robot.permanent_wrench_composer
    def norms():
        return dict(force_n=composer.composed_force_as_torch[ids].flatten(1).norm(dim=1).tolist(),
                    torque_nm=composer.composed_torque_as_torch[ids].flatten(1).norm(dim=1).tolist())
    audit=dict(contract=WAVE_RESET_CONTROLLER_CONTRACT,before=norms())
    env.scene.reset(ids)
    for asset in assets:
        if asset.num_joints:
            zero=torch.zeros_like(asset.data.joint_pos[ids])
            asset.set_joint_velocity_target(zero,env_ids=ids)
            asset.set_joint_effort_target(zero,env_ids=ids)
    audit['after_scene_reset']=norms()
    return audit


def settle_neutral_wave_controllers(env, steps=60,*,diagnostic_callback=None):
    """Hold the newly installed neutral pose with current support every substep.

    No rewards, transitions or policy actions are collected here. Managers
    must already have captured the new joint/root poses. A zero command holds
    them; normal action application recomputes base gravity/COM support and
    the articulation writer recomputes joint gravity compensation as usual.
    """
    env.action_manager.process_action(torch.zeros_like(env.action_manager.action))
    for index in range(steps):
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        if diagnostic_callback is not None:
            diagnostic_callback(index+1)


class DevelopmentSuccessGuard:
    """Compare the same development cases without consulting final outcomes.

    Strict mode stops on any regional count loss. Optional exact paired mode
    distinguishes that loss from physical repeat noise. Its familywise alpha
    is spent over comparisons and regions; neither mode sees final outcomes.
    """
    def __init__(self, minimum_region_success_rate=0., *, regression_significance=0.):
        if not 0<=minimum_region_success_rate<=1:
            raise ValueError('Development success floor must be within0..1')
        self.minimum_region_success_rate=minimum_region_success_rate
        if not 0<=regression_significance<1:
            raise ValueError('Regression significance must be within0..1, excluding1')
        self.regression_significance=regression_significance
        self.cases = None
        self.best = None
        self.best_wave = None
        self.best_actor_updates = None
        self.best_success = None
        self.comparisons = 0
        self.paired_reference_success = None
        self.paired_reference_counts = None
        self.paired_reference_actor_updates = None
        self.paired_reference_wave = None

    def evaluate(self, layouts, results, wave_index, *, actor_updates=None):
        if len(layouts) != len(results) or not layouts:
            raise ValueError('Development layouts and physical results differ')
        keys=[json.dumps(row, sort_keys=True) for row in layouts]
        cases = sorted(keys)
        success={key:bool(result and result['success']) for key,result in zip(keys,results)}
        if len(success)!=len(layouts):raise ValueError('Development requires identical initial cases with distinct entries')
        if self.cases is not None and cases != self.cases:
            raise ValueError('Development regression requires identical initial cases')
        counts = {}
        for row, result in zip(layouts, results):
            region = row['layout']['target_region']
            count = counts.setdefault(region, dict(attempts=0, successes=0))
            count['attempts'] += 1
            count['successes'] += int(bool(result and result['success']))
        raw_loss = self.best is not None and any(
            count['successes'] < self.best[region]['successes']
            for region, count in counts.items())
        paired={};threshold=None
        regression=raw_loss
        if self.best is not None and self.regression_significance:
            self.comparisons+=1
            # Sum over k>=1 of1/(k*(k+1)) is1: the declared familywise alpha
            # covers repeated development looks, including this whole region set.
            threshold=self.regression_significance/(self.comparisons*(self.comparisons+1)*len(counts))
            for region in counts:
                region_keys=[key for key,row in zip(keys,layouts) if row['layout']['target_region']==region]
                # Compare to the first, preselected development baseline.
                # Choosing the best noisy past observation as the null baseline
                # would itself bias the exact test toward false regressions.
                lost=sum(self.paired_reference_success[key] and not success[key] for key in region_keys)
                gained=sum(not self.paired_reference_success[key] and success[key] for key in region_keys)
                discordant=lost+gained
                p=(sum(math.comb(discordant,i) for i in range(lost,discordant+1))/2**discordant
                   if discordant else 1.)
                paired[region]=dict(lost_successes=lost,gained_successes=gained,
                    one_sided_exact_p=p,significant_loss=lost>gained and p<=threshold)
            regression=any(v['significant_loss'] for v in paired.values())
        compared_actor_updates=(self.paired_reference_actor_updates if self.regression_significance
                                else self.best_actor_updates)
        cause=('physical_reproducibility_loss_without_actor_update'
               if regression and actor_updates is not None and actor_updates==compared_actor_updates
               else 'policy_performance_loss' if regression else None)
        if self.paired_reference_success is None:
            self.paired_reference_success=success
            self.paired_reference_counts=counts
            self.paired_reference_actor_updates=actor_updates
            self.paired_reference_wave=wave_index
        if self.best is None or not raw_loss:
            self.cases, self.best, self.best_wave = cases, counts, wave_index
            self.best_actor_updates=actor_updates
            self.best_success=success
        baseline_failed=any(count['successes']/count['attempts']<self.minimum_region_success_rate
                            for count in counts.values())
        return dict(regression=regression,regression_cause=cause,
                    baseline_failed=baseline_failed, by_region=counts,
                    best_by_region=self.best, best_wave=self.best_wave,
                    minimum_region_success_rate=self.minimum_region_success_rate,
                    regional_raw_loss=raw_loss,paired_comparison=paired,
                    regression_significance=self.regression_significance,
                    spent_region_significance=threshold,
                    paired_reference_wave=self.paired_reference_wave,
                    paired_reference_by_region=self.paired_reference_counts,
                    regression_rule=('fixed_initial_baseline_exact_paired_alpha_spending'
                                     if self.regression_significance else 'strict_regional_count'),
                    final_outcomes_used=False)


def evaluate_development_wave(guard,layouts,results,wave_index,*,actor_updates,completed=True):
    if not completed:
        return dict(evaluated=False,reason='requested_stop_during_development',final_outcomes_used=False)
    return dict(evaluated=True,**guard.evaluate(layouts,results,wave_index,actor_updates=actor_updates))


def measured_wave_mask(active, numerical_failure, diagnostics, last, step):
    """Quarantine replaced numerical states without rewriting actual past rows.

    The failed requested attempt remains a denominator failure. Its corrupted
    action->replacement observation is never a measured transition for Q/HDF.
    Healthy terminal rows are still measured; the caller deactivates them only
    after recording their actual terminal observations.
    """
    if numerical_failure.dtype!=torch.bool or numerical_failure.shape!=active.shape:
        raise ValueError('Numerical wave mask must match active environments')
    failed=active&numerical_failure
    for i in torch.where(failed)[0].tolist():
        last[i]=dict(steps=step+1,success=False,unsafe=True,invalid_reset=False,
            time_out=False,numerical_failure=True,
            excluded_corrupted_transition=True,
            numerical_causes=[k for k,v in diagnostics.items() if bool(v[i])],
            last_valid_physics_result=last[i])
    return active&~numerical_failure


def observe_measured_held_rows(pilot,stages,ids,previous,terminal,reward,terminated,clocks,measured):
    """Keep physical rows and their per-environment context on identical IDs.

    Numerical quarantine can remove a row after act(). Its anchor and held
    waypoint must also leave observe(); retaining the pre-step context mixes
    identities or crashes the entire healthy learner batch.
    """
    keep=measured[ids]
    if not bool(keep.any()):return 0
    valid_ids=ids[keep]
    pilot.stage=stages.held_context(valid_ids)
    pilot.anchor=stages.anchors[valid_ids].clone()
    pilot.observe(tuple(v[keep] for v in previous),terminal['policy'][valid_ids],
        torch.cat((terminal['policy'],terminal['critic']),-1)[valid_ids],
        reward[valid_ids],terminated[valid_ids],clocks[keep])
    return len(valid_ids)


class BatchedBaseStages:
    def __init__(self,coordinates,templates,raw):
        self.stages=[StagedBaseHoldDiagnostic(coordinates,templates,row[None]) for row in raw]
        self.coordinates=coordinates
        self.anchors=torch.zeros(len(raw),2,device=raw.device,dtype=raw.dtype)
        self.target_xy=torch.cat([s.target_xy for s in self.stages])
        self.target_yaw=raw.new_tensor([s.target_yaw for s in self.stages])
        self._stable_steps=torch.zeros(len(raw),device=raw.device,dtype=torch.long)
        self._held=torch.zeros(len(raw),device=raw.device,dtype=torch.bool)

    def update(self,raw,linear,angular,active,step):
        ids=torch.where(active)[0]
        if not len(ids):return ids
        # One geometry batch and one diagnostic transfer. Per-environment
        # scalar reads formerly synchronized the GPU for every error/speed.
        _,_,xy,yaw,_=self.coordinates.current(raw[ids])
        position_error=(xy-self.target_xy[ids]).norm(dim=-1)
        difference=yaw-self.target_yaw[ids]
        yaw_error=torch.atan2(difference.sin(),difference.cos()).abs()
        linear_speed=linear[ids,:2].norm(dim=-1)
        angular_speed=angular[ids].norm(dim=-1)
        stable=(position_error<.008)&(yaw_error<.02)&(linear_speed<.01)&(angular_speed<.025)
        self._stable_steps[ids]=torch.where(self._held[ids],self._stable_steps[ids],
            torch.where(stable,self._stable_steps[ids]+1,0))
        new=ids[~self._held[ids]&(self._stable_steps[ids]>=15)]
        if len(new):
            self.anchors[new]=self.coordinates.box_anchor(raw[new])
            self._held[new]=True
        diagnostics=torch.stack((position_error,yaw_error,linear_speed,angular_speed,
                                  self._stable_steps[ids]),-1).cpu().tolist()
        for i,values in zip(ids.tolist(),diagnostics):
            stage=self.stages[i]
            stage.position_error,stage.yaw_error,stage.linear_speed,stage.angular_speed=values[:4]
            stage.stable_steps=int(values[4])
            if stage.phase=='approach' and stage.stable_steps>=15:
                stage.phase='held_grasp';stage.manipulation_start=step
        return ids[self._held[ids]]

    def held_context(self,ids):
        selected=[self.stages[i] for i in ids.tolist()]
        if not selected or any(s.phase!='held_grasp' or s.manipulation_start is None for s in selected):
            raise ValueError('Batched replay excludes unconfirmed base approaches')
        return SimpleNamespace(phase='held_grasp',manipulation_start=True,
            target_xy=torch.cat([s.target_xy for s in selected]),
            target_yaw=self.anchors.new_tensor([s.target_yaw for s in selected]),
            name=selected[0].name,templates=selected[0].templates)

    def clocks(self,ids,step):
        return torch.tensor([self.stages[i].manipulation_index(step) for i in ids.tolist()],
                            device=ids.device)

    def approach_commands(self,raw,active):
        result=raw.new_zeros(len(raw),24)
        ids=torch.where(active&~self._held)[0]
        if len(ids):
            joint,torso,_,_,_=self.coordinates.current(raw[ids])
            goal=torch.cat((joint,torso,self.target_xy[ids],self.target_yaw[ids,None],
                            raw.new_full((len(ids),2),-1.)),-1)
            result[ids,:3]=self.coordinates.decode(raw[ids],goal)[:,:3]
            result[ids,20:22]=-1.
        return result


def restore_batched_inferred_scene(env,actors,*,capture_reset_diagnostics=False,
                                  zero_passive_roller_velocity_probe=False):
    """Whole-wave reset from neutral source observations, never success states.

    Every scene has its own dynamic boxes, rack-relative sampled layout and
    robot initial pose. This separate helper leaves legacy single replay intact.
    """
    from isaaclab.utils.math import quat_from_matrix,quat_inv,quat_mul,quat_apply
    from ..demo_replay import _rotation_matrix
    from ..scene.spawn import physical_asset_names,physical_pool_id,logical_cells
    from ..scene.reset_kinematics import refresh_teleported_articulations
    from ..state.schema import ACTUATED_BODY_JOINTS
    if actors.shape!=(env.num_envs,464) or not torch.isfinite(actors).all():
        raise ValueError('One finite neutral layout per environment is required')
    ids=torch.arange(env.num_envs,device=env.device)
    actors=actors.to(env.device)
    rack=env.scene['rack'].data.root_pose_w
    root_q=quat_mul(rack[:,3:],quat_inv(quat_from_matrix(_rotation_matrix(actors[:,71:77]))))
    root_p=rack[:,:3]-quat_apply(root_q,actors[:,68:71])
    robot=env.scene['robot']
    names=physical_asset_names()
    assets=[robot,*[env.scene[name] for name in names]]
    env._batched_reset_control_audit=reset_wave_controller_state(env,assets,ids)
    root=robot.data.root_state_w.clone();root[:,:3]=root_p;root[:,3:7]=root_q;root[:,7:]=0
    robot.write_root_state_to_sim(root,env_ids=ids)
    q=robot.data.default_joint_pos.clone()
    joints,_=robot.find_joints(list(ACTUATED_BODY_JOINTS),preserve_order=True)
    q[:,joints]=actors[:,:20]
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.set_joint_position_target(q,env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(q),env_ids=ids)
    env._multi_box_active[:]=False
    env._multi_box_box_type_ids[:]=-1;env._multi_box_region_ids[:]=-1;env._multi_box_pool_ids[:]=-1
    cells=logical_cells(env.cfg.multi_box)
    for name in names:
        asset=env.scene[name];parked=asset.data.default_root_state.clone()
        parked[:,:3]+=env.scene.env_origins
        asset.write_root_state_to_sim(parked,env_ids=ids)
        if hasattr(asset.data,'default_joint_pos'):
            qbox=asset.data.default_joint_pos.clone()
            asset.write_joint_state_to_sim(qbox,torch.zeros_like(qbox),env_ids=ids)
            asset.set_joint_position_target(qbox,env_ids=ids)
    tokens=actors[:,86:350].reshape(-1,12,22)
    for logical in range(12):
        for kind in (0,1):
            selected=torch.where((tokens[:,logical,0]>.5)&(tokens[:,logical,3:5].argmax(-1)==kind))[0]
            if not len(selected):continue
            token=tokens[selected,logical];pool=physical_pool_id(cells[logical],kind)
            env._multi_box_active[selected,logical]=True
            env._multi_box_box_type_ids[selected,logical]=kind
            env._multi_box_region_ids[selected,logical]=token[:,8:12].argmax(-1)
            env._multi_box_pool_ids[selected,logical]=pool
            asset=env.scene[names[pool]];state=asset.data.default_root_state[selected].clone()
            state[:,:3]=root_p[selected]+quat_apply(root_q[selected],token[:,12:15])
            state[:,3:7]=quat_mul(root_q[selected],quat_from_matrix(_rotation_matrix(token[:,15:21])))
            state[:,7:]=0
            asset.write_root_state_to_sim(state,env_ids=selected)
    env._multi_box_counts[:]=env._multi_box_active.sum(-1)
    env._multi_box_grasp_target_override=actors[:,400:412].argmax(-1)
    refresh_teleported_articulations(env,assets,ids)
    env.sim.forward();env.scene.update(env.step_dt)
    env._refresh_robot_kinematics()
    for manager in (env.action_manager,env.observation_manager,env.reward_manager,env.termination_manager):manager.reset(ids)
    if hasattr(env,'_multi_box_privileged_grasp'):env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids);env.episode_length_buf[:]=0
    if capture_reset_diagnostics:
        from ..scene.reset_diagnostics import passive_roller_snapshot, startup_normal_contact_snapshot
        from ..debug.contact_sensors import BELT_CONTACT_SENSOR_NAMES, CONTACT_SENSOR_NAMES, V2_OBSTACLE_SENSOR_NAME
        env._batched_reset_box_diagnostics=dict(before_neutral_hold=
            measured_initial_box_failures(env,actors,names,failures_only=False,include_link_states=True),
            neutral_hold_trace=[],neutral_normal_contact_trace=[],
            passive_rollers_before_velocity_probe=passive_roller_snapshot(env))
    if zero_passive_roller_velocity_probe:
        from ..scene.reset_diagnostics import zero_passive_roller_velocities
        zero_passive_roller_velocities(env,ids)
    if capture_reset_diagnostics:
        env._batched_reset_box_diagnostics['passive_rollers_before_neutral_hold']=passive_roller_snapshot(env)
    def capture_hold_step(index):
        if index <= 12 or index in (16,32,60):
            env._batched_reset_box_diagnostics['neutral_normal_contact_trace'].append(dict(
                physics_step=index,elapsed_s=index*env.physics_dt,
                **startup_normal_contact_snapshot(env,names,BELT_CONTACT_SENSOR_NAMES,
                    (V2_OBSTACLE_SENSOR_NAME,*CONTACT_SENSOR_NAMES))))
        if index in (1,2,4,8,16,32):
            env._batched_reset_box_diagnostics['neutral_hold_trace'].append(dict(
                physics_step=index,elapsed_s=index*env.physics_dt,
                boxes=measured_initial_box_failures(env,actors,names,
                    failures_only=False,include_link_states=True)))
    settle_neutral_wave_controllers(env,
        diagnostic_callback=capture_hold_step if capture_reset_diagnostics else None)
    if capture_reset_diagnostics:
        env._batched_reset_box_diagnostics['after_neutral_hold']=\
            measured_initial_box_failures(env,actors,names,failures_only=False,include_link_states=True)
        env._batched_reset_box_diagnostics['passive_rollers_after_neutral_hold']=passive_roller_snapshot(env)
    composer=robot.permanent_wrench_composer
    env._batched_reset_control_audit['after_neutral_hold']=dict(
        force_n=composer.composed_force_as_torch[ids].flatten(1).norm(dim=1).tolist(),
        torque_nm=composer.composed_torque_as_torch[ids].flatten(1).norm(dim=1).tolist())
    env._refresh_robot_kinematics()
    for manager in (env.action_manager,env.observation_manager,env.reward_manager,env.termination_manager):manager.reset(ids)
    env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids)
    return env.observation_manager.compute()


def wait_for_original_layouts(env, observation, invalid_before):
    """Wait for requested resets, never for replacements of failed cases."""
    settling=env._multi_box_reset_settling
    maximum=max(1,math.ceil(3*float(env.cfg.multi_box.reset_settle_timeout_seconds)/env.step_dt))
    failed=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    for steps in range(maximum+1):
        failed|=settling.invalid_count!=invalid_before
        if bool((settling.ready|failed).all()) or steps==maximum:break
        observation,_,terminated,truncated,info=env.step(torch.zeros_like(env.action_manager.action))
        failed|=terminated|truncated|info['transition_numerical_failure']
    failed|=~settling.ready
    return observation,steps,failed


def wait_for_original_surrounding_boxes(env,observation,expected,names,invalid_before,failed):
    """Require eight stable ticks per intact environment, with a bounded wait.

    An unstable/replaced case cannot block another environment or become
    replay. Every failed requested case is retained in evaluation counts.
    """
    settling=env._multi_box_reset_settling
    stable_ticks=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    rows=torch.arange(env.num_envs,device=env.device)[:,None]
    failed=failed.clone()
    for steps in range(91):
        failed|=(settling.invalid_count!=invalid_before)|~settling.ready
        failed|=~(env._multi_box_active==expected).all(-1)
        velocities=torch.stack([env.scene[name].data.root_vel_w for name in names],1)
        measured=velocities[rows,env._multi_box_pool_ids.clamp_min(0)]
        stable=torch.isfinite(measured).all(-1)&(measured[...,:3].norm(dim=-1)<.01)&(measured[...,3:].norm(dim=-1)<.05)
        stable=(stable|~expected).all(-1)&~failed
        stable_ticks=torch.where(stable,stable_ticks+1,0)
        if bool(((stable_ticks>=8)|failed).all()) or steps==90:break
        observation,_,terminated,truncated,info=env.step(torch.zeros_like(env.action_manager.action))
        failed|=terminated|truncated|info['transition_numerical_failure']
    return observation,steps,failed,stable_ticks


def measured_initial_box_failures(env,actors,names,*,failures_only=True,include_link_states=False):
    """Read why an original box fails geometry/stability; never alter its state.

    A replaced case is explicitly labelled. Its parked old asset must not be
    mistaken for the requested box's original physical failure trajectory.
    This final guard sample supplements counts, not pre-respawn observations.
    """
    from ..scene.spawn import physical_pool_id,logical_cells
    from ..geometry.pose import quat_apply,quat_conjugate,normalize_quaternion,replace_invalid_poses
    from ..geometry.rack import box_shelf_clearance_m
    from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
    from ....workcell.workcell_layout import scale
    tokens=actors[:,86:350].reshape(-1,12,22)
    poses=torch.stack([env.scene[name].data.root_pose_w for name in names],1)
    velocities=torch.stack([env.scene[name].data.root_vel_w for name in names],1)
    rack,invalid_rack=replace_invalid_poses(env.scene['rack'].data.root_pose_w)
    rows=torch.arange(env.num_envs,device=env.device)
    failures=[]
    def finite_list(value):
        values=value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value
        return [float(v) if math.isfinite(float(v)) else None for v in values]
    for cell in logical_cells(env.cfg.multi_box):
        logical=cell.logical_id;active=tokens[:,logical,0]>.5
        for kind,name in enumerate(('small','medium')):
            selected=active&(tokens[:,logical,3:5].argmax(-1)==kind)
            if not selected.any():continue
            pool=physical_pool_id(cell,kind);pose=poses[:,pool];velocity=velocities[:,pool]
            safe,invalid_box=replace_invalid_poses(pose)
            finite=~invalid_box&~invalid_rack&torch.isfinite(velocity).all(-1)
            type_ids=rows.new_full((env.num_envs,),kind);region_ids=rows.new_full((env.num_envs,),cell.region_id)
            footprint=finite&env._multi_box_reset_settling._footprint_in_region(safe,type_ids,region_ids)
            on_shelf=finite&env._multi_box_reset_settling._on_assigned_shelf(safe,type_ids,region_ids)
            clearance=box_shelf_clearance_m(safe,rack,BOX_DIMENSIONS_M[name],shelf=cell.shelf,rack_scale=scale('rack'))
            local=quat_apply(quat_conjugate(normalize_quaternion(rack[:,3:])),safe[:,:3]-rack[:,:3])
            stable=finite&(velocity[:,:3].norm(dim=-1)<.01)&(velocity[:,3:].norm(dim=-1)<.05)
            same_pool=env._multi_box_active[:,logical]&(env._multi_box_pool_ids[:,logical]==pool)
            chosen=selected&~(footprint&on_shelf&stable&same_pool) if failures_only else selected
            for i in torch.where(chosen)[0].tolist():
                asset=env.scene[names[pool]];joints=getattr(asset.data,'joint_pos',None)
                sample=dict(environment=i,logical_id=logical,original_pool_id=pool,
                    original_asset_still_active=bool(same_pool[i]),finite=bool(finite[i]),
                    invalid_box_pose=bool(invalid_box[i]),invalid_rack_pose=bool(invalid_rack[i]),
                    footprint_in_region=bool(footprint[i]),on_assigned_shelf=bool(on_shelf[i]),
                    stable_at_guard=bool(stable[i]),rack_local_root_xyz_m=finite_list(local[i]) if finite[i] else None,
                    shelf_clearance_m=float(clearance[i]) if finite[i] else None,
                    linear_speed_mps=float(velocity[i,:3].norm()) if finite[i] else None,
                    angular_speed_radps=float(velocity[i,3:].norm()) if finite[i] else None,
                    box_pose_world=finite_list(pose[i]),box_velocity_world=finite_list(velocity[i]),
                    joint_names=list(getattr(asset,'joint_names',[])),
                    joint_positions_rad=None if joints is None else finite_list(joints[i]))
                if include_link_states:
                    sample.update(body_names=list(asset.body_names),
                        body_link_pose_world=[finite_list(row) for row in asset.data.body_link_pose_w[i].detach().cpu().tolist()],
                        body_link_velocity_world=[finite_list(row) for row in asset.data.body_link_vel_w[i].detach().cpu().tolist()],
                        joint_velocities_radps=finite_list(asset.data.joint_vel[i]))
                failures.append(sample)
    return failures


def settle_batched_layouts(env, actors, *, allow_partial=False,capture_reset_diagnostics=False,
                           zero_passive_roller_velocity_probe=False):
    """Reject replaced/unsettled targets and every invalid surrounding box."""
    from ..scene.spawn import physical_asset_names
    observation=restore_batched_inferred_scene(env,actors,capture_reset_diagnostics=capture_reset_diagnostics,
        zero_passive_roller_velocity_probe=zero_passive_roller_velocity_probe)
    invalid_before=env._multi_box_reset_settling.invalid_count.clone()
    names=physical_asset_names();settling=env._multi_box_reset_settling
    expected=actors[:,86:350].reshape(-1,12,22)[:,:,0]>.5
    if capture_reset_diagnostics:
        from ..scene.reset_diagnostics import ResetFailureCapture
        settling.failure_capture=ResetFailureCapture(env)
    try:
        observation,steps,failed_during_settle=wait_for_original_layouts(env,observation,invalid_before)
        observation,tick,failed_during_settle,stable_ticks=wait_for_original_surrounding_boxes(
            env,observation,expected,names,invalid_before,failed_during_settle)
        captured=[] if settling.failure_capture is None else settling.failure_capture.records
    finally:
        settling.failure_capture=None
    surrounding_unsettled=stable_ticks<8
    if not allow_partial and bool((failed_during_settle|surrounding_unsettled).any()):
        raise ValueError('Batched layout failed or surrounding boxes did not settle')
    rack_error=(observation['policy'][:,68:77]-actors[:,68:77]).norm(dim=-1)
    guard=dict(expected=expected.to(torch.int).tolist(),actual=env._multi_box_active.to(torch.int).tolist(),
        rack_error=rack_error.tolist(),invalid=(settling.invalid_count-invalid_before).tolist(),
        footprint_invalid=settling.footprint_invalid_count.tolist(),ready=settling.ready.tolist(),
        base_pose=env.scene['robot'].data.root_pose_w.tolist(),rack_pose=env.scene['rack'].data.root_pose_w.tolist(),
        controller_reset=env._batched_reset_control_audit)
    valid=(env._multi_box_active==expected).all(-1)&torch.isfinite(rack_error)&(rack_error<=.025) \
        &(settling.invalid_count==invalid_before)&settling.ready&~failed_during_settle&~surrounding_unsettled
    if not allow_partial and not bool(valid.all()):
        raise ValueError('Batched layout was replaced during settling; replay prohibited: '+str(guard))
    footprint_failures=[]
    for logical in range(12):
        active=expected[:,logical]
        if not active.any():continue
        pools=env._multi_box_pool_ids[:,logical].clamp_min(0)
        poses=torch.stack([env.scene[name].data.root_pose_w for name in names],1)
        pose=poses[torch.arange(env.num_envs,device=env.device),pools]
        types=env._multi_box_box_type_ids[:,logical].clamp_min(0)
        regions=env._multi_box_region_ids[:,logical].clamp_min(0)
        footprint=settling._footprint_in_region(pose,types,regions)&settling._on_assigned_shelf(pose,types,regions)
        footprint_failures.append(dict(logical_id=logical,environment_ids=torch.where(active&~footprint)[0].tolist()))
        valid&=footprint|~active
        if not allow_partial and not bool((footprint|~active).all()):
            raise ValueError(f'Batched surrounding box {logical} left its assigned shelf/region')
    ids=torch.arange(env.num_envs,device=env.device)
    env.episode_length_buf[:]=0
    env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    guard['valid_original_layout']=valid.tolist()
    guard['failed_during_settle']=failed_during_settle.tolist()
    guard['surrounding_stable_ticks']=stable_ticks.tolist()
    guard['surrounding_unsettled']=surrounding_unsettled.tolist()
    guard['box_footprint_or_shelf_failures']=footprint_failures
    guard['measured_box_failure_details']=measured_initial_box_failures(env,actors,names)
    if capture_reset_diagnostics:
        guard['reset_failure_diagnostics']=dict(**env._batched_reset_box_diagnostics,
            first_invalid_before_respawn=captured,final_guard_is_not_pre_respawn_trajectory=True,
            physical_state_unchanged=not zero_passive_roller_velocity_probe,
            passive_roller_velocity_probe=zero_passive_roller_velocity_probe,
            box_base_poses_randomization_physics_parameters_success_and_safety_unchanged=True)
    print('[BATCH LAYOUT GUARD] '+str(guard),flush=True)
    return env.observation_manager.compute(),steps+tick,valid,guard
