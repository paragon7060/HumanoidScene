"""Independent physical base stages sharing one SAC; no live motion teacher."""
import json
from types import SimpleNamespace

import torch

from .staged_base_hold import StagedBaseHoldDiagnostic


class DevelopmentSuccessGuard:
    """Compare the same development cases without consulting final outcomes.

    A loss in any region stops the learner. The caller saves actual Q/replay
    before returning, so actor recovery can use a separate, immutable run.
    """
    def __init__(self, minimum_region_success_rate=0.):
        if not 0<=minimum_region_success_rate<=1:
            raise ValueError('Development success floor must be within0..1')
        self.minimum_region_success_rate=minimum_region_success_rate
        self.cases = None
        self.best = None
        self.best_wave = None

    def evaluate(self, layouts, results, wave_index):
        if len(layouts) != len(results) or not layouts:
            raise ValueError('Development layouts and physical results differ')
        cases = sorted(json.dumps(row, sort_keys=True) for row in layouts)
        if self.cases is not None and cases != self.cases:
            raise ValueError('Development regression requires identical initial cases')
        counts = {}
        for row, result in zip(layouts, results):
            region = row['layout']['target_region']
            count = counts.setdefault(region, dict(attempts=0, successes=0))
            count['attempts'] += 1
            count['successes'] += int(bool(result and result['success']))
        regression = self.best is not None and any(
            count['successes'] < self.best[region]['successes']
            for region, count in counts.items())
        if not regression:
            self.cases, self.best, self.best_wave = cases, counts, wave_index
        baseline_failed=any(count['successes']/count['attempts']<self.minimum_region_success_rate
                            for count in counts.values())
        return dict(regression=regression, baseline_failed=baseline_failed, by_region=counts,
                    best_by_region=self.best, best_wave=self.best_wave,
                    minimum_region_success_rate=self.minimum_region_success_rate,
                    final_outcomes_used=False)


class BatchedBaseStages:
    def __init__(self,coordinates,templates,raw):
        self.stages=[StagedBaseHoldDiagnostic(coordinates,templates,row[None]) for row in raw]
        self.coordinates=coordinates
        self.anchors=torch.zeros(len(raw),2,device=raw.device,dtype=raw.dtype)

    def update(self,raw,linear,angular,active,step):
        held=[]
        for i in torch.where(active)[0].tolist():
            stage=self.stages[i]
            was_held=stage.phase=='held_grasp'
            stage.update(raw[i:i+1],linear[i:i+1],angular[i:i+1],step)
            if stage.phase=='held_grasp':
                held.append(i)
                if not was_held:
                    self.anchors[i]=self.coordinates.box_anchor(raw[i:i+1])[0]
        return torch.tensor(held,device=raw.device,dtype=torch.long)

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
        for i in torch.where(active)[0].tolist():
            if self.stages[i].phase=='approach':
                result[i]=self.stages[i].action(raw[i:i+1])[0]
        return result


def restore_batched_inferred_scene(env,actors):
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
    names=physical_asset_names();cells=logical_cells(env.cfg.multi_box)
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
    refresh_teleported_articulations(env,[env.scene[name] for name in names],ids)
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt)
    env._refresh_robot_kinematics()
    for manager in (env.action_manager,env.observation_manager,env.reward_manager,env.termination_manager):manager.reset(ids)
    if hasattr(env,'_multi_box_privileged_grasp'):env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids);env.episode_length_buf[:]=0
    for _ in range(60):
        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
    env._refresh_robot_kinematics()
    for manager in (env.action_manager,env.observation_manager,env.reward_manager,env.termination_manager):manager.reset(ids)
    env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids)
    return env.observation_manager.compute()


def settle_batched_layouts(env, actors):
    """Reject replaced/unsettled targets and every invalid surrounding box."""
    from ...runners.train_asymmetric_sac import _settle_initial_resets
    from ..scene.spawn import physical_asset_names
    observation=restore_batched_inferred_scene(env,actors)
    invalid_before=env._multi_box_reset_settling.invalid_count.clone()
    observation,steps=_settle_initial_resets(env,observation)
    names=physical_asset_names();settling=env._multi_box_reset_settling
    expected=actors[:,86:350].reshape(-1,12,22)[:,:,0]>.5
    stable_ticks=0
    for tick in range(90):
        velocities=torch.stack([env.scene[name].data.root_vel_w for name in names],1)
        pools=env._multi_box_pool_ids.clamp_min(0)
        rows=torch.arange(env.num_envs,device=env.device)[:,None]
        measured=velocities[rows,pools]
        stable=torch.isfinite(measured).all(-1)&(measured[...,:3].norm(dim=-1)<.01)&(measured[...,3:].norm(dim=-1)<.05)
        stable_ticks=stable_ticks+1 if bool((stable|~expected).all()) else 0
        if stable_ticks>=8:break
        observation,_,terminated,truncated,info=env.step(torch.zeros_like(env.action_manager.action))
        if bool((terminated|truncated|info['transition_numerical_failure']).any()):
            raise ValueError('Batched layout failed while surrounding boxes settled')
    else:
        raise ValueError('Batched surrounding boxes did not settle')
    rack_error=(observation['policy'][:,68:77]-actors[:,68:77]).norm(dim=-1)
    guard=dict(expected=expected.to(torch.int).tolist(),actual=env._multi_box_active.to(torch.int).tolist(),
        rack_error=rack_error.tolist(),invalid=(settling.invalid_count-invalid_before).tolist(),
        footprint_invalid=settling.footprint_invalid_count.tolist(),ready=settling.ready.tolist(),
        base_pose=env.scene['robot'].data.root_pose_w.tolist(),rack_pose=env.scene['rack'].data.root_pose_w.tolist())
    print('[BATCH LAYOUT GUARD] '+str(guard),flush=True)
    if not torch.equal(env._multi_box_active,expected) or (rack_error>.025).any() \
            or (settling.invalid_count!=invalid_before).any():
        raise ValueError('Batched layout was replaced during settling; replay prohibited: '+str(guard))
    for logical in range(12):
        active=expected[:,logical]
        if not active.any():continue
        pools=env._multi_box_pool_ids[:,logical].clamp_min(0)
        poses=torch.stack([env.scene[name].data.root_pose_w for name in names],1)
        pose=poses[torch.arange(env.num_envs,device=env.device),pools]
        types=env._multi_box_box_type_ids[:,logical].clamp_min(0)
        regions=env._multi_box_region_ids[:,logical].clamp_min(0)
        valid=settling._footprint_in_region(pose,types,regions)&settling._on_assigned_shelf(pose,types,regions)
        if not bool((valid|~active).all()):
            raise ValueError(f'Batched surrounding box {logical} left its assigned shelf/region')
    ids=torch.arange(env.num_envs,device=env.device)
    env.episode_length_buf[:]=0
    env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    return env.observation_manager.compute(),steps+tick
