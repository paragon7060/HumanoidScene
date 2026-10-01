"""Replay a VR joint reference through current control; this is not a SAC policy.

Legacy observations do not contain initial flap or pending drive state.
Scene restoration is explicitly inferred and subsequent transitions are measured.
"""
import torch


def restore_inferred_scene(env, observation):
    if env.num_envs != 1:
        raise ValueError('Inferred VR restoration is a single-environment diagnostic')
    from isaaclab.utils.math import quat_from_matrix, quat_inv, quat_mul, quat_apply
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
    from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names,physical_pool_id,logical_cells
    from kuavo_isaaclab_scene.rl.multi_box.state.schema import ACTUATED_BODY_JOINTS
    ids=torch.tensor([0],device=env.device)
    actor=observation[None].to(env.device)
    rack=env.scene['rack'].data.root_pose_w.clone()
    rack_b=actor[:,68:77]
    rb_q=quat_from_matrix(_rotation_matrix(rack_b[:,3:]))
    root_q=quat_mul(rack[:,3:],quat_inv(rb_q))
    root_p=rack[:,:3]-quat_apply(root_q,rack_b[:,:3])
    robot=env.scene['robot']
    root=robot.data.root_state_w.clone();root[:,:3]=root_p;root[:,3:7]=root_q;root[:,7:]=0
    robot.write_root_state_to_sim(root,env_ids=ids)
    q=robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
    joint_ids,_=robot.find_joints(list(ACTUATED_BODY_JOINTS),preserve_order=True)
    q[:,joint_ids]=actor[:,:20]
    robot.write_joint_state_to_sim(q,v,env_ids=ids)
    robot.set_joint_position_target(q,env_ids=ids)
    tokens=actor[:,86:350].reshape(1,12,22)
    env._multi_box_active[:]=False
    env._multi_box_box_type_ids[:]=-1;env._multi_box_region_ids[:]=-1;env._multi_box_pool_ids[:]=-1
    names=physical_asset_names();cells=logical_cells(env.cfg.multi_box)
    for name in names:
        asset=env.scene[name];parked=asset.data.default_root_state.clone();parked[:,:3]+=env.scene.env_origins
        asset.write_root_state_to_sim(parked,env_ids=ids)
        if hasattr(asset.data,'default_joint_pos'):
            asset.write_joint_state_to_sim(asset.data.default_joint_pos.clone(),torch.zeros_like(asset.data.default_joint_pos),env_ids=ids)
            asset.set_joint_position_target(asset.data.default_joint_pos.clone(),env_ids=ids)
    for logical in torch.where(tokens[0,:,0]>.5)[0].tolist():
        token=tokens[0,logical];kind=int(token[3:5].argmax());region=int(token[8:12].argmax())
        pool=physical_pool_id(cells[logical],kind)
        env._multi_box_active[0,logical]=True;env._multi_box_box_type_ids[0,logical]=kind;env._multi_box_region_ids[0,logical]=region;env._multi_box_pool_ids[0,logical]=pool
        asset=env.scene[names[pool]];state=asset.data.default_root_state.clone()
        state[:,:3]=root_p+quat_apply(root_q,token[None,12:15]);state[:,3:7]=quat_mul(root_q,quat_from_matrix(_rotation_matrix(token[None,15:21])));state[:,7:]=0
        asset.write_root_state_to_sim(state,env_ids=ids)
    env._multi_box_counts[:]=env._multi_box_active.sum(-1)
    env._multi_box_grasp_target_override=torch.tensor([int(observation[400:412].argmax())],device=env.device)
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics import refresh_teleported_articulations
    refresh_teleported_articulations(env,[env.scene[name] for name in names],ids)
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt)
    env._refresh_robot_kinematics()
    env.action_manager.reset(ids);env.observation_manager.reset(ids);env.reward_manager.reset(ids);env.termination_manager.reset(ids)
    if hasattr(env,'_multi_box_privileged_grasp'): env._multi_box_privileged_grasp.reset(ids)
    env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids)
    env.episode_length_buf[ids]=0
    print('[INFERRED VR SEED] logical',env._multi_box_active.to(torch.long).argmax(-1).tolist(),'root/rack before settling',env.scene['robot'].data.root_pose_w[0].tolist(),rack[0].tolist(),flush=True)
    for tick in range(60):
        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
    env._refresh_robot_kinematics()
    env.action_manager.reset(ids);env.observation_manager.reset(ids);env.reward_manager.reset(ids);env.termination_manager.reset(ids)
    env._multi_box_privileged_grasp.reset(ids);env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
    env._multi_box_reset_settling.reset(ids)
    return env.observation_manager.compute(),rack


class VRJointTracker:
    def __init__(self,env,demo,rack,*,orientation_mode='full',contact_torso_forward_m=0.,
                 close_distance_m=.035,coordinated_close=False,reference_grippers=False):
        from kuavo_isaaclab_scene.rl.multi_box.state.schema import ACTUATED_BODY_JOINTS
        self.env=env;self.demo=demo;self.rack=rack;self.index=0
        if not 0<=contact_torso_forward_m<=.08:
            raise ValueError('Contact torso assist must be within0..8cm')
        self.contact_torso_forward_m=contact_torso_forward_m
        if not .003<=close_distance_m<=.035:
            raise ValueError('VR closing gate must be within3..35mm')
        self.close_distance_m,self.coordinated_close=close_distance_m,coordinated_close
        self.reference_grippers=reference_grippers
        self.phase=torch.zeros(1,dtype=torch.long,device=env.device);self.close_ticks=self.phase.clone();self.solvers=[]
        self.slices={};i=0
        for name in env.action_manager.active_terms:
            width=env.action_manager.get_term(name).action_dim;self.slices[name]=slice(i,i+width);i+=width
        self.upper=env.action_manager.get_term('upper_body')
        self.upper_columns=[ACTUATED_BODY_JOINTS.index(n) for n in self.upper._joint_names]
        from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import KinematicGraspExplorer
        from kuavo_isaaclab_scene.teleop.urdf_arm_ik import UrdfArm
        from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
        self.final_guide=KinematicGraspExplorer(env,demo,grasp_goal='demo',lift_distance_m=.08,base_clearance_m=.55,torso_forward_m=0,
                                              orientation_mode=orientation_mode)
        for side,solver in zip(('left','right'),self.final_guide.solvers):
            solver.configure_urdf(UrdfArm(resolve_robot_model().urdf_path,side))
            from kuavo_isaaclab_scene.teleop.teleop_servo import RESPONSIVE
            solver.response=RESPONSIVE
        self.solvers=self.final_guide.solvers
        self.lift_goal=None
        self.rest_seeded=False
        self.contact_goal=None;self.contact_rotation=None;self.confirm_ticks=0
        self.switch_index=int(torch.where((demo['critic_obs'][:,464+35:464+37]>.5).all(-1))[0][0])
    def act(self,observation):
        from isaaclab.utils.math import quat_from_matrix,quat_inv,quat_mul,quat_apply
        from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
        from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import planar_position
        idx=min(self.index,len(self.demo['actor_obs'])-1)
        desired=self.demo['next_actor_obs'][idx:idx+1].to(self.env.device)
        action=torch.zeros_like(self.env.action_manager.action)
        action[:,self.slices['upper_body']]=((desired[:,self.upper_columns]-self.upper.processed_actions)/self.upper._scale).clamp(-1,1)
        torso=self.env.action_manager.get_term('height')
        desired_xz=planar_position(desired[:,:2],torso._links)
        action[:,self.slices['height']]=((desired_xz-torso.processed_actions)/(torso.cfg.speed_m_s*self.env.step_dt)).clamp(-1,1)
        action[:,20:22]=-1.  # Keep jaws open until the live geometric handoff.
        if self.reference_grippers:
            action[:,20:22]=self.demo['action'][idx:idx+1,20:22].to(self.env.device)
        ref_rack=desired[:,68:77];ref_q=quat_mul(self.rack[:,3:],quat_inv(quat_from_matrix(_rotation_matrix(ref_rack[:,3:]))))
        ref_p=self.rack[:,:3]-quat_apply(ref_q,ref_rack[:,:3])
        root=self.env.scene['robot'].data.root_pose_w
        position_error=quat_apply(quat_inv(root[:,3:]),ref_p-root[:,:3])
        base=self.env.action_manager.get_term('base')
        action[:,:2]=(.5*position_error[:,:2]/base._scale[:2]).clamp(-.5,.5)
        heading=quat_apply(quat_mul(quat_inv(root[:,3:]),ref_q),root.new_tensor([[1.,0,0]]))
        action[:,2]=(torch.atan2(heading[:,1],heading[:,0])/base._scale[2]).clamp(-.5,.5)
        self.phase[:]=int(bool((action[:,20:22]>0).all()))
        if self.index>=self.switch_index:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import entry_geometry,target_token,retarget_grasp_goal,observed_close_ticks
            from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
            guide=self.final_guide
            if not self.rest_seeded:
                for solver in self.solvers:
                    solver._urdf_rest=solver._numpy(self.env.scene['robot'].data.joint_pos[0,solver._joint_ids])[solver._urdf_order].copy()
                self.rest_seeded=True
            tcp,centers,stage,outward=entry_geometry(observation,guide.front_y)
            token,valid=target_token(observation)
            rotation=_rotation_matrix(token[:,15:21])
            goals,stage,offset=retarget_grasp_goal(centers,stage,outward,rotation,guide.goal_offset,'demo')
            target_rotation=rotation[:,None]@guide.relative_rotation[None]
            quaternion=guide.quat_from_matrix(target_rotation.reshape(-1,3,3)).reshape(-1,2,4)
            close=(goals-tcp[...,:3]).norm(dim=-1)<self.close_distance_m
            if self.reference_grippers:
                close=self.demo['action'][idx:idx+1,20:22].to(self.env.device)>0
            pinching=self.env._multi_box_privileged_grasp_step.pinch.hand_pinching
            from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
            tcp_rotation=_rotation_matrix(tcp[...,3:])
            if self.contact_goal is None:
                self.contact_goal=tcp[...,:3].clone();self.contact_rotation=tcp_rotation.clone()
            self.contact_goal=torch.where(pinching[...,None],self.contact_goal,tcp[...,:3])
            self.contact_rotation=torch.where(pinching[...,None,None],self.contact_rotation,tcp_rotation)
            goals=torch.where(pinching[...,None],self.contact_goal,goals)
            target_rotation=torch.where(pinching[...,None,None],self.contact_rotation,target_rotation)
            quaternion=guide.quat_from_matrix(target_rotation.reshape(-1,3,3)).reshape(-1,2,4)
            close |= pinching
            if self.coordinated_close and not bool(pinching.any()):
                close=close.all(-1,keepdim=True).expand_as(close)
            self.confirm_ticks=self.confirm_ticks+1 if bool(pinching.all()) else 0
            self.close_ticks=observed_close_ticks(self.close_ticks,close,observation)
            if self.lift_goal is None and self.confirm_ticks>=3:
                self.lift_goal=tcp[...,:3].clone();self.lift_goal[...,2]+=.08
            goals=goals if self.lift_goal is None else self.lift_goal
            action.zero_()
            if self.contact_torso_forward_m:
                # Bring the arm parents closer while retaining upright pitch.
                # The hand goals stay at the box; existing physical torso
                # travel/rate and measured collision limits remain active.
                target_xz=desired_xz.clone();target_xz[:,0]+=self.contact_torso_forward_m
                action[:,self.slices['height']]=(2*(target_xz-torso.processed_actions)/torso.cfg.speed_m_s).clamp(-.3,.3)
            for hand,solver in enumerate(self.solvers):
                columns=guide.columns[hand]
                solver._joint_command[:]=self.upper.processed_actions[:,columns]
                solver._joint_velocity[:]=observation[:,20:40][:,guide.velocity_columns[hand]]
                solver.process_actions(torch.cat((goals[:,hand],quaternion[:,hand]),-1))
                delta=(solver._joint_command-self.upper.processed_actions[:,columns])/self.upper._scale[:,columns]
                action[:,[self.slices['upper_body'].start+c for c in columns]]=delta.clamp(-1,1)
                action[:,self.slices[('left_gripper','right_gripper')[hand]]]=torch.where(close[:,hand,None] if self.lift_goal is None else torch.ones_like(close[:,hand,None]),1.,-1.)
            self.phase[:]=2 if self.lift_goal is not None else 1
        self.index+=1
        return action
    def reset(self,done): pass


def settle_reference_scene(env, demo, *, settle_all=False):
    """Reject a respawn instead of replaying onto a different target."""
    from ...runners.train_asymmetric_sac import _settle_initial_resets
    observation, rack = restore_inferred_scene(env, demo['actor_obs'][0])
    invalid_before = env._multi_box_reset_settling.invalid_count.clone()
    observation, steps = _settle_initial_resets(env, observation)
    expected = demo['actor_obs'][0,86:350].reshape(12,22)[:,0] > .5
    if settle_all and expected.sum()>1:
        from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
        pools=env._multi_box_pool_ids[0,env._multi_box_active[0]].tolist()
        names=physical_asset_names();stable_ticks=0
        for tick in range(90):
            velocities=torch.stack([env.scene[names[p]].data.root_vel_w[0] for p in pools])
            stable=bool(torch.isfinite(velocities).all() and
                (velocities[:,:3].norm(dim=-1)<.01).all() and
                (velocities[:,3:].norm(dim=-1)<.05).all())
            stable_ticks=stable_ticks+1 if stable else 0
            if stable_ticks>=8:
                break
            observation,_,terminated,truncated,_=env.step(torch.zeros_like(env.action_manager.action))
            if bool((terminated|truncated).any()):
                raise ValueError('A varied scene terminated while surrounding boxes were settling')
        else:
            raise ValueError('Surrounding boxes did not settle; this layout is not trainable')
        env.episode_length_buf[:]=0
        env._multi_box_privileged_grasp.reset(torch.tensor([0],device=env.device))
        env._multi_box_privileged_grasp_counter=-1;env._multi_box_grasp_safety_counter=-1
        observation=env.observation_manager.compute()
        steps+=tick
    actual = env._multi_box_active[0]
    rack_error = float((observation['policy'][0,68:77]
                       - demo['actor_obs'][0,68:77].to(env.device)).norm())
    print('[VR SCENE GUARD]', {'expected_target': torch.where(expected)[0].tolist(),
        'actual_target': torch.where(actual)[0].tolist(), 'rack_error': rack_error,
        'invalid_resets': (env._multi_box_reset_settling.invalid_count-invalid_before).tolist(),
        'footprint_invalid_total': env._multi_box_reset_settling.footprint_invalid_count.tolist(),
        'on_assigned_shelf': env._multi_box_reset_settling.on_assigned_shelf.tolist()}, flush=True)
    if (not torch.equal(actual.cpu(), expected.cpu()) or rack_error > .025
            or (env._multi_box_reset_settling.invalid_count != invalid_before).any()):
        raise ValueError('VR scene was replaced during settling; replay/Q import prohibited')
    if settle_all:
        from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
        names=physical_asset_names();settling=env._multi_box_reset_settling;checks=[]
        for logical in torch.where(actual)[0].tolist():
            pool=int(env._multi_box_pool_ids[0,logical])
            pose=env.scene[names[pool]].data.root_pose_w
            types=env._multi_box_box_type_ids[:,logical]
            regions=env._multi_box_region_ids[:,logical]
            footprint=bool(settling._footprint_in_region(pose,types,regions).all())
            on_shelf=bool(settling._on_assigned_shelf(pose,types,regions).all())
            checks.append(dict(logical=logical,footprint_in_region=footprint,on_assigned_shelf=on_shelf))
        print('[VR ALL BOX GUARD]',checks,flush=True)
        if any(not check['footprint_in_region'] or not check['on_assigned_shelf'] for check in checks):
            raise ValueError('A surrounding box settled outside its assigned shelf/region')
    return observation, rack, steps


def select_reference_episode(batch, index):
    """Select one complete demonstration without crossing reset boundaries."""
    ends = torch.where(batch['terminated'].bool())[0].tolist()
    if not 0 <= index < len(ends):
        raise ValueError(f'Expected episode index 0..{len(ends)-1}, got {index}')
    start = 0 if index == 0 else ends[index-1]+1
    stop = ends[index]+1
    return {key: value[start:stop] for key, value in batch.items()}
