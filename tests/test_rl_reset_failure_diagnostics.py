"""Original reset evidence survives partial respawns without changing physics."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import (
    ResetFailureCapture, validate_reset_diagnostic_request, zero_passive_roller_velocities,
)
from kuavo_isaaclab_scene.rl.multi_box.scene.reset_settling import IsaacResetSettling
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells, physical_asset_names
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def environment():
    spec = MultiBoxSpec()
    p, q = logical_cells(spec)[0].local_pose('small')
    pose = torch.tensor([(*p, *q)]).repeat(4, 1)
    pose[1, 1] -= 1.
    pose[3, 3] = float('nan')
    velocity = torch.zeros(4, 6)
    velocity[2, 0] = .02
    rack = torch.zeros(4, 7); rack[:, 3] = 1
    robot = rack.clone(); robot[:, 0] = 10.
    box = SimpleNamespace(joint_names=['flap'], data=SimpleNamespace(
        root_pose_w=pose, root_vel_w=velocity, joint_pos=torch.ones(4, 1)))
    scene = dict(rack=SimpleNamespace(data=SimpleNamespace(root_pose_w=rack)),
                 robot=SimpleNamespace(data=SimpleNamespace(root_pose_w=robot)))
    scene.update({name:box for name in physical_asset_names()})
    active = torch.zeros(4, 12, dtype=torch.bool); active[:, 0] = True
    return SimpleNamespace(num_envs=4, device='cpu', common_step_counter=0,
        cfg=SimpleNamespace(multi_box=spec), scene=scene,
        _multi_box_active=active, _multi_box_pool_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_box_type_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_region_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_rack_local_positions=torch.zeros(4,12,3))


def advance(env, tracker):
    env.common_step_counter += 1
    return tracker.measure(.1)


def test_first_failure_is_measured_before_respawn_and_survives_replacement_failures():
    env = environment(); tracker = IsaacResetSettling(env)
    tracker.failure_capture = ResetFailureCapture(env)
    original_pose = env.scene[tracker.names[0]].data.root_pose_w.clone()
    advance(env,tracker)
    assert tracker.failure_capture.records == []  # original first-step contact grace
    advance(env,tracker)
    captured = deepcopy(tracker.failure_capture.records)
    assert [r['environment'] for r in captured] == [1,3]
    assert captured[0]['box_pose_world'][1] == float(original_pose[1,1])
    assert captured[0]['left_region'] and not captured[0]['timed_out']
    assert captured[0]['robot_pose_world'][0] == 10.
    assert captured[1]['invalid_box_pose'] and captured[1]['box_pose_world'][3] is None
    assert captured[1]['rack_local_root_xyz_m'] is None
    json.dumps(captured, allow_nan=False)
    # This is the same tracker reset used by partial respawn. Park the old
    # asset and provoke a replacement failure; original evidence stays exact.
    tracker.reset(torch.tensor([1,3]))
    env.scene[tracker.names[0]].data.root_pose_w[1,1] = -100.
    advance(env,tracker); advance(env,tracker)
    assert tracker.failure_capture.records == captured
    assert tracker.invalid_count[1] == 2
    # Capture is bounded by the number of requested environments, not retries.
    assert len(tracker.failure_capture.records) <= env.num_envs


def test_timeout_is_distinct_from_geometry_and_capture_does_not_mutate_scene():
    env = environment(); tracker = IsaacResetSettling(env)
    tracker.failure_capture = ResetFailureCapture(env)
    box = env.scene[tracker.names[0]]
    original = [v.clone() for v in (box.data.root_pose_w,box.data.root_vel_w,box.data.joint_pos)]
    for _ in range(22): advance(env,tracker)
    cases = {r['environment']:r for r in tracker.failure_capture.records}
    assert 0 not in cases and tracker.ready[0]
    assert cases[2]['timed_out'] and not cases[2]['left_region']
    assert cases[2]['footprint_in_region'] and cases[2]['on_assigned_shelf']
    assert not cases[2]['stable']
    assert cases[2]['box_velocity_world'][0] == pytest.approx(.02)
    for before,after in zip(original,(box.data.root_pose_w,box.data.root_vel_w,box.data.joint_pos)):
        torch.testing.assert_close(before,after,equal_nan=True)


def test_disabled_capture_keeps_reset_outcomes_identical():
    plain,observed = environment(),environment()
    a,b = IsaacResetSettling(plain),IsaacResetSettling(observed)
    assert a.failure_capture is None
    b.failure_capture = ResetFailureCapture(observed)
    for _ in range(22):
        x,y = advance(plain,a),advance(observed,b)
        for name in x.__dataclass_fields__:
            torch.testing.assert_close(getattr(x,name),getattr(y,name))
    for name in ('invalid_count','region_invalid_count','shelf_invalid_count',
                 'footprint_invalid_count','timeout_invalid_count','nonfinite_invalid_count'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('waves,training,steps',[
    ([dict(split='train')],False,1),([dict(split='holdout')],False,1),
    ([dict(split='validation')],True,1),([dict(split='validation')],False,900),
    ([],False,1),
])
def test_diagnostic_mode_cannot_collect_training_or_inspect_independent_final(waves,training,steps):
    with pytest.raises(ValueError,match='frozen DEV'):
        validate_reset_diagnostic_request(waves,enabled=True,training=training,steps=steps)
    validate_reset_diagnostic_request(waves,enabled=False,training=training,steps=steps)


def test_frozen_dev_startup_capture_is_allowed():
    validate_reset_diagnostic_request([dict(split='validation')],enabled=True,training=False,steps=1)


def test_passive_roller_probe_removes_spin_only_in_selected_environment():
    class Asset:
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.tensor([[1.,2.],[3.,4.]]),
                joint_vel=torch.tensor([[5.,6.],[7.,8.]]))
            self.velocity_target=torch.full((2,2),9.)
            self.effort_target=torch.full((2,2),10.)
        def write_joint_velocity_to_sim(self,value,env_ids):self.data.joint_vel[env_ids]=value
        def set_joint_velocity_target(self,value,env_ids):self.velocity_target[env_ids]=value
        def set_joint_effort_target(self,value,env_ids):self.effort_target[env_ids]=value
    roller,box=Asset(),Asset()
    env=SimpleNamespace(scene=SimpleNamespace(articulations=dict(rack_roller_deck_02=roller,mb_s2_small_5=box)))
    assert zero_passive_roller_velocities(env,torch.tensor([1]))==['rack_roller_deck_02']
    assert roller.data.joint_pos.tolist()==[[1.,2.],[3.,4.]]
    assert roller.data.joint_vel.tolist()==[[5.,6.],[0.,0.]]
    assert roller.velocity_target.tolist()==[[9.,9.],[0.,0.]]
    assert roller.effort_target.tolist()==[[10.,10.],[0.,0.]]
    assert box.data.joint_vel.tolist()==[[5.,6.],[7.,8.]]


def test_passive_roller_probe_rejects_a_scene_without_roller_decks():
    with pytest.raises(ValueError,match='live rack roller'):
        zero_passive_roller_velocities(SimpleNamespace(scene=SimpleNamespace(articulations={})),torch.tensor([0]))
