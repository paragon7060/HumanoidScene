"""Run the real multi-flap measurement on CPU tensors, without physics or rendering."""

import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.flap_grasp import FlapGrasp
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


@pytest.fixture
def state(monkeypatch):
    geometry = ModuleType("kuavo_isaaclab_scene.rl.mdp.geometry")
    def rotate(q, p):
        v = q[..., 1:]
        cross = 2 * torch.linalg.cross(v, p)
        return p + q[..., :1] * cross + torch.linalg.cross(v, cross)
    geometry.rotate = rotate
    geometry.unrotate = lambda q, p: rotate(q * torch.tensor([1., -1., -1., -1.]), p)
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)
    # Two cloned cells, two boxes, different active boxes and rigid rotations.
    q = torch.tensor([[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5]])
    origin = torch.tensor([[0., 0., 0.], [8., 0., 0.]])
    local_pos = torch.tensor([[[-.2, 0., 1.], [.2, 0., 1.]]]).expand(2, -1, -1)
    pos = rotate(q[:, None].expand(-1, 2, -1), local_pos) + origin[:, None]
    boxes = [NS(find_bodies=lambda *a, **kw: ([0, 1], []),
                data=NS(body_link_pos_w=pos.clone(), body_link_quat_w=q[:, None].expand(-1, 2, -1))) for _ in range(2)]
    # Right fingers straddle flap0 at its middle height, NOT its upper band.
    fingers = torch.zeros(2, 4, 3)
    for jaw, offset in enumerate((-.01, .01)):
        fingers[:, 2 + jaw] = pos[:, 0] + rotate(q, torch.tensor([[offset, 0., 0.]]).expand(2, -1))
    fingers[:, :2] = origin[:, None] + torch.tensor([0., 1., 1.])
    tools = torch.stack((origin + torch.tensor([0., 1., 1.]),
                         pos[:, 0] + rotate(q, torch.tensor([[.03, 0., 0.]]).expand(2, -1))), 1)
    spec = task_spec("pick", control_mode="arms-only", box_names=("a", "b"), grasp_mode="flap_top")
    scene = {}
    for i in range(4):
        point = torch.full((2, 1, 4, 3), float('nan'))
        force = torch.zeros(2, 1, 4, 3)
        if i >= 2:
            sign = -1 if i == 2 else 1
            for env in range(2):
                point[env, 0, env*2] = pos[env, 0] + rotate(q[env], torch.tensor([sign*.002, 0., 0.]))
                force[env, 0, env*2] = rotate(q[env], torch.tensor([sign*1., 0., 0.]))
        scene[f"grasp_contact_{i}"] = NS(data=NS(contact_pos_w=point, force_matrix_w=force,
                                               net_forces_w=force.sum(2)))
    geom = NS(center=(0., 0., 0.), half_size=(.002, .1, .055))
    t = NS(spec=spec, boxes=boxes, num_envs=2, n=2, device="cpu", ids=torch.arange(2),
           active_box=torch.tensor([0, 1]), tools=tools,
           robot=NS(find_bodies=lambda *a, **kw: ([0, 1, 2, 3], []), data=NS(body_link_pos_w=fingers)),
           cfg=NS(geometry={name: NS(flaps={f: geom for f in spec.grasp_flaps}) for name in spec.box_names}),
           _env=NS(scene=scene, common_step_counter=0, step_dt=1/30))
    return t, FlapGrasp(t)


def test_right_hand_accepts_previously_left_assigned_flap_and_clones_rotate_correctly(state):
    t, grasp = state
    grasp.measure()
    assert t.grasped.tolist() == [True, True]
    assert t.contact_flap_index[:, 1].tolist() == [0, 0]
    assert not t.hand_grasp_flags[:, 0].any()
    torch.testing.assert_close(t.hand_target_distance[:, 1], torch.full((2,), .028), atol=1e-6, rtol=1e-5)
    assert not t.unexpected_finger_force.any()


def test_two_jaws_on_different_flaps_do_not_make_a_grasp(state):
    t, grasp = state
    sensor = t._env.scene['grasp_contact_3'].data
    for env in range(2):
        index = env * 2
        sensor.force_matrix_w[env, 0, index+1] = sensor.force_matrix_w[env, 0, index]
        sensor.force_matrix_w[env, 0, index] = 0
    grasp.measure()
    assert not t.grasped.any()


def test_distance_target_can_switch_without_reassigning_held_contact(state):
    t, grasp = state
    grasp.measure()
    t.tools[:, 1] = t.boxes[0].data.body_link_pos_w[:, 1]
    t._env.common_step_counter += 1
    grasp.measure()
    assert t.nearest_flap_index[:, 1].tolist() == [1, 1]
    assert t.contact_flap_index[:, 1].tolist() == [0, 0]
    assert t.grasped.all()


def test_top_band_option_rejects_middle_contacts_and_nan_never_acquires(state):
    from dataclasses import replace
    t, grasp = state
    t.spec = replace(t.spec, flap_contact_region="top_band")
    grasp.measure()
    assert not t.grasped.any()
    t.spec = replace(t.spec, flap_contact_region="surface")
    t._env.common_step_counter += 1
    for sensor in t._env.scene.values():
        sensor.data.contact_pos_w[:] = float('nan')
    grasp.measure()
    assert not t.grasped.any()
