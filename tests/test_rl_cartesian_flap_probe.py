from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.tensor_arm_kinematics import TensorArmKinematics
from kuavo_isaaclab_scene.rl.multi_box.experiments.cartesian_flap_probe import (
    FrozenCartesianFlapProbe, install_frozen_cartesian_flap_probe, contact_region_offsets,
    cartesian_flap_probe_contract)
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates


def rotation6(R):
    return torch.cat((R[..., :, 0], R[..., :, 1]), -1)


@pytest.mark.parametrize('hand', [0, 1])
def test_tcp_jacobian_predicts_independent_measured_joint_perturbations(hand):
    model = TensorArmKinematics(dtype=torch.float64)
    q = torch.zeros(1, 20, dtype=torch.float64)
    q[:, :4] = torch.tensor([.18, -.16, -.02, .2])
    for h, columns in enumerate(model.columns):
        q[:, columns] = ((model.lower[h] + model.upper[h]) / 2).clamp(-.4, .4)
    p, R, J = model.fk(q)
    eps = 1e-6
    for local, column in enumerate(model.columns[hand]):
        plus, minus = q.clone(), q.clone()
        plus[:, column] += eps; minus[:, column] -= eps
        pp, Rp, _ = model.fk(plus); pm, Rm, _ = model.fk(minus)
        derivative = (pp[:, hand] - pm[:, hand]) / (2 * eps)
        assert torch.allclose(derivative, J[:, hand, :3, local], atol=1e-8, rtol=0)
        skew = ((Rp[:, hand] - Rm[:, hand]) / (2 * eps)) @ R[:, hand].transpose(-1, -2)
        angular = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), -1)
        assert torch.allclose(angular, J[:, hand, 3:, local], atol=1e-8, rtol=0)


def fixture():
    model = TensorArmKinematics()
    raw = torch.zeros(2, 464)
    raw[:, 439] = 1
    raw[:, 71:77] = torch.tensor([1., 0, 0, 0, 1, 0])
    p, R, _ = model.fk(raw[:, :20])
    raw[:, 50:68] = torch.cat((p, rotation6(R)), -1).flatten(1)
    supplemental = torch.zeros(2, 38)
    relation = supplemental[:, :36].reshape(2, 2, 2, 9)
    relation[..., 0] = .1
    relation[..., 3:] = torch.tensor([1., 0, 0, 0, 1, 0])
    supplemental[:, 36] = 1
    coordinates = PoseGoalCoordinates(exact_projected_base=True)
    joints, torso, _, _, _ = coordinates.current(raw)
    center = torch.cat((joints[0], torso[0], torch.zeros(2)))
    goals = torch.zeros(2, 21); goals[:, 19:] = -1
    actor = torch.zeros(2, 518); critic = torch.zeros(2, 578)
    agent = SimpleNamespace(anchor_and_scale=lambda _: (torch.zeros(2, 19), torch.full((2, 19), .3)),
        action_projector=SimpleNamespace(entropy_mask=lambda _: torch.ones(2, 21)))
    stage = SimpleNamespace(target_xy=torch.zeros(2, 2), target_yaw=torch.zeros(2))
    pilot = SimpleNamespace(coordinates=coordinates, center=center, scale=torch.ones(21), stage=stage, agent=agent)
    physical = held_goal_coordinates(coordinates, raw, center + goals, stage)
    return pilot, raw, torch.zeros(2, 523), (physical, (actor, critic, goals)), supplemental


def test_guide_records_projected_goals_and_preserves_every_nonarm_body_channel():
    pilot, raw, critic, original, supplemental = fixture()
    before = [raw.clone(), supplemental.clone(), original[0].clone(), original[1][2].clone()]
    tracker = FrozenCartesianFlapProbe(128, raw)
    command, (actor, context, goals) = tracker.step(pilot, raw, critic, original,
        torch.tensor([5, 19]), supplemental)
    assert tracker.phase[5] >= 0 and tracker.phase[19] >= 0 and tracker.phase[0] == -1
    assert torch.equal(command[:, [0, 1, 2, 3, 18, 19, 22, 23]], original[0][:, [0, 1, 2, 3, 18, 19, 22, 23]])
    assert goals[:, :19].abs().max() <= .300001
    assert set(goals[:, 19:].flatten().tolist()) <= {-1., 1.}
    decoded = held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * goals, pilot.stage)
    assert torch.equal(decoded, command)
    assert actor is original[1][0] and context is original[1][1]
    assert all(torch.equal(a, b) for a, b in zip(before, [raw, supplemental, original[0], original[1][2]]))
    assert tracker.statistics['confirmed_lift_episodes'] == 0


def test_mismatched_live_tool_frame_refused_before_any_guide_command():
    pilot, raw, critic, original, supplemental = fixture()
    raw[:, 50] += .02
    with pytest.raises(ValueError, match='does not match'):
        FrozenCartesianFlapProbe(128, raw).step(pilot, raw, critic, original,
            torch.tensor([5, 19]), supplemental)


@pytest.mark.parametrize('contacts', ['empty_closed', 'same_flap', 'opposing'])
def test_lift_requires_eight_real_opposing_pinch_ticks_not_closed_fraction(contacts):
    pilot, raw, critic, original, supplemental = fixture()
    raw[:, 46:50] = 1
    if contacts != 'empty_closed':
        critic[:, 464 + 35:464 + 37] = 1
        critic[:, 464 + 37] = 1
        critic[:, 464 + (40 if contacts == 'opposing' else 39)] = 1
    ids = torch.tensor([5, 19])
    tracker = FrozenCartesianFlapProbe(128, raw)
    tracker.phase[ids] = 1
    for _ in range(7):
        tracker.step(pilot, raw, critic, original, ids, supplemental)
        assert tracker.statistics['confirmed_lift_episodes'] == 0
    tracker.step(pilot, raw, critic, original, ids, supplemental)
    assert tracker.statistics['confirmed_lift_episodes'] == (2 if contacts == 'opposing' else 0)


def test_learning_rejected_and_dedicated_hooks_restored():
    class Pilot:
        training = True; actor_updates = 0; critic_updates = 0; replay = SimpleNamespace(size=0)
        def act(self, *args, **kwargs):
            raise AssertionError('Learning should be refused before act')
        def report(self): return {}
    manifest = SimpleNamespace(checkpoint_manifest_fields=lambda: {})
    original = Pilot.act, Pilot.report, manifest.checkpoint_manifest_fields
    restore = install_frozen_cartesian_flap_probe(Pilot, manifest)
    try:
        with pytest.raises(ValueError, match='cannot train'):
            Pilot().act(None, None, 0)
        assert manifest.checkpoint_manifest_fields()['frozen_cartesian_flap_probe']['Q_import_eligible'] is False
        assert Pilot().report()['frozen_cartesian_flap_probe']['privileged_pinch_used_for_teacher_hold_and_lift'] is True
    finally:
        restore()
    assert (Pilot.act, Pilot.report, manifest.checkpoint_manifest_fields) == original


def test_cartesian_probe_cannot_supply_original_waypoint_or_Q(tmp_path):
    import json
    from prepare_size_workplaces import closed_probe
    run = tmp_path / 'run'; run.mkdir()
    (tmp_path / 'launch.json').write_text(json.dumps({'run': str(run),
        'command': ['--checkpoint', str(tmp_path / 'checkpoint.pt'), '--waypoints', str(tmp_path / 'waypoints.json')]}))
    (tmp_path / 'status.json').write_text(json.dumps({'training_exit_code': 0, 'training_pid': 2147483647}))
    (run / 'status.json').write_text(json.dumps({'status': 'complete'}))
    (run / 'manifest.json').write_text(json.dumps({'frozen_cartesian_flap_probe': {'standalone_SAC': False}}))
    with pytest.raises(ValueError, match='Changed Cartesian'):
        closed_probe(run)


def region_fixture():
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import ActorFeatures
    pilot, raw, critic, original, supplemental = fixture()
    layout = ActorFeatures(464, 'grasp_target_no_history')
    raw[:, layout.token_start] = 1
    raw[:, layout.token_start + 3] = 1
    raw[:, layout.token_start + 5:layout.token_start + 8] = torch.tensor([.266, .185, .13])
    raw[:, layout.mask_start] = 1
    raw[:, layout.target_start] = 1
    return pilot, raw, critic, original, supplemental


def test_contact_region_preserves_valid_off_center_grasp_and_stays_inside_panel():
    relation = torch.zeros(1, 2, 9)
    relation[..., 3:] = torch.tensor([1., 0, 0, 0, 1, 0])
    # An observed successful contact can be6.7cm from the midpoint, yet
    # inside the panel. The second hand starts beyond both tangent edges.
    relation[0, 0, :3] = torch.tensor([.0005, .067, .0036])
    relation[0, 1, :3] = torch.tensor([.03, -.2, -.3])
    half = torch.tensor([[[.0015, .09, .05], [.0015, .09, .05]]])
    point = contact_region_offsets(relation, half)
    assert torch.allclose(point[0, 0], torch.tensor([0., -.067, -.0036]), atol=1e-7, rtol=0)
    assert torch.allclose(point[0, 1], torch.tensor([0., .085, .045]), atol=1e-7, rtol=0)
    assert point[0, 0].norm() > .018
    assert (point.abs() <= half).all()


def test_region_guide_cannot_reopen_an_original_production_permitted_neural_close():
    pilot, raw, critic, original, supplemental = region_fixture()
    goals = original[1][2].clone(); goals[:, 19:21] = 1
    original = (held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * goals, pilot.stage),
                (original[1][0], original[1][1], goals))
    tracker = FrozenCartesianFlapProbe(128, raw, contact_region=True)
    command, (_, _, executed) = tracker.step(pilot, raw, critic, original, torch.tensor([5, 19]), supplemental)
    assert torch.equal(executed[:, 19:21], goals[:, 19:21])
    assert torch.equal(command[:, 20:22], torch.ones(2, 2))
    assert tracker.statistics['original_neural_close_retained_rows'] == 2
    assert tracker.statistics['extra_assisted_close_rows'] == 0
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * executed, pilot.stage))
    assert torch.equal(command[:, [0, 1, 2, 3, 18, 19, 22, 23]], original[0][:, [0, 1, 2, 3, 18, 19, 22, 23]])
    assert executed[:, :19].abs().max() <= .300001


def test_region_point_is_selected_once_while_midpoint_observations_remain_unchanged():
    pilot, raw, critic, original, supplemental = region_fixture()
    relation = supplemental[:, :36].reshape(2, 2, 2, 9)
    relation[..., 1] = .07
    tracker = FrozenCartesianFlapProbe(128, raw, contact_region=True)
    ids = torch.tensor([5, 19])
    tracker.step(pilot, raw, critic, original, ids, supplemental)
    points = tracker.contact_offsets[ids].clone()
    assert torch.allclose(points[..., 1], torch.full((2, 2), -.07))
    relation[..., 1] += .005
    before = supplemental.clone()
    tracker.step(pilot, raw, critic, original, ids, supplemental)
    assert torch.equal(tracker.contact_offsets[ids], points)
    assert torch.equal(supplemental, before)
    assert cartesian_flap_probe_contract(contact_region=True)['supplemental_midpoint_observations_unchanged']


def test_real_pinch_holds_measured_arm_orientation_instead_of_rotating_the_flap():
    pilot, raw, critic, original, supplemental = region_fixture()
    model = TensorArmKinematics()
    for hand, columns in enumerate(model.columns):
        raw[:, columns] = (model.lower[hand] + model.upper[hand]) / 2
    p, R, _ = model.fk(raw[:, :20])
    raw[:, 50:68] = torch.cat((p, rotation6(R)), -1).flatten(1)
    joints, torso, _, _, _ = pilot.coordinates.current(raw)
    pilot.center = torch.cat((joints[0], torso[0], torch.zeros(2)))
    original = (held_goal_coordinates(pilot.coordinates, raw, pilot.center + original[1][2], pilot.stage), original[1])
    critic[:, 464 + 35] = 1
    critic[:, 464 + 37] = 1
    tracker = FrozenCartesianFlapProbe(128, raw, contact_region=True)
    command, _ = tracker.step(pilot, raw, critic, original, torch.tensor([5, 19]), supplemental)
    assert command[:, 4:11].abs().max() < 1e-5
    assert tracker.statistics['confirmed_lift_episodes'] == 0


def test_URDF_teacher_bypasses_source_range_with_exact_labels_and_original_physical_caps():
    pilot, raw, critic, original, supplemental = region_fixture()
    pilot.scale[1:15] = .001
    tracker = FrozenCartesianFlapProbe(128, raw, contact_region=True, arm_goal_bounds='urdf')
    command, (_, _, goals) = tracker.step(pilot, raw, critic, original, torch.tensor([5, 19]), supplemental)
    assert goals[:, 1:15].abs().max() > 1
    assert command.abs().max() <= 1
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * goals, pilot.stage))
    assert torch.equal(command[:, [0, 1, 2, 3, 18, 19, 22, 23]], original[0][:, [0, 1, 2, 3, 18, 19, 22, 23]])
    target = (pilot.center + pilot.scale * goals)[:, 1:15].reshape(2, 2, 7)
    assert (target >= tracker.kinematics.lower + .01 - 1e-6).all()
    assert (target <= tracker.kinematics.upper - .01 + 1e-6).all()
    measured = torch.stack([raw[:, columns] for columns in tracker.kinematics.columns], 1)
    assert (target - measured).abs().max() <= .020001
    assert tracker.statistics['source_affine_arm_range_bypassed_rows'] == 2
    assert tracker.statistics['affine_arm_goal_clamped_rows'] == 0
    contract = cartesian_flap_probe_contract(contact_region=True, arm_goal_bounds='urdf')
    assert contract['stored_teacher_goals_NOT_SAC_actor_actions']
    assert contract['Q_import_eligible'] is False and contract['standalone_SAC'] is False


@pytest.mark.parametrize('mode,region', [('unknown', True), ('urdf', False)])
def test_URDF_bounds_require_a_known_explicit_contact_region_diagnostic(mode, region):
    with pytest.raises(ValueError, match='URDF arm diagnostic'):
        cartesian_flap_probe_contract(contact_region=region, arm_goal_bounds=mode)


def test_URDF_teacher_still_rejects_learning_before_the_source_actor_runs():
    class Pilot:
        training = False; actor_updates = 1; critic_updates = 0; replay = SimpleNamespace(size=0)
        def act(self, *args, **kwargs):
            raise AssertionError('Changed teacher must reject learned Q before acting')
        def report(self): return {}
    manifest = SimpleNamespace(checkpoint_manifest_fields=lambda: {})
    original = Pilot.act, Pilot.report, manifest.checkpoint_manifest_fields
    restore = install_frozen_cartesian_flap_probe(Pilot, manifest, contact_region=True, arm_goal_bounds='urdf')
    try:
        with pytest.raises(ValueError, match='cannot train'):
            Pilot().act(None, None, 0)
        assert manifest.checkpoint_manifest_fields()['frozen_cartesian_flap_probe']['source_affine_arm_goal_bounds_bypassed']
    finally:
        restore()
    assert (Pilot.act, Pilot.report, manifest.checkpoint_manifest_fields) == original


def test_pending_lift_keeps_every_precontact_command_identical():
    pilot, raw, critic, original, supplemental = region_fixture()
    raw[:, 416:436] = .04
    original = (held_goal_coordinates(pilot.coordinates, raw,
        pilot.center + pilot.scale * original[1][2], pilot.stage), original[1])
    a = FrozenCartesianFlapProbe(128, raw, contact_region=True, arm_goal_bounds='urdf')
    b = FrozenCartesianFlapProbe(128, raw, contact_region=True, arm_goal_bounds='urdf',
        lift_drive='bounded-pending-target')
    ids = torch.tensor([5, 19])
    old, (_, _, oldgoals) = a.step(pilot, raw, critic, original, ids, supplemental)
    new, (_, _, newgoals) = b.step(pilot, raw, critic, original, ids, supplemental)
    assert torch.equal(old, new) and torch.equal(oldgoals, newgoals)
    assert b.statistics['lift_pending_target_integrated_rows'] == 0


@pytest.mark.parametrize('opposing', [False, True])
def test_pending_lift_accumulates_only_with_real_opposing_pinch_and_bounds_joint_lead(opposing):
    pilot, raw, critic, original, supplemental = region_fixture()
    model = TensorArmKinematics()
    for hand, columns in enumerate(model.columns):
        raw[:, columns] = (model.lower[hand] + model.upper[hand]) / 2
    p, R, _ = model.fk(raw[:, :20])
    raw[:, 50:68] = torch.cat((p, rotation6(R)), -1).flatten(1)
    joints, torso, _, _, _ = pilot.coordinates.current(raw)
    pilot.center = torch.cat((joints[0], torso[0], torch.zeros(2)))
    raw[:, 416:436] = .079
    original = (held_goal_coordinates(pilot.coordinates, raw,
        pilot.center + pilot.scale * original[1][2], pilot.stage), original[1])
    critic[:, 464 + 35:464 + 37] = 1
    critic[:, 464 + 37] = 1
    critic[:, 464 + (40 if opposing else 39)] = 1
    ids = torch.tensor([5, 19])
    tracker = FrozenCartesianFlapProbe(128, raw, contact_region=True, arm_goal_bounds='urdf',
        lift_drive='bounded-pending-target')
    tracker.phase[ids] = 2
    tracker.lift_targets_rack[ids] = p + torch.tensor([0., 0., .025])
    command, (_, _, goals) = tracker.step(pilot, raw, critic, original, ids, supplemental)
    targets = (pilot.center + pilot.scale * goals)[:, 1:15].reshape(2, 2, 7)
    measured = torch.stack([raw[:, cols] for cols in model.columns], 1)
    assert (targets >= model.lower + .01).all() and (targets <= model.upper - .01).all()
    assert command.abs().max() <= 1
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw,
        pilot.center + pilot.scale * goals, pilot.stage))
    assert (targets - measured).abs().max() <= (.080001 if opposing else .020001)
    assert tracker.statistics['lift_pending_target_integrated_rows'] == (2 if opposing else 0)
    if opposing:
        assert (targets - measured).abs().max() > .02
        assert tracker.statistics['lift_target_lead_clamped_rows'] == 2
    contract = cartesian_flap_probe_contract(contact_region=True, arm_goal_bounds='urdf',
        lift_drive='bounded-pending-target')
    assert contract['approach_and_insertion_target_rule_unchanged']
    assert contract['name'].endswith('v4') and contract['Q_import_eligible'] is False


@pytest.mark.parametrize('bounds,drive', [('source-affine', 'bounded-pending-target'), ('urdf', 'unknown')])
def test_pending_lift_requires_the_explicit_known_URDF_contract(bounds, drive):
    with pytest.raises(ValueError, match='Pending-target lift'):
        cartesian_flap_probe_contract(contact_region=True, arm_goal_bounds=bounds, lift_drive=drive)
