"""CPU checks for the multi-box deployable-actor/privileged-critic SAC."""

import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import (
    ActorFeatures,
    ActorImitationBuffer,
    AsymmetricReplayBuffer,
    AsymmetricSAC,
    SuccessfulTransitionHistory,
)
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import (
    _ApproachDiagnostics,
    _SafetyDiagnostics,
    _grasp_distance_masks,
    _reset_settling_metrics,
    _reward_breakdown,
    _demo_fraction,
    _sample_warmup_action,
    _settle_initial_resets,
    _termination_snapshot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import (
    _compatible_checkpoint,
    apply_run_profile,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import (
    GraspActionProjector, GuidedDemoWarmup, RELATION_START, ASSIGNMENT_START,
    assigned_flap_center_distance,
)
from kuavo_isaaclab_scene.rl.multi_box.observations import flat_actor_observation_dim
from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import (
    entry_geometry, observed_close_ticks, successful_demo_grasp_offsets,
    retarget_grasp_goal, target_token,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.episode_guidance import EpisodicIKGuidance
from kuavo_isaaclab_scene.rl.multi_box.experiments.imitation_schedule import (
    imitation_fraction, teacher_fraction, validate_teacher_schedule,
)


def test_no_history_actor_keeps_controller_state_and_ignores_previous_action():
    torch.manual_seed(12)
    original = ActorFeatures(464, "grasp_target")
    encoder = ActorFeatures(464, "grasp_target_no_history")
    observations = torch.randn(8, 464)
    expected = original(observations)[:, :-24]
    assert encoder.output_dim == 174
    assert torch.equal(encoder(observations), expected)
    changed = observations.clone()
    changed[:, 440:] = torch.randn(8, 24) * 100
    assert torch.equal(encoder(changed), expected)
    changed[:, 416:440] += 1
    assert not torch.equal(encoder(changed), expected)
    with pytest.raises(ValueError, match='controller-state'):
        ActorFeatures(403, "grasp_target_no_history")


def test_no_history_actor_has_no_gradient_to_its_previous_actions():
    agent = AsymmetricSAC(464, 530, 24, SACConfig(
        hidden=16, actor_feature_mode="grasp_target_no_history"))
    observations = torch.randn(4, 464, requires_grad=True)
    actions = agent.actor(agent.actor_normalizer(agent.actor_features(observations)), deterministic=True)[0]
    actions.sum().backward()
    assert torch.count_nonzero(observations.grad[:, 440:]) == 0
    assert torch.count_nonzero(observations.grad[:, 416:439]) > 0
    state = agent.checkpoint()
    restored = AsymmetricSAC(464, 530, 24, SACConfig(**state['config']))
    restored.restore(state, training=False)
    assert torch.equal(agent.act(observations.detach(), True), restored.act(observations.detach(), True))
    state['inference_only'] = True
    with pytest.raises(ValueError, match='inference-only'):
        restored.restore(state, training=True)
    restored.restore(state, training=False)


def test_live_teacher_continues_after_short_pilot_vr_schedule_ends():
    args = SimpleNamespace(teacher_batch_fraction=.2, teacher_min_batch_fraction=.1,
                           teacher_decay_updates=128_000)
    assert _demo_fraction(.2, 40_000, 15_360) == 0
    assert teacher_fraction(args, 40_000) == pytest.approx(.1375)
    assert teacher_fraction(args, 128_000) == .1
    assert imitation_fraction(.2, 0, 128_000, 128_000) == 0
    guide = EpisodicIKGuidance(4, "cpu", .2, 128_000, minimum_fraction=.1)
    assert guide.fraction(128_000) == .1
    with pytest.raises(ValueError):
        imitation_fraction(.2, .3, 0, 100)


def test_independent_teacher_requires_queries_and_valid_schedule():
    args = SimpleNamespace(teacher_batch_fraction=.2, teacher_min_batch_fraction=.1,
        teacher_decay_updates=128_000, teacher_bc_strength=100,
        online_teacher_labels=False, demo_dataset="demo.hdf5",
        online_ik_episode_fraction=.2, online_ik_min_episode_fraction=.1,
        online_ik_decay_updates=128_000)
    with pytest.raises(ValueError, match="online teacher labels"):
        validate_teacher_schedule(args)
    args.online_teacher_labels = True
    validate_teacher_schedule(args)


def test_teacher_actor_labels_train_without_vr_and_do_not_change_critic_batch():
    torch.manual_seed(9)
    cfg = SACConfig(hidden=16, initial_alpha=1e-7, actor_lr=.001)
    teacher_agent = AsymmetricSAC(4, 7, 2, cfg, "cpu")
    control = AsymmetricSAC(4, 7, 2, cfg, "cpu")
    control.load_state_dict(teacher_agent.state_dict())
    batch = _batch(8)
    labels = {"actor_obs": batch["actor_obs"], "action": torch.ones(8, 2) * .7}
    # The teacher has no reward, critic view or next state to inject into Q.
    torch.manual_seed(17)
    result = teacher_agent.update(batch, teacher=labels, teacher_weight=20)
    torch.manual_seed(17)
    control_result = control.update(batch)
    assert result["demo_bc_weight"] == 0
    assert result["teacher_bc_loss"] > 0 and result["teacher_bc_weight"] == 20
    assert result["q_loss"] == control_result["q_loss"]
    for a, b in zip(teacher_agent.q1.parameters(), control.q1.parameters(), strict=True):
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(
        teacher_agent.actor.parameters(), control.actor.parameters(), strict=True))


def test_rack_peak_body_reports_one_cause_per_failure_before_reset():
    monitor = _SafetyDiagnostics("cpu", rack_body_names=("arm", "gripper"))
    false = torch.zeros(3, dtype=torch.bool)
    safety = SimpleNamespace(invalid_box_pose=false, invalid_flap_pose=false,
        robot_rack_collision=torch.tensor([True, True, True]),
        obstacle_collision=false, workspace_limit=false, box_drop=false,
        box_lift_limit=false, box_speed_limit=false, self_collision=false,
        contact_eligible=torch.tensor([True, True, False]),
        rack_force_n=torch.tensor([20., 30., 100.]), obstacle_force_n=torch.zeros(3),
        rack_body_force_n=torch.tensor([[20., 15.], [11., 30.], [100., 1.]]))
    monitor.record(safety, torch.ones(3, dtype=torch.bool))
    result = monitor.report()
    assert result["unsafe_rack_peak_body/arm"] == 1
    assert result["unsafe_rack_peak_body/gripper"] == 1
    assert result["contact_force/rack_body_arm_max_n"] == 20
    assert result["contact_force/rack_body_gripper_max_n"] == 30


def test_v2_safety_diagnostics_separates_unsafe_causes_and_force_bands():
    monitor = _SafetyDiagnostics("cpu", ("Surface", "RailLeft"))
    false = torch.zeros(3, dtype=torch.bool)
    monitor.record(SimpleNamespace(
        invalid_box_pose=torch.tensor([True, False, False]),
        invalid_flap_pose=torch.tensor([False, True, False]),
        robot_rack_collision=torch.tensor([False, True, False]),
        obstacle_collision=torch.tensor([False, True, False]),
        workspace_limit=false, box_drop=torch.tensor([False, False, True]),
        box_lift_limit=false, box_speed_limit=false, self_collision=false,
        contact_eligible=torch.tensor([True, True, False]),
        rack_force_n=torch.tensor([0.2, 12.0, 100.0]),
        obstacle_force_n=torch.tensor([0.0, 6.0, 100.0]),
        obstacle_target_force_n=torch.tensor([[0.0, 0.0], [6.0, 4.0], [100.0, 100.0]]),
    ), torch.tensor([False, True, True]))

    result = monitor.report()
    assert result["reset_invalid_box_pose"] == 1
    assert result["reset_invalid_flap_pose"] == 1
    assert result["unsafe_cause/robot_rack_collision"] == 1
    assert result["unsafe_cause/obstacle_collision"] == 1
    assert result["unsafe_cause/box_drop"] == 1
    assert result["unsafe_cause/overlap"] == 1
    assert result["unsafe_cause/unattributed"] == 0
    assert result["contact_force/eligible_samples"] == 2
    assert result["contact_force/rack_gt_0p1_n"] == 2
    assert result["contact_force/rack_gt_10p0_n"] == 1
    assert result["contact_force/obstacle_gt_5p0_n"] == 1
    assert result["contact_force/rack_max_n"] == 12.0
    assert result["unsafe_obstacle/Surface"] == 1
    assert result["unsafe_obstacle/RailLeft"] == 0
    assert result["contact_force/obstacle_RailLeft_max_n"] == 4.0


def test_safe_approach_diagnostics_identify_where_rack_collisions_occur():
    monitor = _ApproachDiagnostics("cpu")
    distances = torch.tensor([[0.1, 0.3], [0.4, 0.8], [1.2, 1.4], [0.2, 0.2]])
    safety = SimpleNamespace(
        contact_eligible=torch.tensor([True, True, True, False]),
        robot_rack_collision=torch.tensor([True, False, False, True]),
        obstacle_collision=torch.tensor([False, True, False, False]),
    )
    monitor.record(distances, safety, torch.ones(4, dtype=torch.bool))
    result = monitor.report()
    assert result["front_stage/le_0p25m/samples"] == 1
    assert result["front_stage/le_0p25m/rack_unsafe"] == 1
    assert result["front_stage/le_0p25m/rack_rate"] == 1.0
    assert result["front_stage/0p25_to_0p5m/obstacle_unsafe"] == 1
    assert result["front_stage/1_to_2m/samples"] == 1
    assert result["front_stage/both_under_0p25m"] == 0


def test_finite_simulator_explosion_is_excluded_from_sac_replay():
    finite, plausible = _grasp_distance_masks({
        "matched_flap_distance_m": torch.tensor([[1.2, 1.3], [8.5e6, 1.0], [float("nan"), 1.0]]),
        "front_staging_distance_m": torch.tensor([[0.8, 0.9], [8.3e6, 1.0], [1.0, 1.0]]),
    }, 3)
    assert finite.tolist() == [True, True, False]
    assert plausible.tolist() == [True, False, False]


def _batch(count=32):
    actor = torch.randn(count, 4)
    critic = torch.randn(count, 7)
    return {
        "actor_obs": actor,
        "critic_obs": critic,
        "action": torch.rand(count, 2) * 2 - 1,
        "reward": torch.randn(count),
        "next_actor_obs": actor + 0.1,
        "next_critic_obs": critic - 0.1,
        "terminated": torch.zeros(count, dtype=torch.bool),
    }


def test_asymmetric_replay_keeps_actor_and_critic_views_separate():
    replay = AsymmetricReplayBuffer(5, 4, 7, 2)
    replay.add(**_batch(8))
    assert replay.size == 5
    assert replay.data["actor_obs"].shape == (5, 4)
    assert replay.data["critic_obs"].shape == (5, 7)
    sample = replay.sample(12, "cpu")
    assert sample["next_actor_obs"].shape == (12, 4)
    assert sample["next_critic_obs"].shape == (12, 7)


def test_demo_imitation_decays_after_early_sac_updates():
    assert _demo_fraction(0.2, 0, 30) == pytest.approx(0.2)
    assert _demo_fraction(0.2, 15, 30) == pytest.approx(0.1)
    assert _demo_fraction(0.2, 30, 30) == 0
    assert _demo_fraction(0.2, 50, 30) == 0


def test_target_actor_features_ignore_box_slot_number_and_unselected_boxes():
    features = ActorFeatures(flat_actor_observation_dim(24), "grasp_target")
    obs = torch.zeros(2, flat_actor_observation_dim(24))
    obs[0, 86:108] = torch.arange(22)
    obs[1, 86 + 3 * 22:86 + 4 * 22] = torch.arange(22)
    obs[0, 388] = obs[1, 391] = 1
    obs[0, 400] = obs[1, 403] = 1
    obs[1, 86:108] = 1000
    encoded = features(obs)
    assert encoded.shape == (2, 198)
    torch.testing.assert_close(encoded[0], encoded[1])


def test_frozen_actor_statistics_survive_online_distribution_shift():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, freeze_actor_normalizer=True))
    agent.pretrain_actor(torch.randn(16, 4), torch.zeros(16, 2), steps=1, batch_size=8)
    mean, var = agent.actor_normalizer.mean.clone(), agent.actor_normalizer.var.clone()
    agent.update_normalizers(torch.full((32, 4), 100.0), torch.randn(32, 7))
    torch.testing.assert_close(agent.actor_normalizer.mean, mean)
    torch.testing.assert_close(agent.actor_normalizer.var, var)
    assert agent.critic_normalizer.count > 0


def test_critic_without_entropy_backup_has_no_idle_survival_bonus():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, entropy_backup=False, reward_scale=10))
    with torch.no_grad():
        for network in (agent.target1, agent.target2):
            for parameter in network.parameters():
                parameter.zero_()
    batch = _batch(16)
    batch["reward"].fill_(-0.002)
    result = agent.update(batch)
    assert result["target_value_mean"] == pytest.approx(-0.02)


def test_kinematic_entry_goals_remain_outside_front_plane_before_insertion():
    obs = torch.zeros(1, 440)
    rotation6 = torch.tensor([1., 0., 0., 0., 1., 0.])
    tcp = obs[:, 50:68].reshape(1, 2, 9)
    tcp[..., 3:] = rotation6
    obs[:, 71:77] = rotation6
    relations = obs[:, RELATION_START:ASSIGNMENT_START].reshape(1, 2, 2, 9)
    relations[0, 0, 0, :3] = torch.tensor([-0.2, -0.3, 1.0])
    relations[0, 1, 1, :3] = torch.tensor([0.2, -0.3, 1.0])
    obs[:, ASSIGNMENT_START] = 1
    _, centers, stage, outward = entry_geometry(obs, 0.08)
    assert (centers[..., 1] < 0.08).all()
    torch.testing.assert_close(stage[..., 1], torch.full((1, 2), 0.08))
    torch.testing.assert_close(stage[..., [0, 2]], centers[..., [0, 2]])
    torch.testing.assert_close(outward, torch.tensor([[0., 1., 0.]]))


def test_recorded_success_pose_is_closeable_with_retargeted_goal_not_neutral_center():
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import (
        load_v2_grasp_demonstrations, _rotation_matrix,
    )
    path = Path(__file__).parents[1] / "examples/demos/v2_grasp_quest_success.hdf5"
    demo, _ = load_v2_grasp_demonstrations(path, self_collision_enabled=False)
    obs = demo["actor_obs"]
    offset = successful_demo_grasp_offsets(demo, .08)
    tcp, centers, stage, outward = entry_geometry(obs, .08)
    token, _ = target_token(obs)
    goal, _, _ = retarget_grasp_goal(centers, stage, outward,
        _rotation_matrix(token[:, 15:21]), offset, "demo")
    physical_pinch = demo["critic_obs"][:, obs.shape[1] + 35:obs.shape[1] + 37] > .5
    for hand in range(2):
        rows = torch.where((demo["action"][:, 20 + hand] > 0) & physical_pinch[:, hand])[0]
        row = rows[-1]
        assert (centers[row, hand] - tcp[row, hand, :3]).norm() > .035
        assert (goal[row, hand] - tcp[row, hand, :3]).norm() < 1e-5


def test_symmetric_closing_axis_alignment_ignores_roll_and_antiparallel_sign():
    from kuavo_isaaclab_scene.teleop.teleop_servo import closing_axis_error
    current = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    desired = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    error = closing_axis_error(current, desired)
    torch.testing.assert_close(error[:2], torch.zeros(2, 3))
    torch.testing.assert_close(error[2], torch.tensor([0., 0., torch.pi / 2]))
    # Position can use angular velocity around the closing axis: it produces
    # no change in jaw alignment, unlike either perpendicular direction.
    projection = torch.eye(3) - torch.outer(current[0], current[0])
    torch.testing.assert_close(projection @ current[0], torch.zeros(3))
    torch.testing.assert_close(projection @ desired[2], desired[2])


def test_hypothetical_close_labels_cannot_start_lift_with_actual_open_jaws():
    obs = torch.zeros(3, 464)
    obs[1, 48:50] = 1  # command pending; measured jaws remain open
    obs[2, 46:50] = 1  # actually closing/closed
    ticks = torch.zeros(3, dtype=torch.long)
    close = torch.ones(3, 2, dtype=torch.bool)
    for _ in range(15):
        ticks = observed_close_ticks(ticks, close, obs)
    torch.testing.assert_close(ticks, torch.tensor([0, 0, 15]))
    obs[2, 48] = 0
    assert observed_close_ticks(ticks, close, obs)[2] == 0


def test_teacher_lift_requires_actual_opposing_flap_pinches():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import confirmed_pinch_ticks
    ticks = torch.zeros(4, dtype=torch.long)
    # Empty close, one-hand pinch, same-flap contact, valid opposing capture.
    pinches = torch.tensor([[False, False], [True, False], [True, True], [True, True]])
    flaps = torch.tensor([[-1, -1], [0, -1], [0, 0], [1, 0]])
    for _ in range(15):
        ticks = confirmed_pinch_ticks(ticks, pinches, flaps)
    torch.testing.assert_close(ticks, torch.tensor([0, 0, 0, 15]))
    pinches[3, 1] = False
    assert confirmed_pinch_ticks(ticks, pinches, flaps)[3] == 0


def test_guidance_stays_with_the_episode_and_retires_only_on_reset():
    torch.manual_seed(1)
    guide = EpisodicIKGuidance(100, "cpu", .2, 100)
    ready = torch.ones(100, dtype=torch.bool)
    selected = guide.select(warming_up=False, ready=ready, actor_updates=0)
    assert 0 < selected.sum() < 100
    torch.testing.assert_close(
        guide.select(warming_up=False, ready=ready, actor_updates=100), selected)
    guide.reset(selected, 100)
    assert not guide.mask.any()
    assert selected.any()  # saved transition attribution must not mutate on reset


def test_guidance_successes_exclude_warmup_handoffs_from_sac_from_reset():
    guide = EpisodicIKGuidance(2, "cpu", 0, 100)
    ready = torch.ones(2, dtype=torch.bool)
    success = ready.clone()
    guide.select(warming_up=True, ready=ready, actor_updates=0)
    assert guide.successes(success, warming_up=True) == (2, 0, 0, 0)
    guide.select(warming_up=False, ready=ready, actor_updates=0)
    assert guide.successes(success, warming_up=False) == (0, 0, 2, 0)
    guide.reset(torch.tensor([True, False]), 0)
    guide.select(warming_up=False, ready=ready, actor_updates=0)
    assert guide.successes(success, warming_up=False) == (0, 0, 2, 1)
    guide.mask[1] = True
    guide.select(warming_up=False, ready=ready, actor_updates=0)
    assert guide.successes(success, warming_up=False) == (0, 1, 1, 1)


def test_deferred_stop_survives_foreign_callbacks_without_throwing(monkeypatch):
    import signal
    from kuavo_isaaclab_scene.rl.runners import common

    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(common, "_STOP_REQUESTED", False)
    common.install_stop_handlers(defer=True)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert common.stop_requested()
    common.install_stop_handlers()
    assert not common.stop_requested()
    with pytest.raises(KeyboardInterrupt):
        handlers[signal.SIGINT](signal.SIGINT, None)


def test_saturated_variance_logits_cannot_exceed_configured_exploration_cap():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, initial_policy_std=.15, max_policy_std=.3))
    with torch.no_grad():
        agent.actor.network[-1].bias[2:].fill_(100)
    obs = torch.zeros(4096, 4)
    actions, _ = agent.actor(obs)
    assert actions.std(dim=0).amax().item() < 0.35


def test_demo_reward_is_ignored_by_actor_imitation_and_critic():
    torch.manual_seed(11)
    first = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16))
    second = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16))
    second.restore(first.checkpoint())
    online = _batch(16)
    demo = _batch(4)
    changed_reward = {**demo, "reward": torch.full((4,), 1e6)}
    torch.manual_seed(12)
    report_first = first.update(online, demonstration=demo, demonstration_weight=0.2)
    torch.manual_seed(12)
    report_second = second.update(
        online, demonstration=changed_reward, demonstration_weight=0.2)
    assert report_first["q_loss"] == pytest.approx(report_second["q_loss"])
    assert report_first["actor_loss"] == pytest.approx(report_second["actor_loss"])
    assert report_first["demo_bc_loss"] > 0


def test_demo_actor_pretraining_reduces_action_error_without_critic_update():
    torch.manual_seed(23)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16))
    obs = torch.randn(32, 4)
    action = torch.zeros(32, 2)
    q_before = {name: value.clone() for name, value in agent.q1.state_dict().items()}
    report = agent.pretrain_actor(obs, action, steps=40, batch_size=16)
    assert report["final_mse"] < report["initial_mse"]
    assert all(torch.equal(value, q_before[name])
               for name, value in agent.q1.state_dict().items())


def test_counterfactual_controller_labels_stay_out_of_critic_experience():
    obs = torch.randn(8, 4)
    executed = torch.ones(8, 2)
    labels = -executed
    imitation = ActorImitationBuffer(16, 4, 2)
    imitation.add(actor_obs=obs, action=labels)
    snapshot = imitation.snapshot()
    assert set(snapshot) == {"actor_obs", "action"}
    torch.testing.assert_close(snapshot["action"], labels)
    restored = ActorImitationBuffer(16, 4, 2)
    restored.add(**snapshot)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16))
    batch = restored.sample(8, "cpu")
    before = {key: value.clone() for key, value in agent.q1.state_dict().items()}
    report = agent.pretrain_actor(batch["actor_obs"], batch["action"], steps=50, batch_size=8)
    assert report["final_mse"] < report["initial_mse"]
    assert all(torch.equal(value, before[key]) for key, value in agent.q1.state_dict().items())
    torch.testing.assert_close(executed, torch.ones(8, 2))


def test_imitation_priority_keeps_precise_labels_and_expires_overwritten_rows():
    labels = ActorImitationBuffer(4, 2, 1,
        priority_fn=lambda obs, action: obs[:, 0] > 0, priority_fraction=0.5)
    labels.add(actor_obs=torch.tensor([[1., 0.], [-1., 0.], [-1., 0.], [-1., 0.]]),
               action=torch.zeros(4, 1))
    batch = labels.sample(100, "cpu")
    assert (batch["actor_obs"][:, 0] > 0).sum() >= 50
    labels.add(actor_obs=torch.tensor([[-1., 0.]]), action=torch.zeros(1, 1))
    assert not labels.priority.any()
    assert (labels.sample(100, "cpu")["actor_obs"][:, 0] < 0).all()


def test_persistent_critical_imitation_survives_far_fifo_overwrites_and_restore():
    predicate = lambda obs, action: obs[:, 0] > 0
    labels = ActorImitationBuffer(4, 2, 1, priority_fn=predicate,
                                 priority_fraction=0.5, priority_capacity=3)
    labels.add(actor_obs=torch.tensor([[9., 0.]]), action=torch.tensor([[.7]]))
    labels.add(actor_obs=torch.full((12, 2), -1.), action=torch.zeros(12, 1))
    assert not labels.priority.any() and labels.priority_size == 1
    batch = labels.sample(100, "cpu")
    assert (batch["actor_obs"][:, 0] > 0).sum() == 50
    torch.testing.assert_close(batch["action"][:50], torch.full((50, 1), .7))
    snapshot = labels.snapshot(max_rows=20)
    assert len(snapshot["action"]) <= 20
    restored = ActorImitationBuffer(4, 2, 1, priority_fn=predicate,
                                   priority_fraction=0.5, priority_capacity=3)
    restored.add(**snapshot)
    restored.add(actor_obs=torch.full((4, 2), -1.), action=torch.zeros(4, 1))
    assert (restored.sample(100, "cpu")["actor_obs"][:, 0] > 0).sum() == 50
    # The critical FIFO changes only when new critical labels arrive.
    labels.add(actor_obs=torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]]),
               action=torch.ones(4, 1))
    assert labels.priority_size == 3
    assert set(labels.priority_data["actor_obs"][:, 0].tolist()) == {2., 3., 4.}


def test_teacher_label_checkpoint_loads_only_compatible_finite_actor_labels(tmp_path):
    from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _teacher_labels_from_checkpoint
    from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint
    source = dict(algorithm="asymmetric_sac", actor_obs_dim=4, action_dim=2,
        teacher_imitation=dict(actor_obs=torch.zeros(3, 4), action=torch.ones(3, 2)),
        success_replay={"reward": torch.tensor([123.])}, model={"irrelevant": torch.tensor([999.])})
    path = save_checkpoint(tmp_path, source, 1)
    labels = _teacher_labels_from_checkpoint(path, 4, 2)
    assert set(labels) == {"actor_obs", "action"}
    with pytest.raises(ValueError, match="contract"):
        _teacher_labels_from_checkpoint(path, 5, 2)
    source["teacher_imitation"]["action"][0, 0] = float('nan')
    path = save_checkpoint(tmp_path, source, 2)
    with pytest.raises(ValueError, match="finite"):
        _teacher_labels_from_checkpoint(path, 4, 2)


def test_success_history_keeps_contiguous_tail_without_crossing_reset():
    history = SuccessfulTransitionHistory(2, 3, 4, 7, 2)
    for step in range(5):
        batch = _batch(2)
        batch["reward"].fill_(step)
        batch["terminated"].fill_(step == 4)
        history.add(torch.tensor([0, 1]), **batch)
        if step == 2:
            history.reset(torch.tensor([False, True]))
    tail = history.tails(torch.tensor([0, 1]))
    assert tail["reward"].tolist() == [2, 3, 4, 3, 4]
    assert tail["terminated"].tolist() == [False, False, True, False, True]
    history.reset(torch.tensor([True, False]))
    assert history.tails(torch.tensor([0]))["reward"].numel() == 0
    assert history.tails(torch.tensor([1]))["reward"].tolist() == [3, 4]


def test_std_capped_entropy_target_is_reachable_and_alpha_can_decrease():
    torch.manual_seed(42)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16,
        initial_policy_std=0.02, max_policy_std=0.02))
    # Zero mean at the largest allowed std approximates maximum entropy.
    with torch.no_grad():
        output = agent.actor.network[-1]
        output.weight.zero_()
        output.bias[:2].zero_()
        output.bias[2:].fill_(torch.tensor(0.02).log())
    before = agent.log_alpha.item()
    report = agent.update(_batch(4096))
    assert agent.target_entropy_per_dim == pytest.approx(-2.99308447)
    assert report["policy_entropy_error_mean"] > 0
    assert agent.log_alpha.item() < before
    assert agent.checkpoint()["entropy_contract"]["name"] == "squash_aware_active_dims_v2"
    # Broad default variance retains legacy -1/dim behavior.
    assert AsymmetricSAC(4, 7, 2, SACConfig(hidden=16)).target_entropy_per_dim == -1


def test_saturated_teacher_means_keep_a_reachable_entropy_target():
    torch.manual_seed(42)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16,
        initial_policy_std=.02, max_policy_std=.02))
    with torch.no_grad():
        output = agent.actor.network[-1]
        output.weight.zero_()
        output.bias[:2].fill_(5.)
        output.bias[2:].fill_(torch.tensor(.02).log())
    before = agent.log_alpha.item()
    report = agent.update(_batch(4096))
    assert report["target_entropy_mean"] < -20
    assert report["policy_entropy_error_mean"] > 0
    assert agent.log_alpha.item() < before


def test_temperature_cannot_exceed_its_configured_cap():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16,
        initial_alpha=.001, max_alpha=.001, initial_policy_std=.01, max_policy_std=.02))
    with torch.no_grad():
        output = agent.actor.network[-1]
        output.weight.zero_()
        output.bias[2:].fill_(-5)
    report = agent.update(_batch(4096))
    assert report["policy_entropy_error_mean"] < 0
    assert report["alpha"] <= .00100001


def test_q_normalization_prevents_value_scale_from_overpowering_imitation():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16,
        initial_alpha=1e-9, actor_q_normalize=True, critic_layer_norm=True))
    with torch.no_grad():
        for net in (agent.q1, agent.q2):
            net[-1].weight.zero_()
            net[-1].bias.fill_(1e6)
    report = agent.update(_batch(16))
    assert report["q_value_mean"] > 999_000
    assert report["actor_q_scale"] < 1.01e-6
    assert report["actor_loss"] == pytest.approx(-1, abs=1e-5)
    assert isinstance(agent.q1[1], torch.nn.LayerNorm)


def test_success_seed_import_reuses_only_real_transitions(tmp_path):
    from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _success_experience_from_checkpoint
    from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint
    data = _batch(8)
    data["terminated"][-1] = True
    state = dict(algorithm="asymmetric_sac", actor_obs_dim=4, critic_obs_dim=7,
                 action_dim=2, success_replay=data, model={"invalid": torch.tensor(float('nan'))})
    path = save_checkpoint(tmp_path, state, 1)
    result = _success_experience_from_checkpoint(path, 4, 7, 2)
    assert set(result) == set(data)
    torch.testing.assert_close(result["reward"], data["reward"])
    with pytest.raises(ValueError, match="contract"):
        _success_experience_from_checkpoint(path, 4, 8, 2)
    data["next_critic_obs"][0, 0] = float('nan')
    path = save_checkpoint(tmp_path, state, 2)
    with pytest.raises(ValueError, match="finite"):
        _success_experience_from_checkpoint(path, 4, 7, 2)


def test_data_only_import_allows_optimizer_changes_but_not_reward_changes(tmp_path):
    original = {"exploration": {"actor_lr": 3e-5}, "reward_profile": {"weights": 1},
                "sac_stability": {"critic_layer_norm": False}}
    (tmp_path / "manifest.json").write_text(json.dumps(original))
    path = tmp_path / "checkpoint_00000001.pt"
    updated = dict(original, exploration={"actor_lr": 1e-5},
                   sac_stability={"critic_layer_norm": True})
    _compatible_checkpoint(path, updated, data_only=True)
    with pytest.raises(ValueError, match="exploration"):
        _compatible_checkpoint(path, updated)
    updated["reward_profile"] = {"weights": 2}
    with pytest.raises(ValueError, match="reward_profile"):
        _compatible_checkpoint(path, updated, data_only=True)


def test_critic_warmup_preserves_pretrained_actor_and_entropy_coefficient():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, actor_lr=0.00003))
    agent.pretrain_actor(torch.randn(16, 4), torch.zeros(16, 2), steps=2, batch_size=8)
    actor = {name: value.clone() for name, value in agent.actor.state_dict().items()}
    q = {name: value.clone() for name, value in agent.q1.state_dict().items()}
    alpha = agent.log_alpha.detach().clone()
    report = agent.update(_batch(16), update_actor=False)
    assert not report["actor_updated"]
    assert all(torch.equal(value, actor[name]) for name, value in agent.actor.state_dict().items())
    assert any(not torch.equal(value, q[name]) for name, value in agent.q1.state_dict().items())
    torch.testing.assert_close(agent.log_alpha, alpha)
    assert agent.actor_optimizer.param_groups[0]["lr"] == 0.00003


def test_guided_demo_warmup_keeps_distant_grippers_open_and_resets_noise():
    widths = {"base": 3, "left_gripper": 1, "right_gripper": 1}
    env = SimpleNamespace(
        num_envs=2, device="cpu",
        action_manager=SimpleNamespace(
            total_action_dim=5, active_terms=tuple(widths),
            get_term=lambda name: SimpleNamespace(action_dim=widths[name]),
        ),
    )
    obs = torch.zeros(2, flat_actor_observation_dim(5))
    relations = obs[:, RELATION_START:ASSIGNMENT_START].reshape(2, 2, 2, 9)
    relations[0, 0, 0, 0] = 0.05
    relations[0, 1, 1, 0] = 0.20
    relations[1, 0, 0, 0] = 0.30
    relations[1, 1, 1, 0] = 0.06
    obs[:, ASSIGNMENT_START] = 1
    distances = assigned_flap_center_distance(obs)
    torch.testing.assert_close(distances, torch.tensor([[0.05, 0.20], [0.30, 0.06]]))
    guide = GuidedDemoWarmup(env, noise_scale=0)
    agent = SimpleNamespace(act=lambda _obs, deterministic:
                            torch.ones(2, 5))
    action = guide.act(agent, obs)
    torch.testing.assert_close(action[:, 3:], torch.tensor([[1., -1.], [-1., 1.]]))
    guide.noise.fill_(0.25)
    guide.reset(torch.tensor([True, False]))
    assert guide.noise[0].eq(0).all()
    assert guide.noise[1].eq(0.25).all()


def test_projected_sac_closes_only_near_flaps_in_rollout_and_updates():
    terms = [("base", 3), ("left_gripper", 1), ("right_gripper", 1)]
    projection = GraspActionProjector(terms)
    obs = torch.zeros(2, flat_actor_observation_dim(5))
    relations = obs[:, RELATION_START:ASSIGNMENT_START].reshape(2, 2, 2, 9)
    relations[0, 0, 0, 0] = 0.05
    relations[0, 1, 1, 0] = 0.20
    relations[1, 0, 0, 0] = 0.30
    relations[1, 1, 1, 0] = 0.30
    obs[:, ASSIGNMENT_START] = 1
    raw = torch.ones(2, 5, requires_grad=True)
    projected = projection(obs, raw)
    torch.testing.assert_close(projected[:, 3:], torch.tensor([[1., -1.], [-1., -1.]]))
    projected[:, 3:].sum().backward()
    torch.testing.assert_close(raw.grad[:, 3:], torch.tensor([[1., 0.], [0., 0.]]))
    torch.testing.assert_close(projection.entropy_mask(obs)[:, 3:],
                               torch.tensor([[1., 0.], [0., 0.]]))

    agent = AsymmetricSAC(obs.shape[1], obs.shape[1] + 2, 5,
                          SACConfig(hidden=16), action_projector=projection)
    assert agent.act(obs)[:, 4].eq(-1).all()
    checkpoint = agent.checkpoint()
    assert checkpoint["action_projection"] == GraspActionProjector.name
    with pytest.raises(ValueError, match="projection differs"):
        AsymmetricSAC(obs.shape[1], obs.shape[1] + 2, 5,
                      SACConfig(hidden=16)).restore(checkpoint)


def test_warmup_limits_continuous_actions_but_explores_binary_grippers():
    widths = {"base": 3, "left_gripper": 1, "right_gripper": 1, "head": 2}
    env = SimpleNamespace(
        num_envs=100,
        device="cpu",
        action_manager=SimpleNamespace(
            total_action_dim=sum(widths.values()),
            active_terms=tuple(widths),
            get_term=lambda name: SimpleNamespace(action_dim=widths[name]),
        ),
    )
    action = _sample_warmup_action(env, 0.35)
    assert action.shape == (100, 7)
    assert bool((action[:, :3].abs() <= 0.35).all())
    assert bool((action[:, 5:].abs() <= 0.35).all())
    assert set(action[:, 3:5].unique().tolist()) == {-1.0, 1.0}


def test_asymmetric_sac_updates_and_restores_without_privileged_actor_input():
    torch.manual_seed(7)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=32))
    batch = _batch()
    agent.update_normalizers(batch["actor_obs"], batch["critic_obs"])
    before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
    report = agent.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())
    assert any(
        not torch.equal(value, before[name])
        for name, value in agent.actor.state_dict().items()
    )
    # Deployment action requires exactly the actor view; privileged features
    # are accepted only by the Q networks during training.
    assert agent.act(torch.zeros(3, 4)).shape == (3, 2)
    assert agent.q1[0].in_features == 7 + 2

    state = agent.checkpoint()
    restored = AsymmetricSAC(4, 7, 2, SACConfig(hidden=32))
    restored.restore(state)
    torch.testing.assert_close(
        restored.act(torch.zeros(3, 4), deterministic=True),
        agent.act(torch.zeros(3, 4), deterministic=True),
    )


def test_asymmetric_sac_exploration_floor_and_diagnostics():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, min_alpha=0.01))
    with torch.no_grad():
        agent.log_alpha.fill_(torch.tensor(0.001).log())
    report = agent.update(_batch())
    assert report["alpha"] >= 0.01 - 1e-7
    assert torch.isfinite(torch.tensor(report["policy_logp_mean"]))
    assert report["policy_action_std_mean"] > 0


def test_asymmetric_sac_rejects_checkpoint_dimension_mismatch():
    state = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16)).checkpoint()
    target = AsymmetricSAC(4, 8, 2, SACConfig(hidden=16))
    try:
        target.restore(state)
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("Mismatched critic dimensions must be rejected")


def test_v2_reward_breakdown_is_reconstructed_before_sac_storage():
    env = SimpleNamespace(
        num_envs=2,
        _multi_box_grasp_reward_breakdown=SimpleNamespace(
            terms={
                "progress": torch.tensor([0.2, -0.1]),
                "success_event": torch.tensor([0.0, 3.0]),
            },
            total=torch.tensor([0.2, 2.9]),
        ),
    )
    terms, total = _reward_breakdown(env)
    assert set(terms) == {"progress", "success_event"}
    torch.testing.assert_close(total, torch.tensor([0.2, 2.9]))


def test_v2_reward_breakdown_rejects_inconsistent_total():
    env = SimpleNamespace(
        num_envs=1,
        _multi_box_grasp_reward_breakdown=SimpleNamespace(
            terms={"progress": torch.tensor([0.2])},
            total=torch.tensor([0.3]),
        ),
    )
    with pytest.raises(RuntimeError, match="do not sum"):
        _reward_breakdown(env)


def test_v2_terminal_snapshot_separates_failures_from_timeout():
    class Manager:
        active_terms = ["success", "unsafe", "time_out"]
        _term_cfgs = [
            SimpleNamespace(time_out=False),
            SimpleNamespace(time_out=False),
            SimpleNamespace(time_out=True),
        ]
        values = {
            "success": torch.tensor([True, False, False]),
            "unsafe": torch.tensor([False, True, False]),
            "time_out": torch.tensor([False, False, True]),
        }

        def get_term(self, name):
            return self.values[name]

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        termination_manager=Manager(),
    )
    terms, terminated, truncated = _termination_snapshot(env)
    assert set(terms) == {"success", "unsafe", "time_out"}
    assert terminated.tolist() == [True, True, False]
    assert truncated.tolist() == [False, False, True]


def test_v2_reset_settling_metrics_expose_rejection_causes():
    settling = SimpleNamespace(
        ready=torch.tensor([True, False, False]),
        invalid=torch.tensor([False, True, False]),
        invalid_count=torch.tensor([0, 2, 0]),
        region_invalid_count=torch.tensor([0, 2, 0]),
        footprint_invalid_count=torch.tensor([0, 1, 0]),
        shelf_invalid_count=torch.tensor([0, 2, 0]),
        timeout_invalid_count=torch.tensor([0, 0, 0]),
        nonfinite_invalid_count=torch.tensor([0, 1, 0]),
    )
    metrics = _reset_settling_metrics(SimpleNamespace(
        _multi_box_reset_settling=settling))
    assert metrics == {
        "reset_ready_envs": 1,
        "reset_settling_envs": 1,
        "reset_invalid_total": 2,
        "reset_region_invalid_total": 2,
        "reset_footprint_invalid_total": 1,
        "reset_shelf_invalid_total": 2,
        "reset_timeout_invalid_total": 0,
        "reset_nonfinite_invalid_total": 1,
    }


def test_v2_initial_reset_settling_uses_zero_actions_until_every_env_is_ready():
    class Environment:
        step_dt = 0.1
        cfg = SimpleNamespace(multi_box=SimpleNamespace(
            reset_settle_timeout_seconds=1.0))
        action_manager = SimpleNamespace(action=torch.ones(2, 3))
        _multi_box_reset_settling = SimpleNamespace(
            ready=torch.tensor([False, False]),
            invalid=torch.tensor([False, False]),
            invalid_count=torch.zeros(2, dtype=torch.long),
            region_invalid_count=torch.zeros(2, dtype=torch.long),
            footprint_invalid_count=torch.zeros(2, dtype=torch.long),
            shelf_invalid_count=torch.zeros(2, dtype=torch.long),
            timeout_invalid_count=torch.zeros(2, dtype=torch.long),
            nonfinite_invalid_count=torch.zeros(2, dtype=torch.long),
        )

        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(action.clone())
            if len(self.actions) == 1:
                self._multi_box_reset_settling.ready[0] = True
            if len(self.actions) == 2:
                self._multi_box_reset_settling.ready[1] = True
            return {"policy": torch.full((2, 1), len(self.actions))}, None, None, None, None

    env = Environment()
    observations, steps = _settle_initial_resets(
        env, {"policy": torch.zeros(2, 1)})
    assert steps == 2
    assert observations["policy"].tolist() == [[2], [2]]
    assert all(torch.equal(action, torch.zeros(2, 3)) for action in env.actions)


def test_v2_initial_reset_settling_can_start_with_ready_majority():
    class Environment:
        num_envs = 10
        step_dt = 0.1
        cfg = SimpleNamespace(multi_box=SimpleNamespace(
            reset_settle_timeout_seconds=0.1))
        action_manager = SimpleNamespace(action=torch.ones(10, 3))
        _multi_box_reset_settling = SimpleNamespace(
            ready=torch.tensor([True] * 9 + [False]),
            invalid=torch.zeros(10, dtype=torch.bool),
        )

        def step(self, action):
            assert torch.equal(action, torch.zeros(10, 3))
            return {"policy": torch.zeros(10, 1)}, None, None, None, None

    observations, steps = _settle_initial_resets(
        Environment(), {"policy": torch.ones(10, 1)})
    assert steps == 4
    assert torch.equal(observations["policy"], torch.zeros(10, 1))


def test_v2_sac_pilot_profile_is_bounded_but_performs_updates():
    args = SimpleNamespace(
        smoke_test=False, pilot=True, num_envs=4096, max_iterations=2000,
        rollout_steps=128, batch_size=1024, replay_capacity=250_000,
        learning_starts=100_000, warmup_vector_steps=450,
        updates_per_step=4, save_interval=50,
    )
    apply_run_profile(args)
    assert args.num_envs == 64
    assert args.max_iterations == 20
    assert args.rollout_steps == 32
    assert args.batch_size == 512
    assert args.replay_capacity == 50_000
    assert args.learning_starts == 4_096
    assert args.warmup_vector_steps == 64
    assert args.updates_per_step == 1
    assert args.save_interval == 5


def test_v2_sac_profiles_are_mutually_exclusive():
    args = SimpleNamespace(smoke_test=True, pilot=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_run_profile(args)


def test_same_dimension_checkpoint_cannot_resume_with_changed_flap_goal(tmp_path):
    original = {name: None for name in (
        "task_family", "schema_version", "skill", "algorithm", "robot_model",
        "gripper", "actions", "observations", "observation_contract",
        "critic_mapping", "reward_profile", "exploration", "demonstrations",
        "self_collision",
    )}
    original["observation_contract"] = "nearest_surface_tcp_frame_v1"
    (tmp_path / "manifest.json").write_text(json.dumps(original))
    checkpoint = tmp_path / "checkpoint_00000001.pt"
    checkpoint.touch()
    updated = dict(original, observation_contract="neutral_flap_center_tcp_frame_v1")
    with pytest.raises(ValueError, match="observation_contract"):
        _compatible_checkpoint(checkpoint, updated)


def test_checkpoint_allows_new_expert_episode_fraction_but_not_changed_actor_features(tmp_path):
    original = {"exploration": {"actor_feature_mode": "grasp_target", "initial_policy_std": .01,
                                "critic_warmup_updates": 4000}}
    (tmp_path / "manifest.json").write_text(json.dumps(original))
    checkpoint = tmp_path / "checkpoint_00000001.pt"
    checkpoint.touch()
    updated = {"exploration": original["exploration"] | {
        "online_ik_episode_fraction": .2, "critic_warmup_updates": 500}}
    _compatible_checkpoint(checkpoint, updated)
    updated["exploration"]["actor_feature_mode"] = "flat"
    with pytest.raises(ValueError, match="exploration"):
        _compatible_checkpoint(checkpoint, updated)
