"""Actual mean clipping, size separation and side-effect-free observations."""
from copy import deepcopy

import pytest
import torch

from test_rl_servo_critic import decoder_fixture
from test_rl_urdf_full_arm_sac import full
from kuavo_isaaclab_scene.rl.multi_box.debug.servo_policy_diagnostics import measured_policy_servo_diagnostics


def test_measured_state_sizes_and_actual_sampler_clipping_without_policy_or_rng_change():
    raw, encoder = decoder_fixture(8); agent = full(encoder)
    anchor = torch.zeros(8, 19); anchor[4:, 1:15] = .7
    agent.executed_body_anchor = lambda measured: anchor
    executed = agent.act(raw, True)
    previous = (raw, torch.zeros(8, 4), executed)
    before = deepcopy(agent.state_dict()); rng = torch.get_rng_state().clone()
    layouts = [dict(layout=dict(target_region='shelf_2_left', target_box_type='small' if i < 4 else 'medium')) for i in range(8)]
    result = measured_policy_servo_diagnostics(agent, previous, torch.arange(8), layouts)
    assert result['states'] == 8
    assert result['by_region_size']['shelf_2_left/small']['arms']['greedy_step_clip_zero_gradient_fraction'] == 0
    assert result['by_region_size']['shelf_2_left/medium']['arms']['greedy_step_clip_zero_gradient_fraction'] == 1
    assert result['by_region_size']['shelf_2_left/medium']['arms']['greedy_all_active_step_gradients_zero_states'] == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, agent.state_dict()[k]) for k, v in before.items())
    assert torch.equal(executed, agent.act(raw, True))


@pytest.mark.parametrize('ids', [torch.tensor([0]*8), torch.arange(8)+1])
def test_diagnostics_reject_ambiguous_or_wrong_wave_identity(ids):
    raw, encoder = decoder_fixture(8); agent = full(encoder)
    previous = (raw, torch.zeros(8, 4), agent.act(raw, True))
    layouts = [dict(layout=dict(target_region='shelf_2_left', target_box_type='small')) for _ in range(8)]
    with pytest.raises(ValueError):measured_policy_servo_diagnostics(agent, previous, ids, layouts)
