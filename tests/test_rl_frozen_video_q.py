"""Q annotations bind measured action rows to their own frozen DEV frames."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from frozen_video_q import FrozenVideoQ


class CriticAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = nn.Linear(4, 1), nn.Linear(4, 1)
        with torch.no_grad():
            self.q1.weight.fill_(1); self.q2.weight.fill_(.5)
            self.q1.bias.zero_(); self.q2.bias.zero_()

    def critic_normalizer(self, critic):
        return critic + 1

    def critic_action_features(self, actor, actions):
        return torch.cat((actions, actor.sum(-1, keepdim=True)), -1)


def pilot():
    return SimpleNamespace(agent=CriticAgent(), training=False, actor_updates=8, critic_updates=32)


def measured():
    return (torch.tensor([[1., 2.], [20., 30.], [3., 4.]]),
            torch.tensor([[4., 6.], [40., 60.], [8., 9.]]),
            torch.tensor([[.2], [.9], [.4]]))


def test_q_uses_actual_goal_encoder_and_correct_noncontiguous_environment_rows():
    p = pilot(); q = FrozenVideoQ(p, [0, 4, 9]); previous = measured()
    original = [v.clone() for v in previous]; rng = torch.get_rng_state().clone()
    q.prepare(17, torch.tensor([9, 1, 4]), previous)
    assert q.frame(17, 0)['q_min'] is None
    assert q.frame(17, 0)['reason'] == 'critic_trained_on_held_grasp_only'
    assert q.frame(17, 9)['control_step'] == 18
    assert q.frame(17, 9)['q1'] == pytest.approx(15.2)
    assert q.frame(17, 9)['q_min'] == pytest.approx(7.6)
    assert q.frame(17, 4)['q1'] == pytest.approx(26.4)
    assert q.frame(17, 4)['q_min'] == pytest.approx(13.2)
    assert all(torch.equal(a, b) for a, b in zip(previous, original))
    assert torch.equal(rng, torch.get_rng_state())
    proof = q.close_proof()
    assert proof['frozen_model_unchanged'] and proof['optimizer_updates'] == 0
    assert not proof['raw_replay_HDF_read'] and not proof['raw_observations_actions_stored']


def test_approach_has_no_critic_value_and_a_frame_cannot_take_another_steps_q():
    q = FrozenVideoQ(pilot(), [4]); q.prepare(3, torch.tensor([], dtype=torch.long), None)
    assert q.frame(3, 4)['phase'] == 'base_approach'
    with pytest.raises(ValueError, match='out of step'): q.frame(4, 4)
    with pytest.raises(ValueError, match='no measured'): q.prepare(5, torch.tensor([4]), None)
    with pytest.raises(ValueError, match='rows differ'): q.prepare(5, torch.tensor([4, 4, 9]), measured())


def test_frozen_proof_rejects_training_counters_and_silent_parameter_changes():
    p = pilot(); p.training = True
    with pytest.raises(ValueError, match='frozen evaluation'): FrozenVideoQ(p, [0])
    p.training = False; q = FrozenVideoQ(p, [0]); p.critic_updates += 1
    with pytest.raises(ValueError, match='updates occurred'): q.close_proof()
    p.critic_updates -= 1
    with torch.no_grad(): p.agent.q1.bias.add_(1)
    with pytest.raises(ValueError, match='model changed'): q.close_proof()


def test_nonfinite_q_is_explicitly_unavailable_and_never_a_success_value():
    p = pilot()
    with torch.no_grad(): p.agent.q1.bias.fill_(float('inf'))
    q = FrozenVideoQ(p, [9]); q.prepare(3, torch.tensor([9, 1, 4]), measured())
    frame = q.frame(3, 9)
    assert frame['q1'] is frame['q2'] is frame['q_min'] is None
    assert frame['reason'] == 'nonfinite_Q_estimate'


def test_terminal_video_frame_keeps_its_own_pre_action_q_and_closed_model_proof(tmp_path):
    import json
    import numpy as np
    from selected_scene_videos import SelectedSceneVideos

    p = pilot(); video = SelectedSceneVideos.__new__(SelectedSceneVideos)
    video.q_trace = FrozenVideoQ(p, [0]); video.scope = dict(label='DEV', split='validation', workplace_search=False)
    video.env = SimpleNamespace(device='cpu', scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    video.renderer = SimpleNamespace(meshes=[], frame=lambda *a: np.zeros((720, 960, 3), dtype=np.uint8))
    video.meshes = []; video.capture_every = 10; video.last_steps = {}; video.output = tmp_path; video.wave = 0
    video.actor_updates = 8; video.critic_updates = 32
    labels = []
    video.cv2 = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0,
        rectangle=lambda *a: None, putText=lambda frame, text, *a: labels.append(text))
    frames = []; released = []
    video.writers = {0: SimpleNamespace(write=lambda frame: frames.append(frame.copy()), release=lambda: released.append(True))}
    video.records = [dict(environment=0, region='shelf_2_left', frames=0, raw_file='test_raw.mp4')]
    video.finalize_video = lambda record, *a: record.update(browser_file='test_h264.mp4')
    active = torch.tensor([True]); false = torch.tensor([False]); distance = torch.tensor([[.02, .03]])
    initial = tuple(v[:1].clone() for v in measured())
    video.prepare_q(0, torch.tensor([0]), initial)
    video.capture(0, active, false, false, false, distance)
    video.capture(0, active, false, false, false, distance)
    final = tuple(v.clone() for v in initial); final[2].fill_(.4)
    video.prepare_q(10, torch.tensor([0]), final)
    video.capture(10, active, torch.tensor([True]), false, false, distance)
    assert len(frames) == 2 and len(released) == 1
    record = video.records[0]
    assert [v['control_step'] for v in record['frame_Q_values']] == [1, 11]
    assert record['frame_Q_values'][0]['q_min'] == pytest.approx(7.6)
    assert record['frame_Q_values'][1]['q_min'] == pytest.approx(7.7)
    assert record['terminal_frame_before_reset'] and record['terminal_flags']['success']
    assert any('min Q=7.700' in label for label in labels)
    video.close([dict(success=True)])
    saved = json.loads((tmp_path/'test_h264.json').read_text())
    assert saved['frozen_Q_proof']['frozen_model_unchanged'] and saved['frames'] == 2
    assert 'actor_obs' not in json.dumps(saved) and 'critic_obs' not in json.dumps(saved)
