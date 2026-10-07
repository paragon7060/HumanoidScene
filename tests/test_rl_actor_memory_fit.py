"""Actor initialization must reject reward-bearing or unbound TRAIN state."""
from copy import deepcopy

import pytest
import torch

from test_rl_actor_train_memory import fixtures
from prepare_actor_memory_servo import prepare
from fit_actor_train_memory import fit


@pytest.mark.parametrize('fault', ['actor', 'Q', 'optimizer', 'replay', 'memory', 'already_fitted'])
def test_actor_initialization_rejects_learning_or_memory_identity_change(fault):
    initial, experience, memory, audit = fixtures()
    state, replay = prepare(initial, experience, memory, audit)
    before, old_replay = deepcopy(state), deepcopy(replay)
    if fault == 'actor': state['actor_updates'] = 1
    elif fault == 'Q': state['critic_updates'] = 1
    elif fault == 'optimizer': state['optimizers'][0]['state'] = {0: dict(step=torch.tensor(1))}
    elif fault == 'replay': replay['executed_goal_transitions']['reward'] = torch.ones(1)
    elif fault == 'memory': replay['actor_training_memory']['source_checkpoint_SHA256'] = 'a'*64
    else: state['actor_memory_initialization'] = dict(kind='previous_offline_fit')
    with pytest.raises(ValueError):
        fit(state, replay, steps=1, batch_size=64, learning_rate=1e-4, seed=7)
    # No candidate fit or mutation may occur before rejection.
    assert all(torch.equal(v, state['model'][k]) for k,v in before['model'].items())
    assert all(torch.equal(v, replay['executed_goal_transitions'][k])
               for k,v in old_replay['executed_goal_transitions'].items() if k != 'reward')


@pytest.mark.parametrize('option,value', [('steps',0),('batch_size',65),
    ('learning_rate',float('nan')),('seed',True)])
def test_actor_fit_rejects_invalid_limits_before_loading_models(option,value):
    options=dict(steps=1,batch_size=64,learning_rate=1e-4,seed=7);options[option]=value
    with pytest.raises(ValueError):fit({}, {}, **options)
