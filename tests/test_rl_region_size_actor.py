"""Only component continuity and isolation; not production SAC/physics proof."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.common import mlp,ObservationNormalizer,optimize
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import RegionRoutedNetwork
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_size_actor import RegionSizeRoutedNetwork,SCOPES
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS


def setup():
    norm=ObservationNormalizer(518)
    norm.mean[89:94]=torch.tensor([1.,0.,.266,.185,.13]);norm.var[89:94]=0
    source=RegionRoutedNetwork(mlp(518,42,16),norm)
    raw=torch.zeros(12,518)
    for i,(region,size) in enumerate(SCOPES*2):
        raw[i,94+REGIONS.index(region)]=1
        raw[i,89+int(size=='medium')]=1
        raw[i,91:94]=torch.tensor([.32,.22,.185] if size=='medium' else [.266,.185,.13])
    with torch.no_grad():
        for i,h in enumerate(source.heads):h[-1].bias.add_(i*.1)
    return norm,source,raw


def test_clone_keeps_same_full_batch_outputs_for_all_six_scopes_and_input_state():
    norm,source,raw=setup();before=deepcopy(source.state_dict())
    split=RegionSizeRoutedNetwork(source,norm);x=norm(raw)
    assert torch.equal(split.scope_indices(x),torch.arange(12)%6)
    assert torch.equal(split(x),source(x))
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in before.items())
    assert split(x[:0]).shape==(0,42)


def test_clipped_medium_type_is_distinguished_without_changing_normalizer():
    norm,source,raw=setup();split=RegionSizeRoutedNetwork(source,norm)
    assert split.type_patterns.tolist()==[[0.,0.],[-10.,10.]]
    medium=raw[:1];x=norm(medium)
    recovered=x[:,89:91]*norm.var[89:91].clamp_min(1e-4).sqrt()+norm.mean[89:91]
    assert recovered.argmax(-1).item()==0  # Incorrect inverse clipped argmax.
    assert split.scope_indices(x).item()==0  # Correct middle-right medium.


def test_new_size_head_changes_do_not_modify_other_size_or_other_regions():
    norm,source,raw=setup();split=RegionSizeRoutedNetwork(source,norm);x=norm(raw)
    before=split(x).detach().clone()
    with torch.no_grad():split.heads[5][-1].bias[:19].add_(.001)
    after=split(x).detach();chosen=split.scope_indices(x)==5
    assert not torch.equal(before[chosen,:19],after[chosen,:19])
    assert torch.equal(before[~chosen],after[~chosen])
    assert torch.equal(before[chosen,19:],after[chosen,19:])


def test_absent_size_heads_keep_existing_Adam_state_without_gradient():
    norm,source,raw=setup();split=RegionSizeRoutedNetwork(source,norm)
    optimizer=torch.optim.Adam(split.parameters(),lr=.001)
    optimize(optimizer,split(norm(raw)).square().mean(),split.parameters())
    weights=deepcopy(split.state_dict())
    moments={p:deepcopy(optimizer.state[p]) for h in split.heads[:5] for p in h.parameters()}
    x=norm(raw[5:6]);optimize(optimizer,(split(x)-2).square().mean(),split.parameters())
    for i in range(5):
        for p in split.heads[i].parameters():
            assert p.grad is None
            assert all(torch.equal(v,optimizer.state[p][key]) for key,v in moments[p].items())
        for k,v in weights.items():
            if k.startswith(f'heads.{i}.'):assert torch.equal(v,split.state_dict()[k])


def test_stale_normalizer_and_unsupported_upper_medium_are_rejected():
    norm,source,raw=setup();norm.mean[94]=.1
    with pytest.raises(ValueError,match='same frozen'):RegionSizeRoutedNetwork(source,norm)
    norm.mean[94]=0;split=RegionSizeRoutedNetwork(source,norm)
    upper=raw[2:3].clone();upper[:,89:91]=torch.tensor([0.,1.])
    with pytest.raises(ValueError,match='Upper-shelf'):split(norm(upper))
