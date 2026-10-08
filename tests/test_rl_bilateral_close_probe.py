from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.bilateral_close_probe import (
    close_bilateral_when_allowed,install_frozen_bilateral_close_probe,
    validate_bilateral_close_probe)


def test_only_bilateral_eligible_jaws_change_and_recorded_goals_match_execution():
    physical=torch.arange(4*24,dtype=torch.float32).reshape(4,24)/100
    physical[:,20:22]=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    goals=torch.randn(4,21);goals[:,19:21]=physical[:,20:22]
    actor=torch.randn(4,518);critic=torch.randn(4,578)
    originals=[x.clone() for x in (physical,goals,actor,critic)]
    near=torch.tensor([[True,True],[True,False],[False,True],[True,True]])
    statistics=dict(held_rows_checked=0,both_near_rows=0,jaw_overridden_rows=0)
    result,(ao,co,executed)=close_bilateral_when_allowed((physical,(actor,critic,goals)),near,statistics)
    assert result[:,20:22].tolist()==[[1.,1.],[-1.,1.],[1.,-1.],[1.,1.]]
    assert torch.equal(result[:,:20],physical[:,:20]) and torch.equal(result[:,22:],physical[:,22:])
    assert torch.equal(executed[:,:19],goals[:,:19])
    assert torch.equal(executed[:,19:21],result[:,20:22])
    assert ao is actor and co is critic
    assert all(torch.equal(a,b) for a,b in zip((physical,goals,actor,critic),originals))
    assert statistics==dict(held_rows_checked=4,both_near_rows=2,jaw_overridden_rows=1)


def wave():
    pairs=[('shelf_2_left','small',16),('shelf_2_left','medium',16),
        ('shelf_2_right','small',16),('shelf_2_right','medium',16),
        ('shelf_3_left','small',32),('shelf_3_right','small',32)]
    return [dict(split='train',layouts=[dict(layout=dict(target_region=r,target_box_type=k))
        for r,k,n in pairs for _ in range(n)])]


@pytest.mark.parametrize('bad',['training','DEV','small_only','short','missing_workplace'])
def test_training_or_reduced_scope_rejected(bad):
    waves=wave();training=False;steps=900;workplace={'validated':True}
    if bad=='training':training=True
    elif bad=='DEV':waves[0]['split']='validation'
    elif bad=='small_only':
        for r in waves[0]['layouts']:r['layout']['target_box_type']='small'
    elif bad=='short':steps=899
    else:workplace=None
    with pytest.raises(ValueError):validate_bilateral_close_probe(waves,training=training,steps=steps,workplace=workplace)
    assert validate_bilateral_close_probe(wave(),training=False,steps=900,workplace={'validated':True})['Q_import_eligible'] is False


def test_hooks_are_process_local_restored_and_refuse_training_before_act():
    class Pilot:
        training=True;actor_updates=0;critic_updates=0;replay=SimpleNamespace(size=0)
        def act(self):raise AssertionError('Original act must not run in training')
        def report(self):return {'source':True}
    module=SimpleNamespace(checkpoint_manifest_fields=lambda: {'source':True})
    original=(Pilot.act,Pilot.report,module.checkpoint_manifest_fields)
    restore=install_frozen_bilateral_close_probe(Pilot,module)
    try:
        with pytest.raises(ValueError):Pilot().act()
        assert module.checkpoint_manifest_fields()['frozen_bilateral_close_probe']['standalone_SAC'] is False
        assert Pilot().report()['frozen_bilateral_close_probe']['actual_statistics']['held_rows_checked']==0
    finally:restore()
    assert (Pilot.act,Pilot.report,module.checkpoint_manifest_fields)==original
