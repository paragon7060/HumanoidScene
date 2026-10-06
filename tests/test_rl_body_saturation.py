"""Keep valid controls while recovering gradients lost through saturated tanh."""
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.body_saturation import (
    VARIANT,body_saturation_config,body_saturation_penalty)


def test_soft_mean_gradient_recovers_when_tanh_gradient_is_zero_and_ignores_fixed_goals():
    mean=torch.zeros(3,19);mean[0,:4]=torch.tensor([10.,-10.,2.,-2.]);mean[1,0]=40.;mean.requires_grad_(True)
    active=torch.ones_like(mean,dtype=torch.bool);active[1]=False;active[2]=False
    saturation,report=body_saturation_penalty(mean,active,body_saturation_config(VARIANT))
    gradient=torch.autograd.grad(saturation,mean,retain_graph=True)[0]
    assert gradient[0,0]>0 and gradient[0,1]<0 and gradient[0,2:4].eq(0).all()
    assert gradient[1:].eq(0).all() and report['body_saturation_saturated_coordinates']==2
    assert torch.autograd.grad(mean.tanh().sum(),mean)[0][0,:2].eq(0).all()
    assert report['body_active_abs_mean_max']==10.
    assert float(saturation.detach())==pytest.approx(.001*(7**2+7**2)/19/3)
    before=mean.detach().clone();command=mean.tanh().detach()
    body_saturation_penalty(mean,active,body_saturation_config(VARIANT))
    assert torch.equal(mean.detach(),before) and torch.equal(mean.tanh().detach(),command)


def test_no_active_coordinates_or_interior_means_have_zero_finite_loss():
    for mean,active in [(torch.full((2,19),30.),torch.zeros(2,19,dtype=torch.bool)),
                        (torch.full((2,19),2.5),torch.ones(2,19,dtype=torch.bool))]:
        mean.requires_grad_(True);loss,_=body_saturation_penalty(mean,active,body_saturation_config(VARIANT))
        assert loss==0 and torch.autograd.grad(loss,mean)[0].eq(0).all()


@pytest.mark.parametrize('bad',('width','mask','nonfinite','empty','config'))
def test_invalid_penalty_inputs_fail_closed(bad):
    mean=torch.zeros(2,19);active=torch.ones_like(mean,dtype=torch.bool);config=body_saturation_config(VARIANT)
    if bad=='width':mean=mean[:,:18]
    if bad=='mask':active=active.float()
    if bad=='nonfinite':mean[0,0]=float('nan')
    if bad=='empty':mean=mean[:0];active=active[:0]
    if bad=='config':config=config|dict(weight=.1)
    with pytest.raises(ValueError):body_saturation_penalty(mean,active,config)
    assert body_saturation_config('off') is None


def test_managed_actor_regularization_is_TRAIN_only(tmp_path):
    import json
    from test_rl_cpu_physics_training import inputs
    from batched_staged_goal_with_drive import validate_managed_physics_device
    waves,contract=inputs();wp=tmp_path/'waves.json';mp=tmp_path/'manifest.json'
    wp.write_text(json.dumps(waves));mp.write_text(json.dumps(contract))
    flags=['--waves-json',str(wp),'--training-manifest',str(mp),'--body-saturation-penalty',VARIANT]
    assert validate_managed_physics_device('cpu',flags+['--cpu-physics-training','--training'],learner_device='cuda:0')['training']
    wp.write_text(json.dumps(waves[:1]))
    with pytest.raises(ValueError,match='unchanged other physics'):
        validate_managed_physics_device('cpu',flags+['--frozen-physics-backend-eval','--no-training'])
