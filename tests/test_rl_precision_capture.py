"""Precision geometry and strict actor-only reward migration, without Isaac."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.metrics.potentials import grasp_reward_potentials
from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
    with_contact_reward_profile, frozen_actor_reward_contract, configured_reward_weights,
    contact_reward_weights, learning_termination_mask,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import (
    with_precision_capture_profile, frozen_capture_actor_contract, configured_capture_geometry,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.weights import MultiBoxRewardWeights


def source_contract():
    return with_contact_reward_profile(dict(
        reward_profile=dict(weights=asdict(MultiBoxRewardWeights()), capture_scale_m=.10,
            approach_scale_m=.22, assignment_scale_m=1/12, geometry_profile='unchanged'),
        observations=dict(policy=[464], critic=[530]), actions=dict(upper_body=14),
        terminal_contract=dict(rack_force=10., obstacle_force=5., min_pad_force=5.),
        flap_dynamics=dict(unchanged=True), randomization=dict(box_and_base=True)))


def test_opt_in_identity_changes_only_capture_and_preserves_frozen_actor_contract():
    old=source_contract();unchanged=deepcopy(old)
    new=with_precision_capture_profile(old)
    assert old==unchanged and new!=old
    assert frozen_capture_actor_contract(new)==old
    assert frozen_actor_reward_contract(new)==frozen_actor_reward_contract(old)
    assert configured_reward_weights(new['reward_profile'])==contact_reward_weights()
    assert configured_capture_geometry(old['reward_profile'])==dict(capture_scale_m=.10,capture_aggregation='mean')
    assert configured_capture_geometry(new['reward_profile'])==dict(capture_scale_m=.025,capture_aggregation='weak-hand')
    terminated=torch.tensor([False,True]);timeout=~terminated
    assert learning_termination_mask(new['reward_profile'],terminated,timeout).all()
    assert torch.equal(terminated,torch.tensor([False,True]))
    with pytest.raises(ValueError):with_precision_capture_profile(new)


@pytest.mark.parametrize('field,value',[
    ('capture_scale_m',.03),('precision_capture',{'name':'fake'}),
    ('contact_shaping',{}),('weights',{'discount':.999}),
])
def test_modified_profile_cannot_strip_unknown_reward_for_actor_migration(field,value):
    bad=with_precision_capture_profile(source_contract());bad['reward_profile'][field]=value
    with pytest.raises(ValueError):frozen_actor_reward_contract(bad)
    with pytest.raises(ValueError):configured_capture_geometry(bad['reward_profile'])


def test_capture_compatibility_keeps_safety_and_action_changes_visible():
    old=source_contract();new=with_precision_capture_profile(old)
    new['terminal_contract']['rack_force']=99.
    assert frozen_capture_actor_contract(new)!=old
    new=with_precision_capture_profile(old);new['actions']['upper_body']=13
    assert frozen_capture_actor_contract(new)!=old


def potentials(error,**kwargs):
    n=len(error);matched=torch.full((n,2),.02)
    return grasp_reward_potentials(torch.full((n,2,2),.1),matched,
        torch.ones(n,2),error,torch.full((n,2),.02),torch.full((n,2),.01),torch.zeros(n),**kwargs)


def test_actual_per_hand_capture_only_changes_capture_and_emphasizes_weaker_hand():
    error=torch.tensor([[0.,0.],[0.,1.],[.01,.06],[.01,.03]],requires_grad=True)
    old=potentials(error);new=potentials(error,**configured_capture_geometry(
        with_precision_capture_profile(source_contract())['reward_profile']))
    torch.testing.assert_close(old['capture'],torch.exp(-error/.10).mean(-1),rtol=0,atol=0)
    assert new['capture'][0].item()==1.
    assert new['capture'][1].item()==pytest.approx(.25)
    assert new['capture'][3]>new['capture'][2]
    for name in old:
        if name!='capture':torch.testing.assert_close(old[name],new[name],rtol=0,atol=0)
    gradient=torch.autograd.grad(new['capture'][2],error)[0][2]
    assert (gradient<0).all() and torch.isfinite(gradient).all()


def test_finite_distance_scales_and_known_aggregation_required():
    for scale in (0.,float('nan'),float('inf')):
        with pytest.raises(ValueError):potentials(torch.zeros(1,2),capture_scale_m=scale)
    with pytest.raises(ValueError):potentials(torch.zeros(1,2),capture_aggregation='best-hand')


def test_real_adapter_raw_geometry_uses_actual_tips_and_only_capture_changes():
    # Load the production geometry method unchanged; simulator setup imports
    # are unnecessary for exercising real batched relative-pose calculations.
    import ast
    import __future__
    from pathlib import Path
    from types import SimpleNamespace
    from kuavo_isaaclab_scene.rl.multi_box.geometry import relative_pose
    from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import closest_flap_surface, GRASP_ASSIGNMENT_SCALE_M
    from kuavo_isaaclab_scene.rl.multi_box.geometry.pose import quat_apply
    from kuavo_isaaclab_scene.rl.multi_box.metrics import opposing_flap_reach_assignment
    from kuavo_isaaclab_scene.rl.multi_box.metrics.potentials import front_staging_potential, FRONT_STAGE_CLEARANCE_M, MetricScaleConfig
    from kuavo_isaaclab_scene.workcell.workcell_layout import RACK_RAW_BOUNDS_M, scale as workcell_scale
    path=Path(__file__).resolve().parents[1]/'src/kuavo_isaaclab_scene/rl/multi_box/state/isaac_privileged_grasp.py'
    cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='IsaacPrivilegedGraspAdapter')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_raw_metrics')
    namespace=dict(torch=torch,relative_pose=relative_pose,closest_flap_surface=closest_flap_surface,
        grasp_reward_potentials=grasp_reward_potentials,
        GRASP_ASSIGNMENT_SCALE_M=GRASP_ASSIGNMENT_SCALE_M,opposing_flap_reach_assignment=opposing_flap_reach_assignment,
        quat_apply=quat_apply,front_staging_potential=front_staging_potential,
        FRONT_STAGE_CLEARANCE_M=FRONT_STAGE_CLEARANCE_M,RACK_RAW_BOUNDS_M=RACK_RAW_BOUNDS_M,workcell_scale=workcell_scale)
    from kuavo_isaaclab_scene.rl.multi_box.metrics import GraspRawMetrics
    namespace['GraspRawMetrics']=GraspRawMetrics
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
    flap=pose[:,None].expand(-1,2,-1).clone();flap[0,:,0]=torch.tensor([-.15,.15]);flap[:,:,2]=1.
    tcp=flap.clone();tcp[0,1,1]=.09
    tips=torch.tensor([[[[-.15,-.004,1.],[-.15,.004,1.]],[[.15,.08,1.],[.15,.10,1.]]]])
    adapter=SimpleNamespace(num_envs=1,device='cpu',tcp=SimpleNamespace(center_pose_w=tcp,tips_w=tips,definition=True),
        env=SimpleNamespace(scene={'rack':SimpleNamespace(data=SimpleNamespace(root_pose_w=pose))}))
    centers=torch.zeros(1,2,3);halves=torch.tensor([[[.05,.002,.05],[.05,.002,.05]]]);axes=torch.ones(1,2,dtype=torch.long)
    outputs=[]
    for scale,aggregation in ((.10,'mean'),(.025,'weak-hand')):
        adapter.reward_scale=MetricScaleConfig(grasp_approach_m=.22,grasp_capture_m=scale)
        adapter.capture_aggregation=aggregation
        outputs.append(namespace['_raw_metrics'](adapter,pose,flap,centers,halves,axes,torch.zeros(1)))
    old,new=outputs
    assert new[2].tolist()==[[0,1]]
    assert new[0].capture_error_m.item()==pytest.approx(.04)
    assert old[1]['capture'].item()==pytest.approx(.5*(1+torch.exp(torch.tensor(-.8)).item()))
    assert new[1]['capture'].item()==pytest.approx(.25+.75*torch.exp(torch.tensor(-3.2)).item())
    for name in old[1]:
        if name!='capture':torch.testing.assert_close(old[1][name],new[1][name],rtol=0,atol=0)
    for index in (2,3,4):torch.testing.assert_close(old[index],new[index],rtol=0,atol=0)
