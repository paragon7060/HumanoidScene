import math
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('capture_scale_bounds_audit',
    Path(__file__).resolve().parents[1]/'scripts/rl/audit_capture_scale_bounds.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
score_bounds_from_saved_mean=audit.score_bounds_from_saved_mean


@pytest.mark.parametrize('candidate_scale', [.025,.05])
def test_intervals_contain_every_feasible_pair_and_attain_endpoints(candidate_scale):
    m=torch.linspace(0,1,101,dtype=torch.float64)
    bounds=score_bounds_from_saved_mean(m,candidate_scale=candidate_scale)
    t0=(2*m-1).clamp_min(0)
    grid=t0[:,None]+(m-t0)[:,None]*torch.linspace(0,1,2001,dtype=torch.float64)
    power=.1/candidate_scale
    large=(2*m[:,None]-grid).pow(power);small=grid.pow(power)
    mean=(large+small)*.5;weak=.25*large+.75*small
    for name,values in [('narrow_mean',mean),('weak_blend',weak)]:
        low=bounds[name+'_lower'];high=bounds[name+'_upper']
        assert ((values>=low[:,None]-1e-12)&(values<=high[:,None]+1e-12)).all()
        assert torch.allclose(values.max(-1).values,high,atol=1e-12,rtol=0)
        assert torch.allclose(values.min(-1).values,low,atol=1e-7,rtol=0)
        assert low[0]==high[0]==0 and low[-1]==high[-1]==1


def test_balanced_pair_and_asymmetric_stationary_point_are_included():
    m=torch.tensor([.5,.8,.95],dtype=torch.float64)
    bounds=score_bounds_from_saved_mean(m)
    assert (bounds['weak_blend_lower']<=m.pow(4)).all()
    assert (bounds['weak_blend_upper']>=m.pow(4)).all()
    assert bounds['weak_blend_lower'][1]<.25+.75*.6**4
    assert bounds['weak_blend_lower'][1]<.8**4


def test_almost_equal_scales_remain_finite_without_exponent_overflow():
    values=score_bounds_from_saved_mean(torch.linspace(0,1,100),candidate_scale=.0999999999)
    assert all(torch.isfinite(v).all() for v in values.values())


@pytest.mark.parametrize('mean,scale', [([-.01],.025),([1.01],.025),([math.nan],.025),([.5],0),([.5],.1),([.5],math.inf)])
def test_nonphysical_or_non_narrow_candidate_is_rejected(mean,scale):
    with pytest.raises(ValueError):score_bounds_from_saved_mean(torch.tensor(mean),candidate_scale=scale)
