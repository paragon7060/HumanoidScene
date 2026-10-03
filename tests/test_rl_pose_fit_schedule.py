"""Fit refinement is explicit and leaves the original constant-rate default intact."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('pose_fit_schedule',
    Path(__file__).parents[1]/'scripts/rl/fit_v2_pose_student.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_default_constant_rate_and_opt_in_refinement_endpoints():
    schedule=module.fit_learning_rate
    assert [schedule(i,20000,3e-4) for i in [0,1,10000,19999]]==[3e-4]*4
    values=[schedule(i,101,3e-4,3e-6) for i in range(101)]
    assert values[0]==3e-4 and values[-1]==3e-6
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert schedule(0,1,3e-4)==3e-4


@pytest.mark.parametrize('step,steps,initial,final',[
    (0,0,3e-4,None),(10,10,3e-4,None),(-1,10,3e-4,None),
    (0,10,0.,None),(0,10,float('nan'),None),(0,10,3e-4,float('inf')),
    (0,10,3e-4,4e-4),(0,10,3e-4,0.),(0,1,3e-4,3e-6)])
def test_invalid_fit_schedule_cannot_produce_an_untrained_or_ambiguous_prior(step,steps,initial,final):
    with pytest.raises(ValueError):module.fit_learning_rate(step,steps,initial,final)
