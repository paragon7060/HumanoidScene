"""Explicit dynamics identity for fresh-Q held-base solver experiments."""
import math


def staged_solver_contract(name):
    if name not in ('TGS','PGS'):raise ValueError('Expected TGS or PGS')
    return dict(solver=name,solver_type={'TGS':1,'PGS':0}[name],
                physics_dt_s=1/120,control_dt_s=1/30)


def configure_staged_physics(cfg, contract):
    """Legacy data is TGS; a changed solver requires an explicit fresh contract.

    This changes neither asset properties nor iteration counts. The staged
    learner includes this identity in its strict checkpoint/replay contract.
    An old actor can initialize new Q; old dynamics' Q/replay cannot resume.
    """
    dynamics=contract.get('physics_dynamics')
    if dynamics is None:
        if cfg.sim.physx.solver_type!=1:raise ValueError('Legacy staged physics requires TGS')
        return
    if not isinstance(dynamics,dict) or dynamics!=staged_solver_contract(dynamics.get('solver')):
        raise ValueError('Staged physics dynamics contract differs')
    if not math.isclose(cfg.sim.dt,dynamics['physics_dt_s'],abs_tol=1e-12,rel_tol=0) \
            or not math.isclose(cfg.sim.dt*cfg.decimation,dynamics['control_dt_s'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Staged physics/control timestep differs')
    cfg.sim.physx.solver_type=dynamics['solver_type']
