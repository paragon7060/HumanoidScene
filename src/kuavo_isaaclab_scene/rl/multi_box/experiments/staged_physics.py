"""Explicit dynamics identity for fresh-Q held-base solver experiments."""
import math


def frozen_prior_lift_contract(contract):
    """Old terminal labels are used only to restore a frozen actor prior.

    Staged Q/replay uses the unmodified current contract and fresh critics.
    Other terminal fields remain strict. Old measured data does not become
    current-reward Q data through this actor-only compatibility path.
    """
    from ..geometry.rack import GRASP_LIFT_CLEARANCE_CONTRACT
    terminal=contract.get('terminal_contract',{})
    reference=terminal.get('proof_lift_reference')
    if reference is None:return contract
    if reference!=GRASP_LIFT_CLEARANCE_CONTRACT:raise ValueError('Unknown proof-lift reference')
    offset=terminal.get('proof_lift_support_offset_m')
    if offset is None or not math.isfinite(offset) or offset<0:
        raise ValueError('Missing or invalid active support offset')
    return contract|dict(terminal_contract={k:v for k,v in terminal.items()
                         if k not in {'proof_lift_reference','proof_lift_support_offset_m'}})


def require_current_lift_contract(contract):
    from ..geometry.rack import grasp_lift_terminal_contract
    terminal=contract.get('terminal_contract',{})
    expected=grasp_lift_terminal_contract()
    if any(terminal.get(key)!=value for key,value in expected.items()):
        raise ValueError('Current grasp requires active support clearance; old Q/replay must not resume')


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
