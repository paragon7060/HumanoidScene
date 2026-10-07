"""Explicit dynamics identity for fresh-Q held-base solver experiments."""
import math


def frozen_prior_lift_contract(contract):
    """Old terminal labels are used only to restore a frozen actor prior.

    Staged Q/replay uses the unmodified current contract and fresh critics.
    The reviewed 10cm reset-relative drop guard is also ignored for this
    actor-only match. Other terminal fields remain strict; old measured
    data does not become current-reward Q data through this path.
    """
    from ..geometry.box_drop import frozen_drop_actor_contract
    contract = frozen_drop_actor_contract(contract)
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


CPU_PHYSICS_BACKEND = 'CPU_PhysX_v1'


def staged_solver_contract(name, *, physics_backend=None):
    if name not in ('TGS','PGS'):raise ValueError('Expected TGS or PGS')
    if physics_backend not in (None, CPU_PHYSICS_BACKEND):
        raise ValueError('Unknown staged physics backend')
    result = dict(solver=name,solver_type={'TGS':1,'PGS':0}[name],
                physics_dt_s=1/120,control_dt_s=1/30)
    if physics_backend is not None:result['physics_backend']=physics_backend
    return result


def frozen_cpu_actor_contract(contract):
    """Only a frozen controller may ignore the reviewed CPU backend marker.

    Current Q/replay always retains the backend and drop markers. The reviewed
    10cm drop guard is the only safety exception for this frozen actor match;
    solver, timesteps, actions, observations and rewards remain strict.
    """
    from ..geometry.box_drop import frozen_drop_actor_contract
    contract = frozen_drop_actor_contract(contract)
    dynamics=contract.get('physics_dynamics',{})
    if not isinstance(dynamics,dict) or 'physics_backend' not in dynamics:return contract
    if dynamics!=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND):
        raise ValueError('Unknown CPU backend cannot bypass frozen actor compatibility')
    return contract|dict(physics_dynamics=staged_solver_contract('PGS'))


def configure_staged_physics(cfg, contract):
    """Legacy data is TGS; a changed solver requires an explicit fresh contract.

    This changes neither asset properties nor iteration counts. The staged
    learner includes this identity in its strict checkpoint/replay contract.
    An old actor can initialize new Q; old dynamics' Q/replay cannot resume.
    Restore the recorded drop guard, including None for historical labels.
    """
    from ..geometry.box_drop import configure_grasp_drop, configured_drop_limit
    if hasattr(cfg, 'multi_box'):
        configure_grasp_drop(cfg, contract)
    elif configured_drop_limit(contract) is not None:
        raise ValueError('Box-drop contract requires a multi-box configuration')
    dynamics=contract.get('physics_dynamics')
    if dynamics is None:
        if cfg.sim.physx.solver_type!=1:raise ValueError('Legacy staged physics requires TGS')
        return
    if not isinstance(dynamics,dict) or dynamics!=staged_solver_contract(
            dynamics.get('solver'), physics_backend=dynamics.get('physics_backend')):
        raise ValueError('Staged physics dynamics contract differs')
    if dynamics.get('physics_backend') == CPU_PHYSICS_BACKEND and str(cfg.sim.device) != 'cpu':
        raise ValueError('CPU physics Q/replay requires actual CPU simulation')
    if not math.isclose(cfg.sim.dt,dynamics['physics_dt_s'],abs_tol=1e-12,rel_tol=0) \
            or not math.isclose(cfg.sim.dt*cfg.decimation,dynamics['control_dt_s'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Staged physics/control timestep differs')
    cfg.sim.physx.solver_type=dynamics['solver_type']
