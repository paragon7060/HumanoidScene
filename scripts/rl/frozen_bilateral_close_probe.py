#!/usr/bin/env python3
"""Same whole frozen TRAIN128; isolate near-jaw choice from neural body motion."""
import json
from pathlib import Path
import sys


def main():
    argv=sys.argv[1:]
    required=('--no-training','--cpu-workplace-probe','--base-waypoint-probe',
        '--unmeasured-size-workplace-probe')
    if any(flag not in argv for flag in required) or '--training' in argv:
        raise ValueError('Use only the explicit full frozen supported-size workplace route')
    forbidden=('--cpu-physics-training','--base-attitude-gain-probe','--base-substep-trace-env-indices',
        '--grasp-observation-audit','--frozen-physics-backend-eval','--jaw-behavior',
        '--body-behavior','--contact-stability-probe','--tgs-zero-velocity-probe',
        '--contact-last-probe','--pgs-probe','--gripper-drive-probe',
        '--centered-world-probe','--reset-failure-diagnostics','--packed-background-probe')
    if any(a.split('=')[0] in forbidden for a in argv):
        raise ValueError('Isolate jaw choice from learning and other physical/controller probes')
    def value(flag):
        return argv[argv.index(flag)+1]
    if value('--device')!='cpu' or value('--learner-device')!='cpu':
        raise ValueError('This frozen diagnostic imports no GPU Q/replay')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import validate_cpu_workplace_probe
    from kuavo_isaaclab_scene.rl.multi_box.experiments.bilateral_close_probe import (
        validate_bilateral_close_probe,install_frozen_bilateral_close_probe)
    waves=json.loads(Path(value('--waves-json')).read_text())
    physical=json.loads(Path(value('--training-manifest')).read_text())
    workplace=validate_cpu_workplace_probe(waves,physical,enabled=True,device='cpu',
        training=False,steps=int(value('--steps')),waypoint_enabled=True,
        explicit_frozen=True,unmeasured_size_probe=True)
    validate_bilateral_close_probe(waves,training=False,steps=int(value('--steps')),workplace=workplace)
    import torch
    state=torch.load(value('--checkpoint'),map_location='cpu',weights_only=True)
    if (state.get('artifact_type')!='staged_actual_flap_regional_actor_memory_servo_retention_sac_v1'
            or state.get('actor_updates')!=0 or state.get('critic_updates')!=0):
        raise ValueError('Keep the same pristine regional actor for the closure-only comparison')
    del state
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments import policy_manifest
    import train_batched_staged_goal
    restore=install_frozen_bilateral_close_probe(StagedGoalSACPilot,policy_manifest)
    try:
        return train_batched_staged_goal.main()
    finally:
        restore()


if __name__=='__main__':
    raise SystemExit(main())
