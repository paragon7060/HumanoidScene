#!/usr/bin/env python3
"""Whole original frozen TRAIN128; diagnose state-dependent flap contact servo."""
import sys


def main():
    # Reuse all strict original128/model/physics guards of the jaw diagnostic;
    # substitute only its process-local hook before the runner starts.
    import frozen_bilateral_close_probe as guarded
    from kuavo_isaaclab_scene.rl.multi_box.experiments import bilateral_close_probe as base
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cartesian_flap_probe import install_frozen_cartesian_flap_probe
    original = base.install_frozen_bilateral_close_probe
    base.install_frozen_bilateral_close_probe = install_frozen_cartesian_flap_probe
    try:
        return guarded.main()
    finally:
        base.install_frozen_bilateral_close_probe = original


if __name__ == '__main__':
    raise SystemExit(main())
