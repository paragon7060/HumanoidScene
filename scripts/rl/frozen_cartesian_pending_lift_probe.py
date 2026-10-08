#!/usr/bin/env python3
"""Whole frozen TRAIN128; bound pending arm-target integration during real lift."""


def main():
    import frozen_bilateral_close_probe as guarded
    from kuavo_isaaclab_scene.rl.multi_box.experiments import bilateral_close_probe as base
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cartesian_flap_probe import install_frozen_cartesian_flap_probe
    original = base.install_frozen_bilateral_close_probe
    def install(pilot, manifest):
        return install_frozen_cartesian_flap_probe(pilot, manifest, contact_region=True,
            arm_goal_bounds='urdf', lift_drive='bounded-pending-target')
    base.install_frozen_bilateral_close_probe = install
    try:
        return guarded.main()
    finally:
        base.install_frozen_bilateral_close_probe = original


if __name__ == '__main__':
    raise SystemExit(main())
