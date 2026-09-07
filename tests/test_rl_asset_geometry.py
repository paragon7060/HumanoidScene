"""Optional USD-only geometry tests; no SimulationApp or rendering.

Run in a Python environment where the USD bindings are already available.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pxr.Usd")
from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry


def test_physical_wrapper_nested_scale_and_flap_size():
    root = Path(__file__).resolve().parents[1]
    cfg = SimpleNamespace(spawn=SimpleNamespace(
        usd_path=str(root / "src/kuavo_isaaclab_scene/assets/MediumBox_atlas.usda"), scale=(1., 1., 1.)))
    g = box_geometry(cfg, ("flap_right", "flap_left"))
    assert g.half_size[:2] == pytest.approx((.16, .11), abs=1e-6)
    assert g.half_size[2] == pytest.approx(.185925 / 2, abs=1e-6)
    assert g.flaps["flap_right"].half_size[2] == pytest.approx(.055, abs=1e-6)
    assert g.flaps["flap_right"].body_path.endswith("/flap_right")
    cfg.spawn.scale = (2., 1., 1.)
    scaled = box_geometry(cfg)
    assert scaled.half_size[0] == pytest.approx(2*g.half_size[0])
    assert scaled.half_size[2] == pytest.approx(g.half_size[2])
