"""Guard estimates from silently replacing manufacturer-specified inertials."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from kuavo_isaaclab_scene.core.paths import ASSET_DIR


def test_hand_estimates_only_cover_missing_inertials_with_physical_inertia():
    folder = ASSET_DIR / "kuavo_s200062"
    links = {e.get("name"): e for e in ET.parse(folder / "urdf/biped_s200062.urdf").findall("link")}
    estimates = json.loads((folder / "teleop_inertials.json").read_text())["links"]
    for name, values in estimates.items():
        assert links[name].find("inertial") is None, name
        assert 0 < values["mass_kg"] <= .5
        inertia = np.array(values["diagonal_inertia_kg_m2"])
        assert np.all(inertia > 0)
        assert 2 * inertia.max() <= inertia.sum() + 1e-12
        assert np.all(np.isfinite(values["com_m"]))
    for side in "lr":
        total = sum(v["mass_kg"] for k, v in estimates.items()
                    if k.startswith(f"{side}_") or k.startswith(f"zarm_{side}"))
        assert .5 < total < 1.0
