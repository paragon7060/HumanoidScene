import json

import pytest

from kuavo_isaaclab_scene.robot_material_config import load_robot_material_settings


EXPECTED_VISUAL_LINKS = {
    "base_link",
    "head_camera_base",
    "head_camera_depth",
    "head_radar",
    "waist_camera",
    "waist_camera_base",
    "waist_yaw_link",
    "zhead_1_link",
    "zhead_2_link",
    *(f"leg_{side}{index}_link" for side in "lr" for index in range(1, 7)),
    *(f"zarm_{side}{index}_link" for side in "lr" for index in range(1, 8)),
}


def test_default_palette_covers_every_kuavo_visual_link() -> None:
    settings = load_robot_material_settings()
    assert settings.name == "industrial_blue"
    assert settings.enabled
    assert set(settings.materials) == {
        "shell",
        "graphite",
        "accent_blue",
        "joint_metal",
        "sensor",
    }
    assert all(settings.material_for_link(link) is not None for link in EXPECTED_VISUAL_LINKS)


def test_original_palette_disables_runtime_override() -> None:
    settings = load_robot_material_settings("original")
    assert not settings.enabled
    assert settings.materials == {}
    assert settings.material_for_link("base_link") is None


def test_invalid_material_reference_is_rejected(tmp_path) -> None:
    payload = {
        "default": "bad",
        "presets": {
            "bad": {
                "materials": {
                    "shell": {
                        "diffuse_color": [0.2, 0.3, 0.4],
                        "roughness": 0.5,
                        "metallic": 0.0,
                    }
                },
                "link_rules": [{"pattern": "base_link", "material": "missing"}],
            }
        },
    }
    path = tmp_path / "bad_material.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown material"):
        load_robot_material_settings(config_path=path)
