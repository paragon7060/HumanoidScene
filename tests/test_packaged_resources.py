"""Checks that required deployment resources are present in the package."""

from pathlib import Path

from kuavo_isaaclab_scene.paths import ASSET_DIR, CONFIG_DIR, PACKAGE_CONFIG_DIR


def test_required_assets_are_packaged() -> None:
    required = (
        ASSET_DIR / "Rack.usd",
        ASSET_DIR / "SmallBox.usd",
        ASSET_DIR / "MediumBox.usd",
        ASSET_DIR / "LargeBox.usd",
        ASSET_DIR / "XLargeBox.usd",
        ASSET_DIR / "button_station.usda",
        ASSET_DIR / "kuavo5" / "usd" / "kuavo5_fixed.usd",
    )
    assert all(path.is_file() for path in required)


def test_default_runtime_configs_are_packaged() -> None:
    assert (CONFIG_DIR / "workcell_layout.json").is_file()
    assert (CONFIG_DIR / "rack_box_poses.json").is_file()
    assert (CONFIG_DIR / "grippers.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "workcell_layout.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "rack_box_poses.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "grippers.json").is_file()


def test_deployment_and_wheel_fallback_configs_are_synchronized() -> None:
    repository_configs = Path(__file__).resolve().parents[1] / "configs"
    for name in ("workcell_layout.json", "rack_box_poses.json", "grippers.json"):
        assert (repository_configs / name).read_bytes() == (
            PACKAGE_CONFIG_DIR / name
        ).read_bytes()
