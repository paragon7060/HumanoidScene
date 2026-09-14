"""Canonical package resource paths.

All runtime modules resolve assets and versioned default configurations from
the installed package rather than assuming a repository working directory.
Mutable deployments may redirect configuration files with the existing
``KUAVO_WORKCELL_LAYOUT`` and ``KUAVO_RACK_BOX_POSES`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


# Keep resource and subprocess paths independent of the caller's subpackage.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_IMPORT_ROOT = PACKAGE_ROOT.parent
ASSET_DIR = PACKAGE_ROOT / "assets"
PACKAGE_CONFIG_DIR = PACKAGE_ROOT / "configs"

# Every runtime and example should spawn the final measured, photo-textured
# wrappers.  The original *.usd and intermediate *_physical.usda files remain
# implementation dependencies of these composed assets only.
BOX_ATLAS_ASSETS: dict[str, Path] = {
    "small": ASSET_DIR / "SmallBox_atlas.usda",
    "medium": ASSET_DIR / "MediumBox_atlas.usda",
    "large": ASSET_DIR / "LargeBox_atlas.usda",
    "xlarge": ASSET_DIR / "XLargeBox_atlas.usda",
}

# Static, hand-editable roller-deck asset: all three shelf tiers merged
# into one file (each tier keeps its own PhysicsArticulationRootAPI; see
# workcell/rack_rollers.py). Built once by that module's build script and
# then committed; the running scene only reads it and never regenerates or
# overwrites it, so manual edits (e.g. aligning a carved shelf_ramp recess)
# persist.
RACK_ROLLER_ASSET: Path = ASSET_DIR / "rack_roller.usda"
# Runtime composition keeps the rack frame as one kinematic rigid body and
# relocates the three roller articulations beside it.  GPU PhysX contact views
# can then filter the rack body without parenting articulations under it.
RACK_ROLLER_RUNTIME_ASSET: Path = ASSET_DIR / "rack_roller_runtime.usda"


def _runtime_config_dir() -> Path:
    """Resolve mutable deployment config before packaged fallback defaults."""
    override = os.environ.get("KUAVO_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    working_tree_config = Path.cwd() / "configs"
    if working_tree_config.is_dir():
        return working_tree_config.resolve()
    return PACKAGE_CONFIG_DIR


CONFIG_DIR = _runtime_config_dir()


def default_artifacts_dir() -> Path:
    """Use the checkout's artifacts folder, or the working directory for wheels."""
    checkout = PACKAGE_IMPORT_ROOT.parent
    if PACKAGE_IMPORT_ROOT.name == "src" and (checkout / "pyproject.toml").is_file():
        return checkout / "artifacts"
    # A wheel must not write evaluation output into site-packages.
    return Path.cwd() / "artifacts"


def require_resource(path: Path, description: str) -> Path:
    """Return a package resource path or raise an actionable error."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path
