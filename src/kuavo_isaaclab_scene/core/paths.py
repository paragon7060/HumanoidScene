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
