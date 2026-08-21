"""Canonical package resource paths.

All runtime modules resolve assets and versioned default configurations from
the installed package rather than assuming a repository working directory.
Mutable deployments may redirect configuration files with the existing
``KUAVO_WORKCELL_LAYOUT`` and ``KUAVO_RACK_BOX_POSES`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
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


def require_resource(path: Path, description: str) -> Path:
    """Return a package resource path or raise an actionable error."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path
