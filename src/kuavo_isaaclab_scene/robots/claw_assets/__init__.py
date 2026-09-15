"""Reusable S200062-derived claw; pure asset metadata needs no Isaac runtime."""

from .package import ClawAsset, append_claw_branch, load_claw_asset

__all__ = ["ClawAsset", "append_claw_branch", "load_claw_asset"]
