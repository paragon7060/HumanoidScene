#!/usr/bin/env python3
"""Finalize packaged S200062/S56 USDs with physical four-bar closure joints.

Run with the Isaac Lab conda Python; this is a build step, not a simulation.
Only the small root layer is authored; existing mesh layers are not rewritten.
"""
from pathlib import Path
import argparse
import os
import sys


def main():
    # Isaac's bundled USD libraries normally become importable only after Kit
    # starts. Load those same libraries directly for this offline build step.
    try:
        from pxr import Usd
    except ImportError:
        import importlib.util
        spec = importlib.util.find_spec("isaacsim")
        if spec is None:
            raise RuntimeError("Use a Python environment containing Isaac Sim")
        sim_dir = Path(next(iter(spec.submodule_search_locations)))
        libs = sorted((sim_dir / "extscache").glob("omni.usd.libs-*"))
        if not libs:
            raise RuntimeError("Isaac Sim USD libraries not found")
        if os.environ.get("KUAVO_USD_BOOTSTRAPPED") == "1":
            raise RuntimeError("Cannot load the bundled USD runtime")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(libs[-1]), env.get("PYTHONPATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join([
            str(Path(sys.prefix) / "lib"), str(libs[-1] / "bin"),
            env.get("LD_LIBRARY_PATH", "")])
        env["KUAVO_USD_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kuavo_isaaclab_scene.core.paths import ASSET_DIR
    from kuavo_isaaclab_scene.robots.twofinger_linkage import author_closed_linkages, require_closed_linkages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate existing assets without writing")
    args = parser.parse_args()
    paths = args.paths or [ASSET_DIR / model / "usd" / f"{model}_fixed.usd"
                          for model in ("kuavo_s200062", "kuavo_s56_twofinger")]
    for path in paths:
        stage = Usd.Stage.Open(str(path.resolve()))
        if not args.check:
            author_closed_linkages(stage)
            stage.GetRootLayer().Save()
        require_closed_linkages(stage.GetDefaultPrim())
        print(f"Validated four closed loops: {path}")


if __name__ == "__main__":
    main()
