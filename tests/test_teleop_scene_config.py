"""CPU-only tests: do not import Isaac Sim or start an application."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.teleop.teleop_scene_config import apply_scene_config


def test_default_scene_is_untouched():
    cfg = SimpleNamespace(scene=SimpleNamespace(original=True))
    assert apply_scene_config(None, cfg) == ()
    assert vars(cfg.scene) == {"original": True}


def test_select_scene_and_register_recording_object(tmp_path):
    path = tmp_path / "scene.py"
    path.write_text("RECORDING_OBJECTS = ('demo_cube',)\n"
                    "def configure(cfg):\n"
                    "    cfg.scene.demo_cube = 'rigid-object-config'\n"
                    "    cfg.scene.old_box = None\n")
    cfg = SimpleNamespace(scene=SimpleNamespace(old_box="box"))
    assert apply_scene_config(path, cfg) == ("demo_cube",)
    assert cfg.scene.old_box is None
    assert cfg.scene.demo_cube == "rigid-object-config"


@pytest.mark.parametrize("source, message", [
    ("RECORDING_OBJECTS = ()", "configure"),
    ("RECORDING_OBJECTS = ('bad/key',)\ndef configure(cfg): pass", "unique scene"),
    ("RECORDING_OBJECTS = ('cube', 'cube')\ndef configure(cfg): pass", "unique scene"),
    ("RECORDING_OBJECTS = ('missing',)\ndef configure(cfg): pass", "missing recording"),
])
def test_invalid_contract_fails_before_environment_creation(tmp_path, source, message):
    path = tmp_path / "scene.py"
    path.write_text(source)
    with pytest.raises(ValueError, match=message):
        apply_scene_config(path, SimpleNamespace(scene=SimpleNamespace()))


def test_collector_configures_scene_before_creating_env():
    source = Path(__file__).parents[1] / "src/kuavo_isaaclab_scene/teleop/collect_quest_teleop.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {node.func.id: node.lineno for node in ast.walk(main)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in {"apply_scene_config", "ManagerBasedRLEnv"}}
    assert calls["apply_scene_config"] < calls["ManagerBasedRLEnv"]
