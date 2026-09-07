"""Parallel RL configuration/architecture checks without importing Isaac Sim."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

from kuavo_isaaclab_scene.rl.envs.parallel_cfg import ParallelEnvCfg


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
RL = SOURCE / "kuavo_isaaclab_scene" / "rl"
SCENES = RL / "scenes"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path, tree):
    """Resolve relative imports without importing their simulator dependencies."""
    relative = path.relative_to(SOURCE).with_suffix("")
    module = ".".join(relative.parts)
    package = module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            expression = "." * node.level + (node.module or "")
            base = resolve_name(expression, package) if node.level else expression
            yield base, node.lineno
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}", node.lineno


@pytest.mark.parametrize("num_envs", [1, 2, 8, 64, 256])
@pytest.mark.parametrize("spacing", [5.0, 8.0, 12.5])
def test_parallel_configuration_preserves_explicit_batch_size(num_envs, spacing):
    cfg = ParallelEnvCfg(num_envs=num_envs, env_spacing=spacing)
    assert cfg.scene_kwargs() == {
        "num_envs": num_envs,
        "env_spacing": spacing,
        "replicate_physics": True,
        "filter_collisions": True,
        "clone_in_fabric": False,
    }


@pytest.mark.parametrize("num_envs", [0, -1, True, False, 1.5, "8"])
def test_invalid_environment_counts_are_rejected(num_envs):
    with pytest.raises(ValueError, match="positive integer"):
        ParallelEnvCfg(num_envs=num_envs).scene_kwargs()


@pytest.mark.parametrize("spacing", [-1.0, 0.0, 4.999, float("inf"), float("-inf"), float("nan")])
def test_cells_require_finite_spacing_of_at_least_five_metres(spacing):
    with pytest.raises(ValueError, match="env_spacing"):
        ParallelEnvCfg(env_spacing=spacing).scene_kwargs()


def test_collision_filtering_and_usd_spawner_constraints_are_explicit():
    with pytest.raises(ValueError, match="collision filtering"):
        ParallelEnvCfg(filter_collisions=False).scene_kwargs()
    with pytest.raises(ValueError, match="clone_in_fabric"):
        ParallelEnvCfg(clone_in_fabric=True).scene_kwargs()
    # The slower non-replicated fallback remains available without disabling
    # collision isolation, useful when debugging per-environment USD changes.
    kwargs = ParallelEnvCfg(replicate_physics=False).scene_kwargs()
    assert kwargs["replicate_physics"] is False
    assert kwargs["filter_collisions"] is True


@pytest.mark.parametrize("num_envs,spacing,extent", [(1, 8.0, 30.0), (9, 5.0, 30.0),
                                                      (64, 8.0, 128.0), (17, 6.0, 60.0)])
def test_global_floor_covers_the_cloned_grid(num_envs, spacing, extent):
    assert ParallelEnvCfg(num_envs=num_envs, env_spacing=spacing).ground_extent == extent


def test_rl_modules_do_not_import_interactive_scene_or_teleoperation_pipelines():
    forbidden = (
        "kuavo_isaaclab_scene.envs.manager_env",
        "kuavo_isaaclab_scene.envs.scene",
        "kuavo_isaaclab_scene.envs.manager_mdp",
        "kuavo_isaaclab_scene.evaluation",
        "kuavo_isaaclab_scene.teleop",
    )
    violations = []
    for path in sorted(RL.rglob("*.py")):
        for imported, line in _imports(path, _tree(path)):
            if any(imported == banned or imported.startswith(banned + ".") for banned in forbidden):
                violations.append(f"{path.relative_to(ROOT)}:{line}: {imported}")
    assert not violations, "RL must remain independent of general pipelines:\n" + "\n".join(violations)


def test_minimal_rl_scene_directly_inherits_isaaclab_scene_configuration():
    path = SCENES / "scene_cfg.py"
    tree = _tree(path)
    imported_names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    scene_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert scene_classes, "RL scene assembly needs its own scene configuration class."
    bases = [imported_names.get(base.id, base.id) for node in scene_classes for base in node.bases
             if isinstance(base, ast.Name)]
    assert "isaaclab.scene.InteractiveSceneCfg" in bases
    assert all(not name.endswith("RobustWorkcellSceneCfg") for name in bases)


def test_scene_assembly_has_no_factory_movers_or_remote_usd_dependencies():
    entity_names = set()
    usd_expressions = []
    network_references = []
    for path in sorted(SCENES.rglob("*.py")):
        tree = _tree(path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name) and node.value.id == "scene"):
                entity_names.add(node.attr)
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and node.func.id == "setattr"
                        and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "scene" and isinstance(node.args[1], ast.Constant)):
                    entity_names.add(node.args[1].value)
                usd_expressions.extend(keyword.value for keyword in node.keywords if keyword.arg == "usd_path")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(("omniverse://", "http://", "https://")):
                    network_references.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id in {"NUCLEUS_ASSET_ROOT_DIR", "ISAAC_NUCLEUS_DIR"}:
                network_references.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert {"robot", "rack", "rack_visual", "conveyor_surface", "button_station"} <= entity_names
    assert not entity_names.intersection({"factory", "moving_robot", "moving_human", "human", "conveyor"})
    assert usd_expressions, "Keep local robot/rack/box assets in the independent scene assembly."
    assert not network_references, "RL scene geometry must not depend on network USD: " + ", ".join(network_references)


def test_shared_floor_is_explicitly_global_for_collision_filtering():
    tree = _tree(SCENES / "workcell.py")
    ground_assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Attribute) and target.attr == "ground"
                                  and isinstance(target.value, ast.Name) and target.value.id == "scene"
                                  for target in node.targets)]
    assert len(ground_assignments) == 1
    call = ground_assignments[0].value
    assert isinstance(call, ast.Call)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords["prim_path"]) == "/World/Ground"
    assert ast.literal_eval(keywords["collision_group"]) == -1
