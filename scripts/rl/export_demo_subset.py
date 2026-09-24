#!/usr/bin/env python3
"""Export selected Quest demonstration episodes into one new, compressed file.

Reads the V2 grasp transition files written by
`kuavo_isaaclab_scene.recording.rl_transition_recorder` and copies whole
episodes without changing a single stored value.  Files are merged only when
they share the same environment contract, because SAC replays these
transitions as if the training environment had produced them.  No Isaac, GPU
or torch import; the source files are opened read-only.
"""

import argparse
import json
from pathlib import Path

import h5py

from kuavo_isaaclab_scene.recording.rl_transition_recorder import TRANSITION_FIELDS


FORMAT = "kuavo_v2_grasp_sac_transitions"
# Everything that changes what a stored transition means.  The controller
# mapping and the seed describe how an operator produced the demonstration,
# not the observation/action/reward contract, so they may differ per file.
CONTRACT_KEYS = (
    "action_encoding",
    "action_dim",
    "action_terms",
    "actor_obs_dim",
    "critic_obs_dim",
    "critic_mapping",
    "control_dt",
    "robot_model",
    "gripper",
    "gripper_close_force_n",
    "rack_rollers",
    "reward_source",
    "multi_box",
    "task",
)


def read_manifest(file, path):
    """Return the manifest of a demonstration file, rejecting other formats."""
    if file.attrs.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return json.loads(file.attrs["manifest_json"])


def contract_differences(manifest, reference):
    """Return the contract keys that stop two files from being merged."""
    return [key for key in CONTRACT_KEYS if manifest.get(key) != reference.get(key)]


def merged_manifest(manifests):
    """Keep values shared by every source file and mark the ones that differ.

    Contract keys are validated to be equal before this runs, so only
    descriptive keys such as the controller mapping or the seed can become
    `"mixed"`.  Per-episode provenance keeps the exact source values.
    """
    merged = dict(manifests[0])
    for key in list(merged):
        if any(manifest.get(key) != merged[key] for manifest in manifests[1:]):
            merged[key] = "mixed"
    return merged


def selected_episodes(file, *, success_only=True, allow_incomplete=False):
    """Return episode names to copy; interrupted recordings are skipped."""
    chosen = []
    for name in sorted(file["episodes"]):
        attrs = file["episodes"][name].attrs
        if "end_reason" not in attrs and not allow_incomplete:
            continue
        if success_only and not bool(attrs.get("success", False)):
            continue
        chosen.append(name)
    return chosen


def _copy_episode(episode, group, index, path, name, compression_level):
    transitions = episode["transitions"]
    missing = set(TRANSITION_FIELDS) - set(transitions)
    if missing:
        raise ValueError(f"{path}:{name} is missing transition fields {sorted(missing)}")
    lengths = {transitions[field].shape[0] for field in TRANSITION_FIELDS}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError(f"{path}:{name} has empty or inconsistent transition lengths")
    target = group.create_group(f"episode_{index:06d}")
    copied = target.create_group("transitions")
    for field in TRANSITION_FIELDS:
        data = transitions[field]
        copied.create_dataset(
            field, data=data[...], dtype=data.dtype, chunks=True,
            compression="gzip", compression_opts=compression_level)
    target.attrs["num_transitions"] = int(lengths.pop())
    for key in ("success", "end_reason"):
        if key in episode.attrs:
            target.attrs[key] = episode.attrs[key]
    # Keep the provenance of every copied episode so a merged file can still be
    # traced back to the session that recorded it.
    target.attrs["source_file"] = Path(path).name
    target.attrs["source_episode"] = name


def export(inputs, output, *, success_only=True, allow_incomplete=False,
           compression_level=4):
    """Copy the selected episodes into a new file and return their summary."""
    if not inputs:
        raise ValueError("At least one input file is required.")
    if not 0 <= compression_level <= 9:
        raise ValueError("gzip compression level must be between 0 and 9.")
    reference = reference_path = None
    summary = []
    with h5py.File(output, "x") as out:
        episodes = out.create_group("episodes")
        manifests = []
        for path in inputs:
            with h5py.File(path, "r") as source:
                manifest = read_manifest(source, path)
                if reference is None:
                    reference, reference_path = manifest, path
                else:
                    differences = contract_differences(manifest, reference)
                    if differences:
                        raise ValueError(
                            f"{path} does not share the environment contract of "
                            f"{reference_path}: {differences}")
                names = selected_episodes(
                    source, success_only=success_only, allow_incomplete=allow_incomplete)
                if names:
                    manifests.append(manifest)
                for name in names:
                    _copy_episode(source["episodes"][name], episodes, len(summary),
                                  path, name, compression_level)
                    summary.append({
                        "file": Path(path).name,
                        "episode": name,
                        "transitions": int(
                            source["episodes"][name]["transitions"]["reward"].shape[0]),
                        "success": bool(source["episodes"][name].attrs.get("success", False)),
                        "controller_mapping": manifest.get("controller_mapping"),
                        "seed": manifest.get("seed"),
                    })
        if not summary:
            raise ValueError("No episode matched the selection; nothing was exported.")
        out.attrs["format"] = FORMAT
        out.attrs["format_version"] = 1
        out.attrs["manifest_json"] = json.dumps(merged_manifest(manifests), sort_keys=True)
        out.attrs["exported_from_json"] = json.dumps(summary, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Recorded demonstration files.")
    parser.add_argument("--output", required=True, type=Path,
                        help="New file to create; an existing path is never overwritten.")
    parser.add_argument("--all-episodes", action="store_true",
                        help="Copy every complete episode instead of successful ones only.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Also copy episodes whose recording was interrupted.")
    parser.add_argument("--compression-level", type=int, default=4,
                        help="gzip level 0-9 for the copied transition datasets.")
    args = parser.parse_args()
    summary = export(args.inputs, args.output, success_only=not args.all_episodes,
                     allow_incomplete=args.allow_incomplete,
                     compression_level=args.compression_level)
    transitions = sum(row["transitions"] for row in summary)
    print(f"Exported {len(summary)} episode(s), {transitions} transitions, "
          f"{args.output.stat().st_size / 1e6:.1f} MB -> {args.output}")
    for index, row in enumerate(summary):
        print(f"  episode_{index:06d} <- {row['file']}:{row['episode']} "
              f"({row['transitions']} transitions, success={row['success']})")


if __name__ == "__main__":
    main()
