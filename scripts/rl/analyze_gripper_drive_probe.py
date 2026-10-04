#!/usr/bin/env python3
"""Write a descriptive, paired report without accessing Isaac, CUDA, or Drive."""
import argparse
import json
from pathlib import Path

from kuavo_isaaclab_scene.rl.multi_box.experiments.gripper_drive_report import summarize_gripper_drive_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.run_dir/name).read_text())
    report = summarize_gripper_drive_probe(read('manifest.json'), read('metrics.json'),
        read('gripper_drive_audit.json'), read('status.json'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
