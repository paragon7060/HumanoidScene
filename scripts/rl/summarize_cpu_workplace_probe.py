#!/usr/bin/env python3
"""Read a finished managed workplace search; keep repeated cases explicit."""
import argparse,json,os
from pathlib import Path
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args();parent=args.experiment_dir.resolve()
    if parent.stat().st_uid!=os.getuid():parser.error('Only an owned managed experiment may be summarized')
    status=json.loads((parent/'status.json').read_text());pid=status['training_pid']
    if Path('/proc',str(pid)).exists() or status.get('training_exit_code')!=0:
        parser.error('Wait for the original writer to stop normally; observation timeout is not termination')
    run=Path(json.loads((parent/'launch.json').read_text())['run'])
    if run.parent!=parent:parser.error('Managed run must belong to the requested experiment')
    manifest=json.loads((run/'manifest.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
    result=summarize_workplace_results(manifest,metrics)
    args.output_json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('unique_fresh_TRAIN_cases','physical_candidate_attempts',
        'promising_candidates_for_fresh_TRAIN_recheck','all_regions_have_a_measured_success_candidate')},ensure_ascii=False))


if __name__=='__main__':main()
