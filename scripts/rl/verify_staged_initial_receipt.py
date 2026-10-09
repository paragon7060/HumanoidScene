"""Record the exact named initialization input; never claim runtime equality.

The separate whole-DEV verifier compares its protected runtime model to this
input. No live HDF/replay, GPU/Isaac or model updates are used here.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes
from export_eval_q_videos import restored_agent
from summarize_batched_staged_run import read_snapshot


def named_initial_state(ready, managed):
    path = Path(ready['initial_checkpoint'])
    command = managed['command']
    if command.count('--checkpoint') != 1:
        raise ValueError('Exactly one explicit named initialization input required')
    index = command.index('--checkpoint')
    if index + 1 == len(command) or Path(command[index + 1]).absolute() != path.absolute():
        raise ValueError('Actual launch checkpoint differs from the verified named input')
    blob = owned_stable_bytes(path)
    if hashlib.sha256(blob).hexdigest() != ready['checkpoint_SHA256']:
        raise ValueError('Named initialization SHA256 changed')
    state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if state.get('actor_updates') != 0 or state.get('critic_updates') != 0:
        raise ValueError('Named input is not a fresh zero-update initialization')
    def finite(value):
        if isinstance(value, torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value, dict):return all(finite(x) for x in value.values())
        if isinstance(value, (list, tuple)):return all(finite(x) for x in value)
        return True
    if not finite(state):
        raise ValueError('Named initialization contains nonfinite tensors')
    if hashlib.sha256(owned_stable_bytes(path)).hexdigest() != ready['checkpoint_SHA256']:
        raise ValueError('Named input changed during verification')
    return path, state


def verify(launch_path, ready_path, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Receipt verification must be CPU-only')
    launch = read_snapshot(launch_path)
    parent = Path(launch['parent'])
    managed, status = [read_snapshot(parent/name) for name in ['launch.json', 'status.json']]
    run = Path(managed['run'])
    gpu = launch['gpu']
    for key, hint in [('training_pid', str(run)), ('supervisor_pid', str(parent))]:
        proc = Path('/proc', str(status[key]))
        if not proc.exists() or proc.stat().st_uid != os.getuid() \
                or hint.encode() not in (proc/'cmdline').read_bytes() \
                or f'CUDA_VISIBLE_DEVICES={gpu}'.encode() not in (proc/'environ').read_bytes().split(b'\0'):
            raise ValueError('Original writer/supervisor identity or GPU mask changed')
    ready = read_snapshot(ready_path)
    if ready['checkpoint_SHA256'] != launch['source_checkpoint_SHA256']:
        raise ValueError('Launch provenance and initialization proof differ')
    path, state = named_initial_state(ready, managed)
    restored_agent(state)  # Validate the actual artifact, goal contract and all model buffers.
    proof = dict(recorded_at=datetime.now().astimezone().isoformat(),
        protected_checkpoint=str(path), checkpoint_SHA256=ready['checkpoint_SHA256'],
        named_initial_checkpoint_all_tensors_finite=True,
        runtime_end_saved_model_equality_PENDING=True,
        source='verified_named_CPU_initializer_not_runtime_checkpoint',
        actual_command_named_checkpoint_and_launch_SHA_verified=True,
        original_live_writer_and_supervisor_owner_run_CUDA_verified=True,
        original_input_unmodified=True, no_GPU_or_live_HDF_replay_read=True,
        no_training_changes_process_signals_or_new_authentication=True,
        goal_not_complete=True)
    if output.exists():
        old = read_snapshot(output)
        if any(old.get(k) != proof[k] for k in ['protected_checkpoint', 'checkpoint_SHA256',
                                               'named_initial_checkpoint_all_tensors_finite', 'source']):
            raise ValueError('Existing receipt has a different identity; preserve both sources')
        return old
    pending = output.with_suffix('.pending.json')
    pending.write_text(json.dumps(proof, indent=2)+'\n')
    pending.replace(output)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ['launch', 'ready', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    proof = verify(args.launch, args.ready, args.output)
    print(json.dumps({k:v for k,v in proof.items() if k!='protected_checkpoint'}))


if __name__ == '__main__':main()
