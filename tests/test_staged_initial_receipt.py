import hashlib
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from verify_staged_initial_receipt import named_initial_state


def fixture(tmp_path, **changes):
    path=tmp_path/'checkpoint_00000000.pt'
    state=dict(actor_updates=0,critic_updates=0,model={'weight':torch.ones(2)})|changes
    torch.save(state,path)
    return dict(initial_checkpoint=str(path),checkpoint_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()),dict(command=['python','runner.py','--checkpoint',str(path)])


def test_named_receipt_does_not_claim_runtime_model_or_modify_input(tmp_path):
    ready,managed=fixture(tmp_path);path=Path(ready['initial_checkpoint']);before=path.stat()
    result,state=named_initial_state(ready,managed)
    assert result==path and state['actor_updates']==0
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)


@pytest.mark.parametrize('failure',['hash','different_path','duplicate_input','learned_model','nonfinite'])
def test_changed_or_ambiguous_initial_provenance_is_rejected(tmp_path,failure):
    changes=({'critic_updates':1} if failure=='learned_model' else
             {'model':{'weight':torch.tensor(float('nan'))}} if failure=='nonfinite' else {})
    ready,managed=fixture(tmp_path,**changes)
    if failure=='hash':ready['checkpoint_SHA256']='0'*64
    elif failure=='different_path':managed['command'][-1]=str(tmp_path/'other'/'checkpoint_00000000.pt')
    elif failure=='duplicate_input':managed['command']+=['--checkpoint',ready['initial_checkpoint']]
    with pytest.raises(ValueError):named_initial_state(ready,managed)
