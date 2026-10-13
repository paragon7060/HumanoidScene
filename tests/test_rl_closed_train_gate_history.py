"""TRAIN evidence survives subsequent TRAIN/DEV report replacement."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts' / 'rl'))
from closed_train_gate_history import archive_train_gate_summary


def fixture(wave=1):
    summary=dict(name='existing_perceived_contact_gates_current_wave_v1',
        counts=dict(selected_rows=12, both_projected_closed_rows=5),
        by_environment=[dict(environment=5, counts=dict(selected_rows=12,
                            both_projected_closed_rows=5))],
        overlapping_counts_NOT_exclusive_failure_causes=True,
        closure_or_command_NOT_physical_pinch=True)
    outcomes=[dict(wave=wave, split='train', environment=i, complete=True)
              for i in range(8)]
    return summary, outcomes


def test_two_train_waves_remain_in_final_metrics_with_null_dev_learner_report():
    history=[]
    first, rows=fixture()
    archive_train_gate_summary(history,1,'train',first,rows)
    second, rows2=fixture(2)
    second['counts']['selected_rows']=second['by_environment'][0]['counts']['selected_rows']=17
    archive_train_gate_summary(history,2,'train',second,rows2)
    archive_train_gate_summary(history,3,'validation',None,[])
    final=json.loads(json.dumps(dict(learner=dict(perceived_contact_gate_summary_current_wave=None),
                                    TRAIN_contact_gate_history=history)))
    assert [r['wave'] for r in final['TRAIN_contact_gate_history']]==[1,2]
    assert [r['summary']['counts']['selected_rows'] for r in history]==[12,17]
    first['by_environment'].clear()
    assert history[0]['summary']['by_environment'][0]['environment']==5


def test_no_train_report_and_dev_final_are_not_fabricated_as_zero_train_evidence():
    history=[]; summary, rows=fixture()
    for split in ['validation','final']:
        archive_train_gate_summary(history,1,split,summary,rows)
    archive_train_gate_summary(history,1,'train',None,rows)
    assert history==[]


def test_interrupted_wave_is_not_recorded_as_complete_or_physical_success():
    history=[]; summary, rows=fixture(); rows[7]['complete']=False
    original=deepcopy((summary,rows))
    archive_train_gate_summary(history,1,'train',summary,rows)
    assert not history[0]['complete']
    assert history[0]['counts_NOT_exclusive_failure_causes_or_physical_pinch']
    assert (summary,rows)==original
    archive_train_gate_summary(history,1,'train',summary,rows)
    assert len(history)==1
    rows[7]['complete']=True
    with pytest.raises(ValueError,match='immutable'):
        archive_train_gate_summary(history,1,'train',summary,rows)


@pytest.mark.parametrize('damage',['aggregate','duplicate_id','unknown_id'])
def test_inconsistent_scalar_evidence_is_rejected_without_partial_archive(damage):
    history=[];summary,rows=fixture()
    if damage=='aggregate':summary['counts']['selected_rows']+=1
    elif damage=='duplicate_id':summary['by_environment'].append(deepcopy(summary['by_environment'][0]))
    else:summary['by_environment'][0]['environment']=99
    with pytest.raises(ValueError):archive_train_gate_summary(history,1,'train',summary,rows)
    assert history==[]


def test_actual_contact_gate_report_archive_does_not_change_control_or_rng():
    import torch
    from test_rl_perceived_precise_feedback import setup,step
    a,b=setup(),setup();a[1][:,46:48]=b[1][:,46:48]=.98
    history=[]; before=torch.get_rng_state().clone()
    for clock in range(30):
        xa,xb=step(a,clock),step(b,clock)
        assert torch.equal(xa[0],xb[0]) and torch.equal(xa[1][2],xb[1][2])
        if clock==14:
            rows=[dict(wave=1,split='train',environment=i,complete=True)
                  for i in range(len(a[-1].phase))]
            archive_train_gate_summary(history,1,'train',a[-1].gate_report(),rows)
    assert a[-1].statistics==b[-1].statistics
    assert torch.equal(a[-1].gate_counts,b[-1].gate_counts)
    assert torch.equal(a[-1].phase,b[-1].phase)
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.cuda.is_initialized()
