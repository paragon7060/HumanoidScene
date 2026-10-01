"""A zero process exit must not let a failed grasp continue a pilot series."""
import importlib.util
import json
from pathlib import Path
import sys


def load_script():
    scripts=Path(__file__).resolve().parents[1]/'scripts/rl'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('residual_drive_pilot_test',scripts/'reference_residual_with_drive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_frozen_success_gate_rejects_collision_even_with_verified_backup(tmp_path):
    module=load_script();source=tmp_path/'residual_run';source.mkdir()
    (tmp_path/'status.json').write_text(json.dumps({'run_dir':str(source),'training_exit_code':0,'final_upload_verified':True}))
    metric={'completed_attempt':True,'outcomes':{'success':0,'unsafe':1,'invalid_reset':0,'time_out':0}}
    (source/'metrics.json').write_text(json.dumps(metric))
    (source/'checkpoint_00000100.pt').write_bytes(b'checkpoint')
    assert module.verified_success(tmp_path) is None
    metric['outcomes'].update(success=1,unsafe=0)
    (source/'metrics.json').write_text(json.dumps(metric))
    assert module.verified_success(tmp_path)==source/'checkpoint_00000100.pt'
    assert module.verified_success(tmp_path,require_checkpoint=False)==source


def test_series_stops_after_first_failed_frozen_actor(tmp_path,monkeypatch):
    module=load_script();parent=tmp_path/'new-series';calls=[]
    def simulate(command,trial,*args,**kwargs):
        source=trial/'residual_mock';source.mkdir();calls.append(command)
        (trial/'status.json').write_text(json.dumps({'run_dir':str(source),'training_exit_code':0,'final_upload_verified':True}))
        train=len(calls)==1
        (source/'metrics.json').write_text(json.dumps({'completed_attempt':True,
            'outcomes':{'success':int(train),'unsafe':int(not train),'invalid_reset':0,'time_out':0}}))
        if train:(source/'checkpoint_00000100.pt').write_bytes(b'checkpoint')
        return 0
    monkeypatch.setattr(module,'supervise',simulate)
    monkeypatch.setattr(sys,'argv',['pilot','--experiment-dir',str(parent),'--episodes','3',
        '--python',sys.executable,'--remote-root','test-remote:HumanoidScene-RL','--residual-sac'])
    assert module.main()==2
    assert len(calls)==2  # train, frozen eval; no second training episode
    assert '--no-residual-training' in calls[1]
    assert json.loads((parent/'status.json').read_text())['phase']=='performance_gate_failed'
