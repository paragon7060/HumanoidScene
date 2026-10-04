"""Keep old lift labels out of new Q, while allowing a frozen actor prior."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import grasp_lift_terminal_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
    frozen_prior_lift_contract,require_current_lift_contract)


def test_only_frozen_prior_can_restore_the_old_terminal_contract(monkeypatch):
    monkeypatch.setenv('KUAVO_RACK_ROLLERS','1')
    old=dict(terminal_contract=dict(success='exact_grasp_success',
        safety_thresholds={'rack_contact_force_n':10.},timeouts_bootstrap=True),
        observations={'policy':[464]},self_collision={'enabled':False})
    current=old|dict(terminal_contract=old['terminal_contract']|grasp_lift_terminal_contract())
    before=deepcopy(current)
    require_current_lift_contract(current)
    assert frozen_prior_lift_contract(current)==old and current==before
    with pytest.raises(ValueError,match='old Q/replay'):require_current_lift_contract(old)
    changed=current|dict(terminal_contract=current['terminal_contract']|dict(proof_lift_support_offset_m=0.))
    with pytest.raises(ValueError,match='old Q/replay'):require_current_lift_contract(changed)
    unknown=current|dict(terminal_contract=current['terminal_contract']|dict(proof_lift_reference='unknown'))
    with pytest.raises(ValueError,match='Unknown'):frozen_prior_lift_contract(unknown)
    monkeypatch.setenv('KUAVO_RACK_ROLLERS','0')
    with pytest.raises(ValueError,match='old Q/replay'):require_current_lift_contract(current)
    require_current_lift_contract(current|dict(terminal_contract=current['terminal_contract']|grasp_lift_terminal_contract()))


def test_prepared_travel_manifest_reapplies_exact_height_without_double_extension():
    from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import configure_upright_travel_profile
    def cfg():return SimpleNamespace(actions=SimpleNamespace(height=SimpleNamespace(height_range_m=(0.,.4))))
    original={'action_contract':'s63_upright_torso_xz_fixed_pitch_v1'}
    manifest=configure_upright_travel_profile(cfg(),original,.06)
    fresh=cfg()
    assert configure_upright_travel_profile(fresh,manifest,.06)==manifest
    assert fresh.actions.height.height_range_m==(0.,.46)
    assert configure_upright_travel_profile(fresh,manifest,.06)==manifest
    assert fresh.actions.height.height_range_m==(0.,.46)
    with pytest.raises(ValueError,match='reviewed'):configure_upright_travel_profile(cfg(),manifest,.08)


def test_standard_ppo_and_sac_resume_reject_old_lift_labels(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2 import _compatible_checkpoint as ppo
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint as sac
    checkpoint=tmp_path/'checkpoint.pt'
    (tmp_path/'manifest.json').write_text(json.dumps({'terminal_contract':{'success':'exact_grasp_success'}}))
    with pytest.raises(ValueError,match='proof_lift_contract'):
        ppo(checkpoint,{'proof_lift_contract':grasp_lift_terminal_contract()})
    requested={'terminal_contract':{'success':'exact_grasp_success'}|grasp_lift_terminal_contract()}
    with pytest.raises(ValueError,match='terminal_contract'):sac(checkpoint,requested)
    with pytest.raises(ValueError,match='terminal_contract'):sac(checkpoint,requested,data_only=True)
