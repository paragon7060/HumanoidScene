"""Frozen TRAIN diagnosis of jaw choice; never a learned policy or Q seed."""
from collections import Counter
from copy import deepcopy

import torch


def bilateral_close_probe_contract():
    return dict(name='frozen_bilateral_nominal_near_close_diagnostic_v1',
        scope='original_full_TRAIN128_workplace_candidates8',
        action_change='request_both_jaws_closed_only_when_both_production_gates_allow',
        body_neural_goals_and_existing_12cm_gate_unchanged=True,
        privileged_contacts_used_for_action=False,
        standalone_SAC=False, Q_import_eligible=False,
        original_randomization_success_safety_and_controller_preserved=True)


def validate_bilateral_close_probe(waves, *, training, steps, workplace):
    expected=Counter({('shelf_2_left','small'):16,('shelf_2_left','medium'):16,
        ('shelf_2_right','small'):16,('shelf_2_right','medium'):16,
        ('shelf_3_left','small'):32,('shelf_3_right','small'):32})
    if (training or steps!=900 or not workplace or len(waves)!=1
            or waves[0].get('split')!='train' or
            Counter((r['layout'].get('target_region'),r['layout'].get('target_box_type'))
                for r in waves[0]['layouts'])!=expected):
        raise ValueError('Jaw diagnosis requires the whole frozen supported-size TRAIN128 workplace')
    return bilateral_close_probe_contract()


@torch.no_grad()
def close_bilateral_when_allowed(result, near, statistics):
    """Keep the original executed body; align the stored jaw goal with execution."""
    physical, previous=result
    actor, critic, goals=previous
    n=len(physical)
    if (physical.shape!=(n,24) or goals.shape!=(n,21) or near.shape!=(n,2)
            or near.dtype!=torch.bool or len(actor)!=n or len(critic)!=n):
        raise ValueError('Require aligned physical24, executed-goal21 and two production masks')
    if not all(torch.isfinite(x).all() for x in (physical,goals,actor,critic)):
        raise ValueError('Nonfinite physical records cannot establish a jaw comparison')
    eligible=near.all(-1)
    command=physical.clone();executed=goals.clone()
    changed=eligible & (physical[:,20:22]<=0).any(-1)
    command[eligible,20:22]=1.
    executed[eligible,19:21]=1.
    statistics['held_rows_checked']+=n
    statistics['both_near_rows']+=int(eligible.sum())
    statistics['jaw_overridden_rows']+=int(changed.sum())
    return command,(actor,critic,executed)


def install_frozen_bilateral_close_probe(pilot_class, manifest_module):
    """Install a process-local, reversible hook in this dedicated diagnostic."""
    original_act=pilot_class.act;original_report=pilot_class.report
    original_manifest=manifest_module.checkpoint_manifest_fields
    stats=dict(held_rows_checked=0,both_near_rows=0,jaw_overridden_rows=0)
    contract=bilateral_close_probe_contract()

    def act(self,*args,**kwargs):
        if self.training or self.actor_updates or self.critic_updates or self.replay.size:
            raise ValueError('Bilateral close diagnostic cannot optimize or use learned Q/replay')
        result=original_act(self,*args,**kwargs)
        near=self.agent.action_projector.entropy_mask(result[1][0])[:,19:21].bool()
        return close_bilateral_when_allowed(result,near,stats)

    def report(self):
        return original_report(self)|dict(frozen_bilateral_close_probe=deepcopy(contract)|
            dict(actual_statistics=deepcopy(stats)))

    def manifest(*args,**kwargs):
        return original_manifest(*args,**kwargs)|dict(frozen_bilateral_close_probe=deepcopy(contract))

    pilot_class.act=act;pilot_class.report=report
    manifest_module.checkpoint_manifest_fields=manifest

    def restore():
        pilot_class.act=original_act;pilot_class.report=original_report
        manifest_module.checkpoint_manifest_fields=original_manifest
    return restore
