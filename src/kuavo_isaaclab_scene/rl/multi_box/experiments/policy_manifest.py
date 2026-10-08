"""Record the loaded policy, independently of historical initialization metadata."""
from copy import deepcopy
import json


def checkpoint_manifest_fields(training_contract, checkpoint, *, artifact_type):
    """Publish the checkpoint contract before the immutable manifest is archived.

    Simulator setup uses the physical fields of ``training_contract``. Its
    nested goal can still describe a much older nominal initializer. Keep that
    provenance separately and record the policy actually selected by dispatch.
    Runtime restore checks and ``agent.yaml`` remain authoritative; this helper
    changes no model, optimizer, observations, physical settings or replay.
    """
    policies = [(key, checkpoint.get(key)) for key in
        ('goal_contract', 'physical_body_contract') if
        isinstance(checkpoint.get(key), dict) and
        checkpoint[key].get('name') == artifact_type]
    if checkpoint.get('artifact_type') != artifact_type or len(policies) != 1:
        raise ValueError('Manifest policy must match the loaded checkpoint and selected runner')
    contract_key, goal = policies[0]
    counts = {key: checkpoint.get(key) for key in ('actor_updates', 'critic_updates')}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('Checkpoint learning counters are missing or malformed')
    # Avoid publishing a non-finite or non-serializable policy description.
    json.dumps(goal, allow_nan=False)
    return {contract_key: deepcopy(goal)} | dict(
        initialization_source_goal_contract=deepcopy(training_contract.get('goal_contract')),
        policy_contract_source='loaded_policy_checkpoint',
        policy_contract_key=contract_key,
        policy_checkpoint_state=dict(artifact_type=artifact_type, **counts),
        authoritative_runtime_policy_contract_file='agent.yaml',
        initialized_not_trained=all(value == 0 for value in counts.values()))
