"""Keep actually successful full paths visible in the actor imitation batch."""

import torch


def sample_teacher_with_success(teacher, success, count, device, fraction=0.0):
    """Mix executed-success labels with correction labels, without Q metadata.

    ``fraction`` is inside the existing teacher batch, not an additional BC
    weight or the offline VR-demo fraction. A zero default preserves previous
    callers. Sampling only actor/action avoids copying unused critic tensors.
    """
    if count < 1 or not 0 <= fraction <= 1:
        raise ValueError("Success imitation needs a positive count and fraction in [0,1]")
    actual = round(count * fraction) if success.size else 0
    if not teacher.size:
        if not success.size:
            raise ValueError("No teacher or executed-success labels are available")
        actual = count
    result = teacher.sample(count - actual, device) if count > actual else None
    if actual:
        ids = torch.randint(success.size, (actual,), device=success.data["actor_obs"].device)
        labels = {key: success.data[key][ids].to(device) for key in ("actor_obs", "action")}
        result = labels if result is None else {
            key: torch.cat((result[key], labels[key])) for key in labels}
    return result, actual
