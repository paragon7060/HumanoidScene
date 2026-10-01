"""Explicit selected target for grasp scenes that also contain distractors."""
import torch


def select_grasp_target(active, override=None):
    if active.ndim!=2 or not bool(active.any(-1).all()):
        raise ValueError('Every grasp scene needs at least one active box')
    first=active.to(torch.long).argmax(-1)
    if override is None:
        if not bool((active.sum(-1)==1).all()):
            raise ValueError('Multiple active grasp boxes require an explicit target')
        return first
    if override.shape!=first.shape or override.dtype!=torch.long:
        raise ValueError('Explicit grasp targets require one integer index per environment')
    explicit=override>=0
    if bool((override>=active.shape[1]).any()) or bool(((active.sum(-1)!=1)&~explicit).any()):
        raise ValueError('Multiple active grasp boxes require a valid explicit target')
    target=torch.where(explicit,override,first)
    rows=torch.arange(len(active),device=active.device)
    if not bool(active[rows,target].all()):
        raise ValueError('The explicit grasp target must be active')
    return target


class ExplicitGraspTargetSelector:
    """Use the same task command in deployable perception and privileged truth."""
    def __init__(self, source, fallback):
        self.source,self.fallback=source,fallback

    def select(self, observation, selectable_boxes, waiting_env_ids):
        from ..hierarchy.types import HighLevelSelection
        selected=self.fallback.select(observation,selectable_boxes,waiting_env_ids)
        override=self.source()
        if override is None:
            return selected
        values=override[waiting_env_ids];explicit=values>=0
        if bool((values>=selectable_boxes.shape[1]).any()):
            raise ValueError('Explicit task target exceeds the perceived box set')
        rows=torch.arange(len(values),device=values.device)
        if not bool((selectable_boxes[rows,values.clamp_min(0)]|~explicit).all()):
            raise ValueError('Explicit task target is not selectable in current perception')
        return HighLevelSelection(torch.where(explicit,values,selected.box_index))
