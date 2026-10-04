"""Measured frozen-policy workplace comparisons; never matching SAC replay."""
import math


def validate_waypoint_probe(waves, *, enabled=False, training=False):
    rows=[row for wave in waves for row in wave['layouts']]
    supplied=any('waypoint_probe' in row for row in rows)
    if training and (enabled or supplied):
        raise ValueError('Waypoint candidates are frozen-only and cannot train Q')
    if supplied and not enabled:
        raise ValueError('Waypoint candidates require the explicit frozen probe flag')
    if not enabled:return
    for row in rows:
        candidate=row.get('waypoint_probe')
        if not isinstance(candidate,dict) or set(candidate)!={'name','offset_xy_yaw'}:
            raise ValueError('Every probe row needs a named offset_xy_yaw candidate')
        name=candidate['name'];offset=candidate['offset_xy_yaw']
        if not isinstance(name,str) or not name or len(name)>80:
            raise ValueError('Waypoint candidate name must be a short nonempty string')
        if not isinstance(offset,list) or len(offset)!=3 or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in offset):
            raise ValueError('Waypoint offset needs three finite XY/yaw numbers')
        if max(abs(offset[0]),abs(offset[1]))>.15 or abs(offset[2])>math.pi/12:
            raise ValueError('Frozen waypoint offsets exceed15cm/15degrees')
    for wave in waves:
        identities=[(row['layout']['seed'],row['waypoint_probe']['name']) for row in wave['layouts']]
        if len(set(identities))!=len(identities):
            raise ValueError('Probe must have distinct case/candidate identities')


def apply_waypoint_probe(stages, layouts):
    """Move approach/held targets only, after strict source-policy restoration.

    Robot/box initial states and source waypoint templates stay unchanged.
    Both tensor caches and per-environment contexts must use the same targets.
    """
    if len(layouts)!=len(stages.stages):raise ValueError('Waypoint row count differs')
    for stage,row in zip(stages.stages,layouts):
        candidate=row['waypoint_probe'];dx,dy,yaw=candidate['offset_xy_yaw']
        stage.target_xy=stage.target_xy+stage.target_xy.new_tensor([dx,dy])
        stage.target_yaw+=yaw
        stage.waypoint_probe=dict(name=candidate['name'],offset_xy_yaw=list(candidate['offset_xy_yaw']))
    stages.target_xy=stages.target_xy.new_tensor([s.target_xy[0].tolist() for s in stages.stages])
    stages.target_yaw=stages.target_yaw.new_tensor([s.target_yaw for s in stages.stages])
