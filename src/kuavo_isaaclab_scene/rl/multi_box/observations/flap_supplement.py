"""Optional perceived panel-midpoint relations alongside a nominal policy view."""
import torch

from ..geometry import pose_to_position_rotation_6d, relative_pose
from ..geometry.grasp import (
    GRASP_ASSIGNMENT_SCALE_M, opposing_flap_reach_assignment, perceived_flap_center_poses,
)
from ..geometry.pose import replace_invalid_poses


SUPPLEMENTAL_GROUP = 'actual_flap_relations'
SUPPLEMENTAL_DIM = 38


def supplemental_perception_contract():
    return dict(name='perceived_articulated_flap_midpoint_relations_v1',
        observation_group=SUPPLEMENTAL_GROUP,dimension=SUPPLEMENTAL_DIM,
        order=['left_hand_right_flap_pose9','left_hand_left_flap_pose9',
               'right_hand_right_flap_pose9','right_hand_left_flap_pose9',
               'left_hand_flap_assignment_one_hot2'],
        pose_representation='hand_relative_midpoint_position_m_rotation6d',
        source='replaceable_perceived_panel_midpoints_and_orientations',
        current_simulator_backend='articulated_link_pose_plus_known_local_midpoint',
        nominal_policy_group_preserved=True,contact_forces_or_success_labels_in_actor=False,
        terminal_capture='observation_manager_before_autoreset',new_sensors_or_physics_steps=False)


def actual_flap_relations(centers, sizes, types, tcp, valid):
    """Finite38D perceived features; invalid poses never enter quaternion math."""
    n=len(centers)
    if centers.shape!=(n,2,7) or tcp.shape!=(n,2,7) or valid.shape!=(n,) \
            or valid.dtype!=torch.bool:
        raise ValueError('Actual flap relations need two perceived panels/hands and a valid mask')
    centers,invalid_panels=replace_invalid_poses(centers)
    tcp,invalid_hands=replace_invalid_poses(tcp)
    valid=valid&~invalid_panels.any(-1)&~invalid_hands.any(-1)
    poses,distances=perceived_flap_center_poses(centers,sizes,types,tcp)
    relations=pose_to_position_rotation_6d(relative_pose(tcp[:,:,None],poses))
    _,assignment=opposing_flap_reach_assignment(distances,GRASP_ASSIGNMENT_SCALE_M)
    one_hot=torch.nn.functional.one_hot(assignment[:,0],2).to(relations)
    return torch.cat((relations.flatten(1),one_hot),-1)*valid[:,None]
