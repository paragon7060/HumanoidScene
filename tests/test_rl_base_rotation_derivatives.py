"""Quaternion pose derivatives must use the world frame and preserve sign symmetry."""
import numpy as np
import pytest

from summarize_closed_base_substeps import rotation_velocity_world


def test_constant_world_pitch_and_quaternion_sign_changes():
    dt=1/120;t=np.arange(200)*dt;angle=.8*t
    q=np.column_stack((np.cos(angle/2),np.zeros_like(t),np.sin(angle/2),np.zeros_like(t)))
    q[::3]*=-1
    actual=rotation_velocity_world(q,dt)
    assert np.allclose(actual,np.tile([0.,.8,0.],(199,1)),atol=1e-10)


def test_velocity_is_world_frame_at_quarter_turn_spawn_heading():
    dt=1/120;t=np.arange(200)*dt;angle=.5*t
    c=np.cos(angle/2);s=np.sin(angle/2);h=2**-.5
    # Rotation about world Y, followed by the original root's90degree yaw.
    q=np.column_stack((h*c,h*s,h*s,h*c))
    assert np.allclose(rotation_velocity_world(q,dt),np.tile([0.,.5,0.],(199,1)),atol=1e-10)


def test_substep_motion_can_resolve_a_control_rate_alias():
    dt=1/120;t=np.arange(401)*dt
    # A30Hz bounded pitch oscillation appears stationary every fourth sample.
    angle=.01*np.sin(2*np.pi*30*t)
    q=np.column_stack((np.cos(angle/2),np.zeros_like(t),np.sin(angle/2),np.zeros_like(t)))
    substeps=rotation_velocity_world(q,dt)
    control=rotation_velocity_world(q[::4],4*dt)
    assert np.max(np.abs(control))<1e-10
    assert np.max(np.abs(substeps[:,1]))==pytest.approx(1.2)


@pytest.mark.parametrize('q,dt',[(np.zeros((2,4)),.01),(np.ones((1,4)),.01),
    (np.full((2,4),np.nan),.01),(np.ones((2,4)),0.)])
def test_missing_or_invalid_motion_cannot_be_reported_as_valid(q,dt):
    with pytest.raises(ValueError):rotation_velocity_world(q,dt)
