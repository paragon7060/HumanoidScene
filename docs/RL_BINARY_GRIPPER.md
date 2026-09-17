# RL binary gripper control

The shared RL action configuration executes one gripper scalar per active
hand as a binary command:

- `0`: open
- `1`: close

The policy still emits a scalar through the continuous action vector used by
RSL-RL, SAC and diffusion. Values at or below zero execute `0`; positive
values execute `1`. This keeps exploration balanced around a new policy's
zero mean while exposing only `0` or `1` in the gripper action term.

Both endpoints pass through the measured directional position mapping and
the existing target filter. A close command therefore preserves the calibrated
real-gripper motion instead of jumping directly to a raw joint endpoint.

During close, the position PD remains active and a sensor-free geometric
feedforward torque is added. The configured total is 50 N, or 25 N equivalent
force at each jaw. The torque follows the four-bar linkage leverage at the
current joint angle. It is removed on open and at the empty mechanical closed
stop.

This is a commanded force equivalent, not measured force regulation. Actual
contact force can differ with object geometry, friction and contact solver
compliance. The RL path adds no contact sensors and performs only a two-jaw
table interpolation per physics step. Quest force diagnostics retain their
separate contact-feedback controller.

Change `force_control.close_force_n` in
`src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json` to tune the
nominal total squeeze. The runtime divides that value equally between both
jaws, so the default 50 N produces 25 N per jaw.
Changing binary control invalidates older checkpoint action configs and the
alternative-runner dataset encoding is now `manager_mixed_binary_gripper_v2`.
