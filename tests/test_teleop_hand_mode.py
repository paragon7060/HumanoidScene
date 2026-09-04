import numpy as np

from kuavo_isaaclab_scene.teleop.teleop_hand_mode import (
    HandModeSwitch, LongPress, HandCommands, HandGripper, HandTrackingGuard, hand_packet,
)


def hand(index_distance=.09, middle_distance=.1):
    wrist = np.array([.3, .2, 1., 1., 0., 0., 0.])
    thumb = wrist.copy()
    index, middle = thumb.copy(), thumb.copy()
    index[0] += index_distance
    middle[0] += middle_distance
    return {"wrist": wrist, "thumb_tip": thumb, "index_tip": index, "middle_tip": middle}


def pump(switch, start, stop, squeeze, hands=False, controllers=True, head=True):
    events = []
    for now in np.arange(start, stop, .05):
        event = switch.update(float(now), squeeze, hands_ready=hands, controllers_ready=controllers, head_ready=head)
        if event:
            events.append(event)
    return events


def test_switch_requires_long_press_countdown_and_continuously_valid_hands():
    switch = HandModeSwitch()
    assert pump(switch, 0, .9, 1.) == []
    assert pump(switch, .9, 1.5, 0.) == []
    assert pump(switch, 1.5, 3., 1.) == ["begin"]
    assert switch.mode == "controllers" and switch.pending
    assert pump(switch, 3., 6., None) == []  # timeout alone cannot start hands
    assert pump(switch, 6., 6.3, None, hands=True, controllers=False) == []
    assert pump(switch, 6.3, 6.5, None, hands=False, controllers=False) == []
    assert pump(switch, 6.5, 7.2, None, hands=True, controllers=False) == ["ready"]
    assert switch.mode == "hands" and not switch.pending


def test_return_requires_release_and_no_repeat_or_fallback_on_tracking_loss():
    switch = HandModeSwitch("hands")
    assert pump(switch, 0, 1.5, 1.) == ["begin"]
    assert pump(switch, 1.5, 7., 1.) == []
    assert switch.mode == "hands" and switch.pending
    assert pump(switch, 7., 7.8, 0.) == ["ready"]
    assert switch.mode == "controllers"
    assert pump(switch, 7.8, 10., None, hands=True, controllers=False) == []
    assert switch.mode == "controllers"  # no implicit source switch


def test_switch_cancel_and_stall_do_not_auto_start():
    switch = HandModeSwitch()
    pump(switch, 0, 1.5, 1.)
    switch.cancel()
    assert pump(switch, 1.5, 7., None, hands=True) == []
    assert switch.mode == "controllers"
    hold = LongPress()
    assert not hold.update(0., 1.)
    assert not hold.update(3., 1.)  # app stall is not observed sustained input
    assert not hold.update(3.05, None)
    assert not hold.update(3.10, 1.)


def test_hand_source_needs_fresh_wrist_and_fingertips():
    tracked = hand()
    assert hand_packet(tracked).shape == (2, 7)
    for missing in ("wrist", "thumb_tip", "index_tip"):
        assert hand_packet({k: v for k, v in tracked.items() if k != missing}) is None
    tracked["index_tip"][:] = np.nan
    assert hand_packet(tracked) is None


def test_middle_finger_commands_do_not_conflict_with_index_grasp():
    commands = HandCommands()
    for now in np.arange(0., 2., .05):
        assert commands.update(now, {"left": hand(.02, .02), "right": hand(.02, .02)}) == []
    events = []
    for now in np.arange(2., 4., .05):
        events.extend(commands.update(now, {"left": hand(), "right": hand(.09, .01)}))
    assert events == ["toggle"]
    assert commands.active == {"left": False, "right": True}
    # Missing tracking does not release the gesture latch.
    commands.update(4., {"left": None, "right": None})
    for now in np.arange(4.05, 5.5, .05):
        assert commands.update(now, {"left": hand(), "right": hand(.09, .01)}) == []


def test_gripper_hysteresis_loss_command_hold_and_transition_rearm():
    gripper = HandGripper()
    assert gripper.update("left", hand(.02)) == -1
    for sample in (None, {}, hand(.06)):
        assert gripper.update("left", sample) == -1
    assert gripper.update("left", hand(.1), hold=True) == -1
    gripper.sync({"left": -1, "right": 1})
    assert gripper.update("left", hand(.1)) == -1  # switching must not drop held box
    assert gripper.update("left", hand(.02)) == -1  # match current jaw state to arm
    assert gripper.update("left", hand(.1)) == 1
    assert gripper.update("right", hand(.02)) == 1
    assert gripper.update("right", hand(.1)) == 1
    assert gripper.update("right", hand(.02)) == -1


def test_tracking_loss_stops_after_two_seconds_only_while_following():
    guard = HandTrackingGuard()
    assert not guard.update(0., following=True, valid=False)
    assert not guard.update(1.9, following=True, valid=False)
    assert guard.update(2.1, following=True, valid=False)
    assert not guard.update(3., following=False, valid=False)
    assert not guard.update(5., following=True, valid=True)
