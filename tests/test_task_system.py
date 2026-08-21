from kuavo_isaaclab_scene.task_system import (
    ConveyorSlotManager,
    PlacementMode,
    RackConveyorTask,
    TaskPhase,
)


SLOTS = tuple((0.65 + 0.26 * index, -0.58, 0.78) for index in range(4))


def test_direct_placement_reserves_empty_infeed():
    manager = ConveyorSlotManager(SLOTS)
    plan = manager.reserve("kuavo", "rack_a_0", {})
    assert plan.mode is PlacementMode.PLACE
    assert plan.drop_slot == 0
    assert plan.target == SLOTS[0]
    manager.commit(plan)


def test_occupied_infeed_pushes_contiguous_queue_to_gap():
    manager = ConveyorSlotManager(SLOTS)
    positions = {"other_0": SLOTS[0], "other_1": SLOTS[1]}
    plan = manager.reserve("kuavo", "rack_a_0", positions)
    assert plan.mode is PlacementMode.PUSH_THEN_PLACE
    assert plan.empty_slot == 2
    assert plan.push_box_ids == ("other_1", "other_0")
    assert abs(plan.push_distance - 0.26) < 1.0e-6


def test_full_conveyor_blocks_insertion():
    manager = ConveyorSlotManager(SLOTS)
    positions = {f"other_{index}": slot for index, slot in enumerate(SLOTS)}
    plan = manager.reserve("kuavo", "rack_a_0", positions)
    assert plan.mode is PlacementMode.BLOCKED


def test_second_worker_waits_for_active_infeed_reservation():
    manager = ConveyorSlotManager(SLOTS)
    first = manager.reserve("worker_a", "box_a", {})
    second = manager.reserve("worker_b", "box_b", {})
    assert first.mode is PlacementMode.PLACE
    assert second.mode is PlacementMode.WAIT


def test_button_is_gated_by_all_rack_boxes():
    task = RackConveyorTask(("a0", "a1", "b0"))
    task.update_transferred(("a0", "b0", "foreign"))
    assert task.phase is TaskPhase.TRANSFERRING
    assert not task.press_button()
    assert not task.conveyor_enabled

    task.update_transferred(("a0", "a1", "b0", "foreign"))
    assert task.phase is TaskPhase.WAITING_FOR_BUTTON
    assert task.press_button()
    assert task.phase is TaskPhase.COMPLETE
    assert task.conveyor_enabled
    task.update_transferred(())
    assert task.transferred_box_ids == {"a0", "a1", "b0"}


def test_complete_six_tote_task_with_two_foreign_boxes():
    rack_boxes = tuple(f"rack_{index}" for index in range(6))
    task = RackConveyorTask(rack_boxes)
    manager = ConveyorSlotManager(
        tuple((0.65 + 0.26 * index, -0.58, 0.78) for index in range(9))
    )
    positions = {"foreign_0": manager.slot_centers[0], "foreign_1": manager.slot_centers[1]}

    for rack_box in rack_boxes:
        plan = manager.reserve("kuavo", rack_box, positions)
        assert plan.mode is PlacementMode.PUSH_THEN_PLACE
        occupied = manager.occupied_slots(positions)
        box_to_slot = {box_id: slot_id for slot_id, box_id in occupied.items()}
        for pushed_box in plan.push_box_ids:
            positions[pushed_box] = manager.slot_centers[box_to_slot[pushed_box] + 1]
        positions[rack_box] = plan.target
        manager.commit(plan)
        task.update_transferred(tuple(manager.occupied_slots(positions).values()))

    assert task.phase is TaskPhase.WAITING_FOR_BUTTON
    assert len(manager.occupied_slots(positions)) == 8
    assert task.press_button()
    assert task.conveyor_enabled
