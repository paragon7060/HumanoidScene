import json
import asyncio
import socket

import numpy as np
import pytest
import websockets

from kuavo_isaaclab_scene.browser_teleop_bridge import (
    BrowserTeleopBridge,
    parse_tracking_message,
    webxr_pose_to_kuavo,
)


def test_webxr_position_axes_convert_to_kuavo_base():
    pose = webxr_pose_to_kuavo([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(pose[:3], [-3.0, -1.0, 2.0])
    np.testing.assert_allclose(pose[3:], [1.0, 0.0, 0.0, 0.0], atol=1.0e-6)


def test_tracking_packet_accepts_controller_shaped_hands():
    pose = [0.2, 1.2, -0.5, 1.0, 0.0, 0.0, 0.0]
    packet = {
        "type": "tracking",
        "head": [0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0],
        "left_hand": {"wrist": pose, "thumb_tip": pose, "index_tip": pose},
        "right_hand": {"wrist": pose, "thumb_tip": pose, "index_tip": pose},
    }
    sample = parse_tracking_message(json.dumps(packet), received_at=12.0)
    assert sample is not None
    assert sample.received_at == 12.0
    assert sample.head is not None
    assert sample.left_hand is not None and set(sample.left_hand) == {"wrist", "thumb_tip", "index_tip"}
    assert sample.right_hand is not None and "wrist" in sample.right_hand


def test_invalid_tracking_packet_is_ignored():
    assert parse_tracking_message("not-json") is None
    assert parse_tracking_message('{"type":"camera"}') is None


def test_websocket_bridge_exchanges_tracking_and_camera_bytes():
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except PermissionError:
        pytest.skip("local sockets are disabled by this execution sandbox")
    bridge = BrowserTeleopBridge(port=port)
    bridge.start()

    async def exchange():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            pose = [0.2, 1.2, -0.5, 1.0, 0.0, 0.0, 0.0]
            await client.send(json.dumps({"type": "tracking", "head": pose, "left_hand": {"wrist": pose}}))
            for _ in range(20):
                if bridge.latest().head is not None:
                    break
                await asyncio.sleep(0.01)
            assert bridge.latest().head is not None
            bridge.publish_jpeg(b"jpeg-test")
            assert await asyncio.wait_for(client.recv(), timeout=1.0) == b"jpeg-test"

    try:
        asyncio.run(exchange())
    finally:
        bridge.close()
