import json
import asyncio
import socket

import numpy as np
import pytest
import websockets

from kuavo_isaaclab_scene.browser_teleop_bridge import (
    BrowserTeleopBridge,
    PROTOCOL_VERSION,
    pack_frame_packet,
    parse_tracking_message,
    unpack_frame_packet,
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
        "protocol_version": PROTOCOL_VERSION,
        "sequence": 7,
        "timestamp_ms": 123.0,
        "head": [0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0],
        "left_hand": {"wrist": pose, "thumb_tip": pose, "index_tip": pose},
        "right_hand": {"wrist": pose, "thumb_tip": pose, "index_tip": pose},
    }
    sample = parse_tracking_message(json.dumps(packet), received_at=12.0)
    assert sample is not None
    assert sample.sequence == 7
    assert sample.client_timestamp_ms == 123.0
    assert sample.received_at == 12.0
    assert sample.head is not None
    assert sample.left_hand is not None and set(sample.left_hand) == {"wrist", "thumb_tip", "index_tip"}
    assert sample.right_hand is not None and "wrist" in sample.right_hand


def test_invalid_tracking_packet_is_ignored():
    assert parse_tracking_message("not-json") is None
    assert parse_tracking_message('{"type":"camera"}') is None
    assert parse_tracking_message('{"type":"tracking","protocol_version":1}') is None


def test_versioned_frame_packet_round_trip():
    packet = pack_frame_packet(
        b"jpeg-test",
        frame_sequence=4,
        tracking_sequence=9,
        client_timestamp_ms=1234.5,
        encode_ms=2.25,
        server_fps=29.5,
    )
    header, jpeg = unpack_frame_packet(packet)
    assert header["version"] == PROTOCOL_VERSION
    assert header["frame_sequence"] == 4
    assert header["tracking_sequence"] == 9
    assert header["client_timestamp_ms"] == 1234.5
    assert jpeg == b"jpeg-test"


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
            await client.send(json.dumps({"type": "client_hello", "protocol_version": PROTOCOL_VERSION}))
            hello = json.loads(await asyncio.wait_for(client.recv(), timeout=1.0))
            assert hello == {"type": "server_hello", "protocol_version": PROTOCOL_VERSION}
            pose = [0.2, 1.2, -0.5, 1.0, 0.0, 0.0, 0.0]
            await client.send(
                json.dumps(
                    {
                        "type": "tracking",
                        "protocol_version": PROTOCOL_VERSION,
                        "sequence": 1,
                        "timestamp_ms": 10.0,
                        "head": pose,
                        "left_hand": {"wrist": pose},
                        "views": [],
                    }
                )
            )
            for _ in range(20):
                if bridge.latest().head is not None:
                    break
                await asyncio.sleep(0.01)
            assert bridge.latest().head is not None
            bridge.publish_frame(
                b"jpeg-test",
                tracking_sequence=1,
                client_timestamp_ms=10.0,
                encode_ms=1.0,
                server_fps=30.0,
            )
            packet = await asyncio.wait_for(client.recv(), timeout=1.0)
            _, jpeg = unpack_frame_packet(packet)
            assert jpeg == b"jpeg-test"

    try:
        asyncio.run(exchange())
    finally:
        bridge.close()
