"""The transport filler player: gated on agent state, paced one frame per tick.

Runs against a fake transport, so there is no QUIC, no engine, and no call.
`livekit.rtc` is only needed for the frame type; when the SDK is not installed
(a bare checkout of this template) a stub stands in for it.
"""

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:  # pragma: no cover - depends on the checkout, not on the code under test
    from livekit import rtc  # noqa: F401
except ImportError:  # pragma: no cover

    @dataclass
    class _AudioFrame:
        data: bytes
        sample_rate: int
        num_channels: int
        samples_per_channel: int

    _rtc = types.ModuleType("livekit.rtc")
    _rtc.AudioFrame = _AudioFrame
    _livekit = types.ModuleType("livekit")
    _livekit.rtc = _rtc
    sys.modules.setdefault("livekit", _livekit)
    sys.modules["livekit.rtc"] = _rtc

from thinking_audio import TransportThinkingAudio


class _FakeTransport:
    sample_rate = 8000

    def __init__(self) -> None:
        self.frames: list = []

    async def send_frame(self, frame) -> None:
        self.frames.append(frame)


class _Session:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event, cb) -> None:
        self.handlers.setdefault(event, []).append(cb)

    def emit(self, event, ev) -> None:
        for cb in self.handlers.get(event, []):
            cb(ev)


@dataclass
class _StateEvent:
    new_state: str


def _player(transport, **kw) -> TransportThinkingAudio:
    return TransportThinkingAudio(transport, sample_rate=8000, delay=0.0, **kw)


@pytest.mark.asyncio
async def test_plays_only_while_thinking() -> None:
    transport = _FakeTransport()
    player = _player(transport)
    session = _Session()
    player.attach(session)

    session.emit("agent_state_changed", _StateEvent("listening"))
    await asyncio.sleep(0.08)
    assert transport.frames == []  # listening is silent

    session.emit("agent_state_changed", _StateEvent("thinking"))
    await asyncio.sleep(0.12)
    assert transport.frames, "no filler audio while the LLM was thinking"

    session.emit("agent_state_changed", _StateEvent("speaking"))
    sent = len(transport.frames)
    await asyncio.sleep(0.12)
    # Nothing more once the reply starts — the filler must not play under TTS.
    assert len(transport.frames) == sent

    await player.aclose()


@pytest.mark.asyncio
async def test_frames_are_20ms_and_paced_in_real_time() -> None:
    transport = _FakeTransport()
    player = _player(transport)
    player.start()
    await asyncio.sleep(0.22)
    await player.aclose()

    assert transport.frames
    for frame in transport.frames:
        assert frame.samples_per_channel == 160  # 20 ms at 8 kHz
        assert len(frame.data) == 320  # pcm16
    # Paced, not dumped: ~11 frames in 220 ms, never the whole 1.2 s loop.
    assert len(transport.frames) <= 14


@pytest.mark.asyncio
async def test_a_failing_transport_does_not_raise_into_the_call() -> None:
    class _Broken(_FakeTransport):
        async def send_frame(self, frame) -> None:
            raise RuntimeError("connection lost")

    player = _player(_Broken())
    player.start()
    await asyncio.sleep(0.06)
    await player.aclose()  # the task swallowed it; closing is still clean
