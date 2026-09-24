"""Filler audio while the LLM thinks.

A caller cannot see a spinner. Between the end of their sentence and the first
byte of TTS there is dead air — on a phone line that reads as a dropped call,
and callers start saying "hello? hello?" into it. A soft sound during that gap
tells them the agent is still there.

There are two transports, and they need two different players:

  * **LiveKit room** (`agent.py dev/start`, LiveKit Cloud): use the SDK's own
    `BackgroundAudioPlayer`. It publishes a second audio track to the room and
    mixes ambience + thinking clips into it. It NEEDS a room.

  * **TeleQuick QUIC** (`telequick_worker.py`): there is no room and no
    second track — `ctx.room` is a veneer, and the call has exactly one audio
    lane to the caller. `TransportThinkingAudio` writes the filler straight
    onto that lane, one 20 ms frame per 20 ms tick.

`start_thinking_sound(ctx, session)` picks the right one and returns an object
with `aclose()`. Nothing here is required by the rest of the starter: if the
sound cannot start (no room, no transport), the call runs on in silence and the
reason is logged once.

The default sound is synthesized, so the repo carries no audio asset. Point
`CC_THINKING_WAV` at a mono 16-bit WAV (hold music, keyboard typing, a hum) to
use your own.
"""

from __future__ import annotations

import array
import asyncio
import contextlib
import logging
import math
import os
import wave

logger = logging.getLogger("thinking-audio")

# Wait this long inside "thinking" before playing anything. A fast turn is
# answered well before this, and filler on top of a 300 ms gap sounds nervous,
# not attentive.
DEFAULT_DELAY = 0.4
DEFAULT_VOLUME = 0.25
FRAME_MS = 20


def _synth_loop(sample_rate: int, volume: float) -> array.array:
    """A soft two-tone pulse every 1.2 s: present, but not a ringtone.

    Each pulse is a raised-cosine envelope over a sine — the envelope is what
    keeps it from clicking at the edges, which is the whole difference between
    "the agent is thinking" and "the line is broken".
    """
    period_s = 1.2
    pulses = ((0.00, 420.0, 0.09), (0.22, 320.0, 0.09))  # (start, hz, seconds)
    samples = array.array("h", [0] * int(sample_rate * period_s))
    peak = int(32767 * max(0.0, min(1.0, volume)))
    for start_s, hz, dur_s in pulses:
        first = int(start_s * sample_rate)
        count = int(dur_s * sample_rate)
        for i in range(count):
            envelope = 0.5 * (1 - math.cos(2 * math.pi * i / count))
            value = peak * envelope * math.sin(2 * math.pi * hz * i / sample_rate)
            samples[first + i] = int(value)
    return samples


def _load_wav(path: str, sample_rate: int, volume: float) -> array.array:
    """Read a mono 16-bit WAV and resample it to the wire rate.

    Linear resampling is crude, but this is a looped filler clip on an 8 kHz
    telephony leg — the loop is the audible thing, not the interpolation.
    """
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(f"{path}: need mono 16-bit PCM")
        src_rate = wav.getframerate()
        src = array.array("h")
        src.frombytes(wav.readframes(wav.getnframes()))
    gain = max(0.0, min(1.0, volume))
    if src_rate == sample_rate:
        return array.array("h", [int(s * gain) for s in src])
    ratio = src_rate / sample_rate
    out = array.array("h", [0] * int(len(src) / ratio))
    for i in range(len(out)):
        pos = i * ratio
        low = int(pos)
        high = min(low + 1, len(src) - 1)
        frac = pos - low
        out[i] = int((src[low] * (1 - frac) + src[high] * frac) * gain)
    return out


def load_clip(sample_rate: int, *, volume: float = DEFAULT_VOLUME) -> array.array:
    path = os.environ.get("CC_THINKING_WAV")
    if path:
        try:
            return _load_wav(path, sample_rate, volume)
        except Exception:  # a bad file is not a dead call
            logger.exception(
                "thinking audio: cannot read %s — using the synth clip", path
            )
    return _synth_loop(sample_rate, volume)


class TransportThinkingAudio:
    """Thinking filler on the TeleQuick QUIC lane (no room, no mixer).

    Frames go to `transport.send_frame` directly, NOT to `session.output.audio`:
    the audio output counts segments and answers each flush with exactly one
    `on_playback_finished`, which is what advances the SDK's turn loop. Filler
    pushed through it would add segments the turn loop never asked for.

    Stopping is just "stop pushing". We never call `signal_clear`, because clear
    is barge-in: it would drop the caller's downstream buffer and put a playout
    report in the queue that the next real flush would read as its own. Pacing
    one frame per tick is what makes that safe — when the LLM answers, at most
    one 20 ms frame of filler is still in flight ahead of the reply.
    """

    def __init__(
        self,
        transport,  # telequick MediaTransport
        *,
        sample_rate: int = 8000,
        volume: float = DEFAULT_VOLUME,
        delay: float = DEFAULT_DELAY,
    ) -> None:
        from livekit import rtc  # keep import cost off the hot path

        self._rtc = rtc
        self._transport = transport
        self._sr = sample_rate
        self._delay = delay
        self._spf = int(sample_rate * FRAME_MS / 1000)
        self._clip = load_clip(sample_rate, volume=volume)
        self._task: asyncio.Task | None = None
        self._closed = False

    def attach(self, session) -> None:  # AgentSession
        session.on("agent_state_changed", self._on_state)

    def _on_state(self, ev) -> None:  # AgentStateChangedEvent
        state = getattr(ev, "new_state", None)
        if state == "thinking":
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._closed or (self._task is not None and not self._task.done()):
            return
        self._task = asyncio.ensure_future(self._play())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def aclose(self) -> None:
        self._closed = True
        self.stop()

    async def _play(self) -> None:
        try:
            await asyncio.sleep(self._delay)
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            offset = 0
            frame_bytes = self._spf * 2
            clip = self._clip.tobytes()
            while not self._closed:
                # Sleep to an absolute deadline, not a fixed 20 ms: a fixed
                # sleep drifts late by however long each send took, and the
                # filler would then arrive slower than real time.
                deadline += FRAME_MS / 1000
                chunk = clip[offset : offset + frame_bytes]
                offset += frame_bytes
                if len(chunk) < frame_bytes:  # wrap the loop
                    offset = frame_bytes - len(chunk)
                    chunk += clip[:offset]
                await self._transport.send_frame(
                    self._rtc.AudioFrame(
                        data=chunk,
                        sample_rate=self._sr,
                        num_channels=1,
                        samples_per_channel=self._spf,
                    )
                )
                await asyncio.sleep(max(0.0, deadline - loop.time()))
        except asyncio.CancelledError:
            pass
        except Exception:  # filler must never end a call
            logger.exception("thinking audio: playback stopped")


class _RoomThinkingAudio:
    """`BackgroundAudioPlayer` wrapped so both paths close the same way."""

    def __init__(self, player) -> None:
        self._player = player

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._player.aclose()


def _transport_for(session):
    """The TeleQuick transport this session is bound to, or None."""
    try:
        from livekit.plugins import telequick
    except ImportError:
        return None
    # Published after 0.0.12; older wheels only have the bound audio output,
    # which holds the same transport.
    getter = getattr(telequick, "current_transport", None)
    if getter is not None:
        transport = getter()
        if transport is not None:
            return transport
    out = getattr(getattr(session, "output", None), "audio", None)
    return getattr(out, "_transport", None)


async def start_thinking_sound(ctx, session, *, ambience: bool = False):
    """Play filler while the agent thinks. Returns a handle with `aclose()`.

    Picks the room player on LiveKit and the transport player over TeleQuick.
    Returns None when neither is available (e.g. `console` mode), after logging
    why — the call itself is never blocked on this.
    """
    room = getattr(ctx, "room", None)
    if getattr(room, "local_participant", None) is not None:
        from livekit.agents import (
            AudioConfig,
            BackgroundAudioPlayer,
            BuiltinAudioClip,
        )

        try:
            player = BackgroundAudioPlayer(
                ambient_sound=(
                    AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.5)
                    if ambience
                    else None
                ),
                # Two clips so a long think does not loop one sample audibly.
                thinking_sound=[
                    AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
                    AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
                ],
            )
            await player.start(room=room, agent_session=session)
        except Exception:
            logger.exception("thinking audio: room player did not start")
            return None
        logger.info("thinking audio: room player started")
        return _RoomThinkingAudio(player)

    transport = _transport_for(session)
    if transport is None:
        logger.info("thinking audio: no room and no transport — staying silent")
        return None
    player = TransportThinkingAudio(
        transport, sample_rate=getattr(transport, "sample_rate", 8000)
    )
    player.attach(session)
    logger.info("thinking audio: transport player attached")
    return player
