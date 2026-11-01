"""Stats and filler-audio tests that need no models, no network, and no room.

The metric objects are stand-ins with the same attribute names livekit-agents
uses, so these run on a bare checkout.
"""

import array
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from call_stats import CallStats, _pct
from thinking_audio import _synth_loop, load_clip


@dataclass
class EOUMetrics:
    speech_id: str
    end_of_utterance_delay: float


@dataclass
class LLMMetrics:
    speech_id: str
    ttft: float
    prompt_tokens: int
    completion_tokens: int
    prompt_cached_tokens: int = 0


@dataclass
class TTSMetrics:
    speech_id: str
    ttfb: float
    characters_count: int
    audio_duration: float


@dataclass
class STTMetrics:
    audio_duration: float


class _Session:
    """Only what CallStats.attach uses."""

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event, cb) -> None:
        self.handlers.setdefault(event, []).append(cb)

    def emit(self, event, ev) -> None:
        for cb in self.handlers.get(event, []):
            cb(ev)


def _stats() -> tuple[CallStats, _Session]:
    stats = CallStats(
        models={"llm": "gpt-4.1-mini", "stt": "nova-3", "tts": "gpt-4o-mini-tts"},
        forward_to_clutchcall=False,
    )
    session = _Session()
    stats.attach(session)
    return stats, session


def _turn(session, speech_id: str, *, prompt: int, completion: int, chars: int) -> None:
    session.emit("metrics_collected", EOUMetrics(speech_id, 0.2))
    session.emit("metrics_collected", LLMMetrics(speech_id, 0.4, prompt, completion))
    session.emit("metrics_collected", TTSMetrics(speech_id, 0.9, chars, 3.0))


def test_tokens_accumulate_across_turns() -> None:
    stats, session = _stats()
    _turn(session, "s1", prompt=100, completion=20, chars=80)
    _turn(session, "s2", prompt=140, completion=30, chars=95)

    t = stats.totals
    assert t.turns == 2
    assert t.prompt_tokens == 240
    assert t.completion_tokens == 50
    assert t.total_tokens == 290
    assert t.tts_characters == 175
    assert t.llm_requests == 2


def test_stt_audio_and_summary_render() -> None:
    stats, session = _stats()
    _turn(session, "s1", prompt=10, completion=2, chars=8)
    session.emit("metrics_collected", STTMetrics(4.5))

    assert stats.totals.stt_audio_s == 4.5
    text = "\n".join(stats.summary_lines())
    assert "gpt-4.1-mini" in text
    assert "12 tokens" in text  # 10 prompt + 2 completion
    assert "4.5 s audio" in text


def test_a_metric_without_a_field_does_not_raise() -> None:
    """Field names move between SDK versions; a missing one reads as zero."""
    stats, session = _stats()

    @dataclass
    class ThinLLMMetrics:
        speech_id: str
        ttft: float

    session.emit("metrics_collected", ThinLLMMetrics("s1", 0.3))
    assert stats.totals.llm_requests == 1
    assert stats.totals.prompt_tokens == 0


def test_percentiles() -> None:
    assert _pct([], 0.5) == 0.0
    assert _pct([0.1, 0.2, 0.3], 0.5) == 0.2
    assert _pct([0.1, 0.2, 0.3], 0.95) == 0.3


def test_synth_clip_is_one_loop_of_quiet_pcm16() -> None:
    clip = _synth_loop(8000, volume=0.25)
    assert isinstance(clip, array.array) and clip.typecode == "h"
    assert len(clip) == int(8000 * 1.2)  # one 1.2 s loop
    peak = max(abs(s) for s in clip)
    assert 0 < peak <= int(32767 * 0.25) + 1  # audible, but well under full scale
    # Pulses fade in and out, so the loop starts and ends at silence: looping
    # it must not click.
    assert clip[0] == 0 and clip[-1] == 0


def test_bad_wav_path_falls_back_to_the_synth_clip(monkeypatch) -> None:
    monkeypatch.setenv("CC_THINKING_WAV", "/nonexistent/thinking.wav")
    assert len(load_clip(8000)) == int(8000 * 1.2)


# -- engine marks: reply_start and turn -------------------------------------


@dataclass
class _Meta:
    model_name: str
    model_provider: str


@dataclass
class _StateEvent:
    new_state: str


@dataclass
class _Speech:
    id: str


def _fwd_stats() -> tuple[CallStats, _Session]:
    stats = CallStats(
        models={"llm": "gpt-4.1-mini", "stt": "nova-3", "tts": "gpt-4o-mini-tts"},
        providers={"llm": "openai", "stt": "deepgram", "tts": "openai"},
    )
    session = _Session()
    session.current_speech = None
    stats.attach(session)
    return stats, session


def _marks(conn, op):
    return [m for m in conn.sent if m[0] == op]


def test_reply_start_is_sent_once_per_speech(cc_conn) -> None:
    _, session = _fwd_stats()
    session.current_speech = _Speech("sp_1")
    session.emit("agent_state_changed", _StateEvent("speaking"))
    session.emit("agent_state_changed", _StateEvent("listening"))
    session.emit("agent_state_changed", _StateEvent("speaking"))  # same speech
    session.current_speech = _Speech("sp_2")
    session.emit("agent_state_changed", _StateEvent("speaking"))
    session.emit("agent_state_changed", _StateEvent("thinking"))  # not speaking

    assert _marks(cc_conn, "reply_start") == [
        ("reply_start", 42, "sp_1"),
        ("reply_start", 42, "sp_2"),
    ]


def test_reply_start_without_a_speech_omits_the_id(cc_conn) -> None:
    _, session = _fwd_stats()
    session.emit("agent_state_changed", _StateEvent("speaking"))
    assert _marks(cc_conn, "reply_start") == [("reply_start", 42, None)]


def test_one_turn_per_user_turn_and_none_for_the_greeting(cc_conn) -> None:
    _, session = _fwd_stats()
    # Greeting: LLM + TTS, no EOU metric.
    session.emit("metrics_collected", LLMMetrics("greet", 0.3, 10, 5))
    session.emit("metrics_collected", TTSMetrics("greet", 0.5, 20, 2.0))
    _turn(session, "s1", prompt=100, completion=20, chars=80)
    _turn(session, "s2", prompt=140, completion=30, chars=95)

    turns = _marks(cc_conn, "turn")
    assert len(turns) == 2
    f = turns[0][2]
    assert f["speech_id"] == "s1" and f["index"] == 1 and f["mode"] == "pipeline"
    assert f["endpointing_ms"] == pytest.approx(200)
    assert f["llm_ttft_ms"] == pytest.approx(400)
    assert f["tts_ttfb_ms"] == pytest.approx(900)
    assert f["latency_ms"] == pytest.approx(1500)  # eou + ttft + ttfb
    assert f["llm_provider"] == "openai" and f["llm_model"] == "gpt-4.1-mini"
    assert f["asr_provider"] == "deepgram" and f["tts_provider"] == "openai"
    assert turns[1][2]["index"] == 2


def test_turn_prefers_metric_metadata_over_the_fallbacks(cc_conn) -> None:
    @dataclass
    class MetaLLM(LLMMetrics):
        metadata: _Meta = None

    _, session = _fwd_stats()
    session.emit("metrics_collected", EOUMetrics("s1", 0.2))
    session.emit("metrics_collected", MetaLLM("s1", 0.4, 1, 1, metadata=_Meta("m-x", "p-x")))
    session.emit("metrics_collected", TTSMetrics("s1", 0.9, 8, 1.0))
    f = _marks(cc_conn, "turn")[0][2]
    assert f["llm_model"] == "m-x" and f["llm_provider"] == "p-x"


def test_realtime_metric_sends_a_realtime_turn(cc_conn) -> None:
    @dataclass
    class RealtimeModelMetrics:
        ttft: float

    _, session = _fwd_stats()
    session.emit("metrics_collected", RealtimeModelMetrics(0.7))
    (m,) = _marks(cc_conn, "turn")
    f = m[2]
    assert f["mode"] == "realtime" and f["index"] == 1
    assert f["rt_ttfa_ms"] == pytest.approx(700)
    assert f["latency_ms"] == pytest.approx(700)


def test_a_failing_connection_does_not_raise_into_the_call(cc_conn) -> None:
    def boom(*a, **k):
        raise RuntimeError("stream closed")

    cc_conn.send_turn = boom
    cc_conn.send_reply_start = boom
    stats, session = _fwd_stats()
    session.emit("agent_state_changed", _StateEvent("speaking"))
    _turn(session, "s1", prompt=1, completion=1, chars=1)
    assert stats.totals.turns == 1


def test_no_marks_without_a_clutchcall_connection() -> None:
    # No cc_conn fixture: plain LiveKit (or no plugin) must be a silent no-op.
    stats, session = _fwd_stats()
    session.emit("agent_state_changed", _StateEvent("speaking"))
    _turn(session, "s1", prompt=1, completion=1, chars=1)
    assert stats.totals.turns == 1
