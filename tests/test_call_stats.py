"""Stats and filler-audio tests that need no models, no network, and no room.

The metric objects are stand-ins with the same attribute names livekit-agents
uses, so these run on a bare checkout.
"""

import array
import sys
from dataclasses import dataclass
from pathlib import Path

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
