"""Per-call stats: latency, token use, and what each turn cost you.

`livekit-agents` emits one `metrics_collected` event per pipeline stage (STT,
end-of-turn, LLM, TTS). This module turns that stream into two things:

  * a ONE-LINE-PER-TURN console view, so you can watch where a slow turn went
    while the call is live, and
  * a call summary at hangup: total tokens in/out, characters synthesized,
    audio transcribed, and the p50/p95 of each latency stage.

Both work the same on LiveKit Cloud and over the ClutchCall transport. When the
agent runs over ClutchCall, the same numbers are ALSO forwarded to the engine
(`send_usage`), so they land on the call timeline and in ClickHouse next to the
CDR — that path is best-effort and never raises into the call.

Attribute names on the metric objects differ slightly across SDK versions, so
every read goes through `_get` with a default. A missing field shows as 0, not
as a crashed call.

    stats = CallStats(models={"llm": "gpt-4.1-mini", "stt": "nova-3"})
    stats.attach(session)
    ...
    ctx.add_shutdown_callback(stats.log_summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("call-stats")


def _get(obj: object, name: str, default: float = 0.0) -> float:
    value = getattr(obj, name, None)
    return default if value is None else value


def _pct(values: list[float], q: float) -> float:
    """p50/p95 over a short list. Nearest-rank, no interpolation, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[idx]


@dataclass
class _Turn:
    """What we know so far about one speech turn, keyed by the SDK speech id."""

    eou_delay: float = 0.0
    ttft: float = 0.0
    ttfb: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    characters: int = 0


@dataclass
class Totals:
    turns: int = 0
    llm_requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    tts_requests: int = 0
    tts_characters: int = 0
    tts_audio_s: float = 0.0
    stt_audio_s: float = 0.0
    eou_delays: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    ttfbs: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CallStats:
    """Collects the session's metrics, prints them, and forwards usage."""

    def __init__(
        self,
        *,
        models: dict[str, str] | None = None,
        providers: dict[str, str] | None = None,
        forward_to_clutchcall: bool = True,
        log: logging.Logger | None = None,
    ) -> None:
        # Used only as the fallback label when a metric carries no metadata of
        # its own (SDK < 1.8 does not attach model/provider to every stage).
        self._models = models or {}
        self._providers = providers or {}
        self._forward = forward_to_clutchcall
        self._log = log or logger
        self.totals = Totals()
        self._open: dict[str, _Turn] = {}

    # -- wiring -----------------------------------------------------------
    def attach(self, session) -> None:  # AgentSession
        session.on("metrics_collected", self._on_metrics)

    # -- the metrics stream ----------------------------------------------
    def _on_metrics(self, ev) -> None:  # MetricsCollectedEvent
        m = getattr(ev, "metrics", ev)
        kind = type(m).__name__
        try:
            if "EOU" in kind:
                self._on_eou(m)
            elif "LLM" in kind:
                self._on_llm(m)
            elif "TTS" in kind:
                self._on_tts(m)
            elif "STT" in kind:
                self._on_stt(m)
        except Exception:  # telemetry never breaks a call
            self._log.exception("stats: dropped a %s metric", kind)

    def _turn(self, m: object) -> _Turn:
        key = str(getattr(m, "speech_id", "") or "-")
        return self._open.setdefault(key, _Turn())

    def _on_eou(self, m: object) -> None:
        delay = _get(m, "end_of_utterance_delay")
        self._turn(m).eou_delay = delay
        self.totals.eou_delays.append(delay)

    def _on_llm(self, m: object) -> None:
        turn = self._turn(m)
        turn.ttft = _get(m, "ttft")
        turn.prompt_tokens = int(_get(m, "prompt_tokens"))
        turn.cached_tokens = int(_get(m, "prompt_cached_tokens"))
        turn.completion_tokens = int(_get(m, "completion_tokens"))

        t = self.totals
        t.llm_requests += 1
        t.prompt_tokens += turn.prompt_tokens
        t.cached_tokens += turn.cached_tokens
        t.completion_tokens += turn.completion_tokens
        t.ttfts.append(turn.ttft)

        self._send_usage(
            "llm",
            {
                "prompt_tokens": turn.prompt_tokens,
                "prompt_cached_tokens": turn.cached_tokens,
                "completion_tokens": turn.completion_tokens,
            },
            m,
        )

    def _on_tts(self, m: object) -> None:
        turn = self._turn(m)
        turn.ttfb = _get(m, "ttfb")
        turn.characters = int(_get(m, "characters_count"))

        t = self.totals
        t.tts_requests += 1
        t.tts_characters += turn.characters
        t.tts_audio_s += _get(m, "audio_duration")
        t.ttfbs.append(turn.ttfb)

        self._send_usage("tts", {"characters": turn.characters}, m)

        # The first synthesis of a turn is the point the caller starts hearing
        # the reply, so that is where the turn line is worth printing. Later
        # segments of the same turn only add to the totals.
        key = str(getattr(m, "speech_id", "") or "-")
        done = self._open.pop(key, None)
        if done is not None:
            t.turns += 1
            self._log_turn(t.turns, done)

    def _on_stt(self, m: object) -> None:
        self.totals.stt_audio_s += _get(m, "audio_duration")
        self._send_usage("stt", {"audio_seconds": _get(m, "audio_duration")}, m)

    # -- display ----------------------------------------------------------
    def _log_turn(self, n: int, turn: _Turn) -> None:
        cached = f", {turn.cached_tokens} cached" if turn.cached_tokens else ""
        # eou + ttft + ttfb is what the caller waits through before the first
        # byte of the reply plays: the number to watch on a phone call.
        reply = turn.eou_delay + turn.ttft + turn.ttfb
        self._log.info(
            "turn %d | eou %.0f ms · llm ttft %.0f ms (%d→%d tok%s) · "
            "tts ttfb %.0f ms (%d ch) | caller waited %.0f ms",
            n,
            turn.eou_delay * 1000,
            turn.ttft * 1000,
            turn.prompt_tokens,
            turn.completion_tokens,
            cached,
            turn.ttfb * 1000,
            turn.characters,
            reply * 1000,
        )

    def summary_lines(self) -> list[str]:
        t = self.totals
        cached = f" ({t.cached_tokens} cached)" if t.cached_tokens else ""
        return [
            "──── call summary ────",
            f"turns        {t.turns}",
            f"llm  {self._models.get('llm', 'llm'):<16} "
            f"{t.prompt_tokens} prompt{cached} + {t.completion_tokens} completion "
            f"= {t.total_tokens} tokens over {t.llm_requests} requests",
            f"stt  {self._models.get('stt', 'stt'):<16} {t.stt_audio_s:.1f} s audio",
            f"tts  {self._models.get('tts', 'tts'):<16} "
            f"{t.tts_characters} characters → {t.tts_audio_s:.1f} s audio "
            f"over {t.tts_requests} requests",
            "latency p50/p95  "
            f"eou {_pct(t.eou_delays, 0.5) * 1000:.0f}/{_pct(t.eou_delays, 0.95) * 1000:.0f} ms · "
            f"ttft {_pct(t.ttfts, 0.5) * 1000:.0f}/{_pct(t.ttfts, 0.95) * 1000:.0f} ms · "
            f"ttfb {_pct(t.ttfbs, 0.5) * 1000:.0f}/{_pct(t.ttfbs, 0.95) * 1000:.0f} ms",
        ]

    async def log_summary(self) -> None:
        """Shutdown callback: print the totals for the finished call."""
        for line in self.summary_lines():
            self._log.info(line)

    # -- ClutchCall forwarding (no-op off the ClutchCall transport) -------
    def _send_usage(self, stage: str, amounts: dict, m: object) -> None:
        if not self._forward:
            return
        try:
            from livekit.plugins import clutchcall
        except ImportError:
            self._forward = False  # not installed — stop trying
            return
        conn = clutchcall.current_connection()
        cid = clutchcall.current_call_id()
        if conn is None or cid is None:
            return  # running on a LiveKit room, not over ClutchCall
        meta = getattr(m, "metadata", None)
        model = getattr(meta, "model_name", None) or self._models.get(stage, "")
        provider = getattr(meta, "model_provider", None) or self._providers.get(
            stage, ""
        )
        try:
            conn.send_usage(cid, stage, amounts, provider=provider, model=model)
        except Exception:  # reporting must never break the call
            self._log.exception("stats: %s usage report failed", stage)
