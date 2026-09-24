"""End-to-end demo: a LiveKit agent over TeleQuick's transport that reports
its context metadata, tool calls, and per-stage token usage.

This is the example used to prove the whole observability path on a real phone
call: the engine delivers per-agent context as ctx.job.metadata, the agent runs
its own tool loop and its own STT/LLM/TTS on its own provider keys, and both the
tool calls and the usage are forwarded over the control stream so they land on
the TeleQuick timeline (voice.agent.tool_called / voice.agent.usage) and in
ClickHouse.

Run (the plugin is mounted so this uses the current build):

    docker run --rm --name cc-demo-agent \
      -v $PWD/src:/demo -v <plugin>/python:/plug \
      -e PYTHONPATH=/demo:/plug \
      -e CC_APP_KEY=... -e CC_SECRET=... -e CC_AGENT=demo-lk-example \
      -e CC_HOST=engine.telequick.dev \
      -e OPENAI_API_KEY=... -e DEEPGRAM_API_KEY=... \
      cc-starter-agent python /demo/demo_metrics_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo-agent")

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import telequick, deepgram, openai, silero

from call_stats import CallStats
from thinking_audio import start_thinking_sound

LLM_MODEL = "gpt-4.1-mini"
STT_MODEL = "nova-2-phonecall"
TTS_MODEL = "gpt-4o-mini-tts"

_vad = None


def _get_vad():
    global _vad
    if _vad is None:
        _vad = silero.VAD.load()
    return _vad


def _report_tool(name: str, args: dict, result: dict) -> None:
    """Forward a tool call to TeleQuick so it lands on the timeline."""
    conn = telequick.current_connection()
    cid = telequick.current_call_id()
    if conn is None or cid is None:
        return
    try:
        conn.send_tool_call(cid, name, args, result)
        log.info("reported tool call %s -> TeleQuick", name)
    except Exception:  # noqa: BLE001 — reporting must never break the call
        log.exception("tool report failed")


class DemoAgent(Agent):
    def __init__(self, meta: dict) -> None:
        env = meta.get("env", "unknown")
        team = meta.get("team", "unknown")
        super().__init__(
            llm=openai.LLM(model=LLM_MODEL),
            instructions=(
                "You are a brief, friendly order-status assistant answering a phone call. "
                f"Your deployment context is env={env}, team={team} — mention the env once "
                "in your greeting so the caller knows the context arrived. "
                "Speak plainly: no markdown, one to two short sentences per turn. "
                "If the caller asks about an order, call the check_order_status tool with "
                "the order id they give (if they don't give one, use 1234). "
                "Then tell them the status in a natural sentence."
            ),
        )

    @function_tool
    async def check_order_status(self, context: RunContext, order_id: str) -> str:
        """Look up the delivery status of a customer order.

        Args:
            order_id: The order number the caller gives, e.g. "1234".
        """
        result = {"order_id": order_id, "status": "shipped", "eta": "Tuesday"}
        _report_tool("check_order_status", {"order_id": order_id}, result)
        return json.dumps(result)


async def entrypoint(ctx) -> None:
    # Per-agent (and per-call) context the engine delivered on dispatch.
    raw_meta = getattr(getattr(ctx, "job", None), "metadata", "") or ""
    try:
        meta = json.loads(raw_meta) if raw_meta else {}
    except Exception:  # noqa: BLE001
        meta = {}
    log.info("call start: room=%s metadata=%s", getattr(ctx.room, "name", "?"), meta)

    session = AgentSession(
        stt=deepgram.STT(model=STT_MODEL),
        llm=openai.LLM(model=LLM_MODEL),
        # Name the model. `openai.TTS(voice=...)` defaults to tts-1, which
        # measured ttfb 2230 ms against gpt-4o-mini-tts at 895 ms on the same
        # text and network — about 1.3 s of extra silence on every turn.
        # Fixed in agent.py by 8d88a81; this second pipeline kept the default.
        tts=openai.TTS(model=TTS_MODEL, voice="alloy"),
        vad=_get_vad(),
    )

    # Per-stage usage → TeleQuick, and the same numbers on the console.
    # livekit-agents emits them on our own provider keys; forwarding them gives
    # the platform token analytics, and printing them gives the operator a
    # per-turn view of where the latency went.
    stats = CallStats(
        models={"llm": LLM_MODEL, "stt": STT_MODEL, "tts": TTS_MODEL},
        providers={"llm": "openai", "stt": "deepgram", "tts": "openai"},
        log=log,
    )
    stats.attach(session)
    ctx.add_shutdown_callback(stats.log_summary)

    await session.start(agent=DemoAgent(meta), room=ctx.room)

    # Filler audio over the same QUIC lane while the LLM thinks — the caller
    # hears that the agent is still there instead of dead air.
    thinking = await start_thinking_sound(ctx, session)
    if thinking is not None:
        ctx.add_shutdown_callback(thinking.aclose)

    await session.generate_reply(
        instructions="Greet the caller in one short sentence and ask how you can help.",
    )


async def main() -> int:
    cfg = telequick.WebTransportConfig(
        engine_url=f"https://{os.environ.get('CC_HOST', 'engine.telequick.dev')}",
        app_key=os.environ["CC_APP_KEY"],
        app_secret=os.environ["CC_SECRET"],
        agent_name=os.environ.get("CC_AGENT", "demo-lk-example"),
        sample_rate=8000,
        num_channels=1,
        codec="pcm16",
    )
    log.info("registering agent=%s on %s", cfg.agent_name, os.environ.get("CC_HOST"))
    await telequick.run_agent_worker(
        cfg,
        host=os.environ.get("CC_HOST", "engine.telequick.dev"),
        port=int(os.environ.get("CC_PORT", "443")),
        entrypoint=entrypoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
