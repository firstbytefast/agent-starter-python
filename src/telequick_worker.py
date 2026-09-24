"""Run the starter agent over TeleQuick's QUIC transport instead of a LiveKit room.

The agent code in `agent.py` is unmodified livekit-agents code — this worker is
the only TeleQuick-specific file. `telequick.run_agent_worker` opens one
authenticated WebTransport/QUIC connection to the engine, registers the agent
by name, and runs the ordinary `my_agent` entrypoint for every phone call the
engine dispatches. Media rides QUIC datagrams (8 kHz pcm16 telephony); no
LiveKit server, room, or WebRTC is involved.

Required environment:
    CC_APP_KEY / CC_SECRET   TeleQuick media app credential
    CC_AGENT                 agent name to register (default: salesbot)
    CC_HOST / CC_PORT        engine endpoint (default: engine.telequick.dev:443)
    OPENAI_API_KEY           LLM + TTS
    DEEPGRAM_API_KEY         STT

Usage:
    python src/telequick_worker.py
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from livekit.plugins import telequick

from agent import my_agent


async def main() -> int:
    host = os.environ.get("CC_HOST", "engine.telequick.dev")
    port = int(os.environ.get("CC_PORT", "443"))
    cfg = telequick.WebTransportConfig(
        engine_url=f"https://{host}",
        app_key=os.environ["CC_APP_KEY"],
        app_secret=os.environ["CC_SECRET"],
        agent_name=os.environ.get("CC_AGENT", "salesbot"),
        sample_rate=8000,
        num_channels=1,
        codec="pcm16",
    )
    print(
        f"[telequick] starting worker agent={cfg.agent_name} on "
        f"{host}:{port}/media/{cfg.app_key} — awaiting calls",
        flush=True,
    )
    await telequick.run_agent_worker(
        cfg,
        host=host,
        port=port,
        entrypoint=my_agent,
        verify=not os.environ.get("CC_NO_VERIFY"),
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
