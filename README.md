<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents Starter - Python

A complete starter project for building voice AI apps with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/).

The starter project includes:

- A simple voice AI assistant, ready for extension and customization
- A voice AI pipeline built on [LiveKit Inference](https://docs.livekit.io/agents/models/inference), providing zero-configuration access to [models](https://docs.livekit.io/agents/models) from top labs
  - Uses the fast, open-weight Gemma 4 31B model, [hosted by LiveKit](https://docs.livekit.io/agents/models/llm/livekit/) and tuned for optimal performance in voice AI, as the default LLM
  - Supports more than 50 models from OpenAI, Cartesia, Deepgram, and other providers
  - Access to a wide range of other models, including [Realtime models](https://docs.livekit.io/agents/models/realtime), through extensive plugin ecosystem
- Eval suite based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/start/testing/)
- [LiveKit Turn Detector](https://docs.livekit.io/agents/logic/turns/turn-detector/), an end-of-turn model that listens to the user's audio directly, combining semantic understanding with acoustic cues for state-of-the-art accuracy across 14 languages
- [Background voice cancellation](https://docs.livekit.io/transport/media/noise-cancellation/)
- Deep session insights from LiveKit [Agent Observability](https://docs.livekit.io/deploy/observability/)
- A Dockerfile ready for [production deployment to LiveKit Cloud](https://docs.livekit.io/deploy/agents/)

This starter app is compatible with any [custom web/mobile frontend](https://docs.livekit.io/frontends/) or [telephony](https://docs.livekit.io/telephony/).

## Using coding agents

This project is designed to work with coding agents like [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), and [Codex](https://openai.com/codex/).

For your convenience, LiveKit offers both a CLI and an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/) that can be used to browse and search its documentation. The [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk docs`) works with any coding agent that can run shell commands. Install it for your platform:

**macOS:**

```console
brew install livekit-cli
```

**Linux:**

```console
curl -sSL https://get.livekit.io/cli | bash
```

**Windows:**

```console
winget install LiveKit.LiveKitCLI
```

The `lk docs` subcommand requires version 2.15.0 or higher. Check your version with `lk --version` and update if needed. Once installed, your coding agent can search and browse LiveKit documentation directly from the terminal:

```console
lk docs search "voice agents"
lk docs get-page /agents/start/voice-ai-quickstart
```

See the [Using coding agents](https://docs.livekit.io/intro/coding-agents/) guide for more details, including MCP server setup.

The project includes a complete [AGENTS.md](AGENTS.md) file for these assistants. You can modify this file to suit your needs. To learn more about this file, see [https://agents.md](https://agents.md).

## Dev Setup

Create a project from this template with the LiveKit CLI (recommended):

```bash
lk cloud auth
lk agent init my-agent --template agent-starter-python
```

The CLI clones the template and configures your environment. Then follow the rest of this guide from [Run the agent](#run-the-agent).

<details>
<summary>Alternative: Manual setup without the CLI</summary>

Clone the repository and install dependencies to a virtual environment:

```console
cd agent-starter-python
uv sync
```

Sign up for [LiveKit Cloud](https://cloud.livekit.io/) then set up the environment by copying `.env.example` to `.env.local` and filling in the required keys:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

You can load the LiveKit environment automatically using the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/):

```bash
lk cloud auth
lk app env --write --destination .env.local
```

</details>

## Run the agent

Run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

In production, use the `start` command:

```console
uv run python src/agent.py start
```

## Run over TeleQuick (no LiveKit server)

This fork also runs the same agent over [TeleQuick](https://telequick.dev)'s
QUIC transport instead of a LiveKit room: phone calls reach the agent through
the TeleQuick engine, media rides WebTransport/QUIC datagrams, and no LiveKit
server or WebRTC stack is involved. The agent code in `src/agent.py` is
standard livekit-agents code (with LiveKit Inference swapped for model plugins,
as any self-hosted deployment requires); only `src/telequick_worker.py` is
TeleQuick-specific.

```console
docker build -f Dockerfile.telequick -t cc-starter-agent .
docker run -d --name cc-starter-agent \
  -e CC_APP_KEY=ck_... -e CC_SECRET=... -e CC_AGENT=salesbot \
  -e OPENAI_API_KEY=sk-... -e DEEPGRAM_API_KEY=... \
  cc-starter-agent
```

The worker registers the agent by name with the engine and serves inbound
calls dispatched to it (route a DID to the agent with a TeleQuick
external-agent config). The transport plugin is published at
`https://artifacts.clutchcall.dev/pip/simple/` as `livekit-plugins-telequick`.

## Thinking sound and call stats

The agent plays a sound while the LLM thinks, and prints what each turn used.
Two files do this, and both work on LiveKit and over TeleQuick:

- `src/thinking_audio.py` — fills the dead air between the end of the caller's
  sentence and the first byte of the reply. A caller cannot see a spinner, and
  silence on a phone line sounds like a dropped call.
- `src/call_stats.py` — prints one line for each turn, and a summary at the end
  of the call.

The two transports need two different players, and
`start_thinking_sound(ctx, session)` selects the correct one:

| Transport | Player | How it plays |
|-----------|--------|--------------|
| LiveKit room | the SDK `BackgroundAudioPlayer` | publishes a second audio track and mixes the clips into it |
| TeleQuick QUIC | `TransportThinkingAudio` | writes 20 ms frames on the caller's audio lane, because the call has only one lane and no room |

The sound starts 400 ms after the agent enters the `thinking` state, so a fast
turn stays silent. The default sound is synthesized, and the repo holds no audio
file. To use your own sound, set `CC_THINKING_WAV` to the path of a mono 16-bit
WAV file:

```console
docker run -d --name cc-starter-agent \
  -e CC_THINKING_WAV=/app/sounds/hold.wav \
  ... \
  cc-starter-agent
```

`CallStats` reads the `metrics_collected` events that livekit-agents emits for
each stage. It prints one line for each turn while the call runs:

```
turn 3 | eou 180 ms · llm ttft 420 ms (312→48 tok) · tts ttfb 890 ms (96 ch) | caller waited 1490 ms
```

At the end of the call it prints the totals:

```
──── call summary ────
turns        7
llm  gpt-4.1-mini     4210 prompt (1024 cached) + 512 completion = 4722 tokens over 7 requests
stt  nova-3           61.4 s audio
tts  gpt-4o-mini-tts  742 characters → 48.1 s audio over 9 requests
latency p50/p95  eou 170/240 ms · ttft 402/780 ms · ttfb 830/1210 ms
```

Over TeleQuick the same numbers also go to the engine (`send_usage`), so the
token counts land on the call timeline and in ClickHouse next to the CDR. On
LiveKit that step does nothing. Telemetry never stops a call: a bad metric, a
missing field, or a failed report is logged and dropped.

To test these two files, run `uv run pytest tests/test_call_stats.py
tests/test_thinking_audio.py`. The tests use fake metric objects and a fake
transport, so they need no models, no network, and no call.

### TeleQuick mirror

This tree is CANONICAL for both brands. `./mirror-telequick.sh` writes the
TeleQuick-branded copy to `.mirror/telequick/` — plugin name
(`livekit-plugins-telequick`), python module, engine host, env prefixes and file
names all rewritten in one mechanical pass. The mirror is generated and
gitignored: make every change here and re-run the script. The package origin
`artifacts.clutchcall.dev` is protected, because one origin serves both brands.

### Outbound calling (single + batch)

The same worker also serves calls the platform dials OUT — the engine places
the call on your trunk and dispatches the answered leg to your registered
agent exactly like an inbound call (`ctx.room` attributes carry the dialed
number in `sip.trunkPhoneNumber`). Trigger dials with the LiveKit-shaped API
in the plugin (`src/outbound.py`):

```console
export TELEQUICK_API_KEY=mpk_...   # console: Settings → API keys, scope mcp:telephony:write
export TELEQUICK_ORG_ID=...
export CC_TRUNK=my-trunk CC_AGENT_ID=71156156

python src/outbound.py +919986398327                        # one call
python src/outbound.py +91... +91... +91... --cps 2         # batch (paced campaign)
```

```python
api = telequick.TeleQuickAPI()
await api.sip.create_sip_participant(sip_trunk_id="my-trunk",
                                     sip_call_to="+919986398327",
                                     agent_id="71156156")
```

Batch runs as an engine-paced campaign (`create_sip_campaign` — calls per
second + max concurrency enforced server-side; `list_campaigns` /
`abort_campaign` to watch or stop it).

## Frontend & Telephony

Get started quickly with our pre-built frontend starter apps, or add telephony support:

| Platform | Link | Description |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | Web voice AI assistant with React & Next.js |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native iOS, macOS, and visionOS voice AI assistant |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform voice AI assistant app |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Native mobile app with React Native & Expo |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Native Android app with Kotlin & Jetpack Compose |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Voice AI widget for any website |
| **Telephony** | [Documentation](https://docs.livekit.io/telephony/) | Add inbound or outbound calling to your agent |

For advanced customization, see the [complete frontend guide](https://docs.livekit.io/frontends/).

## Tests and evals

This project includes a complete suite of evals, based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/start/testing/). To run them, use `pytest`.

```console
uv run pytest
```

## Using this template repo for your own project

Once you've started your own project based on this repo, you should:

1. **Check in your `uv.lock`**: This file is currently untracked for the template, but you should commit it to your repository for reproducible builds and proper configuration management. (The same applies to `livekit.toml`, if you run your agents in LiveKit Cloud)

2. **Remove the git tracking test**: Delete the "Check files not tracked in git" step from `.github/workflows/tests.yml` since you'll now want this file to be tracked. These are just there for development purposes in the template repo itself.

3. **Add your own repository secrets**: You must [add secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions) for `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` so that the tests can run in CI.

## Deploying to production

This project is production-ready and includes a working `Dockerfile`. To deploy it to LiveKit Cloud or another environment, see the [deploying to production](https://docs.livekit.io/deploy/agents/) guide.

## Self-hosted LiveKit

You can also self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting](https://docs.livekit.io/transport/self-hosting/local/) guide for more information. If you choose to self-host, you'll need to also use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference and will need to remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
