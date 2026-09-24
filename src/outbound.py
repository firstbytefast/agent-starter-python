"""Outbound calling over TeleQuick — single dial and batch campaign.

The receiving side is the SAME register-and-wait worker you already run
(`telequick_worker.py`): the engine dials the number on your trunk, and when
the callee answers it dispatches the call to your registered agent exactly
like an inbound call. `ctx.room` carries the numbers where LiveKit puts them
(`sip.trunkPhoneNumber` = the number this call dialed), so one agent script
serves both directions — branch on it if your outbound flow differs.

This file is only the TRIGGER. It uses the LiveKit-shaped dial-out API in the
transport plugin (`TeleQuickAPI`), which wraps the TeleQuick administration
API — the same surface as @clutchcall/admin-sdk, scoped to dialing.

Required environment:
    TELEQUICK_API_KEY       org-scoped mpk_… key, minted in the console under
                             Settings → API keys with scope mcp:telephony:write
    TELEQUICK_ORG_ID        your organization id
    CC_TRUNK                 trunk id to dial through
    CC_AGENT_ID              the TeleQuick agent (console id) that answers —
                             for an external agent, the config whose handle
                             your worker registers as

Usage:
    # one call
    python src/outbound.py +919986398327

    # batch (a paced campaign; engine enforces CPS + concurrency)
    python src/outbound.py +9199... +9198... +9197... --cps 2 --concurrent 5
"""

import argparse
import asyncio
import os

from livekit.plugins import telequick


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("numbers", nargs="+", help="E.164 numbers to dial")
    p.add_argument("--cps", type=int, default=2, help="batch: calls per second")
    p.add_argument("--concurrent", type=int, default=5, help="batch: max concurrent calls")
    p.add_argument("--from", dest="call_from", default=None, help="caller id override")
    args = p.parse_args()

    api = telequick.TeleQuickAPI()  # TELEQUICK_API_KEY / TELEQUICK_ORG_ID
    trunk = os.environ["CC_TRUNK"]
    agent = os.environ["CC_AGENT_ID"]

    if len(args.numbers) == 1:
        res = await api.sip.create_sip_participant(
            sip_trunk_id=trunk,
            sip_call_to=args.numbers[0],
            agent_id=agent,
            sip_call_from=args.call_from,
        )
        print(f"dialing {args.numbers[0]}: call_sid={res.call_sid} status={res.status}")
        print("your worker receives the job_assign when the callee answers")
        return 0

    res = await api.sip.create_sip_campaign(
        name="starter-outbound",
        numbers=args.numbers,
        sip_trunk_id=trunk,
        agent_id=agent,
        sip_call_from=args.call_from,
        calls_per_second=args.cps,
        max_concurrent_calls=args.concurrent,
    )
    print(f"campaign {res.campaign_id}: status={res.status} loaded={res.loaded_numbers}")
    print("abort with: api.sip.abort_campaign(campaign_id); progress: api.sip.list_campaigns()")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
