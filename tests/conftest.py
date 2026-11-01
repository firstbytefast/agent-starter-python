"""A fake ClutchCall connection, so tests can see what the engine would get."""

import sys
import types

import pytest


class FakeConn:
    """Records every control message the starter sends."""

    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def send_usage(self, cid, stage, fields, provider="", model="") -> None:
        self.sent.append(("usage", cid, stage))

    def send_reply_start(self, cid, speech_id=None) -> None:
        self.sent.append(("reply_start", cid, speech_id))

    def send_filler(self, cid, on) -> None:
        self.sent.append(("filler", cid, on))

    def send_turn(self, cid, **fields) -> None:
        self.sent.append(("turn", cid, fields))


@pytest.fixture
def cc_conn(monkeypatch):
    """Make `livekit.plugins.clutchcall` report one live call (id 42)."""
    conn = FakeConn()
    mod = types.ModuleType("livekit.plugins.clutchcall")
    mod.current_connection = lambda: conn
    mod.current_call_id = lambda: 42
    plugins = types.ModuleType("livekit.plugins")
    plugins.clutchcall = mod
    monkeypatch.setitem(sys.modules, "livekit.plugins", plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.clutchcall", mod)
    return conn
