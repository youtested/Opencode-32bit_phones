"""Tests for the AgentLoop: reasoning-only turns, empty-reply history, and the
degenerate tool-call guard."""

import json
from pathlib import Path
from types import SimpleNamespace

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.providers.base import ProviderEvent, ToolCall
from opencode_py.tools import build_registry


class FakeRotation:
    """Replays a script of steps; each step is an event-callable or an exception."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def stream(self, messages, tools, on_event, on_notice=None):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        step = self.script[idx]
        if isinstance(step, BaseException):
            raise step
        step(on_event)
        return "opencode", "deepseek-v4-flash-free"


def make_loop(rotation):
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    return AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rotation,
        agent="build",
    )


def test_normal_text_answer():
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="text_delta", text="hi there"))])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == "hi there"
    assert not result.error


def test_reasoning_only_turn_keeps_nonempty_history():
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="reasoning_delta", text="thinking..."))])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == ""
    assert result.reasoning == "thinking..."
    assistant = [m for m in loop.get_history() if m.get("role") == "assistant"]
    assert assistant and assistant[-1].get("content") != ""


def test_empty_turn_appends_nonempty_assistant():
    rot = FakeRotation([lambda on: None])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == ""
    history = loop.get_history()
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] != ""


def test_degenerate_tool_call_ends_turn_with_error_no_spin():
    tc = ToolCall(id="1", name="", arguments="{}")
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc]))])
    loop = make_loop(rot)
    result = loop.run_turn("hi")
    assert result.error and "invalid tool call" in result.error
    assert rot.calls == 1


def test_valid_tool_call_runs_and_history_has_tool_role():
    tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="done")),
    ])
    loop = make_loop(rot)
    result = loop.run_turn("find py files")
    assert result.text == "done"
    roles = [m.get("role") for m in loop.get_history()]
    assert "tool" in roles


def test_tool_call_with_missing_id_gets_fallback():
    tc = ToolCall(id="", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="done")),
    ])
    loop = make_loop(rot)
    result = loop.run_turn("find py files")
    assert result.text == "done"
    assert not result.error


def test_assistant_tool_calls_message_content_not_none():
    from opencode_py.agent.parse import assistant_message_from_calls

    msg = assistant_message_from_calls([{"id": "c1", "name": "glob", "arguments": "{}"}])
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "glob"


def test_inband_rate_limit_failover_emits_rotated_with_reason():
    """End-to-end: a primary lane that returns an in-band rate-limit error must
    fail over to the next lane AND the 'rotated' event must report the reason."""
    from opencode_py.providers.rotation import Rotation

    class FakeProvider:
        def __init__(self, events):
            self.events = events

        def stream_chat(self, messages, tools, on_event):
            for e in self.events:
                on_event(e)

    queue = iter([
        FakeProvider([ProviderEvent(kind="error", error="rate limit: try again later")]),
        FakeProvider([ProviderEvent(kind="text_delta", text="fallback answer")]),
    ])
    rot = Rotation(
        lanes=[{"provider": "opencode", "model": "deepseek-v4-flash-free"}, {"provider": "opencode", "model": "big-pickle"}],
        make_provider=lambda pid, m: next(queue),
    )
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    events = []
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rot,
        agent="build",
        on_event=events.append,
    )
    result = loop.run_turn("hello")
    assert result.text == "fallback answer"
    rotated = [e for e in events if e.get("kind") == "rotated"]
    assert rotated
    assert rotated[0]["provider"] == "opencode"
    assert "rate" in rotated[0]["reason"]


def test_primary_overload_does_not_fail_over():
    """A transient overload on the chosen model must NOT rotate to a backup;
    the real cause is surfaced instead."""
    from opencode_py.providers.rotation import Rotation

    class FakeProvider:
        def __init__(self, events):
            self.events = events

        def stream_chat(self, messages, tools, on_event):
            for e in self.events:
                on_event(e)

    queue = iter([
        FakeProvider([ProviderEvent(kind="error", error="server_is_overloaded: retry")]),
        FakeProvider([ProviderEvent(kind="text_delta", text="fallback answer")]),
    ])
    rot = Rotation(
        lanes=[{"provider": "opencode", "model": "deepseek-v4-flash-free"}, {"provider": "opencode", "model": "big-pickle"}],
        make_provider=lambda pid, m: next(queue),
    )
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    events = []
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rot,
        agent="build",
        on_event=events.append,
    )
    result = loop.run_turn("hello")
    assert result.error and "server_is_overloaded" in result.error
    assert not result.text
    assert not [e for e in events if e.get("kind") == "rotated"]


# --------------------------------------------------------------------------
# Nested sub-agent event routing (A5): a grandchild's events must keep their
# own session_id through the parent's bridge, not be re-tagged with the
# direct child's id.
# --------------------------------------------------------------------------

def test_nested_subagent_events_keep_own_session_id():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    bridge = parent._subagent_bridge("child1")
    # an event already tagged by a deeper bridge keeps its own session id
    bridge({"kind": "text_delta", "session_id": "grandchild9", "text": "hi"})
    assert events[-1]["session_id"] == "grandchild9"
    # an untagged event from the direct child gets the child's id
    bridge({"kind": "text_delta", "text": "yo"})
    assert events[-1]["session_id"] == "child1"


def test_subagent_start_keeps_own_id_through_parent_bridge():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    bridge = parent._subagent_bridge("child1")
    bridge({"kind": "subagent_start", "session_id": "grandchild9", "agent": "build", "title": "t"})
    evt = events[-1]
    assert evt["session_id"] == "grandchild9"
    assert evt["title"] == "t"


# --------------------------------------------------------------------------
# A failed sub-agent must still emit `subagent_done` (ok=False) so the TUI
# clears its busy state / running indicator.
# --------------------------------------------------------------------------

def test_subagent_run_failure_emits_subagent_done_ok_false(monkeypatch):
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    parent.provider_factory = lambda: SimpleNamespace()

    def boom(self, prompt):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(AgentLoop, "run_turn", boom)
    result = parent.spawn_task({"prompt": "do it", "description": "sub", "subagent_type": "build"})
    done = [e for e in events if e.get("kind") == "subagent_done"]
    assert done and done[-1]["ok"] is False
    assert done[-1]["session_id"] != parent.session_id
    assert result["error"]
    assert "kaboom" in result["output"]


# --------------------------------------------------------------------------
# tool_denied must carry the tool input so the TUI can render what was denied
# (even when no tool_call event preceded it).
# --------------------------------------------------------------------------

def test_tool_denied_emits_input_arguments():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    fake_perm = SimpleNamespace()
    fake_perm.evaluate = lambda *a: "deny"
    parent.permission = fake_perm
    parent.check_permission(
        "write", "{}", "display", call_id="c1", arguments={"filePath": "x.py"}
    )
    evt = events[-1]
    assert evt["kind"] == "tool_denied"
    assert evt["input"] == {"filePath": "x.py"}
    assert evt["call_id"] == "c1"
