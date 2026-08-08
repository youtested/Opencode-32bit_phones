"""Tests for the second TUI bug-fix round.

Covers: /undo re-entrant call_from_thread crash (A1), delta batching (A2),
tool_call finalizing the assistant bubble (A3), tool-only turns (A4),
busy-guarded slash commands, pruned-session clicks, tool_denied input rows,
the interrupted event, the permission-dialog exit hang, raw config-key
preservation, the model-picker context formatting, and InputBar history
navigation.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from opencode_py.config import Config, save_config
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import ChatView, MessageBubble, collapse_tool_output
from opencode_py.tui.input_bar import InputBar, PromptSubmitted
from opencode_py.tui.settings_screen import SettingsScreen


class FakeEngine:
    agent = "build"
    permission = type("P", (), {"mode": "auto"})()


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


async def _mounted_bubble(run: dict) -> MessageBubble:
    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.append_tool(run)
        bubbles = list(chat.query(MessageBubble))
        return bubbles[-1]


# --------------------------------------------------------------------------
# A1: /undo (and any command that makes the engine emit) must not crash the UI
# thread via a re-entrant call_from_thread.
# --------------------------------------------------------------------------

async def test_undo_command_from_ui_thread_does_not_crash():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        fd, path = tempfile.mkstemp()
        os.write(fd, b"new")
        os.close(fd)
        app.engine._undo_stack.append({"path": path, "original": b"old"})
        app._run_command("/undo")
        await pilot.pause()
        with open(path, "rb") as fh:
            assert fh.read() == b"old"
        os.unlink(path)


async def test_engine_event_from_ui_thread_handled_inline():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "hi"})
        app._flush_deltas()
        await pilot.pause()
        chat = app._chat_for(sid)
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert assistants and assistants[-1].content == "hi"


# --------------------------------------------------------------------------
# A2: deltas are batched into a single render instead of one per token.
# --------------------------------------------------------------------------

async def test_delta_batching_renders_once():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "a"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "b"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "c"})
        # buffered, not yet rendered to a bubble
        assert chat._stream_bubble is None
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].content == "abc"


# --------------------------------------------------------------------------
# A3: a tool_call finalizes the assistant bubble; the next step's text must
# land in a fresh bubble (no merged text, no stale cursor).
# --------------------------------------------------------------------------

async def test_tool_call_finalizes_stream_and_new_text_is_new_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Let me"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": " check"})
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "glob",
                "arguments": {"pattern": "*.py"},
                "call_id": "c1",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        assistants = [b for b in bubbles if b.role == "assistant"]
        assert assistants and assistants[-1].content == "Let me check"
        assert assistants[-1].streaming is False
        tools = [b for b in bubbles if b.role == "tool"]
        assert tools and tools[-1].content.get("tool") == "glob"
        # a new tool-loop step's text must not merge into the previous bubble
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Found"})
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 2
        assert assistants[-1].content == "Found"


async def test_reasoning_then_tool_call_does_not_leave_stream_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "reasoning_delta", "session_id": sid, "text": "think"})
        app._flush_deltas()
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "bash",
                "arguments": {"command": "ls"},
                "call_id": "c2",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        # no empty assistant stream bubble lingering above the tool row
        assert not any(b.role == "assistant" and b.content == "" for b in bubbles)


# --------------------------------------------------------------------------
# A4: a tool-only turn must not claim "no reply from the model".
# --------------------------------------------------------------------------

async def test_tool_only_turn_does_not_report_no_reply():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._turn_had_tools = True
        app._turn_done(result=None)
        chat = app._chat_for(sid)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert not any("no reply from the model" in str(m) for m in metas)


async def test_turn_done_still_reports_no_reply_when_nothing_happened():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._turn_done(result=None)
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("no reply from the model" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Busy guard: mutating slash commands are blocked while a turn runs.
# --------------------------------------------------------------------------

async def test_busy_blocks_mutating_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.on_prompt_submitted(PromptSubmitted("/undo"))
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("still working" in str(m) for m in metas)


async def test_busy_allows_safe_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        ran: list[str] = []
        app._run_command = lambda line: ran.append(line)
        app.on_prompt_submitted(PromptSubmitted("/help"))
        assert ran == ["/help"]


# --------------------------------------------------------------------------
# Pruned sub-agent: clicking its task row must not open an empty chat wired to
# the main engine.
# --------------------------------------------------------------------------

async def test_switch_to_pruned_session_does_not_switch():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._pruned.add("dead")
        app._switch_session("dead")
        assert app._current_session_id == app.session.id


async def test_subagent_done_marks_pruned_session():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = "sub1"
        chat = app._chat_for(sid)
        app._chats[sid] = chat
        app._engines[sid] = FakeEngine()
        app._sessions[sid] = type("S", (), {"completed": None})()
        app._busy_sessions.add(sid)
        app._running_agents[sid] = "t · build"
        app._on_subagent_done(
            {"kind": "subagent_done", "session_id": sid, "agent": "build", "title": "t", "ok": True}
        )
        assert sid in app._pruned
        assert sid not in app._chats


# --------------------------------------------------------------------------
# tool_denied must render a row with the tool input even without a prior
# tool_call event.
# --------------------------------------------------------------------------

async def test_tool_denied_appends_input_row():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event(
            {
                "kind": "tool_denied",
                "session_id": sid,
                "tool": "write",
                "reason": "file not read first",
                "call_id": "c9",
                "input": {"filePath": "x.py"},
            }
        )
        await pilot.pause()
        tools = [b for b in chat.query(MessageBubble) if b.role == "tool"]
        assert tools
        assert tools[-1].content.get("tool") == "write"
        assert tools[-1].content.get("input") == {"filePath": "x.py"}
        assert tools[-1].content.get("output") == "file not read first"


# --------------------------------------------------------------------------
# The interrupted event is surfaced instead of silently dropped.
# --------------------------------------------------------------------------

async def test_interrupted_event_shows_meta_and_marks_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "interrupted", "session_id": sid})
        await pilot.pause()
        assert app._turn_interrupted is True
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("Interrupted" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Permission dialog: quitting the app must unblock the engine thread quickly.
# --------------------------------------------------------------------------

async def test_permission_ask_unblocks_on_exit():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        holder: dict[str, str] = {}

        def worker() -> None:
            holder["result"] = app._permission_ask("run this command?", [])

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        await asyncio.sleep(0.05)
        app._exit_requested.set()
        t.join(timeout=3)
        assert not t.is_alive(), "permission ask hung after exit"
        assert holder.get("result") == "reject"


# --------------------------------------------------------------------------
# Config: save_config must preserve unknown raw keys (mcpServers/plugins/tools).
# --------------------------------------------------------------------------

def test_save_config_preserves_raw_keys(tmp_path):
    cfg = Config.from_dict(
        {
            "model": "opencode/foo",
            "mcpServers": {"local": {"command": "npx"}},
            "plugins": ["@opencode/plugin-ts"],
            "tools": {"bash": {"deny": "*"}},
        },
        Path("."),
    )
    p = tmp_path / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["model"] == "opencode/foo"
    assert data["mcpServers"] == {"local": {"command": "npx"}}
    assert data["plugins"] == ["@opencode/plugin-ts"]
    assert data["tools"] == {"bash": {"deny": "*"}}


def test_save_config_known_keys_override_raw():
    cfg = Config.from_dict({"model": "opencode/old", "theme": "solarized"}, Path("."))
    cfg.theme = "opencode"
    p = Path(tempfile.mkdtemp()) / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["theme"] == "opencode"


# --------------------------------------------------------------------------
# Model picker: "128k"-style context strings must not crash int().
# --------------------------------------------------------------------------

def test_format_context_handles_k_and_junk():
    from opencode_py.tui.model_picker import _format_context

    assert _format_context(128000) == "128,000"
    assert _format_context("128k") == "128,000"
    assert _format_context("1m") == "1,000,000"
    assert _format_context("junk") == "junk"
    assert _format_context(None) == "?"
    assert _format_context(0) == "0"


# --------------------------------------------------------------------------
# Chat view: failed tools surface an error line; long write output collapses.
# --------------------------------------------------------------------------

async def test_error_line_shows_failed_tool_error():
    b = await _mounted_bubble({"tool": "read", "status": "error", "error": "No such file"})
    err = b._error_line(b.content)
    assert err is not None and "No such file" in str(err)


async def test_error_line_hidden_for_denial():
    b = await _mounted_bubble({"tool": "read", "status": "error", "output": "user dismissed"})
    assert b._error_line(b.content) is None


def test_write_render_collapses_long_content():
    long = "\n".join(f"line {i}" for i in range(200))
    collapsed = collapse_tool_output(long, 10, 10 * 80)
    assert collapsed["overflow"] is True
    assert "line 199" not in collapsed["output"]
    short = collapse_tool_output("tiny", 10, 10 * 80)
    assert short["overflow"] is False
    assert short["output"] == "tiny"


async def test_write_tool_block_uses_metadata_content():
    b = await _mounted_bubble(
        {
            "tool": "write",
            "status": "completed",
            "input": {"filePath": "x.py"},
            "metadata": {"content": "print('hi')\n"},
        }
    )
    assert b._tool_block() is True


# --------------------------------------------------------------------------
# Settings: the "small model" picker must not retarget the app engine.
# --------------------------------------------------------------------------

def test_small_model_row_does_not_propagate():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    rows = screen._build_rows()
    model_row = next(r for r in rows if r.label == "model")
    small_row = next(r for r in rows if r.label == "small model")
    assert model_row.propagate is True
    assert small_row.propagate is False


# --------------------------------------------------------------------------
# InputBar history: repeated Up must not clobber the typed draft, Down must
# restore it, and Down/Up with no history must never wipe the input.
# --------------------------------------------------------------------------

async def _mounted_bar(history: list[str], pilot) -> InputBar:
    bar = pilot.app.query_one(InputBar)
    bar._history = list(history)
    bar._hist_index = len(bar._history)
    return bar


async def test_repeated_up_preserves_typed_draft():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c2"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c1"
        assert bar._draft == "my draft"


async def test_down_restores_draft_at_end_of_history():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        bar._handle_arrow("up")
        bar._handle_arrow("up")
        assert bar._handle_arrow("down") is True
        assert bar.input.value == "c2"
        assert bar._handle_arrow("down") is True
        assert bar.input.value == "my draft"
        assert bar._hist_index == 2


async def test_down_with_no_history_keeps_typed_input():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("down") is False
        assert bar.input.value == "typed text"


async def test_up_with_no_history_keeps_typed_input():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("up") is False
        assert bar.input.value == "typed text"
