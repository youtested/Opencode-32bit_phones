"""opencode_py Textual TUI app.

Mirrors opencode's session screen: header status bar (agent/model/provider/
permission), scrollable chat with live tool blocks + diff rendering, and a
prompt input bar. The engine runs in a worker thread; events are bridged to
the UI via call_from_thread.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

from ..agent.loop import AgentLoop
from ..auth import Auth
from ..commands import build_registry as build_command_registry
from ..config import Config, load_config
from ..providers import fetch_zen_models
from ..session import new_session, save_session
from ..tools import build_registry as build_tool_registry
from .chat_view import ChatView
from .command_popup import CommandPopup
from .connect_screen import ConnectScreen
from .input_bar import AgentToggleRequested, CommandSelected, InputBar, PromptSubmitted
from .model_picker import ModelPicker
from .permission_dialog import PermissionDialog
from .status_bar import StatusBar


# Read-only slash commands allowed while a turn is running. Everything else
# (undo/clear/compact/agent/model/theme/...) mutates engine state or the running
# turn and is blocked until the current request finishes.
_SAFE_WHILE_BUSY = {
    "help",
    "config",
    "permissions",
    "sessions",
    "ls",
    "models",
    "review",
    "connect",
    "resume",
}


class OpenCodeTUI(App):
    TITLE = "opencode"
    SUB_TITLE = "opencode_py"

    ENABLE_COMMAND_PALETTE = False  # ctrl+p is bound to Settings instead

    CSS = """
    Screen {
        background: #0a0a0a;
        color: #eeeeee;
    }
    #root {
        layout: vertical;
        height: 1fr;
    }
    #chat-stack {
        layout: vertical;
        height: 1fr;
    }
    ChatView {
        width: 100%;
        height: 1fr;
        padding: 0 2;
        background: #0a0a0a;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    InputBar {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: #0a0a0a;
    }
    .prompt-frame {
        height: auto;
        background: #0a0a0a;
    }
    #prompt-accent {
        width: 1;
        height: 3;
        background: #fab283;
    }
    .prompt-body {
        width: 1fr;
        height: auto;
        padding: 0 0 0 1;
    }
    #prompt-input {
        width: 1fr;
        background: #1e1e1e;
        color: #eeeeee;
        border: none;
        outline: none;
        padding: 0 1;
        height: 3;
        min-height: 3;
        content-align: left middle;
    }
    #prompt-input:focus {
        border: none;
        outline: none;
    }
    #prompt-title {
        width: 1fr;
        height: 1;
        margin-top: 1;
        margin-bottom: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-meta {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-status {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-status.hidden {
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 10;
        padding: 0 1;
        background: #1e1e1e;
        color: #eeeeee;
        border: round #9d7cd8;
        margin: 0 1 1 2;
    }
    #suggestions.hidden {
        display: none;
    }
    StatusBar {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: #0a0a0a;
        color: #808080;
    }
    .cmd-popup {
        width: 74;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .cmd-popup.settings {
        width: 74;
        max-height: 90%;
    }
    .cmd-popup-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-scroll {
        height: 40;
        min-height: 4;
        max-height: 70%;
        border: none;
    }
    .settings-body {
        padding: 1 2;
        color: $text;
    }
    .settings-edit {
        margin: 0 2;
        height: 3;
    }
    .settings-hint {
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    .cmd-popup-actions {
        height: 3;
        align: center middle;
        background: $surface;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+r", "resume", "Resume"),
        Binding("ctrl+t", "toggle_agent", "Switch agent"),
        Binding("ctrl+b", "sessions", "Sessions"),
        Binding("escape", "focus_input", "Input"),
        Binding("ctrl+p", "settings", "Settings"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("ctrl+shift+e", "toggle_thought", "Expand/collapse thought"),
    ]

    def __init__(
        self,
        cfg: Config | None = None,
        engine: AgentLoop | None = None,
        directory: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg or load_config()
        self.directory = directory or Path.cwd()
        from ..globals import Path as GPath
        self.auth = Auth(auth_file=GPath.auth_file())
        if engine is None:
            engine = AgentLoop(
                cfg=self.cfg,
                registry=build_tool_registry(self.cfg),
                directory=self.directory,
                auth=self.auth,
                agent=self.cfg.default_agent or "build",
            )
        self.engine = engine
        self.engine.on_event = self._on_engine_event
        # Shared interrupt flag: Ctrl+C flips this and every engine (main + any
        # spawned sub-agents, which capture the same callable by reference) sees
        # it at the next iteration and stops. It is reset once the turn ends so
        # future turns are not aborted.
        self._interrupt_flag = {"requested": False}
        self.engine.interrupt = self._interrupt_requested
        # Permission "ask" mode: bridge the engine thread to a modal dialog.
        # Sub-agents share the same PermissionEngine instance, so one hook works
        # for all sessions.
        self.engine.permission.ask_callback = self._permission_ask
        self.command_registry = build_command_registry()
        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.engine.agent,
        )
        self.engine.session_id = self.session.id
        self._chats: dict[str, ChatView] = {}
        self._sessions: dict[str, Any] = {self.session.id: self.session}
        self._engines: dict[str, AgentLoop] = {self.session.id: self.engine}
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._busy = False
        self._busy_sessions: set[str] = set()
        self._running_agents: dict[str, str] = {}
        self._turn_had_text = False
        self._turn_had_reasoning = False
        self._turn_had_error = False
        self._turn_had_tools = False
        self._turn_interrupted = False
        self._pruned: set[str] = set()
        # delta batching: text/reasoning deltas are queued (non-blocking) and
        # flushed on a short timer so a fast stream isn't re-rendered per token.
        self._pending: dict[str, dict[str, list[str]]] = {}
        self._delta_timer: Any = None
        self._exit_requested = threading.Event()

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="chat-stack"):
                yield ChatView()
            yield InputBar(
                commands=[
                    {"name": c.name, "description": c.description}
                    for c in self.command_registry.list()
                    if not c.hidden
                ]
            )
            yield StatusBar()

    def on_mount(self) -> None:
        self._update_header()
        status = self.query_one(StatusBar)
        status.set_directory(str(self.directory))
        self._main_chat = self.query_one(ChatView)
        self._chats[self.session.id] = self._main_chat
        self.query_one(InputBar).focus()

    # -- session routing --------------------------------------------------
    def _chat_for(self, session_id: str) -> ChatView:
        """Chat view for a session, creating (hidden) one on first use so a
        spawned sub-agent has a live, switchable conversation."""
        chat = self._chats.get(session_id)
        if chat is not None:
            return chat
        chat = ChatView()
        self._chats[session_id] = chat
        try:
            self.query_one("#chat-stack", Vertical).mount(chat, after=self._main_chat)
        except Exception:
            pass
        chat.display = "none"
        return chat

    def _active_engine(self) -> AgentLoop:
        return self._engines.get(self._current_session_id, self.engine)

    def _active_session(self) -> Any:
        return self._sessions.get(self._current_session_id, self.session)

    def action_sessions(self) -> None:
        from .session_list import SessionList

        rows = []
        for sid, sess in self._sessions.items():
            status = "running" if sid in self._busy_sessions else ""
            rows.append(
                {
                    "id": sid,
                    "title": sess.title,
                    "agent": sess.agent,
                    "status": status,
                }
            )
        self.push_screen(SessionList(rows, current=self._current_session_id), self._on_session_picked)

    def _on_session_picked(self, choice: str | None) -> None:
        if choice:
            self._switch_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def _switch_session(self, session_id: str) -> None:
        if session_id == self._current_session_id:
            return
        if session_id in self._pruned:
            # a finished sub-agent session was pruned (no engine/history left);
            # opening it would create an empty chat wired to the wrong engine
            self.notify("That sub-agent session is finished and closed.")
            self.query_one(InputBar).focus()
            return
        old = self._chats[self._current_session_id]
        self._current_session_id = session_id
        new = self._chat_for(session_id)
        old.display = "none"
        new.display = "block"
        sess = self._sessions.get(session_id)
        if sess and sess.title:
            self.notify(f"Session: {sess.title}")
        self._update_header()
        self.query_one(InputBar).focus()

    def on_open_task_session(self, event: Any) -> None:
        if getattr(event, "sid", None):
            self._switch_session(event.sid)

    # -- running agents ---------------------------------------------------
    def _refresh_running_agents(self) -> None:
        """Show the launched sub-agents in the status line above the prompt,
        like opencode's `Delegating...` indicator (transient, no sidebar)."""
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        bar.set_running_agents(list(self._running_agents.values()))

    # -- engine event bridge ---------------------------------------------
    def _on_engine_event(self, event: dict[str, Any]) -> None:
        # Called from the engine thread; hop to the UI thread.
        if getattr(self, "_thread_id", None) == threading.get_ident():
            # Already on the UI thread (e.g. a /command handler that makes the
            # engine emit an event, like /undo): call_from_thread would raise
            # RuntimeError, so handle inline instead.
            self._handle_event(event)
            return
        kind = event.get("kind")
        if kind in ("text_delta", "reasoning_delta"):
            # Don't block the engine stream on the UI render: queue the delta
            # (non-blocking) and let the flush timer batch it into one render.
            # FIFO scheduling keeps ordering with the blocking events below.
            self._schedule_async(event)
            return
        try:
            self.call_from_thread(self._handle_event, event)
        except RuntimeError:
            # app shutting down / loop gone — drop the event safely
            pass

    def _schedule_async(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_handle(event), loop)
        except (RuntimeError, Exception):
            pass

    async def _async_handle(self, event: dict[str, Any]) -> None:
        with self._context():
            self._handle_event(event)

    def _queue_delta(self, session_id: str, text: str, kind: str) -> None:
        buf = self._pending.setdefault(session_id, {"text": [], "reasoning": []})
        buf[kind].append(text)
        if self._delta_timer is None:
            self._delta_timer = self.set_timer(0.03, self._flush_deltas)

    def _cancel_delta_timer(self) -> None:
        if self._delta_timer is not None:
            try:
                self._delta_timer.stop()
            except Exception:
                pass
            self._delta_timer = None

    def _flush_deltas(self) -> None:
        """Render any buffered text/reasoning deltas (one render per batch)."""
        self._cancel_delta_timer()
        pending = self._pending
        self._pending = {}
        for session_id, buf in pending.items():
            chat = self._chat_for(session_id)
            for t in buf.get("reasoning") or []:
                chat.stream_reasoning_delta(t)
            if buf.get("text"):
                chat.end_reasoning()
                for t in buf["text"]:
                    chat.stream_delta(t)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        session_id = event.get("session_id") or self.session.id
        chat = self._chat_for(session_id)
        status = self.query_one(StatusBar)
        if kind == "step":
            pass
        elif kind == "error":
            self._show_error(event.get("error", "unknown error"), retryable=bool(event.get("retryable")), session_id=session_id)
        elif kind == "text_delta":
            self._turn_had_text = True
            self._queue_delta(session_id, event.get("text", ""), "text")
        elif kind == "reasoning_delta":
            self._turn_had_reasoning = True
            self._queue_delta(session_id, event.get("text", ""), "reasoning")
        elif kind == "tool_call":
            # render any buffered text first so the tool row lands below it
            self._flush_deltas()
            tool = event.get("tool", "?")
            # the current thought/assistant stream is over — finalize it so a
            # multi-step tool loop doesn't merge every step's text into one
            # bubble or leave a stale ▍ cursor
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            self._turn_had_tools = True
            chat.append_tool(
                {
                    "tool": tool,
                    "status": "running",
                    "input": event.get("arguments", {}),
                    "call_id": event.get("call_id", ""),
                }
            )
        elif kind == "tool_start":
            tool_run = {
                "tool": event.get("tool", "?"),
                "status": "running",
                "input": event.get("input", {}),
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(tool_run):
                # No exact call_id match (e.g. the tool_call event never
                # arrived for this session) — append the row instead of leaving
                # a stale placeholder or updating the wrong tool.
                chat.append_tool(tool_run)
        elif kind == "tool_complete":
            chat.update_tool_bubble(
                {
                    "tool": event.get("tool", "?"),
                    "status": "error" if event.get("status") == "error" else "completed",
                    "input": event.get("input", {}),
                    "output": event.get("output", ""),
                    "metadata": event.get("metadata", {}),
                    "call_id": event.get("call_id", ""),
                }
            )
        elif kind == "tool_denied":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error",
                "input": event.get("input", {}),
                "output": event.get("reason") or "permission denied",
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(run):
                # No preceding tool_call (e.g. the read-before-edit guard) —
                # show the denied row so the rejected action is visible.
                chat.append_tool(run)
        elif kind == "interrupted":
            self._flush_deltas()
            self._turn_interrupted = True
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            chat.end_stream()
            chat.append_meta("⏹ Interrupted")
        elif kind == "usage":
            if session_id == self.session.id:
                status.set_usage(event.get("usage") or {})
        elif kind == "rotated":
            reason = event.get("reason") or "provider error"
            self.notify(
                f"{reason} — switched to {event.get('provider', '?')}/{event.get('model', '?')}"
            )
        elif kind == "subagent_start":
            self._on_subagent_start(event)
        elif kind == "subagent_done":
            self._on_subagent_done(event)

    def _on_subagent_start(self, event: dict[str, Any]) -> None:
        from ..session import load_session

        sid = event.get("session_id") or ""
        if not sid:
            return
        if sid not in self._sessions:
            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
            else:
                # register a placeholder so the fallbacks
                # (`_sessions.get(sid) or self.session`) never route a sub-agent
                # turn's history onto the main session file.
                from ..session import Session
                self._sessions[sid] = Session({"id": sid}, directory=str(self.directory))
        sub = self.engine.find_subagent(sid)
        if sub is not None:
            self._engines[sid] = sub
        self._busy_sessions.add(sid)
        self._chats[sid] = self._chat_for(sid)
        title = event.get("title") or "sub-agent"
        agent = event.get("agent") or "build"
        self._running_agents[sid] = f"{title} · {agent}"
        self._refresh_running_agents()
        self.notify(f"Sub-agent started: {title} (Ctrl+B to watch)")

    def _on_subagent_done(self, event: dict[str, Any]) -> None:
        sid = event.get("session_id") or ""
        sess = self._sessions.get(sid)
        if sess is not None:
            sess.completed = time.time()
        self._busy_sessions.discard(sid)
        chat = self._chats.get(sid)
        if chat is not None:
            chat.end_reasoning()
            # drop an empty streaming cursor if the sub-agent replied with no text
            chat.remove_last_stream_bubble()
            chat.end_stream()
        title = event.get("title") or "sub-agent"
        ok = event.get("ok", True)
        self._running_agents.pop(sid, None)
        self._refresh_running_agents()
        if sid != self.session.id and sid != self._current_session_id:
            # prune the finished (hidden) sub-agent session so long-running
            # sessions don't accumulate mounted ChatViews / engine entries.
            # Remember it so a click on its task row shows a notice instead of
            # opening an empty chat wired to the main engine.
            self._pruned.add(sid)
            self._chats.pop(sid, None)
            self._engines.pop(sid, None)
            self._sessions.pop(sid, None)
            if chat is not None:
                try:
                    chat.remove()
                except Exception:
                    pass
        if ok:
            self.notify(f"Sub-agent done: {title}")
        else:
            self.notify(f"Sub-agent failed: {title}", severity="error")

    # -- prompt handling -------------------------------------------------
    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            return
        sid = self._current_session_id
        chat = self._chat_for(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        status = self.query_one(StatusBar)
        # always show what the user typed, then route it
        chat.append_user(value, agent=engine.agent)
        if value.lstrip().startswith("/"):
            self._run_command(value.lstrip())
            return
        if self._busy:
            chat.append_meta("⏳ still working on the previous request…")
            self.notify("Busy — your last request is still running (Ctrl+C to interrupt)")
            return
        # (no eager begin_stream: the assistant bubble appears only when the
        # model actually emits text, keeping thought/tool blocks chronological)
        status.set_streaming(True)
        try:
            bar = self.query_one(InputBar)
            bar.set_busy(True)
        except Exception:
            pass
        self._turn_had_text = False
        self._turn_had_reasoning = False
        self._turn_had_error = False
        self._turn_had_tools = False
        self._turn_interrupted = False
        self._busy = True
        self._busy_sessions.add(sid)
        self._active_turn_session_id = sid

        def run():
            result = None
            try:
                result = engine.run_turn(value)
            except Exception as e:  # never let a worker crash silently
                self.call_from_thread(self._show_error, f"{type(e).__name__}: {e}", False, sid)
            finally:
                self.call_from_thread(self._turn_done, result)

        self.run_worker(run, thread=True)

    def _show_error(self, message: str, retryable: bool = False, session_id: str | None = None) -> None:
        self._turn_had_error = True
        sid = session_id or self.session.id
        chat = self._chat_for(sid)
        self._flush_deltas()
        chat.end_reasoning()
        chat.remove_last_stream_bubble()
        chat.append_meta(f"⚠ {message}")
        hint = " Retry, or check /connect for a model/API key." if retryable else ""
        self.notify(f"error: {message}{hint}", severity="error")
        chat.end_stream()
        # Reset the visual streaming state here too, not only in _turn_done. The
        # worker's `finally` normally calls _turn_done and clears these, but if
        # that call_from_thread ever fails (e.g. app unmount mid-error) the UI
        # must not stay stuck on an "streaming" indicator on the error path.
        self._streaming_visual_reset()

    def _streaming_visual_reset(self) -> None:
        """Defensively clear the streaming-indicator UI state (status bar +
        input bar). Does NOT touch _busy/_busy_sessions: those belong to the
        worker thread and are owned by _turn_done's finally, so resetting them
        here could race an active turn on another session."""
        try:
            self.query_one(StatusBar).set_streaming(False)
        except Exception:
            pass
        try:
            self.query_one(InputBar).set_busy(False)
        except Exception:
            pass

    def _turn_done(self, result: Any = None) -> None:
        self._interrupt_flag["requested"] = False
        sid = self._active_turn_session_id
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        if result is not None and result.provider_id:
            # reflect the lane/model that actually answered (e.g. openrouter)
            engine.provider_id = result.provider_id
            engine.model_id = result.model_id or engine.model_id
            engine.rebuild_rotation()
            self._update_header()
        chat = self._chat_for(sid)
        status = self.query_one(StatusBar)
        self._flush_deltas()
        chat.end_reasoning()
        if not self._turn_had_text and not self._turn_had_reasoning and not self._turn_had_error and not self._turn_had_tools and not self._turn_interrupted:
            # provider returned nothing (no text, no reasoning, no tool call,
            # no error) — drop the empty streaming cursor bubble before
            # end_stream clears its pointer.
            chat.remove_last_stream_bubble()
            chat.append_meta(
                "(no reply from the model — check your connection and /connect "
                "for a working model, or switch rotation in /config)"
            )
            self.notify("No reply from the model.", severity="warning")
        chat.end_stream()
        if self._turn_had_text and not self._turn_had_error:
            # the mode line lives fixed above the prompt box now, not in the chat
            self._update_header()
        status.set_streaming(False)
        self._busy = False
        self._busy_sessions.discard(sid)
        try:
            self.query_one(InputBar).set_busy(False)
        except Exception:
            pass
        session.messages = engine.get_history()
        try:
            save_session(session)
        except Exception as e:
            self.notify(f"Failed to save session: {e}", severity="error")
        # reset the per-turn flags so they never leak into a later session's turn
        self._turn_had_text = False
        self._turn_had_reasoning = False
        self._turn_had_error = False
        self._turn_had_tools = False
        self._turn_interrupted = False
        self.query_one(InputBar).focus()

    # -- command handling -------------------------------------------------
    def _run_command(self, line: str) -> None:
        from ..commands import handle_command
        from ..commands import CommandContext

        # /models is a full-screen, live model list. The bare form is already
        # intercepted by the command popup; with arguments it would fall through
        # to the sync fetch_zen_models() in commands.py and freeze the UI thread,
        # so route it to the picker (which fetches off-thread) instead.
        name = line[1:].split(maxsplit=1)[0] if line.startswith("/") else ""
        if name == "models":
            self._open_model_picker()
            return
        # Mutating commands must not run mid-turn: they'd race the running
        # engine (e.g. /undo popping the undo stack the worker is appending to).
        if self._busy and name not in _SAFE_WHILE_BUSY:
            self._chat_for(self._current_session_id).append_meta(
                "⏳ still working on the previous request…"
            )
            self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
            return

        engine = self._active_engine()
        session = self._active_session()

        def reply(text: str) -> None:
            # persistent chat output + a short toast; /models & friends must
            # not vanish into a transient notification
            self._chat_for(self._current_session_id).append_meta(text)
            self.notify(text.splitlines()[0][:60] if text else "", timeout=3)

        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=session,
            engine=engine,
            worktree=str(self.directory),
            reply=reply,
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, line)
        self._update_header()

    def _preview_command(self, name: str) -> str:
        """Run a read-only command with a collecting reply and return its output."""
        from ..commands import CommandContext, handle_command

        collected: list[str] = []
        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=self._active_session(),
            engine=self._active_engine(),
            worktree=str(self.directory),
            reply=collected.append,
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, f"/{name}")
        return "\n".join(collected)

    def on_command_selected(self, event: CommandSelected) -> None:
        # /models is the full-screen, live-updating provider model list
        if event.name == "models":
            self._open_model_picker()
            return
        cmd = self.command_registry.get(event.name)
        content: str | None = None
        if cmd is not None and cmd.preview:
            content = self._preview_command(event.name).strip() or cmd.description

        def done(result: str | None) -> None:
            bar = self.query_one(InputBar)
            bar.input.focus()
            if result == "run":
                self._run_command(f"/{event.name}")
            else:
                # Esc = back: put the command back in the input, cursor at the end
                bar.input.value = f"/{event.name}"
                bar.input.cursor_position = len(bar.input.value)

        self.push_screen(
            CommandPopup(event.name, event.description, content=content),
            done,
        )

    def _open_model_picker(self) -> None:
        def on_picked(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            provider, _, model = choice.partition("/")
            if not provider or not model:
                return
            self.cfg.provider = provider
            self.cfg.model = model
            engine = self._active_engine()
            engine.provider_id = provider
            engine.model_id = model
            engine.rebuild_rotation()
            self.notify(f"Model set to {provider}/{model}")
            self._update_header()
            from ..config import save_config

            save_config(self.cfg)

        self.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )

    def _open_connect(self, provider: str = "") -> None:
        self.app.push_screen(
            ConnectScreen(auth=self.auth, on_connected=self._on_connected, initial=provider),
            self._on_connect_dismissed,
        )

    def _on_connected(self, provider_id: str) -> None:
        self.notify(f"Saved API key for {provider_id}.")
        self._update_header()

    def _on_connect_dismissed(self, result: str | None) -> None:
        if result:
            self.notify(f"Connected {result}.")

    # -- actions ---------------------------------------------------------
    def on_agent_toggle_requested(self, event: AgentToggleRequested) -> None:
        self.action_toggle_agent()

    def _interrupt_requested(self) -> bool:
        return bool(self._interrupt_flag["requested"])

    def action_interrupt(self) -> None:
        # Flipping the shared flag makes run_turn stop at its next iteration
        # check (loop.py). The worker thread's `finally` then calls _turn_done,
        # which resets the flag and clears _busy — we must NOT call _turn_done
        # here, or the worker would finish concurrently and double-complete.
        if self._busy:
            self.notify("Interrupting...")
            self._interrupt_flag["requested"] = True

    # -- permission dialog (engine thread -> UI) --------------------------
    def _permission_ask(self, description: str, always_patterns: list[str]) -> str:
        """Bridge the engine thread's permission.ask to a modal dialog.

        Runs on the engine worker thread. Pushes the dialog on the UI thread via
        call_from_thread (which blocks until the push returns), then waits for the
        user's decision. Returns "once" / "always" / "reject".
        """
        outcome: dict[str, str] = {}
        decided = threading.Event()

        def on_decision(decision: str) -> None:
            outcome["decision"] = decision
            decided.set()

        try:
            self.call_from_thread(
                self._show_permission_dialog, description, on_decision
            )
        except Exception:
            return "reject"
        # The dialog stays open until the user answers; polling the shutdown
        # event means quitting the app unblocks the engine thread immediately
        # instead of hanging for the timeout below.
        while not self._exit_requested.is_set():
            if decided.wait(timeout=0.5):
                break
        return outcome.get("decision", "reject")

    def _show_permission_dialog(
        self, description: str, on_decision: Any
    ) -> None:
        if not self.is_attached:
            on_decision("deny")
            return
        self.push_screen(PermissionDialog(description, on_decision=on_decision))

    def on_exit_app(self) -> None:
        """Fired when the app quits (ctrl+q / /exit).

        Unblocks any engine thread waiting on a permission dialog and saves the
        in-flight turn's history so an exit mid-turn doesn't lose it.
        """
        self._exit_requested.set()
        if self._busy:
            try:
                sid = self._active_turn_session_id
                engine = self._engines.get(sid)
                sess = self._sessions.get(sid)
                if engine is not None and sess is not None:
                    sess.messages = engine.get_history()
                    save_session(sess)
            except Exception:
                pass

    def action_resume(self) -> None:
        self.query_one(InputBar).focus()

    def action_toggle_agent(self) -> None:
        engine = self._active_engine()
        self._set_agent("plan" if engine.agent != "plan" else "build")

    def action_settings(self) -> None:
        from .settings_screen import SettingsScreen

        self.push_screen(
            SettingsScreen(
                cfg=self.cfg,
                engine=self.engine,
                auth=self.auth,
                session=self.session,
                on_model_change=self._set_model,
            ),
            self._on_settings_done,
        )

    def _on_settings_done(self, result: Any) -> None:
        self.query_one(InputBar).focus()

    def _on_model_picked(self, model: str | None) -> None:
        if model:
            self._set_model(model)

    def action_focus_input(self) -> None:
        self.query_one(InputBar).focus()

    def action_toggle_thought(self) -> None:
        chat = self._chat_for(self._current_session_id)
        chat.toggle_last_reasoning()
        self.query_one(InputBar).focus()

    def _set_model(self, model: str) -> None:
        self.cfg.model = model
        engine = self._active_engine()
        engine.model_id = model
        engine.rebuild_rotation()
        self.notify(f"Model set to opencode/{model}")
        self._update_header()

    def _set_agent(self, agent: str) -> None:
        engine = self._active_engine()
        engine.agent = agent
        sess = self._sessions.get(self._current_session_id)
        if sess is not None:
            sess.agent = agent
        self.notify(f"Agent: {agent}")
        self._update_header()

    def _update_header(self) -> None:
        status = self.query_one(StatusBar)
        engine = self._active_engine()
        header = {
            "agent": engine.agent,
            "model": self.cfg.model,
            "provider": self.cfg.provider,
            "permission_mode": engine.permission.mode,
        }
        status.set_header(**header)
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        if hasattr(bar, "set_header"):
            bar.set_header(**header)


def run_tui(cfg: Config | None = None, directory: Path | None = None) -> None:
    OpenCodeTUI(cfg=cfg, directory=directory).run()


if __name__ == "__main__":
    run_tui()
