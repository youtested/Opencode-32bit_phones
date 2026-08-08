"""Main agentic loop.

Flow per turn:
  1. build messages (system + trimmed history + latest user turn + agent reminder)
  2. stream model -> emit text/reasoning/tool_call events to on_event
  3. if tool calls: permission check -> run tool -> append tool result; loop again
  4. cap iterations (safety) and honor interrupt

build agent: full tools. plan agent: edit/write/bash denied by permissions.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..globals import resolve_worktree
from ..permission import PermissionEngine, merge_permissions
from ..providers import ProviderError, RateLimitError, build_rotation
from ..session import new_session, save_session
from ..tools.registry import Registry
from . import messages as msg_mod
from . import parse as parse_mod
from . import system as system_mod

MAX_STEPS = 50
MAX_UNDO = 20


def _missing_directories(path: Path) -> list[Path]:
    """Return the chain of directories (deepest-first) that don't exist yet,
    walking from `path` upward. Used to undo mkdir(parents=True) side effects."""
    missing: list[Path] = []
    p = path
    while str(p) not in ("", ".") and not p.exists():
        missing.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return missing


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    tool_calls_made: int = 0
    usage: dict[str, int] | None = None
    provider_id: str = ""
    model_id: str = ""
    error: str = ""


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


class AgentLoop:
    def __init__(
        self,
        *,
        cfg: Config,
        registry: Registry,
        directory: Path,
        provider=None,
        auth=None,
        permission_engine: PermissionEngine | None = None,
        on_event: Callable[[dict], None] | None = None,
        agent: str = "build",
        provider_id: str = "opencode",
        model_id: str = "",
        interrupt: Callable[[], bool] | None = None,
        session_id: str | None = None,
        provider_factory: Callable[[], Any] | None = None,
        read_paths: set[str] | None = None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.directory = directory
        self.worktree = resolve_worktree(directory)
        self.auth = auth
        self.agent = agent
        self._prev_agent: str = agent
        self._turn_agent: str = agent
        self.provider_id = provider_id
        self.model_id = model_id
        self.interrupt = interrupt or (lambda: False)
        self.session_id = session_id or uuid.uuid4().hex
        # factory used to build rotations for spawned sub-agents (override in
        # tests); default matches the parent's own rotation construction.
        self.provider_factory = provider_factory or (lambda: build_rotation(cfg, auth))

        self.rotation = provider or build_rotation(cfg, auth)

        self.permission = permission_engine or PermissionEngine.from_config(
            merge_permissions(cfg.permission, agent),
            mode="auto",
        )
        self._events: list[dict] = []
        self.on_event = on_event
        self._history: list[dict] = []
        self._call_seq = 0
        self._read_paths: set[str] = set(read_paths or ())
        self._pending_calls: list[dict] = []
        self._undo_stack: list[dict] = []
        self._usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self.subagents: dict[str, "AgentLoop"] = {}
        # the task tool looks this up lazily so sub-agents can nest
        self.registry.task_spawner = self.spawn_task

    def find_subagent(self, session_id: str) -> "AgentLoop | None":
        """Depth-first search for a sub-agent by session id (nested included)."""
        stack = [self]
        while stack:
            loop = stack.pop()
            if loop.session_id == session_id:
                return loop
            stack.extend(loop.subagents.values())
        return None

    # -- event plumbing --------------------------------------------------
    def _emit(self, kind: str, **kwargs: Any) -> None:
        event = {"kind": kind, **kwargs}
        self._events.append(event)
        if self.on_event:
            self.on_event(event)

    def rebuild_rotation(self) -> None:
        """Rebuild the failover lanes from the current config.

        The rotation is built once at startup, so picking a different model or
        provider at runtime would otherwise keep using the old lanes. Call this
        whenever `cfg.model` / `cfg.provider` change so the next turn uses the
        newly picked model.
        """
        self.rotation = build_rotation(self.cfg, self.auth)

    def _emit_tool(self, kind: str, tool: str, **kwargs: Any) -> None:
        self._emit(kind, tool=tool, **kwargs)

    # -- permission ------------------------------------------------------
    def check_permission(self, tool: str, input_value: str, display: str, call_id: str = "", arguments: dict | None = None) -> bool:
        permission_name = tool
        if tool in ("write", "edit", "apply_patch"):
            permission_name = "edit"
        action = self.permission.evaluate(permission_name, input_value)
        if action == "allow":
            return True
        kwargs: dict[str, Any] = {"reason": "denied by permission", "call_id": call_id}
        if action == "deny":
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
            return False
        # ask
        always_patterns: list[str] = ["*"]
        allowed = self.permission.ask(display, always_patterns)
        if not allowed:
            kwargs = {"reason": "rejected by user", "call_id": call_id}
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
        return allowed

    # -- read-before-edit guard ------------------------------------------
    def _ensure_read_before_edit(self, file_path: str) -> bool:
        """Enforce opencode's 'must Read first' rule for edit/write on existing files."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.directory / path
        if not path.exists():
            return True  # write to a new file is fine
        # tool call tracking: we keep a set of read paths fed by the loop
        return str(path.resolve()) in self._read_paths

    # -- tools ------------------------------------------------------------
    def run_tool(self, name: str, arguments: dict, call_id: str = "") -> dict[str, Any]:
        tool = self.registry.get(name)
        if tool is None:
            return {"output": f"Unknown tool: {name}", "error": True}
        input_value = json.dumps(arguments, sort_keys=True)
        if name == "task":
            # opencode's task input id is the sub-agent type, so a plan
            # agent's `task: {grant: deny}` rule actually matches.
            input_value = str(arguments.get("subagent_type", "build"))
        display = f"{name} {input_value[:120]}"

        if not self.check_permission(name, input_value, display, call_id=call_id, arguments=arguments):
            return {
                "output": f"Permission denied for {name}. Tell the user what to do differently.",
                "error": True,
                "denied": True,
            }

        mutates = name in ("edit", "write", "apply_patch")
        file_path = arguments.get("filePath")
        snapshot: bytes | None = None
        snapshot_path: Path | None = None
        created_dirs: list[Path] = []
        if mutates and file_path:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.directory / p
            snapshot_path = p
            snapshot = p.read_bytes() if p.exists() else None
            # record which parent directories would be newly created by the
            # write's mkdir(parents=True) so undo can clean them back up.
            if snapshot is None:
                created_dirs = _missing_directories(p.parent)

        self._emit_tool("tool_start", name, input=arguments, status="running", call_id=call_id)
        try:
            result = tool.run(arguments)
        except Exception as e:
            result = {"output": f"{name} failed: {e}", "error": True}
        if mutates and snapshot_path is not None:
            self._undo_stack.append(
                {
                    "path": str(snapshot_path),
                    "original": snapshot,
                    "dirs": [str(d) for d in created_dirs],
                }
            )
            if len(self._undo_stack) > MAX_UNDO:
                self._undo_stack.pop(0)
        self._emit_tool(
            "tool_complete",
            name,
            input=arguments,
            status="error" if result.get("error") else "completed",
            output=result.get("output", ""),
            metadata=result.get("metadata", {}),
            call_id=call_id,
        )
        if name == "read":
            file_path = arguments.get("filePath")
            if file_path:
                p = Path(file_path)
                if not p.is_absolute():
                    p = self.directory / p
                self._read_paths.add(str(p.resolve()))
        return result

    # -- sub-agents -------------------------------------------------------
    def _subagent_bridge(self, sub_id: str) -> Callable[[dict], None]:
        """Forward a sub-agent's events to our own on_event, tagged with the
        sub-session id so the UI can route them to the right chat view.

        Nested sub-agents already carry their own `session_id` (tagged by the
        deeper bridge); keep that id so a grandchild's events reach its own chat
        instead of being re-tagged with the direct child's id.
        """

        def forward(event: dict[str, Any]) -> None:
            kind = event.get("kind", "")
            sid = event.get("session_id") or sub_id
            payload = {k: v for k, v in event.items() if k not in ("kind", "session_id")}
            self._emit(kind, session_id=sid, **payload)

        return forward

    def spawn_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a sub-agent in its own session; returns the sub-agent's reply.

        The sub-agent gets its own history/session but shares the parent's
        config, auth, permission engine, directory, and interrupt flag. Events
        stream out tagged with the sub-session id. Nested `task` calls work:
        sub-agents build their own registry, which lazily resolves the same hook.
        """
        from ..tools import build_registry

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return {"output": "task: no prompt provided.", "error": True}
        description = str(arguments.get("description", "")).strip() or "sub-agent"
        subagent_type = str(arguments.get("subagent_type", "")).strip() or "build"

        # A read-only plan agent must not spawn a build sub-agent: the child
        # would inherit read-write tools. Force the child to stay read-only.
        if self.agent == "plan" and subagent_type != "plan":
            subagent_type = "plan"

        sub_session = new_session(
            directory=str(self.directory),
            provider=self.provider_id or self.cfg.provider,
            model=self.model_id or self.cfg.model,
            agent=subagent_type,
            title=description,
            parent_id=self.session_id,
        )
        save_session(sub_session)

        sub = AgentLoop(
            cfg=self.cfg,
            registry=build_registry(self.cfg),
            directory=self.directory,
            provider=self.provider_factory(),
            auth=self.auth,
            permission_engine=self.permission,
            on_event=self._subagent_bridge(sub_session.id),
            agent=subagent_type,
            provider_id=self.provider_id,
            model_id=self.model_id,
            interrupt=self.interrupt,
            session_id=sub_session.id,
            provider_factory=self.provider_factory,
            read_paths=set(self._read_paths),
        )
        self.subagents[sub_session.id] = sub
        self._emit("subagent_start", session_id=sub_session.id, agent=subagent_type, title=description)

        try:
            result = sub.run_turn(prompt)
        except Exception as e:
            # never leave the sub-agent session dangling: persist and report done
            sub_session.messages = sub.get_history()
            sub_session.completed = time.time()
            try:
                save_session(sub_session)
            except Exception:
                pass
            self._emit(
                "subagent_done",
                session_id=sub_session.id,
                agent=subagent_type,
                title=description,
                ok=False,
            )
            return {
                "output": f"sub-agent failed: {e}",
                "error": True,
                "metadata": {
                    "sessionId": sub_session.id,
                    "title": description,
                    "status": "error",
                },
            }
        sub_session.messages = sub.get_history()
        sub_session.completed = time.time()
        try:
            save_session(sub_session)
        except Exception:
            pass

        self._emit(
            "subagent_done",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            ok=not result.error,
        )
        text = result.text or result.error or "(no reply from sub-agent)"
        return {
            "output": text,
            "error": bool(result.error and not result.text),
            "metadata": {
                "sessionId": sub_session.id,
                "title": description,
                "status": "error" if result.error else "completed",
            },
        }

    # -- main turn --------------------------------------------------------
    def run_turn(self, user_text: str) -> TurnResult:
        result = TurnResult()
        self._read_paths = set()
        reset = getattr(self.permission, "reset_doom_tracking", None)
        if reset:
            reset()

        self._history.append({"role": "user", "content": user_text})

        # agent reminder (plan/build-switch)
        reminder = system_mod.agent_reminder(self.agent, self._was_plan())
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )

        history = list(self._history)
        messages = msg_mod.build_messages(history=history[:-1], user_text=user_text, reminder=reminder)
        messages = self._prepend_system(messages, system_prompt)
        messages = msg_mod.trim_history(messages, self.cfg.context_budget)

        tools = self._active_tool_schemas()

        for step in range(MAX_STEPS):
            if self.interrupt():
                self._emit("interrupted")
                break

            self._emit("step", step=step)
            # stream
            self._stream(messages, tools, result, system_prompt)

            if result.error:
                break

            # collect tool calls
            if self._pending_calls:
                calls = self._pending_calls
                self._pending_calls = []
                # Drop degenerate calls (missing name); a model that only emits
                # empty tool calls would otherwise spin a silent 50-step loop.
                calls = [c for c in calls if c.get("name")]
                if not calls:
                    result.error = "model produced an invalid tool call (missing name)"
                    self._emit("error", error=result.error)
                    break
            else:
                # The model replied with text but made no tool call — this is a
                # valid final answer, accept it and end the turn. No retry/nudge
                # loops (they cause multiple model completions for one prompt).
                break

            # Some models emit tool calls without an id. Assign a stable
            # fallback (must match assistant_message_from_calls) so the
            # assistant declaration and the following tool-result messages use
            # the same id; otherwise strict OpenAI-compatible backends reject
            # with "insufficient tool messages following tool_calls". The
            # counter keeps ids unique across every step and turn so the UI can
            # reliably match tool rows by call_id (per-step indices collide and
            # cause duplicate/incorrect tool rows).
            for call in calls:
                if not call.get("id"):
                    self._call_seq += 1
                    call["id"] = f"call_{self._call_seq}"

            # append assistant message with calls to history
            self._history.append(parse_mod.assistant_message_from_calls(calls))
            messages = list(self._history)

            for call in calls:
                if self.interrupt():
                    self._emit("interrupted")
                    return result
                name = call.get("name", "")
                try:
                    arguments = parse_mod.parse_arguments(call.get("arguments", "{}"))
                except Exception:
                    arguments = {"arguments": call.get("arguments", "{}")}

                self._emit_tool("tool_call", name, arguments=arguments, call_id=call.get("id", ""))
                if name in ("edit", "write"):
                    fp = arguments.get("filePath", "")
                    if not self._ensure_read_before_edit(fp):
                        self._emit_tool("tool_denied", name, reason="file not read first", call_id=call.get("id", ""), input=arguments)
                        self._history.append(
                            parse_mod.tool_result_message(
                                call.get("id", ""),
                                name,
                                "You must use the read tool before editing this file.",
                                error=True,
                            )
                        )
                        continue
                tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
                result.tool_calls_made += 1
                self._history.append(
                    parse_mod.tool_result_message(
                        call.get("id", ""),
                        name,
                        tool_result.get("output", ""),
                        error=bool(tool_result.get("error")),
                    )
                )
                messages = list(self._history)

        if result.error:
            self._history.append({"role": "user", "content": f"[system] {result.error}"})
        # Track agent state for build-switch detection
        self._prev_agent = self.agent
        return result

    # -- streaming --------------------------------------------------------
    def _stream(self, messages, tools, result: TurnResult, system_prompt: str) -> None:
        self._pending_calls = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        # Tool-loop requests are rebuilt from raw history (which never holds the
        # system prompt); re-prepend it so every request is well-formed.
        if not messages or messages[0].get("role") != "system":
            messages = self._prepend_system(messages, system_prompt)

        def on_event(evt) -> None:
            kind = evt.kind
            if kind == "text_delta":
                text_parts.append(evt.text)
                self._emit("text_delta", text=evt.text)
            elif kind == "reasoning_delta":
                reasoning_parts.append(evt.text)
                self._emit("reasoning_delta", text=evt.text)
            elif kind == "tool_call":
                for tc in evt.tool_calls or []:
                    self._pending_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            elif kind == "usage":
                u = evt.usage
                # Context-window usage = the last completion's input (full
                # conversation context) + its output — mirrors opencode's
                # input+output+reasoning+cache. Summing across streams would
                # double-count history on multi-step tool loops.
                self._usage_total["input_tokens"] = u.input_tokens
                self._usage_total["output_tokens"] = u.output_tokens
                self._usage_total["total_tokens"] = u.input_tokens + u.output_tokens
                result.usage = dict(self._usage_total)
                usage_evt = dict(self._usage_total)
                ctx = self._model_context_size()
                if ctx:
                    result.usage["context_size"] = ctx
                    usage_evt["context_size"] = ctx
                self._emit("usage", usage=usage_evt)
            elif kind == "error":
                result.error = evt.error
                self._emit("error", error=evt.error)
            elif kind == "done":
                pass

        try:
            provider_id, model_id = self.rotation.stream(
                messages, tools, on_event, on_notice=self._on_rotated
            )
            result.provider_id = provider_id
            result.model_id = model_id
        except RateLimitError as e:
            result.error = f"rate limit: {e}"
            self._emit("error", error=result.error, retryable=True)
        except ProviderError as e:
            result.error = str(e)
            self._emit("error", error=result.error, retryable=bool(e.retryable))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result.error = str(e)
            self._emit("error", error=result.error)

        result.text = "".join(text_parts)
        result.reasoning = "".join(reasoning_parts)
        # append assistant text to history if any
        if result.text:
            if result.reasoning:
                self._history.append({"role": "assistant", "content": result.text, "reasoning_content": result.reasoning})
            else:
                self._history.append({"role": "assistant", "content": result.text})
        elif self._pending_calls:
            pass  # assistant message with calls added by caller
        elif result.error:
            pass
        elif result.reasoning:
            # reasoning-only reply: keep it in history (never an empty message,
            # which strict backends reject as malformed)
            self._history.append(
                {"role": "assistant", "content": result.reasoning, "reasoning_content": result.reasoning}
            )
        else:
            # keep role alternation valid and avoid an empty-content message
            self._history.append({"role": "assistant", "content": "(no response)"})

    def undo_last(self) -> str:
        """Revert the most recent edit/write tool call (file-level snapshot)."""
        if not self._undo_stack:
            return "Nothing to undo."
        entry = self._undo_stack.pop()
        path = Path(entry["path"])
        try:
            if entry["original"] is None:
                if path.exists():
                    path.unlink()
                dirs = entry.get("dirs") or []
                for d in reversed(dirs):
                    d = Path(d)
                    # only remove directories we created and that are now empty
                    try:
                        if d.is_dir() and not any(d.iterdir()):
                            d.rmdir()
                    except OSError:
                        pass
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry["original"])
        except OSError as e:
            return f"Undo failed for {path}: {e}"
        self._emit("undo", path=str(path))
        return f"Reverted {path}."

    def _on_rotated(self, provider_id: str, model_id: str, reason: str = "") -> None:
        """A failover lane succeeded; announce it (with the switch reason)."""
        self._emit("rotated", provider=provider_id, model=model_id, reason=reason)

    def _model_context_size(self) -> int:
        """Context-window size of the active lane (0 when unknown)."""
        from ..providers import model_context_size

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_context_size(pid, mid)

    def _active_tool_schemas(self) -> list[dict]:
        schemas = self.registry.schemas()
        if self.agent == "plan":
            schemas = [
                s
                for s in schemas
                if s["function"]["name"] not in ("bash", "write", "edit", "apply_patch")
            ]
        return schemas

    def _was_plan(self) -> bool:
        return self._prev_agent == "plan"

    def _prepend_system(self, messages: list[dict], system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}] + messages

    # -- session glue -----------------------------------------------------
    def set_history(self, history: list[dict]) -> None:
        self._history = list(history)
        if not history:
            self._usage_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def get_history(self) -> list[dict]:
        return list(self._history)

    def add_placeholder_tool_message(self, output: str) -> None:
        self._history.append({"role": "assistant", "content": output})
