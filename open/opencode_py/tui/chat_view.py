"""Chat view: scrollable message list mirroring opencode's session screen.

Rendering mirrors opencode's TUI (packages/tui/src/routes/session/index.tsx):

  - User messages are a full-width block with a single left border strip in the
    agent accent color, a `backgroundPanel` fill and padding (no title), plus an
    optional ` QUEUED ` badge when the turn hasn't started yet.
  - Assistant text flows as plain markdown indented from the left with a block
    cursor (▍) while streaming, then a muted `▣ Build · model` mode line.
  - Reasoning streams as a spinner `Thinking...` and collapses to a clickable
    `+ Thought: <title>` line (opencode's hide mode); clicking toggles the body.
  - Tools render as compact inline rows (`{icon} {label}`, spinner while running)
    or, for tools that produce a result block (bash output, edit diff, todos,
    questions, apply-patch), a subtle left-bordered block on the panel background.
    Per-tool rendering mirrors opencode: Read shows `↳ Loaded <file>`, Glob/Grep
    show `(N matches)`, etc.
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from .theme import get_theme

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.08  # opencode uses 80ms


class OpenTaskSession(Message):
    """A completed task tool row was clicked; open that sub-session."""

    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid

# tool -> icon. Mirrors opencode's InlineTool usage.
TOOL_ICONS = {
    "bash": "$",
    "shell": "$",
    "execute": "$",
    "read": "→",
    "write": "←",
    "glob": "✱",
    "grep": "✱",
    "webfetch": "%",
    "websearch": "◈",
    "edit": "←",
    "apply_patch": "%",
    "todowrite": "⚙",
    "task": "│",
    "question": "→",
    "skill": "→",
    "mcp": "⊙",
    "notify": "·",
}
# tool -> label used in inline rows / titles.
TOOL_NAMES = {
    "bash": "Shell",
    "shell": "Shell",
    "execute": "Execute",
    "read": "Read",
    "write": "Write",
    "glob": "Glob",
    "grep": "Grep",
    "webfetch": "WebFetch",
    "websearch": "WebSearch",
    "edit": "Edit",
    "apply_patch": "Apply Patch",
    "todowrite": "TodoWrite",
    "task": "Task",
}

# tools that have a dedicated renderer in opencode (everything else is "generic").
_TOOL_DISPLAYS = {
    "bash",
    "glob",
    "read",
    "grep",
    "webfetch",
    "websearch",
    "write",
    "edit",
    "task",
    "apply_patch",
    "todowrite",
    "question",
    "skill",
    "execute",
}

# default markdown styling, mirrored from opencode's theme map
MD = {
    "heading": "markdown_heading",
    "strong": "markdown_strong",
    "bullet": "markdown_list_item",
    "code": "markdown_code",
    "hr": "markdown_hr",
}


def _plain(content: Any) -> RenderableType:
    """Plain flowing markdown with no surrounding box (assistant text)."""
    theme = get_theme("opencode")
    lines = str(content).splitlines()
    out: list[RenderableType] = []
    in_code = False
    buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append(Text("\n".join(buf), style=theme.c("markdown_code_block")))
                buf = []
            in_code = not in_code
            continue
        if in_code:
            buf.append(line)
            continue
        if stripped.startswith("#"):
            out.append(Text(line, style=f"bold {theme.c(MD['heading'])}"))
        elif stripped.startswith(("-", "*", "+", "•")):
            out.append(Text(f"• {line.lstrip(' -*+•')}", style=theme.c("text")))
        elif line.strip():
            out.append(Text(line, style=theme.c("text")))
    if in_code and buf:
        out.append(Text("\n".join(buf), style=theme.c("markdown_code_block")))
    return Group(*out)


def _render_diff(diff_text: str) -> RenderableType:
    theme = get_theme("opencode")
    lines: list[RenderableType] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            lines.append(Text(line, style=theme.c("diff_context")))
        elif line.startswith("@"):
            lines.append(Text(line, style=f"bold {theme.c('diff_hunk_header')} on {theme.c('diff_context_bg')}"))
        elif line.startswith("+"):
            lines.append(Text(line, style=f"{theme.c('diff_added')} on {theme.c('diff_added_bg')}"))
        elif line.startswith("-"):
            lines.append(Text(line, style=f"{theme.c('diff_removed')} on {theme.c('diff_removed_bg')}"))
        else:
            lines.append(Text(line, style=theme.c("text")))
    return Group(*lines)


def collapse_tool_output(output: str, max_lines: int, max_chars: int) -> dict:
    """Mirror opencode's collapse-tool-output: cap lines and chars with '…'."""
    lines = output.split("\n")
    if len(lines) <= max_lines and len(output) <= max_chars:
        return {"output": output, "overflow": False}
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        return {"output": preview[: max(0, max_chars - 1)] + "…", "overflow": True}
    return {"output": "\n".join(lines[:max_lines] + ["…"]), "overflow": True}


def _format_input(inp: dict[str, Any], omit: tuple[str, ...] = ()) -> str:
    """Mirror opencode's `input()` helper: `[key=value, key2=value2]`."""
    parts = []
    for key, value in inp.items():
        if key in omit:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    if not parts:
        return ""
    return "[" + ", ".join(parts) + "]"


def reasoning_summary(text: str) -> dict:
    """Mirror opencode's reasoningSummary: extract a bold `**Title**` block."""
    content = str(text).strip()
    match = re.match(r"^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)", content)
    if not match:
        return {"title": None, "body": content}
    return {"title": match.group(1).strip(), "body": content[match.end():].strip()}


def _tool_display(tool: str) -> str:
    return tool if tool in _TOOL_DISPLAYS else "generic"


class MessageBubble(Static):
    """One chat element (user, assistant, reasoning, mode-line, meta) or a tool run."""

    queued: reactive = reactive(False)  # user message waiting for the turn to start
    streaming: reactive = reactive(False)
    expanded: reactive = reactive(False)  # reasoning body collapsed/expanded

    def __init__(
        self,
        role: str,
        content: Any = "",
        agent: str = "build",
        queued: bool = False,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        self.role = role
        self.agent = agent
        self._message = content
        self._spinner = 0
        self._timer: Any = None
        super().__init__("", **kwargs)
        self.can_focus = role == "reasoning"
        self.set_reactive(MessageBubble.queued, queued)
        self.set_reactive(MessageBubble.streaming, streaming)
        self._refresh()

    def watch_queued(self, value: bool) -> None:
        self._refresh()

    def watch_streaming(self, value: bool) -> None:
        self._refresh()

    def watch_expanded(self, value: bool) -> None:
        self._refresh()

    @property
    def content(self) -> Any:
        """Raw payload (assistant text or tool-run dict)."""
        return self._message

    # -- content ----------------------------------------------------------
    def _build_content(self) -> RenderableType:
        theme = get_theme("opencode")
        if self.role == "user":
            inner: list[RenderableType] = [Text(str(self._message), style=theme.c("text"))]
            if self.queued:
                color = theme.agent_color(self.agent)
                inner.append(Text(f" QUEUED ", style=f"bold on {color}"))
            return Group(*inner)
        if self.role == "assistant":
            group = _plain(self._message)
            if self.streaming:
                return Group(group, Text("▍", style=theme.c("primary")))
            return group
        if self.role == "reasoning":
            return self._build_reasoning()
        if self.role == "assistant_mode":
            # e.g. `▣ Build · model`
            t = Text()
            t.append("▣ ", style=theme.agent_color(self.agent))
            t.append(self.agent.title(), style=theme.c("text"))
            if self._message:
                t.append(f" · {self._message}", style=theme.c("text_muted"))
            return t
        if self.role == "meta":
            return Text(str(self._message), style=theme.c("text_muted"))
        return self._render_tool(self._message) if isinstance(self._message, dict) else _plain(self._message)

    # -- reasoning (mirrors opencode's ReasoningPart, thinking mode "hide") --
    def _build_reasoning(self) -> RenderableType:
        """Collapsed by default: `+ Thought: <title>`. Streaming shows a spinner
        with `Thinking...`; clicking the header toggles the muted markdown body
        (also while the thought is still streaming)."""
        theme = get_theme("opencode")
        summary = reasoning_summary(self._message)
        title = summary["title"]
        body = summary["body"]
        prefix = "- " if self.expanded else "+ "

        if self.streaming:
            header = Text(
                f"{prefix}{SPINNER_FRAMES[self._spinner]} Thinking" + (f": {title}" if title else ""),
                style=theme.c("warning"),
            )
        else:
            header = Text(prefix, style=theme.c("warning"))
            header.append("Thought", style=theme.c("warning"))
            if title:
                header.append(f": {title}", style=theme.c("warning"))
        parts: list[RenderableType] = [header]
        if self.expanded and body:
            parts.append(Text(body, style=theme.c("text_muted")))
        return Group(*parts)

    # -- tool rendering (mirrors opencode's per-tool components) ----------
    def _render_tool(self, tool_run: dict[str, Any]) -> RenderableType:
        display = _tool_display(tool_run.get("tool", "?"))
        fn = getattr(self, f"_render_{display}", None)
        if fn is None:
            fn = self._render_generic
        return fn(tool_run)

    def _status(self, run: dict[str, Any]) -> str:
        return run.get("status", "pending")

    def _error(self, run: dict[str, Any]) -> str:
        return run.get("error") or ""

    def _denied(self, run: dict[str, Any]) -> bool:
        # permission denials may arrive as an error field or as output text
        err = run.get("error") or (run.get("output") or "")
        return any(
            m in err
            for m in (
                "QuestionRejectedError",
                "rejected permission",
                "permission denied",
                "denied by permission",
                "user dismissed",
                "specified a rule",
            )
        )

    def _failed(self, run: dict[str, Any]) -> bool:
        """Mirror opencode's InlineTool.failed: a real error that isn't a denial."""
        if self._denied(run):
            return False
        if run.get("status") == "error":
            return True
        return bool(run.get("error"))

    def _inline(
        self,
        icon: str,
        pending: str,
        label: str,
        *,
        spinner: bool = False,
        complete: bool | str | None = None,
    ) -> RenderableType:
        """Mirror opencode's InlineToolRow: ~ pending / spinner running / icon label.

        Colors mirror opencode: running -> primary, completed -> textMuted,
        failed -> error (red), denied -> strikethrough.
        """
        theme = get_theme("opencode")
        status = self._status(self._message)
        denied = self._denied(self._message)
        failed = self._failed(self._message)
        done = status in ("completed", "error") if complete is None else bool(complete)

        if status == "running" and spinner:
            color = theme.c("primary")
            return Text(f"{SPINNER_FRAMES[self._spinner]} {label}", style=color)
        if not done:
            return Text(f"~ {pending}", style=theme.c("text_muted"))

        color = theme.c("error") if failed else theme.c("text_muted")
        style = f"{color} strike" if denied else color
        return Text(f"{icon} {label}", style=style)

    def _error_line(self, run: dict[str, Any]) -> RenderableType | None:
        """A red error line for a failed tool (None when not a real error)."""
        if not self._failed(run) or self._denied(run):
            return None
        err = (self._error(run) or "").strip()
        if not err:
            return None
        return Text(err, style=get_theme("opencode").c("error"))

    def _render_bash(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        command = str((run.get("input") or {}).get("command", ""))
        output = (run.get("output") or "").strip()
        workdir = str((run.get("input") or {}).get("workdir") or "")
        if status == "running":
            return Text(f"{SPINNER_FRAMES[self._spinner]} {command}", style=theme.c("text"))
        if output:
            lines: list[RenderableType] = []
            if workdir and workdir != ".":
                lines.append(Text(f"# Running in {workdir}", style=theme.c("text_muted")))
            lines.append(Text(f"$ {command}", style=theme.c("text")))
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            lines.append(Text(output_text, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                lines.append(Text(label, style=theme.c("text_muted")))
            return Group(*lines)
        return self._inline("$", "Writing command...", command)

    def _render_read(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        filepath = str((run.get("input") or {}).get("filePath", ""))
        loaded = (run.get("metadata") or {}).get("loaded") or []
        extra = _format_input((run.get("input") or {}), omit=("filePath",))
        row = self._inline(
            "→",
            "Reading file...",
            f"Read {filepath}{extra}",
            spinner=status == "running",
        )
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if status == "completed" and loaded:
            sub = [
                Text(f"↳ Loaded {p}", style=theme.c("text_muted"))
                for p in (loaded if isinstance(loaded, list) else [loaded])
            ]
            return Group(row, *sub)
        return row

    def _render_glob(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("count")
        label = f'Glob "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Finding files...", label)
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_grep(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("matches")
        label = f'Grep "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Searching content...", label)
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_webfetch(self, run: dict[str, Any]) -> RenderableType:
        url = str((run.get("input") or {}).get("url", ""))
        row = self._inline("%", "Fetching from the web...", f"WebFetch {url}")
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_websearch(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        provider = meta.get("provider", "")
        query = str(inp.get("query", ""))
        count = meta.get("numResults")
        label = f'{provider + " " if provider else ""}"{query}"'
        if isinstance(count, int):
            label += f" ({count} results)"
        row = self._inline("◈", "Searching web...", label)
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_write(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        filepath = str((run.get("input") or {}).get("filePath", ""))
        written = (run.get("metadata") or {}).get("content")
        if self._status(run) == "completed" and written:
            content = str(written)
            collapsed = collapse_tool_output(content, 10, 10 * 80)
            output = content if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# Wrote {filepath}", style=theme.c("text_muted")),
                Text(output, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        return self._inline("←", "Preparing write...", f"Write {filepath}")

    def _render_edit(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        filepath = str((run.get("input") or {}).get("filePath", ""))
        diff = (run.get("metadata") or {}).get("diff")
        if self._status(run) == "completed" and diff:
            return Group(
                Text(f"← Edit {filepath}", style=theme.c("text_muted")),
                _render_diff(diff),
            )
        err = self._error_line(run)
        if err is not None:
            replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
            return Group(
                self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}"),
                err,
            )
        replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
        return self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}")

    def _render_apply_patch(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        files = (run.get("metadata") or {}).get("files") or []
        if self._status(run) == "completed" and files:
            lines: list[RenderableType] = []
            for f in files if isinstance(files, list) else [files]:
                if not isinstance(f, dict):
                    continue
                rel = f.get("relativePath", "")
                title = f"← Patched {rel}"
                if f.get("type") == "delete":
                    title = f"# Deleted {rel}"
                elif f.get("type") == "add":
                    title = f"# Created {rel}"
                lines.append(Text(title, style=theme.c("text_muted")))
                patch = f.get("patch")
                if patch:
                    lines.append(_render_diff(patch))
            return Group(*lines)
        return self._inline("%", "Preparing patch...", "Patch")

    def _render_todowrite(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        todos = (run.get("metadata") or {}).get("todos") or (run.get("input") or {}).get("todos") or []
        if self._status(run) == "completed" and todos:
            lines: list[RenderableType] = [Text("# Todos", style=theme.c("text_muted"))]
            for todo in todos if isinstance(todos, list) else [todos]:
                if not isinstance(todo, dict):
                    continue
                status = todo.get("status", "pending")
                mark = "✓" if status == "completed" else ("•" if status == "in_progress" else " ")
                color = theme.c("warning") if status == "in_progress" else theme.c("text_muted")
                lines.append(Text(f"[{mark}] {todo.get('content', '')}", style=color))
            return Group(*lines)
        return self._inline("⚙", "Updating todos...", "Updating todos...")

    def _render_question(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        questions = (run.get("input") or {}).get("questions") or []
        answers = (run.get("metadata") or {}).get("answers")
        count = len(questions) if isinstance(questions, list) else 0
        if self._status(run) == "completed" and answers is not None:
            lines: list[RenderableType] = [Text("# Questions", style=theme.c("text_muted"))]
            for i, q in enumerate(questions if isinstance(questions, list) else []):
                if not isinstance(q, dict):
                    continue
                lines.append(Text(str(q.get("question", "")), style=theme.c("text_muted")))
                ans = answers[i] if isinstance(answers, list) and i < len(answers) else None
                lines.append(
                    Text(
                        ", ".join(ans) if isinstance(ans, list) else str(ans) if ans else "(no answer)",
                        style=theme.c("text"),
                    )
                )
            return Group(*lines)
        return self._inline("→", "Asking questions...", f"Asked {count} question{'s' if count != 1 else ''}")

    def _render_task(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        description = str(inp.get("description", ""))
        icon = "✓" if status == "completed" else "│"
        content = description
        if meta.get("sessionId") and (meta.get("title")):
            content = f"{meta.get('title')}"
        row = self._inline(icon, "Delegating...", content, spinner=status == "running")
        sub = []
        title = str(inp.get("title", ""))
        if title:
            sub.append(Text(f"↳ {title}", style=theme.c("text_muted")))
        return Group(row, *sub) if sub else row

    def _render_execute(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        icon = "✓" if status == "completed" else "│"
        row = self._inline(icon, "execute", "execute", spinner=status == "running")
        calls = (run.get("metadata") or {}).get("toolCalls") or []
        sub = []
        for c in calls if isinstance(calls, list) else []:
            if not isinstance(c, dict):
                continue
            name = c.get("tool", "")
            args = _format_input(c.get("input") or {})
            failed = " (failed)" if c.get("status") == "error" else ""
            sub.append(Text(f"↳ {name}{args}{failed}", style=theme.c("text_muted")))
        return Group(row, *sub) if sub else row

    def _render_skill(self, run: dict[str, Any]) -> RenderableType:
        name = str((run.get("input") or {}).get("name", ""))
        return self._inline("→", "Loading skill...", f'Skill "{name}"')

    def _render_generic(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        tool = run.get("tool", "?")
        output = (run.get("output") or "").strip()
        if self._status(run) == "completed" and output:
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# {tool} {_format_input(run.get('input') or {})}".strip(), style=theme.c("text_muted")),
                Text(output_text, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        label = f"{tool} {_format_input(run.get('input') or {})}".strip()
        return self._inline("⚙", "Running...", label)

    # -- frame (border / background / padding) ----------------------------
    def _apply_frame(self) -> None:
        theme = get_theme("opencode")
        st = self.styles
        if self.role == "user":
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.agent_color(self.agent))
            st.padding = (1, 1, 1, 2)
        elif self.role == "tool" and self._tool_block():
            # opencode BlockTool: left border in the panel background, panel fill.
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.c("background"))
            st.padding = (1, 1, 1, 2)
        else:
            # indented plain text (assistant / mode-line / meta / inline tool).
            # use an invisible border so Textual never paints its default.
            st.background = "transparent"
            st.border_left = ("solid", theme.c("background"))
            st.padding = (0, 0, 0, 3)
        st.margin = (1, 0, 0, 0)

    def _tool_block(self) -> bool:
        """A completed tool run renders as a block iff it produced a result block."""
        if not isinstance(self._message, dict):
            return False
        run = self._message
        if run.get("status") != "completed":
            return False
        name = run.get("tool", "?")
        display = _tool_display(name)
        if display == "bash":
            return bool((run.get("output") or "").strip())
        if display == "edit":
            return bool((run.get("metadata") or {}).get("diff"))
        if display == "apply_patch":
            return bool((run.get("metadata") or {}).get("files"))
        if display == "todowrite":
            return bool((run.get("metadata") or {}).get("todos"))
        if display == "question":
            return (run.get("metadata") or {}).get("answers") is not None
        if display == "write":
            return bool((run.get("metadata") or {}).get("content"))
        if display == "generic":
            return bool((run.get("output") or "").strip())
        return False

    def _tool_collapse(self, run: dict[str, Any]) -> dict:
        """Mirror opencode's per-tool collapse limits (bash 10 lines, generic 3)."""
        output = (run.get("output") or "").strip()
        if not output:
            return {"output": output, "overflow": False}
        display = _tool_display(run.get("tool", "?"))
        if display == "bash":
            return collapse_tool_output(output, 10, 10 * 80)
        return collapse_tool_output(output, 3, 3 * 80)

    def _tool_overflow(self) -> bool:
        return isinstance(self._message, dict) and self._tool_collapse(self._message)["overflow"]

    # -- updates -----------------------------------------------------------
    def _refresh(self) -> None:
        self._apply_frame()
        self.update(self._build_content())

    def _start_spinner(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(SPINNER_INTERVAL, self._tick)

    def _stop_spinner(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        if self.role == "tool" and self._message.get("status") == "running":
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()
        elif self.role == "reasoning" and self.streaming:
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()

    def on_click(self, event: Any) -> None:
        if self.role == "reasoning":
            self.expanded = not self.expanded
        elif self.role == "tool":
            if self.content.get("tool") == "task":
                meta = self.content.get("metadata") or {}
                sid = meta.get("sessionId")
                if sid:
                    self.post_message(OpenTaskSession(sid))
            elif self._tool_overflow():
                self.expanded = not self.expanded

    def on_key(self, event: Any) -> None:
        if self.role == "reasoning" and self.has_focus and event.key in ("enter", "space"):
            event.stop()
            self.expanded = not self.expanded

    def update_tool(self, tool_run: dict[str, Any]) -> None:
        self._message = tool_run
        if tool_run.get("status") == "running":
            self._start_spinner()
        else:
            self._stop_spinner()
            self.streaming = False
        self._refresh()

    def update_text(self, text: str) -> None:
        self._message = text
        self._refresh()

    def update_reasoning(self, text: str) -> None:
        self._message = text
        self._refresh()

    def end_reasoning(self) -> None:
        self.streaming = False
        self._stop_spinner()
        # keep thoughts collapsed by default: show `+ Thought: <title>` and let
        # the user click / Enter / Ctrl+Shift+E to expand the full body
        self.expanded = False
        self._refresh()


class ChatView(VerticalScroll):
    """Scrollable list of message bubbles + streaming cursor."""

    messages: reactive = reactive([])
    streaming: reactive = reactive(False)
    streaming_text: reactive = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_bubble: MessageBubble | None = None
        self._reasoning_bubble: MessageBubble | None = None

    def append_user(self, text: str, agent: str = "build", queued: bool = False) -> None:
        bubble = MessageBubble("user", text, agent=agent)
        bubble.queued = queued
        self.mount(bubble)
        self.scroll_end(animate=False)

    def append_assistant(self, text: str) -> None:
        self.mount(MessageBubble("assistant", text))
        self.scroll_end(animate=False)

    def append_meta(self, text: str) -> None:
        """Persistent command/system output (e.g. /models, /help, /config)."""
        self.mount(MessageBubble("meta", text))
        self.scroll_end(animate=False)

    def append_tool(self, tool_run: dict[str, Any]) -> None:
        bubble = MessageBubble("tool", tool_run)
        if tool_run.get("status") == "running":
            # start the spinner right away so a fresh tool_call row isn't stuck
            # on a static frame until tool_start arrives
            bubble._start_spinner()
        self.mount(bubble)
        self.scroll_end(animate=False)

    def begin_stream(self) -> None:
        self._stream_bubble = MessageBubble("assistant", "")
        self._stream_bubble.streaming = True
        self.mount(self._stream_bubble)
        self.scroll_end(animate=False)

    def stream_delta(self, text: str) -> None:
        if self._stream_bubble is None:
            self.begin_stream()
        self._stream_bubble.update_text(self._stream_bubble.content + text)
        self.scroll_end(animate=False)

    def stream_reasoning_delta(self, text: str) -> None:
        if self._reasoning_bubble is None:
            bubble = MessageBubble("reasoning", "")
            bubble.streaming = True
            bubble._start_spinner()
            # chronological order: each new thought mounts below the previous
            # tool runs (not above a stale empty stream bubble)
            self.mount(bubble)
            self._reasoning_bubble = bubble
        self._reasoning_bubble.update_reasoning(self._reasoning_bubble.content + text)
        self.scroll_end(animate=False)

    def end_reasoning(self) -> None:
        if self._reasoning_bubble is not None:
            self._reasoning_bubble.end_reasoning()
            self._reasoning_bubble = None
        self.scroll_end(animate=False)

    def end_stream(self, text: str = "") -> None:
        if self._stream_bubble is not None:
            if text:
                self._stream_bubble.update_text(text)
            self._stream_bubble.streaming = False
            self._stream_bubble._refresh()
            self._stream_bubble = None
        self.scroll_end(animate=False)

    def remove_last_stream_bubble(self) -> None:
        """Remove the empty streaming bubble left behind when there's no reply.

        A bubble that already holds partial text is KEPT (only its streaming
        cursor is ended) so a mid-stream error doesn't discard model output.
        """
        target = self._stream_bubble
        if target is None:
            last_empty = None
            for child in reversed(tuple(self.query(MessageBubble))):
                if child.role == "assistant" and not child.content and not child.streaming:
                    last_empty = child
                    break
            target = last_empty
        if target is not None:
            if target.content:
                target.streaming = False
            else:
                try:
                    target.remove()
                except Exception:
                    pass
        self._stream_bubble = None
        self.scroll_end(animate=False)

    def find_tool(self, tool: str, call_id: str = "") -> MessageBubble | None:
        candidates = []
        for child in self.query(MessageBubble):
            if child.role == "tool" and child.content.get("tool") == tool:
                candidates.append(child)
        if not candidates:
            return None
        if not call_id:
            return candidates[0]
        for child in candidates:
            if child.content.get("call_id") == call_id:
                return child
        return None

    def update_tool_bubble(self, tool_run: dict[str, Any]) -> None:
        bubble = self.find_tool(tool_run.get("tool", ""), tool_run.get("call_id", ""))
        if bubble:
            bubble.update_tool(tool_run)
            self.scroll_end(animate=False)
            return True
        return False

    def last_reasoning(self) -> MessageBubble | None:
        found = None
        for child in self.query(MessageBubble):
            if child.role == "reasoning":
                found = child
        return found

    def toggle_last_reasoning(self) -> None:
        bubble = self.last_reasoning()
        if bubble is not None:
            bubble.expanded = not bubble.expanded
            bubble.focus()

    def watch_messages(self, value: list) -> None:
        self.refresh()
