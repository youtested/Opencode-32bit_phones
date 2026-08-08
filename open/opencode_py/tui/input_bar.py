"""Input bar: opencode-style prompt with agent-colored accent, meta row, and
a busy spinner line, plus /command autocomplete + arrow-key navigation.

The prompt itself is a growing multi-line textarea (mirrors opencode's
`<textarea>` prompt component): long lines wrap and the box grows up to a
max-height cap instead of scrolling horizontally, so a long message is never
hidden behind the accent strip or the edge of the screen.

  - a solid 1-cell left accent strip in the current agent color
  - the input on the backgroundElement colour
  - a muted meta row underneath: `Agent auto · model provider`
  - a status line with an animated block spinner while the engine is busy

Type `/` and a dropdown of matching slash commands appears below the input.
Arrow keys move through it; Enter opens a centered CommandPopup for the
selected command. Enter submits; Shift+Enter inserts a newline. Up/Down move
the cursor through multi-line text, and fall back to prompt history at the
first/last line. Tab with an empty input toggles the agent.
"""

from __future__ import annotations

import math
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.events import Key
from textual.message import Message
from textual.style import Style as ContentStyle
from textual.strip import Strip
from textual.widgets import Static, TextArea

from .theme import get_theme

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class PromptSubmitted(Message):
    """User pressed Enter with a prompt."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class AgentToggleRequested(Message):
    """User pressed Tab with an empty input; cycle agent."""

    def __init__(self) -> None:
        super().__init__()


class CommandSelected(Message):
    """A slash-command suggestion was chosen; open its centered popup."""

    def __init__(self, name: str, description: str) -> None:
        super().__init__()
        self.name = name
        self.description = description


class PromptTextArea(TextArea):
    """Multi-line prompt input.

    Long lines soft-wrap and the widget grows up to a cap (like opencode's
    `<textarea minHeight=1 maxHeight=...>`), so the prompt never scrolls out of
    view. Enter submits; Shift+Enter inserts a newline. Up/Down are delegated to
    the parent InputBar while command suggestions or prompt history apply.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            show_cursor=True,
            tab_behavior="focus",
            **kwargs,
        )
        self._delegate_arrow: Any = None

    # -- compatibility with the old single-line Input ---------------------
    # The rest of the app (and the tests) talk to `bar.input.value` and
    # `bar.input.cursor_position`; map those onto the textarea.
    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text
        self.resize_to_content()

    @property
    def cursor_position(self) -> int:
        return self.document.get_index_from_location(self.selection.end)

    @cursor_position.setter
    def cursor_position(self, index: int) -> None:
        self.move_cursor(self.document.get_location_from_index(index))

    # -- sizing ------------------------------------------------------------
    def _wrapped_lines(self) -> int:
        """Approximate number of visual lines once soft-wrapped to the width."""
        width = max(1, self.size.width - 2)
        total = 0
        for line in self.text.split("\n"):
            if not line:
                total += 1
            else:
                total += max(1, math.ceil(cell_len(line) / width))
        return total

    def _max_height(self) -> int:
        # mirrors opencode: max(6, terminal height / 3)
        return max(6, self.app.size.height // 3)

    def resize_to_content(self) -> None:
        # min height 3 keeps the original prompt box look; grows as the prompt
        # wraps to multiple lines (like opencode's textarea).
        self.styles.height = max(3, min(self._wrapped_lines(), self._max_height()))

    def render_line(self, y: int) -> Strip:
        """Vertically center the placeholder in the (taller) prompt box.

        Textual's TextArea renders the placeholder at the top row and ignores
        `content-align`, so without this override "Ask anything..." hugs the top
        of the 3-row box instead of sitting in the middle of it.
        """
        if not self.text and self.placeholder:
            placeholder_lines = Content.from_text(self.placeholder).wrap(self.content_size.width)
            offset = max(0, (self.size.height - len(placeholder_lines)) // 2)
            idx = y - offset
            if 0 <= idx < len(placeholder_lines):
                style = self.get_visual_style("text-area--placeholder")
                content = placeholder_lines[idx].stylize(style)
                if self._draw_cursor and idx == 0:
                    theme = self._theme
                    cursor_style = theme.cursor_style if theme else None
                    if cursor_style:
                        content = content.stylize(ContentStyle.from_rich_style(cursor_style), 0, 1)
                return Strip(content.render_segments(self.visual_style), content.cell_length)
            return Strip.blank(self.size.width, self.visual_style.rich_style)
        return super().render_line(y)

    # -- keys --------------------------------------------------------------
    async def _on_key(self, event: Key) -> None:
        key = event.key
        if key == "enter":
            # plain Enter submits (mirrors opencode's onSubmit)
            event.stop()
            event.prevent_default()
            self.post_message(PromptSubmitted(self.text))
            return
        if key in ("shift+enter", "ctrl+enter", "alt+enter", "ctrl+j"):
            # opencode binds these to "insert newline"
            event.stop()
            event.prevent_default()
            self._replace_via_keyboard("\n", *self.selection)
            return
        if key in ("up", "down"):
            handler = self._delegate_arrow
            if handler is not None and handler(key):
                event.stop()
                return
        await super()._on_key(event)


class InputBar(Vertical):
    """Prompt input with a left accent strip, meta row and status line."""

    def __init__(self, commands: list[dict[str, str]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._hist_index: int = -1
        self._draft = ""
        self.commands = commands or []
        self._suggestions: list[str] = []
        self._sel: int = 0
        self._navigated = False
        self._busy = False
        self._spinner = 0
        self._timer: Any = None
        self._agents: list[str] = []
        self.agent = "build"
        self.model = ""
        self.provider = ""
        self.permission_mode = "auto"

    def compose(self) -> ComposeResult:
        yield Static("", id="suggestions", classes="suggestions hidden")
        yield Static("", id="prompt-title")
        with Horizontal(classes="prompt-frame"):
            yield Static("", id="prompt-accent")
            with Vertical(classes="prompt-body"):
                yield PromptTextArea(placeholder="Ask anything...", id="prompt-input")
                yield Static("", id="prompt-meta")
        yield Static("", id="prompt-status")

    def on_mount(self) -> None:
        prompt = self.query_one(PromptTextArea)
        prompt._delegate_arrow = self._handle_arrow
        prompt.resize_to_content()
        self._sync_accent_height()
        prompt.focus()

    @property
    def input(self) -> PromptTextArea:
        return self.query_one("#prompt-input", PromptTextArea)

    def _sync_accent_height(self) -> None:
        """Keep the left accent strip the same height as the growing input."""
        try:
            prompt = self.input
            self.query_one("#prompt-accent", Static).styles.height = prompt.styles.height
        except Exception:
            pass

    # -- state --------------------------------------------------------------
    def set_header(
        self,
        *,
        agent: str,
        model: str,
        provider: str,
        permission_mode: str,
    ) -> None:
        theme = get_theme("opencode")
        accent = theme.agent_color(agent)
        self.query_one("#prompt-accent", Static).styles.background = accent
        self.agent = agent
        self.model = model
        self.provider = provider
        self.permission_mode = permission_mode
        self._render_title()
        self._render_meta()

    def _render_title(self) -> None:
        """Fixed `▣ Build · <picked model>` line at the top of the prompt box."""
        theme = get_theme("opencode")
        accent = theme.agent_color(self.agent)
        t = self.query_one("#prompt-title", Static)
        rich = Text()
        rich.append("▣ ", style=accent)
        rich.append(self.agent.title(), style=theme.c("text"))
        if self.model:
            rich.append(f" · {self.model}", style=theme.c("text_muted"))
        t.update(rich)

    def _render_meta(self) -> None:
        theme = get_theme("opencode")
        accent = theme.agent_color(self.agent)
        t = self.query_one("#prompt-meta", Static)
        label = self.agent.title()
        parts: list[tuple[str, str]] = [
            (label, f"bold {accent}"),
        ]
        if self.permission_mode == "auto":
            parts.append((" auto", theme.c("text_muted")))
        parts.append((" ·", theme.c("text_muted")))
        if self.model:
            parts.append((f" {self.model}", theme.c("text")))
        if self.provider:
            parts.append((f" {self.provider}", theme.c("text_muted")))
        rich = Text()
        for text, style in parts:
            rich.append(text, style=style)
        t.update(rich)

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        status = self.query_one("#prompt-status", Static)
        if busy:
            status.remove_class("hidden")
            if self._timer is None:
                self._timer = self.set_interval(0.08, self._tick_spinner)
            self._update_status_line(message)
        else:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            status.update("")
            status.add_class("hidden")

    def set_running_agents(self, agents: list[str]) -> None:
        """Transient list of launched sub-agents shown in the status line while
        they run (opencode's `Delegating...` indicator)."""
        self._agents = agents
        if self._busy:
            self._update_status_line()

    def _tick_spinner(self) -> None:
        self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
        self._update_status_line()

    def _update_status_line(self, message: str = "") -> None:
        if not self._busy:
            return
        theme = get_theme("opencode")
        accent = theme.agent_color(self.agent)
        frame = SPINNER_FRAMES[self._spinner]
        if self._agents:
            text = self._agents[0]
            if len(self._agents) > 1:
                text += f" +{len(self._agents) - 1} more"
        else:
            text = message or "working..."
        self.query_one("#prompt-status", Static).update(
            f"[{accent}]{frame}[/] [dim]{text}[/]"
        )

    # -- suggestions ------------------------------------------------------
    def _suggestion_box(self) -> Static:
        return self.query_one("#suggestions", Static)

    def _update_suggestions(self, value: str) -> None:
        if not value.startswith("/"):
            self._clear_suggestions()
            return
        query = value[1:].lower()
        matches = [
            c["name"]
            for c in self.commands
            if not c.get("hidden") and c["name"].lower().startswith(query)
        ]
        if not matches:
            self._clear_suggestions()
            return
        self._suggestions = matches
        if self._sel >= len(matches):
            self._sel = 0
        box = self._suggestion_box()
        theme = get_theme("opencode")
        lines = []
        for i, name in enumerate(matches):
            if i == self._sel:
                style = f"bold {theme.c('accent')}"
                marker = "● "
            else:
                style = theme.c("text")
                marker = "  "
            lines.append(f"[{style}]{marker}/{name}[/]")
        box.update("\n".join(lines))
        box.remove_class("hidden")
        box.add_class("visible")

    def _clear_suggestions(self) -> None:
        self._suggestions = []
        self._navigated = False
        box = self._suggestion_box()
        box.update("")
        box.remove_class("visible")
        box.add_class("hidden")

    def _has_command(self, name: str) -> bool:
        return any(c["name"] == name for c in self.commands)

    def _select_name(self, name: str) -> None:
        self._suggestions = [name]
        self._sel = 0

    def _show_popup(self) -> None:
        name = self._suggestions[self._sel]
        desc = next((c["description"] for c in self.commands if c["name"] == name), "")
        self._clear_suggestions()
        self.input.value = ""
        self.post_message(CommandSelected(name, desc))

    # -- events -----------------------------------------------------------
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_suggestions(event.text_area.text)
        event.text_area.resize_to_content()
        self._sync_accent_height()
        event.stop()

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            event.stop()
            return
        if value.startswith("/"):
            parts = value[1:].split(maxsplit=1)
            name = parts[0]
            has_args = len(parts) == 2 and bool(parts[1].strip())
            if self._suggestions:
                # dropdown active -> centered popup for the highlighted command
                event.stop()
                self._show_popup()
                return
            if self._has_command(name) and not has_args:
                # bare known command (e.g. "/models") -> popup with its output
                self._select_name(name)
                event.stop()
                self._show_popup()
                return
        self._navigated = False
        self.input.value = ""
        if value.strip():
            self._history.append(value)
            self._hist_index = len(self._history)
        # not stopped: the PromptSubmitted continues up to the app

    def _handle_arrow(self, key: str) -> bool:
        """Consume Up/Down for suggestions / history; False = let the cursor move."""
        if self._suggestions:
            if key == "up":
                self._sel = (self._sel - 1) % len(self._suggestions)
            else:
                self._sel = (self._sel + 1) % len(self._suggestions)
            self._navigated = True
            self.input.value = f"/{self._suggestions[self._sel]}"
            self.input.cursor_position = len(self.input.value)
            self._update_suggestions(self.input.value)
            return True
        if key == "up":
            if self._history and self.input.cursor_at_first_line:
                if self._hist_index > 0:
                    # only capture the typed draft the first time we leave the
                    # "end" position, so repeated Up presses don't overwrite it
                    if self._hist_index == len(self._history):
                        self._draft = self.input.value
                    self._hist_index -= 1
                    self.input.value = self._history[self._hist_index]
                    self.input.cursor_position = len(self.input.value)
                return True
            return False
        if key == "down":
            if not self.input.cursor_at_last_line:
                return False
            if self._history and self._hist_index < len(self._history) - 1:
                self._hist_index += 1
                self.input.value = self._history[self._hist_index]
                self.input.cursor_position = len(self.input.value)
                return True
            if self._hist_index >= 0 and self._hist_index < len(self._history):
                # navigated into history and now at the end: restore the draft
                self._hist_index = len(self._history)
                if self._draft:
                    self.input.value = self._draft
                    self.input.cursor_position = len(self.input.value)
                return True
            # no history (or never navigated) — never wipe the typed input
            return False
        return False

    def on_key(self, event: Key) -> None:
        if event.key == "tab" and not self.input.value:
            self.post_message(AgentToggleRequested())
            event.stop()
        elif event.key == "escape" and self._suggestions:
            self._clear_suggestions()
            event.stop()
