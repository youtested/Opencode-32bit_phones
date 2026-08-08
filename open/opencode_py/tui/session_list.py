"""Session list: switch between the main session and live sub-agent sessions.

Mirrors opencode's session sidebar. Shows each session's title, agent type and
status (running/completed). Enter switches to the selected session, Esc closes.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from .theme import get_theme

RUNNING_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class SessionList(ModalScreen[str]):
    """Modal popup listing sessions; dismisses with the chosen session id."""

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        current: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sessions = sessions
        self.current = current

    def _rows(self) -> list[tuple[str, str]]:
        theme = get_theme("opencode")
        rows: list[tuple[str, str]] = []
        for i, s in enumerate(self.sessions):
            title = s.get("title") or "(untitled)"
            agent = s.get("agent") or "build"
            status = s.get("status", "")
            mark = "▶" if s.get("id") == self.current else " "
            if status == "running":
                frame = RUNNING_CHARS[i % len(RUNNING_CHARS)]
                state = f"{frame} running"
                color = theme.c("warning")
            elif status == "error":
                state = "error"
                color = theme.c("error")
            else:
                state = ""
                color = theme.c("text_muted")
            row = f"[{theme.c('text')}]{mark} {title}[/]"
            row += f"  [{theme.agent_color(agent)}]· {agent}[/]"
            if state:
                row += f"  [{color}]({state})[/]"
            rows.append((s.get("id", ""), row))
        return rows

    def compose(self) -> ComposeResult:
        theme = get_theme("opencode")
        with Vertical(classes="cmd-popup session-popup"):
            yield Static("  Sessions  ", classes="cmd-popup-title")
            yield Static(
                f"[{theme.c('text_muted')}]Ctrl+B to close · Enter to switch[/]",
                classes="cmd-popup-usage",
            )
            items = [ListItem(Label(row), id=f"row-{sid}") for sid, row in self._rows()]
            yield ListView(*items, id="session-list")

    def on_mount(self) -> None:
        lv = self.query_one("#session-list", ListView)
        for i, s in enumerate(self.sessions):
            if s.get("id") == self.current:
                lv.index = i
                break
        lv.focus()

    def on_list_view_selected(self, event: Any) -> None:
        item = event.item
        if item is not None:
            self.dismiss(str(item.id).removeprefix("row-"))
        else:
            self.dismiss(None)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key == "enter":
            lv = self.query_one("#session-list", ListView)
            if lv.highlighted_child is not None:
                self.dismiss(str(lv.highlighted_child.id).removeprefix("row-"))
                event.stop()
