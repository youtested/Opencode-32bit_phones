"""Model picker screen (/models + Settings): live, grouped, auto-refreshing.

Providers are shown in two sections:

  - Free section first: OpenCode Zen, OpenRouter, then the free-tier
    bring-your-own-key providers (Groq, Google, ...), each with its models
    listed underneath (free models first within a provider).
  - Paid section: Anthropic Claude, OpenAI, ... with their live models.

Model lists are fetched live from each provider's `/models` endpoint (only when
an API key is present), fall back to a bundled default when unavailable, and
auto-refresh every REFRESH_SECONDS. Selecting a model dismisses with a
"provider/model" string so settings (and /models) can switch both at once.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListView, ListItem, Static

from ..providers import (
    FREE_PROVIDERS,
    FREE_DEFAULT_MODELS,
    PAID_PROVIDERS,
    fetch_zen_models,
    fetch_openrouter_models,
    fetch_live_models,
)

REFRESH_SECONDS = 60

# (provider id, display name) — free providers first, paid after.
FREE_SECTION: list[tuple[str, str]] = [
    ("opencode", "OpenCode Zen"),
    ("openrouter", "OpenRouter"),
    ("groq", "Groq"),
    ("cerebras", "Cerebras"),
    ("google", "Google AI Studio"),
    ("nvidia", "NVIDIA NIM"),
    ("mistral", "Mistral"),
    ("github", "GitHub Models"),
    ("sambanova", "SambaNova"),
    ("togetherai", "Together"),
    ("ollama", "Ollama (local)"),
]

PAID_SECTION: list[tuple[str, str]] = [
    ("anthropic", "Anthropic Claude"),
    ("openai", "OpenAI"),
    ("deepseek", "DeepSeek"),
    ("xai", "xAI"),
    ("deepinfra", "DeepInfra"),
]

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("free", FREE_SECTION),
    ("paid", PAID_SECTION),
]

# curated fallback when a paid provider has no key / the live list is down.
DEFAULT_PAID_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "deepseek": ["deepseek-chat"],
    "xai": ["grok-2-latest"],
    "deepinfra": ["meta-llama/Meta-Llama-3.3-70B-Instruct"],
}

CSS = """
ModelPicker {
    background: #0a0a0a;
}
#model-picker {
    width: 100%;
    height: 100%;
    layout: vertical;
    padding: 1 2;
}
.screen-title {
    height: auto;
    margin-bottom: 1;
    color: #eeeeee;
    text-style: bold;
}
#models-status {
    height: auto;
    margin-bottom: 1;
    color: #808080;
}
#models-list {
    height: 1fr;
    border: none;
    background: #0a0a0a;
}
.group-header {
    height: auto;
    padding: 1 0 0 1;
    color: #fab283;
    text-style: bold;
}
.model-item {
    height: auto;
    padding: 0 0 0 2;
    color: #eeeeee;
}
#models-actions {
    height: auto;
    padding-top: 1;
    align-horizontal: right;
}
#models-actions Button {
    margin-left: 1;
}
"""


class ModelPicker(ModalScreen[str | None]):
    """Full-screen model list; Enter selects, Esc dismisses, R refreshes."""

    BINDINGS = [
        Binding("r", "refresh_models", "Refresh"),
        Binding("escape", "dismiss_pop", "Close"),
    ]

    def __init__(
        self,
        current: str = "",
        on_select: Callable[[str], None] | None = None,
        cfg: Any = None,
        auth: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.current = current
        self.on_select = on_select
        self.cfg = cfg
        self.auth = auth
        self.models: dict[str, list[dict]] = {}
        self._item_lookup: list[dict] = []
        self._fetching = False
        self._timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker"):
            yield Label("Models", classes="screen-title")
            yield Static("Loading models...", id="models-status")
            yield ListView(id="models-list")
            with Horizontal(id="models-actions"):
                yield Button("Refresh", id="models-refresh", variant="default")
                yield Button("Close", id="models-close", variant="primary")

    def on_mount(self) -> None:
        self.set_loading()
        self._start_worker()
        self._timer = self.set_interval(REFRESH_SECONDS, self._periodic_refresh)

    def on_unmount(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None

    # -- fetching ----------------------------------------------------------
    def set_loading(self) -> None:
        if not self.is_attached:
            return
        try:
            self.query_one("#models-status", Static).update(
                f"Fetching model lists from providers... (auto-refresh every {REFRESH_SECONDS}s)"
            )
        except Exception:
            pass

    def _start_worker(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self.set_loading()
        self.run_worker(self._fetch_models, thread=True)

    def _periodic_refresh(self) -> None:
        self._start_worker()

    def _fetch_models(self) -> None:
        pids = [pid for _, providers in SECTIONS for pid, _ in providers]
        per_provider: dict[str, list[dict]] = {}
        try:
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(self._fetch_provider_models, pid): pid for pid in pids}
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        per_provider[pid] = future.result() or []
                    except Exception:
                        per_provider[pid] = []
        finally:
            self._fetching = False
        self.app.call_from_thread(self.populate, per_provider)

    def _fetch_provider_models(self, pid: str) -> list[dict]:
        if pid == "opencode":
            return fetch_zen_models()
        if pid == "openrouter":
            return fetch_openrouter_models()
        if pid == "ollama":
            return [
                {"id": "llama3.2", "name": "Llama 3.2", "context": 128000, "free": True},
                {"id": "llama3.1", "name": "Llama 3.1", "context": 128000, "free": True},
            ]
        meta = FREE_PROVIDERS.get(pid) or PAID_PROVIDERS.get(pid) or {}
        key = self.auth.get(pid) if self.auth else None
        models = (
            fetch_live_models(pid, key, meta.get("base_url"), meta.get("api_kind", "openai"))
            if meta
            else []
        )
        if models:
            # the whole provider is in the free section, so badge its models FREE
            is_free_section = any(p == pid for p, _ in FREE_SECTION)
            for m in models:
                m["free"] = is_free_section
            return models
        return _fallback_models(pid, has_key=bool(key))

    # -- display -----------------------------------------------------------
    def populate(self, per_provider: dict[str, list[dict]]) -> None:
        # The fetch worker may complete after the screen was dismissed (Esc /
        # Close / model picked). Guard the widget lookups so a pruned screen
        # doesn't raise NoMatches and crash the whole app.
        if not self.is_attached:
            return
        self.models = per_provider
        lv = self.query_one("#models-list", ListView)
        lv.clear()
        self._item_lookup: list[dict] = []

        total_free = 0
        total_paid = 0
        for _, providers in SECTIONS:
            for pid, display in providers:
                items = per_provider.get(pid) or []
                if not items:
                    continue
                lv.append(ListItem(Label(f" {display}:"), classes="group-header"))
                ordered = sorted(items, key=lambda m: (not bool(m.get("free")), m["id"]))
                for m in ordered:
                    idx = f"{pid}/{m['id']}"
                    row_index = len(lv.children)  # absolute ListView index of this row
                    self._item_lookup.append({"row": row_index, "provider": pid, "model": m["id"]})
                    if m.get("free"):
                        total_free += 1
                    else:
                        total_paid += 1
                    ctx = _format_context(m.get("context"))
                    badge = "[green]FREE[/]" if m.get("free") else ""
                    mark = "● " if idx == self.current or m["id"] == self.current else "  "
                    label = Label(f"{mark}{m['id']:<44} ctx={ctx:<9} {badge}", classes="model-item")
                    lv.append(ListItem(label))

        self.query_one("#models-status", Static).update(
            f"{total_free} free, {total_paid} paid — updated {time.strftime('%H:%M:%S')} "
            f"— Enter select · R refresh"
        )

    # -- events ------------------------------------------------------------
    def on_list_view_selected(self, event: Any) -> None:
        if not self._item_lookup:
            return
        index = event.index if event.index is not None else (getattr(event.item, "index", None) or 0)
        for entry in self._item_lookup:
            if entry["row"] == index:
                choice = f"{entry['provider']}/{entry['model']}"
                if self.on_select:
                    self.on_select(choice)
                self.dismiss(choice)
                return

    def action_refresh_models(self) -> None:
        self._start_worker()

    def action_dismiss_pop(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "models-close":
            self.dismiss(None)
        elif bid == "models-refresh":
            self._start_worker()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


def _fallback_models(pid: str, has_key: bool) -> list[dict]:
    """Bundled model list when the live fetch fails or no key is present."""
    if pid in FREE_PROVIDERS:
        mid = FREE_DEFAULT_MODELS.get(pid)
        return [{"id": mid, "name": mid, "context": 0, "free": True}] if mid else []
    out = []
    for mid in DEFAULT_PAID_MODELS.get(pid, []):
        out.append({"id": mid, "name": mid, "context": 0, "free": False})
    return out


def _format_context(value: Any) -> str:
    """Format a context size for display, tolerating "128k"/"1m" strings and junk."""
    if value is None:
        return "?"
    if isinstance(value, str):
        s = value.strip().lower()
        mult = 1
        if s.endswith("k"):
            mult, s = 1000, s[:-1]
        elif s.endswith("m"):
            mult, s = 1000000, s[:-1]
        try:
            return f"{int(float(s) * mult):,}"
        except (ValueError, TypeError):
            return value
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return "?"
