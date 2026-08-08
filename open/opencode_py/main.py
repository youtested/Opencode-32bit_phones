"""opencode_py headless entry point.

Usage:
    opencode_py [dir] [--no-tui] [--auto] [--agent <name>] [--model <id>]
                [--provider <name>] [-m <message>]

Reads a prompt from stdin when no -m/--message is given (like opencode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from . import __version__
from .agent.loop import AgentLoop
from .auth import Auth
from .commands import CommandContext, build_registry as build_command_registry, handle_command
from .config import load_config
from .globals import Path as GPath
from .globals import resolve_worktree
from .session import new_session, save_session
from .tools import build_registry as build_tool_registry


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("directory", required=False, default=".")
@click.option("--no-tui", is_flag=True, help="Force headless (no Textual TUI)")
@click.option("--auto", is_flag=True, help="Auto-approve tool calls (like --auto)")
@click.option("--agent", "agent_name", default="build", help="Agent to use (build|plan)")
@click.option("--model", "model_id", default=None, help="Model id (e.g. big-pickle)")
@click.option("--provider", "provider_name", default=None, help="Provider (e.g. opencode)")
@click.option("-m", "--message", default=None, help="Prompt message (otherwise read stdin)")
@click.option("--print-config", is_flag=True, help="Print resolved config and exit")
@click.option("--check", is_flag=True, help="Ping configured providers and report status")
@click.option("--models", is_flag=True, help="List available Zen models")
@click.option("--version", is_flag=True, help="Print version and exit")
def main(
    directory: str,
    no_tui: bool,
    auto: bool,
    agent_name: str,
    model_id: str | None,
    provider_name: str | None,
    message: str | None,
    print_config: bool,
    check: bool,
    models: bool,
    version: bool,
) -> None:
    if version:
        click.echo(f"opencode_py {__version__}")
        return

    try:
        cfg = load_config()
    except Exception as e:
        click.echo(f"error: failed to load config: {e}", err=True)
        sys.exit(1)

    if provider_name:
        cfg.provider = provider_name
    if model_id:
        cfg.model = model_id
        cfg.provider = provider_name or cfg.provider
    if agent_name:
        cfg.agent = agent_name

    if print_config:
        click.echo(json.dumps(cfg.as_dict(), indent=2))
        return

    worktree = resolve_worktree(Path(directory))

    GPath.init()
    auth = Auth(auth_file=GPath.auth_file())

    if check:
        _run_check(cfg, auth)
        return

    if models:
        _run_models()
        return

    if not no_tui and sys.stdin.isatty() and message is None:
        # interactive terminal: launch the Textual TUI
        from .tui import run_tui

        run_tui(cfg=cfg, directory=Path(worktree))
        return

    def on_engine_event(event: dict) -> None:
        if event.get("kind") == "rotated":
            reason = event.get("reason") or "provider error"
            click.echo(
                f"[{reason} - switched to {event.get('provider')}/{event.get('model')}]",
                err=True,
            )

    engine = AgentLoop(
        cfg=cfg,
        registry=build_tool_registry(cfg),
        directory=Path(worktree),
        auth=auth,
        agent=cfg.agent,
        on_event=on_engine_event,
    )

    session = new_session(
        directory=str(worktree),
        provider=cfg.provider,
        model=cfg.model,
        agent=cfg.agent,
        title=message[:60] if message else "new session",
    )

    def reply(text: str) -> None:
        click.echo(text)

    def get_session() -> Any:
        return session

    def exit_app() -> None:
        save_session(session)
        sys.exit(0)

    ctx = CommandContext(
        config=cfg,
        auth=auth,
        session=session,
        engine=engine,
        worktree=worktree,
        reply=reply,
        get_session=get_session,
        exit_app=exit_app,
        registry=build_command_registry(),
    )

    prompt = message
    if prompt is None:
        prompt = sys.stdin.read()

    run(cfg, engine, ctx, session, prompt, auto)


def run(
    cfg: Any,
    engine: AgentLoop,
    ctx: CommandContext,
    session: Any,
    prompt: str,
    auto: bool,
) -> None:
    if auto:
        engine.permission.mode = "auto"

    if not prompt.strip():
        click.echo("Usage: opencode_py [dir] --message '...'  (or pipe a prompt via stdin)")
        return

    if prompt.lstrip().startswith("/"):
        handled = handle_command(build_command_registry(), ctx, prompt)
        if handled:
            save_session(session)
            return

    session.messages = engine.get_history()
    result = engine.run_turn(prompt)
    session.messages = engine.get_history()
    save_session(session)
    if result.text:
        click.echo(result.text)
    if result.error:
        click.echo(f"error: {result.error}", err=True)
        sys.exit(1)


def _run_check(cfg: Any, auth: Auth) -> None:
    """--check: ping each configured provider lane and report OK/fail."""
    from .providers import check_provider

    results = check_provider(cfg, auth)
    if not results:
        click.echo("No providers configured.")
        return
    failed = 0
    for pid, info in results.items():
        if info.get("ok"):
            click.echo(f"[OK]   {pid:<12} {info.get('model', '')}")
        else:
            failed += 1
            detail = info.get("error") or info.get("status") or "?"
            click.echo(f"[FAIL] {pid:<12} {info.get('model', '')} - {detail}")
    click.echo(f"\n{len(results) - failed}/{len(results)} providers OK")
    if failed:
        sys.exit(1)


def _run_models() -> None:
    """--models: print the live Zen model list (free first)."""
    from .providers import fetch_zen_models

    models = fetch_zen_models()
    free = [m for m in models if m.get("free")]
    paid = [m for m in models if not m.get("free")]
    click.echo("Free models:")
    for m in free:
        ctx_size = f"{m['context']:,}" if m.get("context") else "?"
        click.echo(f"  opencode/{m['id']:<26} ctx={ctx_size}")
    click.echo(f"\nPaid models ({len(paid)}):")
    for m in paid[:20]:
        click.echo(f"  opencode/{m['id']}")
    if len(paid) > 20:
        click.echo(f"  ... and {len(paid) - 20} more")


if __name__ == "__main__":
    main()
