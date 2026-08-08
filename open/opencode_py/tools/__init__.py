"""Tools package: registry builder mirroring opencode's builtin tool order."""

from __future__ import annotations

from typing import Any

from ..config import Config
from . import bash as bash_mod
from . import edit as edit_mod
from . import glob as glob_mod
from . import grep as grep_mod
from . import read as read_mod
from . import task as task_mod
from . import todo as todo_mod
from . import webfetch as webfetch_mod
from . import write as write_mod
from .registry import Registry, Tool, schema_with

TOOL_NAMES = ["bash", "read", "glob", "grep", "edit", "write", "webfetch", "todowrite", "task"]


def build_registry(cfg: Config | None = None) -> Registry:
    cfg = cfg or Config()
    registry = Registry()
    state: dict = {}

    registry.register(bash_mod.tool(
        max_lines=cfg.tool_output_max_lines,
        max_bytes=cfg.tool_output_max_bytes,
        default_timeout=cfg.bash_default_timeout,
    ))
    registry.register(read_mod.tool())
    registry.register(glob_mod.tool())
    registry.register(grep_mod.tool())
    registry.register(edit_mod.tool())
    registry.register(write_mod.tool())
    registry.register(webfetch_mod.tool())
    registry.register(todo_mod.tool(state))
    registry.register(task_mod.tool(registry))

    # config-driven tool toggles: tools.<name> = false removes it (opencode behavior)
    enabled: dict | None = None
    if cfg and cfg.raw:
        enabled = cfg.raw.get("tools")
    if enabled:
        for name in list(registry.names()):
            if enabled.get(name) is False:
                registry._tools.pop(name, None)

    _load_plugins(registry, cfg)
    _load_mcp_servers(registry, cfg)

    return registry


def _load_plugins(registry: Registry, cfg: Config | None) -> None:
    """Plugin-lite: config key "plugins": ["my.tools.module"] where the module
    exposes TOOLS = [{name, description, parameters, run}, ...]."""
    raw = (cfg and cfg.raw) or {}
    import importlib
    import sys
    from pathlib import Path

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for mod_path in raw.get("plugins", []) or []:
        try:
            mod = importlib.import_module(str(mod_path))
            for tool_def in getattr(mod, "TOOLS", []) or []:
                registry.register(
                    Tool(
                        name=tool_def["name"],
                        description=tool_def.get("description", ""),
                        parameters=tool_def.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                        run=tool_def["run"],
                    )
                )
        except Exception as e:
            registry.register(
                Tool(
                    name=f"plugin:{mod_path}",
                    description="plugin failed to load",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=e, mod=mod_path: {
                        "output": f"plugin {mod} failed to load: {err}",
                        "error": True,
                    },
                )
            )


def _load_mcp_servers(registry: Registry, cfg: Config | None) -> None:
    """MCP-lite: config key "mcpServers": {name: {command, args}} -> tools named
    mcp__<name>__<tool>."""
    raw = (cfg and cfg.raw) or {}
    servers = raw.get("mcpServers", {}) or {}
    if not servers:
        return
    from .mcp import MCPError, MCPServer

    for sname, spec in servers.items():
        command = spec.get("command")
        args = spec.get("args") or []
        if not command:
            continue
        server = MCPServer(name=str(sname), command=command, args=args)
        registry.mcp_servers.append(server)
        try:
            remote_tools = server.list_tools()
        except MCPError as e:
            server.close()
            registry.register(
                Tool(
                    name=f"mcp:{sname}",
                    description=f"mcp server {sname} failed to start",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=str(e): {"output": f"mcp {sname}: {err}", "error": True},
                )
            )
            continue
        for t in remote_tools:
            registry.register(
                Tool(
                    name=f"mcp__{sname}__{t['name']}",
                    description=t.get("description") or f"{sname}: {t['name']}",
                    parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                    run=_mcp_run(server, t["name"]),
                )
            )


def _mcp_run(server, remote_name: str):
    def run(arguments: dict) -> dict[str, Any]:
        try:
            return server.run_tool(remote_name, arguments)
        except Exception as e:
            return {"output": f"mcp tool {remote_name} failed: {e}", "error": True}

    return run


__all__ = [
    "Registry",
    "Tool",
    "schema_with",
    "build_registry",
    "TOOL_NAMES",
]
