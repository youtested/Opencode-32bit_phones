"""Tool registry: name -> Tool dataclass with JSON schema + run().

Mirrors opencode's Tool.define pattern. Each tool declares a name, description,
parameter JSON schema, an optional permission key, and a run(input) -> dict.
The run result dict carries `output` (text), plus optional metadata the TUI
renders (e.g. edit diff, bash exit code).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    permission: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.mcp_servers: list[Any] = []

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for all registered tools."""
        out = []
        for tool in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return out


def _param(
    type_: str,
    description: str,
    required: bool = True,
    enum: list[str] | None = None,
    default: Any = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        schema["enum"] = enum
    if default is not None:
        schema["default"] = default
    if not required:
        schema["optional"] = True
    return schema


def schema_with(params: dict[str, dict], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": params, "required": required}
