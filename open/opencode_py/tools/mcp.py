"""Minimal MCP (Model Context Protocol) stdio client.

Speaks newline-delimited JSON-RPC 2.0 over the child's stdin/stdout (the
transport used by the reference MCP servers, e.g. `npx -y @modelcontextprotocol/...`).
Remote tools are exposed to the model as `mcp__<server>__<tool>`.

Pure-Python + subprocess; safe for armv7. No SDK dependency.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from typing import Any


class MCPError(Exception):
    pass


class MCPServer:
    def __init__(self, name: str, command: str, args: list[str], timeout: float = 20.0):
        self.name = name
        self.command = command
        self.args = list(args)
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._counter = 0

    # -- low level --------------------------------------------------------
    def _ensure_started(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        try:
            self.proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise MCPError(f"mcp/{self.name}: could not start '{self.command}': {e}") from e
        try:
            self.call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "opencode_py", "version": "0.1.0"},
                },
            )
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except MCPError:
            self.close()
            raise

    def _send(self, payload: dict) -> None:
        if not self.proc or not self.proc.stdin:
            raise MCPError(f"mcp/{self.name}: not running")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> dict:
        if not self.proc or not self.proc.stdout:
            raise MCPError(f"mcp/{self.name}: not running")
        # Read one line with a hard wall-clock deadline. A buffered TextIOWrapper
        # defeats `select` on the wrapper (it buffers internally while the OS
        # pipe looks empty), so read from the raw fd instead using a small chunk
        # and track bytes ourselves. `select` guarantees some bytes are ready.
        fd = self.proc.stdout.fileno()
        buf = b""
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"mcp/{self.name}: timeout waiting for response")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise MCPError(f"mcp/{self.name}: timeout waiting for response")
            try:
                data = os.read(fd, 65536)
            except (OSError, ValueError) as e:
                raise MCPError(f"mcp/{self.name}: read failed: {e}") from e
            if not data:
                raise MCPError(f"mcp/{self.name}: server closed stdout")
            buf += data
            if len(buf) > 8 * 1024 * 1024:
                raise MCPError(f"mcp/{self.name}: response line too large")
            newline = buf.find(b"\n")
            if newline != -1:
                buf = buf[:newline]
                break
        if not buf.strip():
            raise MCPError(f"mcp/{self.name}: empty response line")
        try:
            return json.loads(buf.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise MCPError(f"mcp/{self.name}: bad response: {e}") from e

    def call(self, method: str, params: dict | None = None) -> dict:
        self._counter += 1
        rid = self._counter
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            msg = self._read_line()
            if msg.get("id") != rid:
                continue  # notification / another request id
            if "error" in msg:
                err = msg["error"]
                raise MCPError(f"mcp/{self.name}: {err.get('message', err)}")
            return msg.get("result") or {}

    # -- public API -------------------------------------------------------
    def list_tools(self) -> list[dict]:
        self._ensure_started()
        res = self.call("tools/list")
        return res.get("tools", []) or []

    def run_tool(self, remote_name: str, arguments: dict) -> dict[str, Any]:
        self._ensure_started()
        res = self.call("tools/call", {"name": remote_name, "arguments": arguments or {}})
        if "isError" in res and res.get("isError"):
            text = _content_text(res) or "mcp tool errored"
            return {"output": text, "error": True}
        text = _content_text(res)
        if text is None:
            text = json.dumps(res.get("content", res))
        return {"output": text}

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


def _content_text(result: dict) -> str | None:
    """Concatenate MCP content blocks: text / resource / image(placeholder)."""
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "resource":
            resource = block.get("resource", {})
            if isinstance(resource, dict) and "text" in resource:
                parts.append(resource["text"])
    return "\n".join(p for p in parts if p) or None
