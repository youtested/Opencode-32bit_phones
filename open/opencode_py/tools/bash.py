"""bash tool: run a shell command in a persistent shell session with timeout."""

from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path
from io import UnsupportedOperation

from .registry import Tool, schema_with

READ_CHUNK = 65536


def _stream_output(
    proc: subprocess.Popen,
    deadline: float,
    max_bytes: int,
) -> tuple[bytes, bool, bool]:
    """Read the process stdout to EOF or deadline, capping the captured bytes.

    Uses the raw fd (not the buffered pipe object, whose internal buffer can
    hold data `select` doesn't see) so nothing gets stuck. Bytes beyond the cap
    are still drained (discarded) so the child never blocks on a full pipe;
    memory stays bounded regardless of output size.
    """
    chunks: list[bytes] = []
    captured = 0
    capped = False
    timed_out = False
    stdout = proc.stdout
    if stdout is None:
        return b"", False, False
    try:
        fd = stdout.fileno()
    except (OSError, ValueError, UnsupportedOperation):
        return b"", False, False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.25))
        if not ready:
            if proc.poll() is not None:
                break
            continue
        try:
            data = os.read(fd, READ_CHUNK)
        except OSError:
            data = b""
        if not data:
            break
        if not capped:
            room = max_bytes - captured
            if room > 0:
                chunks.append(data[:room])
                captured += len(data[:room])
            if len(data) > room:
                capped = True
    return b"".join(chunks), capped, timed_out


def _bash(
    command: str,
    timeout: int = 120,
    workdir: str | None = None,
    max_lines: int = 2000,
    max_bytes: int = 51200,
) -> dict:
    cwd = Path(workdir).resolve() if workdir else Path.cwd()
    if not cwd.exists():
        return {"output": "(no output)", "error": True, "exit_code": 1}

    shell = os.environ.get("SHELL", "/bin/sh")
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=shell,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except Exception as e:
        return {"output": f"failed to run command: {e}", "error": True, "exit_code": 127}

    deadline = time.monotonic() + max(float(timeout), 0.1)
    raw, capped, timed_out = _stream_output(proc, deadline, max_bytes)

    if timed_out:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    if timed_out:
        return {
            "output": raw.decode("utf-8", errors="replace"),
            "exit_code": -1,
            "error": True,
            "metadata": {
                "timeout": True,
                "timed_out": True,
            },
        }

    result: dict = {
        "output": raw.decode("utf-8", errors="replace"),
        "exit_code": proc.returncode or 0,
        "metadata": {"exit_code": proc.returncode or 0, "truncated": capped},
    }
    return _apply_caps(result, max_lines, max_bytes)


def _apply_caps(result: dict, max_lines: int, max_bytes: int) -> dict:
    output = result.get("output", "")
    truncated = bool(result.get("metadata", {}).get("truncated"))
    raw_len = len(output.encode("utf-8", errors="replace"))
    if raw_len > max_bytes:
        trimmed = output.encode("utf-8", errors="replace")[:max_bytes].decode(
            "utf-8", errors="ignore"
        )
        output = trimmed + f"\n... plus {raw_len - max_bytes} more bytes (truncated)"
        truncated = True
    lines = output.splitlines()
    if len(lines) > max_lines:
        output = "\n".join(lines[:max_lines])
        output += f"\n... plus {len(lines) - max_lines} more lines (truncated)"
        truncated = True
    result["output"] = output
    result["metadata"]["truncated"] = truncated
    return result


def tool(max_lines: int = 2000, max_bytes: int = 51200, default_timeout: int = 120) -> Tool:
    description = """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

Be aware: OS: {os}, Shell: {shell}

All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID using `cd <directory> && <command>` patterns - use `workdir` instead.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for that instead."""

    import sys

    description = description.format(os=sys.platform, shell=os.environ.get("SHELL", "bash"))

    def run(input: dict) -> dict:
        command = input["command"]
        timeout_ms = int(input.get("timeout") or default_timeout * 1000)
        timeout = max(0.1, timeout_ms / 1000.0)
        workdir = input.get("workdir")
        return _bash(command, timeout=timeout, workdir=workdir, max_lines=max_lines, max_bytes=max_bytes)

    return Tool(
        name="bash",
        description=description,
        parameters=schema_with(
            {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in milliseconds (default 120000)",
                    "optional": True,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory to run the command in",
                    "optional": True,
                },
            },
            ["command"],
        ),
        run=run,
        permission="bash",
    )
