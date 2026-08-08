"""glob tool: pure-python fnmatch glob with max-results cap."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from .registry import Tool, schema_with

MAX_RESULTS = 100


def _glob(pattern: str, path: str | None = None) -> dict:
    base = Path(path).resolve() if path else Path.cwd()
    if not base.is_dir():
        return {"output": f"Path is not a directory: {base}", "error": True}

    results: list[Path] = []
    try:
        matches = base.glob(pattern)
        for m in matches:
            results.append(m.resolve())
            if len(results) >= MAX_RESULTS:
                break
    except (OSError, ValueError, RecursionError) as e:
        return {"output": f"Glob error: {e}", "error": True}

    truncated = len(results) >= MAX_RESULTS
    if not results:
        return {"output": "No files found"}
    out = "\n".join(str(p) for p in results)
    if truncated:
        out += "\n\n(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)"
    return {"output": out, "metadata": {"count": len(results), "truncated": truncated}}


def tool() -> Tool:
    description = """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths
- Use this tool when you need to find files by name patterns
- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead
- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful."""

    def run(input: dict) -> dict:
        return _glob(input["pattern"], input.get("path"))

    return Tool(
        name="glob",
        description=description,
        parameters=schema_with(
            {
                "pattern": {"type": "string", "description": "The glob pattern to search for"},
                "path": {"type": "string", "description": "The directory to search in", "optional": True},
            },
            ["pattern"],
        ),
        run=run,
        permission="glob",
    )
