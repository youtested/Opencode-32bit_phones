"""write tool: create/overwrite a file."""

from __future__ import annotations

from pathlib import Path

from .registry import Tool, schema_with


def _write(filePath: str, content: str) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    return {
        "output": "Wrote file successfully.",
        "metadata": {"filePath": str(path), "content": content},
    }


def tool() -> Tool:
    description = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""

    def run(input: dict) -> dict:
        return _write(input["filePath"], input["content"])

    return Tool(
        name="write",
        description=description,
        parameters=schema_with(
            {
                "content": {"type": "string", "description": "The content to write to the file"},
                "filePath": {"type": "string", "description": "The absolute path to the file to write"},
            },
            ["content", "filePath"],
        ),
        run=run,
        permission="edit",
    )
