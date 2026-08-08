"""edit tool: exact string replacement + diff verification.

Mirrors opencode's edit tool semantics:
- Requires prior read (tracked per-session by the engine, enforced in the loop).
- Exact oldString match only; returns clear errors otherwise (no silent fuzzy
  substitution — a whitespace-normalized fuzzy match can mis-indent code).
- Verify by reading back the file; report the diff in metadata for the TUI.
"""

from __future__ import annotations

from pathlib import Path

from ..util.diff import create_diff
from .registry import Tool, schema_with


def find_matches(content: str, old: str) -> list[tuple[int, int]]:
    """Return all (start, end) exact-match spans of `old` in `content`."""
    matches = []
    start = 0
    while True:
        idx = content.find(old, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(old)))
        start = idx + 1
    return matches


def _do_edit(path: Path, old: str, new: str, replace_all: bool = False) -> dict:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return {"output": f"Could not read file {path}: {e}", "error": True}

    if old == new:
        return {"output": "oldString and newString are identical. No changes made.", "error": True}
    if not old:
        return {"output": "oldString is empty. Use the write tool to replace the whole file.", "error": True}

    matches = find_matches(content, old)
    if not matches:
        return {"output": "oldString not found in content.", "error": True}
    if len(matches) > 1 and not replace_all:
        return {
            "output": "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match.",
            "error": True,
        }

    if replace_all:
        new_content = content.replace(old, new)
    else:
        start, end = matches[0]
        new_content = content[:start] + new + content[end:]

    # verify by writing then reading back
    try:
        path.write_text(new_content, encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}

    diff = create_diff(content, new_content, old_path=str(path), new_path=str(path))
    return {
        "output": "Edit applied successfully.",
        "metadata": {"diff": diff, "replaceAll": replace_all},
    }


def _edit(filePath: str, oldString: str, newString: str, replaceAll: bool = False) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        return {"output": f"File does not exist: {path}", "error": True}
    return _do_edit(path, oldString, newString, replaceAll)


def tool() -> Tool:
    description = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + colon + space (e.g., `1: `). Everything after that space is the actual file content to match. Never include any part of the line number prefix in the oldString or newString.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `oldString` is not found in the file with an error "oldString not found in content".
- The edit will FAIL if `oldString` is found multiple times in the file with an error "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match." Either provide a larger string with more surrounding context to make it unique or use `replaceAll` to change every instance of `oldString`.
- Use `replaceAll` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""

    def run(input: dict) -> dict:
        return _edit(
            input["filePath"],
            input["oldString"],
            input["newString"],
            replaceAll=bool(input.get("replaceAll", False)),
        )

    return Tool(
        name="edit",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "The absolute path to the file to edit"},
                "oldString": {"type": "string", "description": "The text to replace"},
                "newString": {"type": "string", "description": "The text to replace it with"},
                "replaceAll": {"type": "boolean", "description": "Replace all occurrences (default false)", "optional": True},
            },
            ["filePath", "oldString", "newString"],
        ),
        run=run,
        permission="edit",
    )
