"""read tool: read a file or directory with line numbers + offset/limit."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .registry import Tool, schema_with

MAX_LINES = 2000
MAX_CHARS = 2000
MAX_OUTPUT = 50 * 1024  # 50 KB

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _is_binary_sample(sample: bytes) -> bool:
    if not sample:
        return False
    sample = sample[:1024]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    nonprintable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return nonprintable / len(sample) > 0.30


def _fuzzy_suggestion(path: Path) -> str | None:
    try:
        candidates = [p for p in path.parent.iterdir() if p.is_file()]
    except OSError:
        return None
    name = path.name
    matches = []
    for p in candidates:
        if p.stem == name or name in p.stem or p.stem in name:
            matches.append(p.name)
    return ", ".join(matches[:3]) or None


def _read_error(path: Path, error: BaseException) -> dict:
    suggestion = _fuzzy_suggestion(path)
    msg = f"Could not read file {path}: {error}"
    if suggestion:
        msg += f"\n\nDid you mean one of these?\n{suggestion}"
    return {"output": msg, "error": True}


def _read_file(path: Path, offset: int = 1, limit: int = MAX_LINES) -> dict:
    # image / pdf -> base64 file attachment
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS:
        try:
            data = path.read_bytes()
        except OSError as e:
            return _read_error(path, e)
        mime, _ = mimetypes.guess_type(str(path))
        b64 = base64.b64encode(data).decode()
        return {
            "output": f"<{path.name}><type>{mime}</type><content>data:{mime};base64,{b64}</content>",
            "metadata": {"loaded": [str(path)], "mime": mime},
        }

    # binary detection: sample the head without buffering the whole file
    try:
        with path.open("rb") as f:
            sample = f.read(1024)
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(sample):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}

    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit

    # stream the window line-by-line, holding only the selected lines in memory
    numbered: list[str] = []
    total: int | None = None
    reached_eof = True
    out_chars = 0
    truncated_out = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if lineno > end:
                    reached_eof = False
                    break
                total = lineno
                if lineno <= start:
                    continue
                content = line.rstrip("\n").rstrip("\r")
                if len(content) > MAX_CHARS:
                    content = content[:MAX_CHARS] + f"... (line truncated to {MAX_CHARS} chars)"
                numbered.append(f"{lineno}: {content}")
                capped_chars = (len(content) + 16)
                if out_chars + capped_chars > MAX_OUTPUT:
                    truncated_out = True
                    reached_eof = False
                    break
                out_chars += capped_chars
    except OSError as e:
        return _read_error(path, e)

    body = "\n".join(numbered)
    if truncated_out:
        body = body[:MAX_OUTPUT]

    if reached_eof and total is not None:
        footer = f"(End of file - total {total} lines)"
    else:
        shown = start + len(numbered)
        footer = f"(Showing line {start + 1}-{shown}. Use offset={shown + 1} to continue.)"
    if truncated_out:
        footer += " (Output capped at 50 KB.)"

    return {
        "output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read_directory(path: Path, offset: int = 1, limit: int = MAX_LINES) -> dict:
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return {"output": f"Could not read directory {path}: {e}", "error": True}
    total = len(entries)
    start = max(0, offset - 1)
    selected = entries[start : start + limit]
    lines = [(f"{p.name}/" if p.is_dir() else p.name) for p in selected]
    body = "\n".join(lines)
    if start == 0 and total <= limit:
        footer = f"(Showing {total} entries)"
    else:
        end = min(start + limit, total)
        footer = f"(Showing {start + 1}-{end} of {total} entries. Use offset={end + 1} to continue.)"
    return {
        "output": f"<{path}>…</{path}>\n<type>directory</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read(filePath: str, offset: int = 1, limit: int = MAX_LINES) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        suggestion = _fuzzy_suggestion(path)
        msg = f"Path {path} does not exist."
        if suggestion:
            msg += f"\n\nDid you mean one of these?\n{suggestion}"
        return {"output": msg, "error": True}
    if path.is_dir():
        return _read_directory(path, offset=offset, limit=limit)
    return _read_file(path, offset=offset, limit=limit)


def tool() -> Tool:
    description = """Read a file or directory from the local filesystem. If the path does not exist, an error is returned.

Usage:
- The filePath parameter should be an absolute path.
- By default, this tool returns up to 2000 lines from the start of the file.
- The offset parameter is the line number to start from (1-indexed).
- To read later sections, call this tool again with a larger offset.
- Use the grep tool to find specific content in large files or files with long lines.
- If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern.
- Contents are returned with each line prefixed by its line number as `<line>: <content>`. For example, if a file has contents "foo\n", you will receive "1: foo\n". For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.
- Any line longer than 2000 characters is truncated.
- Call this tool in parallel when you know there are multiple files you want to read.
- Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.
- This tool can read image files and PDFs and return them as file attachments."""

    def run(input: dict) -> dict:
        return _read(
            input["filePath"],
            offset=int(input.get("offset") or 1),
            limit=int(input.get("limit") or MAX_LINES),
        )

    return Tool(
        name="read",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "The absolute path to the file or directory to read"},
                "offset": {"type": "integer", "description": "The line number to start from (1-indexed)", "optional": True},
                "limit": {"type": "integer", "description": "The number of lines to read", "optional": True},
            },
            ["filePath"],
        ),
        run=run,
        permission="read",
    )
