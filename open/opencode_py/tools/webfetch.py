"""webfetch tool: fetch URL -> markdown/text with size caps."""

from __future__ import annotations

import re

import httpx

from .registry import Tool, schema_with

MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT = 30


def _html_to_text(html: str) -> str:
    """Crude HTML -> text (strip tags/scripts/styles). Good enough for v1."""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", "", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|pre|blockquote)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", "", html)
    import html as h

    text = h.unescape(html)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _html_to_markdown(html: str) -> str:
    """Approximate HTML -> markdown. A real turndown port is Phase 2 polish."""
    text = _html_to_text(html)
    return text


def _webfetch(url: str, format: str = "markdown", timeout: int = DEFAULT_TIMEOUT) -> dict:
    if not re.match(r"^https?://", url):
        return {"output": "URL must start with http:// or https://", "error": True}
    upgraded = url.startswith("http://")
    if upgraded:
        url = "https://" + url[len("http://"):]
    headers = {
        "User-Agent": "opencode",
        "Accept": {
            "markdown": "text/markdown, text/plain;q=0.9, text/html;q=0.5, */*;q=0.1",
            "text": "text/plain, text/markdown;q=0.9, text/html;q=0.5, */*;q=0.1",
            "html": "text/html, */*;q=0.8",
        }.get(format, "*/*"),
    }
    try:
        with httpx.Client(timeout=min(timeout, 120), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                parts = []
                size = 0
                truncated_note = ""
                hit_cap = False
                for chunk in resp.iter_bytes():
                    room = MAX_RESPONSE_SIZE - size
                    if room <= 0:
                        hit_cap = True
                        truncated_note = f"\n\n[Response truncated at {MAX_RESPONSE_SIZE} bytes]"
                        break
                    parts.append(chunk[:room])
                    size += len(chunk[:room])
                if hit_cap:
                    # keep draining the rest so the connection can close cleanly
                    for _ in resp.iter_bytes():
                        pass
    except httpx.HTTPStatusError as e:
        msg = f"Fetch failed: HTTP {e.response.status_code}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}
    except httpx.HTTPError as e:
        msg = f"Fetch failed: {e}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}

    data = b"".join(parts)
    if format == "text":
        try:
            body = data.decode("utf-8", errors="replace")
            if "text/html" in content_type:
                body = _html_to_text(body)
        except Exception:
            body = "Failed to decode response"
    elif format == "html":
        body = data.decode("utf-8", errors="replace")
    else:  # markdown (default)
        try:
            body = data.decode("utf-8", errors="replace")
            if "text/html" in content_type:
                body = _html_to_markdown(body)
        except Exception:
            body = "Failed to decode response"
    return {"output": body + truncated_note, "metadata": {"upgraded_to_https": upgraded}}


def tool() -> Tool:
    description = """- Fetches content from a specified URL
- Takes a URL and optional format as input
- Fetches the URL content, converts to requested format (markdown by default)
- Returns the content in the specified format
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: if another tool is present that offers better web fetching capabilities, is more targeted to the task, or has fewer restrictions, prefer using that tool instead of this one.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - Format options: "markdown" (default), "text", or "html"
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large"""

    def run(input: dict) -> dict:
        return _webfetch(input["url"], input.get("format", "markdown"), int(input.get("timeout") or DEFAULT_TIMEOUT))

    return Tool(
        name="webfetch",
        description=description,
        parameters=schema_with(
            {
                "url": {"type": "string", "description": "The URL to fetch content from"},
                "format": {
                    "type": "string",
                    "description": "The format to return the content in",
                    "enum": ["markdown", "text", "html"],
                    "optional": True,
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)", "optional": True},
            },
            ["url"],
        ),
        run=run,
        permission="webfetch",
    )
