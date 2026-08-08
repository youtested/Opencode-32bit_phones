"""History building + context trimming."""

from __future__ import annotations

import json
from typing import Any

from ..util.truncate import estimate_tokens


def build_messages(
    *,
    history: list[dict[str, Any]],
    user_text: str,
    reminder: str | None = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI-style message payload.

    `history` is the prior conversation as OpenAI-style messages. `user_text` is
    the new turn. If `reminder` (plan/build-switch system-reminder) is given it is
    appended to the user message content (mirrors opencode's synthetic part).
    """
    user_content = user_text
    if reminder:
        user_content = f"{user_text}\n\n{reminder}"

    messages = list(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def trim_history(history: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Drop oldest messages until the token estimate fits the budget.

    Always keeps at least the last 2 messages (a user + assistant pair) unless
    the history is smaller.
    """
    if budget <= 0:
        return history
    if len(history) <= 2:
        return history
    total = sum(estimate_tokens(_text(m)) for m in history)
    if total <= budget:
        return history
    trimmed = list(history)
    while len(trimmed) > 2 and sum(estimate_tokens(_text(m)) for m in trimmed) > budget:
        trimmed.pop(0)
    return trimmed


def _text(message: dict[str, Any]) -> str:
    """Serialize a message to text for token estimation.

    Counts content, reasoning, tool_calls (in assistant messages) and tool
    results so trimming under a tight context budget doesn't overtrim — or
    worse, underestimate real usage enough to overflow the window.
    """
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        text_parts.append(
            " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        )
    else:
        text_parts.append(str(content))
    text_parts.append(reasoning)

    # tool_calls embedded in assistant messages (large JSON blobs)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            text_parts.append(json.dumps(tool_calls, sort_keys=True))
        except (TypeError, ValueError):
            text_parts.append(str(tool_calls))

    # tool result metadata that carries substance (id, name)
    for key in ("tool_call_id", "name"):
        val = message.get(key)
        if val:
            text_parts.append(str(val))

    return " ".join(p for p in text_parts if p is not None and p != "")
