"""Native Anthropic Messages API provider (SSE streaming).

Events: message_start / content_block_start / content_block_delta /
content_block_stop / message_delta / message_stop / ping / error.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from ..util.sse import SSEDecoder
from .base import ProviderError, ProviderEvent, RateLimitError, ToolCall, Usage

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

ANTHROPIC_HEADERS = {
    "anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
}


class AnthropicProvider:
    def __init__(
        self,
        *,
        id: str = "anthropic",
        name: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        model: str = "",
        is_free: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ):
        self.id = id
        self.name = name or id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.is_free = is_free
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream",
        }
        headers.update(ANTHROPIC_HEADERS)
        headers.update(self.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/messages"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Anthropic wants messages without "system" role; system goes in `system`.
        system_parts: list[str] = []
        body_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_parts.append(part.get("text", ""))
                continue
            if role == "tool":
                # tool results -> user role with tool_result blocks
                body_messages.append(_to_anthropic_tool_message(content))
                continue
            body_messages.append({"role": role, "content": content})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": body_messages,
            "stream": True,
            "max_tokens": 32000,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [_to_anthropic_tool(t) for t in tools]
            payload["tool_choice"] = {"type": "auto"}
        payload.update(kwargs)
        return payload

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_event: Callable[[ProviderEvent], None] | None = None,
    ) -> None:
        events: list[ProviderEvent] = []
        sink = on_event or (lambda e: events.append(e))
        payload = self.build_payload(messages, tools)
        decoder = SSEDecoder()
        tool_blocks: dict[int, ToolCall] = {}
        usage = Usage()
        stop_reason = ""
        done_emitted = [False]

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            try:
                with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                    self._check_status(resp)
                    for chunk in resp.iter_bytes():
                        for evt in decoder.feed(chunk):
                            self._handle(evt, sink, tool_blocks, usage, done_emitted)
                    # flush any event that arrived without a trailing newline
                    for evt in decoder.close():
                        self._handle(evt, sink, tool_blocks, usage, done_emitted)
            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                raise ProviderError(f"timeout talking to {self.name}: {e}", retryable=True) from e
            except httpx.HTTPStatusError as e:
                self._raise_http(e)
            except httpx.HTTPError as e:
                raise ProviderError(f"network error talking to {self.name}: {e}", retryable=True) from e

        if tool_blocks:
            sink(ProviderEvent(kind="tool_call", tool_calls=list(tool_blocks.values())))
        if usage.total_tokens or usage.input_tokens or usage.output_tokens:
            sink(ProviderEvent(kind="usage", usage=usage))
        if not done_emitted[0]:
            sink(ProviderEvent(kind="done", finish_reason=stop_reason or "stop"))

    def _check_status(self, resp: httpx.Response) -> None:
        if resp.status_code in (200, 201):
            return
        body = ""
        try:
            body = resp.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            try:
                ra = float(retry_after) if retry_after else None
            except ValueError:
                ra = None
            raise RateLimitError(f"{self.name} rate limited (429): {body}", retry_after=ra)
        if resp.status_code >= 500:
            raise ProviderError(
                f"{self.name} server error ({resp.status_code}): {body}", retryable=True, status=resp.status_code
            )
        raise ProviderError(f"{self.name} error ({resp.status_code}): {body}", status=resp.status_code)

    def _raise_http(self, e: httpx.HTTPStatusError) -> None:
        resp = e.response
        try:
            body = resp.text[:500] if resp.text else ""
        except Exception:
            body = ""
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            try:
                ra = float(retry_after) if retry_after else None
            except ValueError:
                ra = None
            raise RateLimitError(f"{self.name} rate limited (429): {body}", retry_after=ra) from e
        if resp.status_code >= 500:
            raise ProviderError(
                f"{self.name} server error ({resp.status_code}): {body}",
                retryable=True,
                status=resp.status_code,
            ) from e
        raise ProviderError(f"{self.name} error ({resp.status_code}): {body}", status=resp.status_code) from e

    def _handle(
        self,
        evt: dict,
        sink: Callable[[ProviderEvent], None],
        tool_blocks: dict[int, ToolCall],
        usage: Usage,
        done_emitted: list[bool] | None = None,
    ) -> None:
        event_type = evt.get("event", "message")
        data = evt.get("data", "")
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return

        if event_type == "message_start":
            msg = obj.get("message", {})
            u = msg.get("usage", {})
            usage.input_tokens = int(u.get("input_tokens", 0) or 0)
            usage.output_tokens = int(u.get("output_tokens", 0) or 0)
            usage.total_tokens = usage.input_tokens + usage.output_tokens
            usage.raw = u
            return
        if event_type == "message_delta":
            delta = obj.get("delta", {})
            if delta.get("stop_reason"):
                if done_emitted is not None:
                    done_emitted[0] = True
                sink(ProviderEvent(kind="done", finish_reason=delta["stop_reason"]))
            u = obj.get("usage", {})
            usage.output_tokens = int(u.get("output_tokens", 0) or 0)
            usage.total_tokens = usage.input_tokens + usage.output_tokens
            return
        if event_type == "content_block_start":
            block = obj.get("content_block", {})
            index = int(obj.get("index", 0))
            if block.get("type") == "tool_use":
                tool_blocks[index] = ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                    index=index,
                )
            return
        if event_type == "content_block_delta":
            index = int(obj.get("index", 0))
            delta = obj.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                sink(ProviderEvent(kind="text_delta", text=delta.get("text", "")))
            elif dtype == "thinking_delta":
                sink(ProviderEvent(kind="reasoning_delta", text=delta.get("thinking", "")))
            elif dtype == "input_json_delta" and index in tool_blocks:
                tool_blocks[index].arguments += delta.get("partial_json", "")
            return
        if event_type == "error":
            err = obj.get("error", {})
            message = err.get("message", "unknown error")
            sink(ProviderEvent(kind="error", error=message))


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style tool schema to Anthropic tool schema."""
    return {
        "name": tool["function"]["name"],
        "description": tool["function"].get("description", ""),
        "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
    }


def _to_anthropic_tool_message(content: Any) -> dict[str, Any]:
    """Convert a tool-result message into an Anthropic user message with tool_result blocks."""
    if isinstance(content, str):
        return {"role": "user", "content": content}
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part.get("tool_call_id", ""),
                        "content": part.get("content", ""),
                    }
                )
        if blocks:
            return {"role": "user", "content": blocks}
    return {"role": "user", "content": str(content)}
