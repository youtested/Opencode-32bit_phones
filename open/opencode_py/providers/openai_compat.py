"""OpenAI-compatible chat completions provider with SSE streaming.

One class parameterized by base_url + api_key + model, reused by Groq, Cerebras,
OpenRouter, Google AI Studio, NVIDIA, Mistral, GitHub Models, Together, SambaNova,
OpenAI, DeepSeek, and local Ollama (/v1/chat/completions).

Streams `chat.completion.chunk` events; accumulates tool_calls deltas.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from ..util.sse import SSEDecoder
from .base import ProviderError, ProviderEvent, RateLimitError, ToolCall, Usage

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI-compat delta.content (a string, or a list of content
    parts used by some gateways) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


class OpenAICompatProvider:
    """OpenAI /v1/chat/completions streaming provider."""

    def __init__(
        self,
        *,
        id: str = "openai",
        name: str | None = None,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        is_free: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        include_usage: bool = True,
    ):
        self.id = id
        self.name = name or id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.is_free = is_free
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        # Some OpenAI-compatible gateways reject the non-standard
        # `stream_options` field; disable it per-provider when needed.
        self.include_usage = include_usage

    # -- helpers ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(kwargs)
        return payload

    # -- streaming -------------------------------------------------------
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_event: Callable[[ProviderEvent], None] | None = None,
    ) -> None:
        """Stream a chat completion; dispatch ProviderEvent to on_event.

        If on_event is None, collects and returns via a list (convenience).
        """
        events: list[ProviderEvent] = []
        sink = on_event or (lambda e: events.append(e))
        self._stream(messages, tools, sink)
        if on_event is None:
            return events  # type: ignore[return-value]

    def _stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        sink: Callable[[ProviderEvent], None],
    ) -> None:
        payload = self.build_payload(messages, tools)
        decoder = SSEDecoder()
        tool_calls: dict[int, ToolCall] = {}
        usage: Usage = Usage()
        done_emitted = [False]

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            try:
                with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                    self._check_status(resp)
                    for chunk in resp.iter_bytes():
                        for evt in decoder.feed(chunk):
                            self._handle_event(evt, sink, tool_calls, usage, done_emitted)
                    # flush any event that arrived without a trailing newline,
                    # otherwise the final chunk's content is silently dropped
                    for evt in decoder.close():
                        self._handle_event(evt, sink, tool_calls, usage, done_emitted)
            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                raise ProviderError(f"timeout talking to {self.name}: {e}", retryable=True) from e
            except httpx.HTTPStatusError as e:
                self._raise_http(e)
            except httpx.HTTPError as e:
                raise ProviderError(f"network error talking to {self.name}: {e}", retryable=True) from e

        if tool_calls:
            sink(ProviderEvent(kind="tool_call", tool_calls=list(tool_calls.values())))
        if usage.total_tokens or usage.input_tokens or usage.output_tokens:
            sink(ProviderEvent(kind="usage", usage=usage))
        if not done_emitted[0]:
            sink(ProviderEvent(kind="done", finish_reason="stop"))

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
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
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

    def _handle_event(
        self,
        evt: dict,
        sink: Callable[[ProviderEvent], None],
        tool_calls: dict[int, ToolCall],
        usage: Usage,
        done_emitted: list[bool] | None = None,
    ) -> None:
        data = evt.get("data", "")
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        if obj.get("object") == "error" or "error" in obj:
            err = obj.get("error", {})
            message = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
            code = err.get("code", "") if isinstance(err, dict) else ""
            if code in ("context_length_exceeded", "insufficient_quota", "server_is_overloaded", "server_error"):
                sink(ProviderEvent(kind="error", error=f"{code}: {message}"))
            else:
                sink(ProviderEvent(kind="error", error=message))
            return

        # usage chunk. Some gateways (e.g. the Zen router) attach a usage
        # object to *every* SSE chunk, not just the final one, so we must NOT
        # return here — the same chunk can also carry a content delta. Only
        # record usage and continue; a pure usage/cost ping has no choices and
        # falls through to the `if not choices` guard below.
        if "usage" in obj and obj.get("usage"):
            u = obj["usage"]
            usage.input_tokens = int(u.get("prompt_tokens", 0) or 0)
            usage.output_tokens = int(u.get("completion_tokens", 0) or 0)
            usage.total_tokens = int(u.get("total_tokens", 0) or 0)
            usage.raw = u

        choices = obj.get("choices") or []
        if not choices:
            # possible cost ping: data: {"choices":[],"cost":...}
            return
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}

        if delta.get("role"):
            pass  # first chunk role marker
        if delta.get("content"):
            content = delta["content"]
            if not isinstance(content, str):
                content = _content_to_text(content)
            if content:
                sink(ProviderEvent(kind="text_delta", text=content))
        if delta.get("reasoning_content"):
            sink(ProviderEvent(kind="reasoning_delta", text=delta["reasoning_content"]))
        if delta.get("reasoning"):
            reason = delta["reasoning"]
            if isinstance(reason, str):
                text = reason
            else:
                text = reason.get("content") or reason.get("text") or ""
            if text:
                sink(ProviderEvent(kind="reasoning_delta", text=text))
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                index = tc.get("index", 0)
                call = tool_calls.setdefault(index, ToolCall(id="", name="", arguments="", index=index))
                if tc.get("id"):
                    call.id = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call.name += fn["name"]
                if fn.get("arguments"):
                    call.arguments += fn["arguments"]
        if finish_reason:
            if done_emitted is not None:
                done_emitted[0] = True
            sink(ProviderEvent(kind="done", finish_reason=finish_reason))

    # -- non-streaming convenience --------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Non-streaming completion, returns the JSON response."""
        payload = self.build_payload(messages, tools)
        payload["stream"] = False
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(self._url(), json=payload, headers=self._headers())
            self._check_status(resp)
            return resp.json()
