"""Tests for the OpenAI-compatible provider: content-part deltas, the
stream_options flag, and flushing the SSE tail on stream end."""

import json
from unittest import mock

from opencode_py.providers.base import ProviderEvent, Usage
from opencode_py.providers.openai_compat import OpenAICompatProvider, _content_to_text


class FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield from self._chunks


class FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, **k):
        return FakeResponse(self._chunks)


def provider(**kwargs):
    return OpenAICompatProvider(base_url="https://example.com", api_key="k", model="m", **kwargs)


def test_handle_content_string():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"content": "hello"}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello"]


def test_handle_content_list_parts():
    p = provider()
    events = []
    evt = {
        "data": json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            }
        )
    }
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello world"]


def test_content_to_text_variants():
    assert _content_to_text("plain") == "plain"
    assert _content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _content_to_text([{"text": "x"}]) == "x"
    assert _content_to_text(["raw"]) == "raw"
    assert _content_to_text({"unexpected": True}) == "{'unexpected': True}"


def test_handle_reasoning_content():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"reasoning_content": "think..."}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think..."]


def test_usage_in_every_chunk_keeps_content():
    # Some gateways (e.g. the Zen router) attach a usage object to every SSE
    # chunk alongside the content delta; the content must not be dropped.
    p = provider()
    events = []
    usage = Usage()
    for data in [
        {"choices": [{"delta": {"reasoning_content": "think"}}], "usage": {"total_tokens": 1}},
        {"choices": [{"delta": {"content": "hi"}}], "usage": {"total_tokens": 2}},
    ]:
        p._handle_event({"data": json.dumps(data)}, events.append, {}, usage)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think"]
    assert [e.text for e in events if e.kind == "text_delta"] == ["hi"]
    assert usage.total_tokens == 2


def test_usage_only_ping_no_content():
    p = provider()
    events = []
    usage = mock.MagicMock()
    usage.input_tokens = usage.output_tokens = usage.total_tokens = 0
    evt = {"data": json.dumps({"choices": [], "cost": "0", "usage": {"total_tokens": 5}})}
    p._handle_event(evt, events.append, {}, usage)
    assert events == []
    assert usage.total_tokens == 5


def test_build_payload_stream_options_on_by_default():
    p = provider()
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert payload["stream_options"] == {"include_usage": True}


def test_build_payload_stream_options_disableable():
    p = provider(include_usage=False)
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert "stream_options" not in payload


def test_stream_flushes_tail_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hello"


def test_stream_tail_done_sentinel_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]",  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hi"
