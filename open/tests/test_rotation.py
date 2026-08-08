"""Tests for the provider failover rotation. Regression coverage for:
- in-band error events treated as "empty response" (swallowing the real cause)
- reasoning-only responses treated as empty and rotated
- non-retryable errors (400) aborting the whole rotation
- misleading rate-limit classification
"""

from opencode_py.providers.base import ProviderError, ProviderEvent, RateLimitError
from opencode_py.providers.rotation import Rotation
from opencode_py.providers.rotation import build_rotation as build_default_rotation


class FakeProvider:
    """Emits a fixed list of events or raises a fixed exception on stream_chat."""

    def __init__(self, events=None, exc=None):
        self.events = events or []
        self.exc = exc

    def stream_chat(self, messages, tools, on_event):
        if self.exc is not None:
            raise self.exc
        for e in self.events:
            on_event(e)


def build_rotation(providers):
    it = iter(providers)

    def make(pid, model):
        return next(it)

    lanes = [{"provider": f"p{i}", "model": "m"} for i in range(len(providers))]
    return Rotation(lanes=lanes, make_provider=make)


def test_first_lane_success_no_notice():
    rot = build_rotation([FakeProvider(events=[ProviderEvent(kind="text_delta", text="hi")])])
    got = []
    notices = []
    pid, mid = rot.stream([], [], got.append, lambda p, m, r: notices.append((p, m, r)))
    assert pid == "p0"
    assert mid == "m"
    assert [e.text for e in got] == ["hi"]
    assert notices == []


def test_inband_error_fails_over_and_reason_surfaces():
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="insufficient_quota: free limit reached")]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    got = []
    notices = []
    pid, _ = rot.stream([], [], got.append, lambda p, m, r: notices.append((p, m, r)))
    assert pid == "p1"
    assert [e.text for e in got] == ["backup"]
    assert notices and notices[0][0] == "p1"
    assert "rate limit" in notices[0][2]


def test_primary_transient_error_does_not_rotate():
    """A transient overload on the user's chosen lane must surface the real
    cause, NOT silently route them onto a backup model."""
    class BoomProvider:
        def stream_chat(self, messages, tools, on_event):
            on_event(ProviderEvent(kind="error", error="server_is_overloaded: busy"))

    rot = build_rotation([
        BoomProvider(),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError on transient overload")
    except ProviderError as e:
        assert "server_is_overloaded" in str(e)


def test_error_only_lane_combined_message_preserves_cause():
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="rate limited gateway")]),
        FakeProvider(events=[]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        text = str(e)
        assert "rate limited gateway" in text
        assert "empty response" in text


def test_reasoning_only_counts_as_output():
    rot = build_rotation([FakeProvider(events=[ProviderEvent(kind="reasoning_delta", text="thinking...")])])
    got = []
    pid, _ = rot.stream([], [], got.append, None)
    assert pid == "p0"
    assert [e.text for e in got] == ["thinking..."]


def test_primary_empty_response_does_not_rotate():
    """An empty reply from the chosen lane is a transient miss — surface it,
    don't silently switch to another model."""
    rot = build_rotation([
        FakeProvider(events=[]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError on empty primary")
    except ProviderError as e:
        assert "empty response" in str(e)


def test_empty_backup_lane_is_skipped():
    """An empty reply from a backup lane must not block the chain."""
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("limit")),
        FakeProvider(events=[]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p2"


def test_all_rate_limited_raises_rate_limit():
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("boom1")),
        FakeProvider(exc=RateLimitError("boom2")),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected RateLimitError")
    except RateLimitError:
        pass


def test_non_retryable_400_fails_over():
    rot = build_rotation([
        FakeProvider(exc=ProviderError("bad model id", status=400)),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p1"


def test_mixed_failures_raise_provider_error_not_rate_limit():
    rot = build_rotation([
        FakeProvider(exc=ProviderError("oops", retryable=True)),
        FakeProvider(exc=RateLimitError("later")),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError:
        pass
    except RateLimitError:
        raise AssertionError("mixed failures must not be reported as rate limit")


def test_all_failed_message_lists_every_lane():
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("rl1")),
        FakeProvider(exc=ProviderError("bad model id", status=400)),
        FakeProvider(exc=ProviderError("timeout", retryable=True)),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        text = str(e)
        assert "rl1" in text and "bad model id" in text and "timeout" in text


def test_buffered_events_replayed_in_order():
    rot = build_rotation([
        FakeProvider(events=[
            ProviderEvent(kind="text_delta", text="a"),
            ProviderEvent(kind="text_delta", text="b"),
            ProviderEvent(kind="done", finish_reason="stop"),
        ]),
    ])
    got = []
    rot.stream([], [], got.append, None)
    assert [e.text for e in got if e.kind == "text_delta"] == ["a", "b"]
    assert any(e.kind == "done" for e in got)


def test_partial_text_then_error_keeps_text():
    rot = build_rotation([
        FakeProvider(events=[
            ProviderEvent(kind="text_delta", text="partial"),
            ProviderEvent(kind="error", error="cut off"),
        ]),
    ])
    got = []
    pid, _ = rot.stream([], [], got.append, None)
    assert pid == "p0"
    kinds = [e.kind for e in got]
    assert "text_delta" in kinds and "error" in kinds


def test_all_failed_hint_mentions_rotation():
    rot = build_rotation([FakeProvider(events=[])])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        assert "rotation" in str(e)


def test_auto_rotation_does_not_duplicate_primary_model():
    """The default opencode provider rotation must keep the user's selected
    model exactly once as the primary lane (a prefixed 'opencode/...' model id
    must never be auto-appended again as a separate free-model lane)."""
    from opencode_py.config import Config

    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "opencode/deepseek-v4-flash-free"
    rot = build_default_rotation(cfg)
    ids = [l.get("model", "").split("/", 1)[-1] for l in rot.lanes]
    assert ids.count("deepseek-v4-flash-free") == 1
    assert ids[0] == "deepseek-v4-flash-free"
