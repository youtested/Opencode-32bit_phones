"""Provider factory + registry + failover rotation.

rotation list: [{"provider": "zen", "model": "..."}, {"provider": "groq", "model": "..."}, ...]
On a real rate limit (429 / "limit reached") the engine tries the next lane.
Transient hiccups (timeout, 5xx, overload, empty reply) keep the current model.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from ..config import Config
from .base import ProviderError, ProviderEvent, RateLimitError
from .openai_compat import OpenAICompatProvider
from .zen import FREE_MODELS, ZEN_BASE_URL, ZenProvider
from .ollama import OllamaProvider
from .anthropic import AnthropicProvider

# free-tier direct providers (OpenAI-compatible). Env var names match auth.py.
FREE_PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "env": ("GROQ_API_KEY",)},
    "cerebras": {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "env": ("CEREBRAS_API_KEY",)},
    "google": {
        "name": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env": ("OPENROUTER_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env": ("NVIDIA_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "mistral": {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "env": ("MISTRAL_API_KEY",)},
    "github": {"name": "GitHub Models", "base_url": "https://models.github.ai/inference", "env": ("GITHUB_TOKEN",)},
    "sambanova": {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "env": ("SAMBANOVA_API_KEY",)},
    "togetherai": {"name": "Together", "base_url": "https://api.together.xyz/v1", "env": ("TOGETHER_API_KEY",)},
}

# default free models per provider (Phase 3 preload)
FREE_DEFAULT_MODELS: dict[str, str] = {
    "zen": "deepseek-v4-flash-free",
    "groq": "llama-3.3-70b-versatile",
    "cerebras": "llama-3.3-70b",
    "google": "gemini-2.5-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia": "nemotron-3-ultra-free",
    "mistral": "codestral-latest",
    "github": "gpt-4o-mini",
    "sambanova": "Meta-Llama-3.3-70B-Instruct",
    "togetherai": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# paid (bring-your-own-key) providers with their own /models endpoints.
PAID_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "env": ("ANTHROPIC_API_KEY",),
        "api_kind": "anthropic",
    },
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "env": ("OPENAI_API_KEY",)},
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "env": ("DEEPSEEK_API_KEY",)},
    "xai": {"name": "xAI", "base_url": "https://api.x.ai/v1", "env": ("XAI_API_KEY",)},
    "deepinfra": {"name": "DeepInfra", "base_url": "https://api.deepinfra.com/v1/openai", "env": ("DEEPINFRA_API_KEY",)},
}


# markers that identify an in-band provider error as a rate limit / quota
# exhaustion (the "reached your limit" case). Everything else is treated as a
# transient failure (overload, capacity) and must NOT rotate off the model.
_RATE_MARKERS = ("rate", "quota", "limit", "429", "too many", "throttl", "insufficient")


def _is_rate_limit_message(message: str) -> bool:
    low = (message or "").lower()
    return any(m in low for m in _RATE_MARKERS)


def _fail_message(provider_id: str, error: Exception) -> str:
    """Message for a surfaced failure; keeps prior lane failures for context."""
    return f"{provider_id}: {error}"


class Rotation:
    """Try lanes in order on rate limits / dead lanes.

    Rotation happens ONLY when the current lane genuinely can't serve:
      - the lane is rate limited / quota exhausted (429 or an in-band
        "limit reached" error), or
      - the lane is permanently broken (bad model id, bad key, dead endpoint).
    Transient failures (timeout, 5xx, network, overload, empty reply) on the
    PRIMARY lane are surfaced instead of silently rotating the user onto a
    different model. Backup lanes that fail for any reason are simply skipped
    so a dead free model never blocks the chain.
    """

    def __init__(self, lanes: list[dict[str, str]], make_provider: Callable[[str, str], Any]):
        self.lanes = lanes
        self.make_provider = make_provider

    @property
    def first(self) -> Any | None:
        if not self.lanes:
            return None
        l = self.lanes[0]
        return self.make_provider(l.get("provider", "zen"), l.get("model", FREE_DEFAULT_MODELS["zen"]))

    def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        on_event: Callable[[ProviderEvent], None],
        on_notice: Callable[[str, str, str], None] | None = None,
    ) -> tuple[str, str]:
        """Stream across lanes; returns (provider_id, model_id) that succeeded.

        `on_notice(provider_id, model, reason)` is called when a lane other
        than the first succeeds (a failover happened), with a short reason for
        the switch so the UI can announce it accurately.

        A lane fails over only when it genuinely can't serve: a real rate limit
        (429 or an in-band "limit reached" error) or a hard error (bad model
        id, bad key, dead endpoint). Transient failures on the user's chosen
        lane (timeout, 5xx, overload, empty reply) are raised so the real cause
        surfaces instead of silently routing the user onto another model.
        Temporarily-down backup lanes are skipped so a dead free model never
        blocks the chain.
        """
        errors: list[str] = []
        saw_rate_limit = False
        saw_other = False
        last_reason = ""
        for index, lane in enumerate(self.lanes):
            provider_id = lane.get("provider", "zen")
            model = lane.get("model", FREE_DEFAULT_MODELS.get(provider_id, FREE_DEFAULT_MODELS["zen"]))
            had_output = False
            had_error = False
            error_message = ""
            buffered: list[ProviderEvent] = []

            def wrapped(evt: ProviderEvent) -> None:
                nonlocal had_output, had_error, error_message
                # text, reasoning, or a real tool call = usable output.
                # Reasoning counts too: a reasoning model cut off before any
                # content must not be treated as "empty" and silently dropped.
                if evt.kind in ("text_delta", "reasoning_delta", "tool_call"):
                    had_output = True
                elif evt.kind == "error":
                    had_error = True
                    error_message = evt.error or error_message
                # Buffer every event and only replay them once this lane is
                # known to have completed successfully. Forwarding streamed text
                # to the UI before we know whether the provider will finish
                # leaks partial output from a lane that later fails, so the next
                # lane's answer is appended on top and the UI shows duplicate
                # responses for one prompt.
                buffered.append(evt)

            try:
                provider = self.make_provider(provider_id, model)
                provider.stream_chat(messages, tools, wrapped)
                if had_error and not had_output:
                    # the lane errored before producing any usable output
                    if _is_rate_limit_message(error_message):
                        raise RateLimitError(f"{provider_id}: {error_message or 'rate limited'}")
                    raise ProviderError(
                        f"{provider_id}: {error_message or 'error response'}",
                        retryable=True,
                    )
                if not had_output:
                    # the lane "succeeded" but produced nothing usable
                    raise ProviderError(f"{provider_id}: empty response", retryable=True)
                for evt in buffered:
                    on_event(evt)
                if index > 0 and on_notice:
                    on_notice(provider_id, model, last_reason or "provider error")
                return provider_id, model
            except RateLimitError as e:
                # Real limit reached (429 or in-band "limit reached") -> the
                # lane is done, move to the next one.
                errors.append(f"{provider_id}: rate limited ({e})")
                saw_rate_limit = True
                last_reason = "rate limited"
                continue
            except ProviderError as e:
                # Only permanently broken lanes (bad model id / bad key / dead
                # endpoint) rotate. Transient failures (timeout, 5xx, overload,
                # empty reply) must NOT silently move the user off the model
                # they picked, especially the primary lane.
                if e.retryable:
                    if index == 0:
                        # the user's chosen model hiccuped — surface the real
                        # cause instead of routing them elsewhere
                        raise ProviderError(
                            f"{provider_id}: {e}\n\nHint: this looks like a"
                            " temporary issue — add another provider to your"
                            " 'rotation' list to fail over, or wait and retry."
                        )
                    # a backup lane that's temporarily down: skip it, the next
                    # one may still answer
                    errors.append(f"{provider_id}: {e}")
                    last_reason = (e.message or str(e))[:120]
                    saw_other = True
                    continue
                # hard (non-retryable) error: model id gone, bad key, dead
                # endpoint — this lane can never answer, rotate on
                errors.append(f"{provider_id}: {e}")
                last_reason = (e.message or str(e))[:120]
                continue
        message = (
            "all providers failed:\n" + "\n".join(errors)
            + "\n\nHint: add another provider to your 'rotation' list to fail over,"
            + " or wait and retry."
        )
        if saw_rate_limit and not saw_other:
            # every failure was a rate limit -> surface as a retryable rate-limit error
            raise RateLimitError(message)
        raise ProviderError(message)


def build_provider(cfg: Config, provider_id: str | None = None, model: str | None = None, auth=None) -> Any:
    """Build a provider instance from config + auth."""
    explicit_provider = provider_id is not None
    provider_id = provider_id or cfg.provider
    model = model or cfg.model
    if not explicit_provider and "/" in model:
        # 'provider/model' shorthand only when the provider isn't already known;
        # avoids breaking model ids that legitimately contain '/' (e.g. OpenRouter).
        provider_id, model = model.split("/", 1)
    elif explicit_provider and model.startswith(provider_id + "/"):
        # Strip the 'provider/' prefix already folded in by _parse_model; it must
        # not be sent to the API (e.g. Zen rejects "opencode/deepseek-v4-flash-free").
        model = model.split("/", 1)[1]
    key = auth.get(provider_id) if auth else None

    if provider_id == "opencode":
        return ZenProvider(api_key=key, model=model)
    if provider_id == "ollama":
        return OllamaProvider(model=model)
    if provider_id == "anthropic":
        return AnthropicProvider(api_key=key, model=model)
    if provider_id == "openai":
        return OpenAICompatProvider(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=key,
            model=model,
            is_free=False,
        )
    if provider_id in FREE_PROVIDERS:
        info = FREE_PROVIDERS[provider_id]
        return OpenAICompatProvider(
            id=provider_id,
            name=info["name"],
            base_url=info["base_url"],
            api_key=key,
            model=model,
            is_free=True,
            extra_headers=info.get("headers", {}),
        )
    # custom provider from config providers.<id>
    custom = cfg.providers.get(provider_id)
    if custom and isinstance(custom, dict):
        base_url = custom.get("base_url") or custom.get("api")
        api_key = custom.get("api_key") or key
        if not base_url:
            raise ProviderError(f"provider {provider_id}: no base_url configured")
        return OpenAICompatProvider(
            id=provider_id,
            name=custom.get("name", provider_id),
            base_url=base_url,
            api_key=api_key,
            model=model,
            extra_headers=custom.get("headers", {}) or {},
        )
    raise ProviderError(f"unknown provider: {provider_id}")


def _has_openrouter_key(auth) -> bool:
    """True if the user has an OpenRouter API key (env or auth.json)."""
    import os

    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    return auth is not None and bool(auth.get("openrouter"))


def build_rotation(cfg: Config, auth=None) -> Rotation:
    """Build a rotation with the picked model as the primary lane.

    - An explicit `rotation` list in config is always honored.
    - Otherwise the current selection (`cfg.provider`/`cfg.model`) is tried
      first, so picking a model at runtime actually uses it.
    - If the user has an OpenRouter API key, the OpenRouter default free model
      is added as a failover lane; for the opencode provider the other free
      models follow too, so an empty/failed response can fail over automatically.
    """
    lanes = list(cfg.rotation)
    if not lanes:
        lanes = [{"provider": cfg.provider, "model": cfg.model}]
        if _has_openrouter_key(auth) and cfg.provider != "openrouter":
            lanes.append({"provider": "openrouter", "model": FREE_DEFAULT_MODELS["openrouter"]})
        if cfg.provider == "opencode":
            # `cfg.model` is stored with the provider prefix folded in (e.g.
            # "opencode/deepseek-v4-flash-free"); compare against the bare id
            # so the currently selected model is NOT duplicated as a lane.
            current = cfg.model.split("/", 1)[-1]
            lanes += [
                {"provider": "opencode", "model": model}
                for model in [m["id"] for m in FREE_MODELS if m["id"] != current]
            ]
    return Rotation(lanes, lambda pid, m: build_provider(cfg, pid, m, auth))


def fetch_zen_models(cache_file=None, ttl_hours: int = 24) -> list[dict]:
    """Live-fetch https://opencode.ai/zen/v1/models with a cached fallback.

    Returns a list of {id, name, context, output, free} sorted with free first.
    """
    import os
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    cache_file = cache_file or GPath.models_file()
    cache_path = Path(cache_file)
    models: list[dict] = []

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            age = time.time() - data.get("ts", 0)
            models = data.get("models", [])
            if age < ttl_hours * 3600 and models:
                return _normalize_models(models)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        resp = httpx.get(f"{ZEN_BASE_URL}/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else data
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"ts": time.time(), "models": models}), encoding="utf-8")
            return _normalize_models(models)
    except Exception:
        pass

    return list(FREE_MODELS)


def fetch_openrouter_models() -> list[dict]:
    """Live-fetch OpenRouter's public model list filtered to free models.

    Returns [{id, name, context, free, provider='openrouter'}] sorted by id.
    The chosen default free model is always included even if the live list is
    unavailable or it happens to be missing.
    """
    import time

    out: list[dict] = []
    default = FREE_DEFAULT_MODELS["openrouter"]
    seen: set[str] = set()

    try:
        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("data", []) if isinstance(data, dict) else data
            for m in raw:
                mid = m.get("id", "")
                if not mid.endswith(":free"):
                    continue
                seen.add(mid)
                out.append(
                    {
                        "id": mid,
                        "name": m.get("name") or mid,
                        "context": (m.get("context_length") or 0) // 1000,
                        "free": True,
                        "provider": "openrouter",
                    }
                )
    except Exception:
        pass

    if default not in seen:
        out.append(
            {
                "id": default,
                "name": default,
                "context": (1000 if "550b" in default else 0),
                "free": True,
                "provider": "openrouter",
            }
        )

    return sorted(out, key=lambda d: d["id"])


def fetch_live_models(
    provider_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    api_kind: str = "openai",
) -> list[dict]:
    """Live-fetch a provider's `GET /models` list.

    Handles OpenAI-compatible endpoints (Bearer auth) and Anthropic
    (x-api-key + anthropic-version). Returns
    ``[{id, name, context, free, provider}]`` (``free=False``) or ``[]`` when
    there is no key, the endpoint fails, or nothing usable comes back. The
    free-focused fetchers for opencode/openrouter are handled separately.
    """
    if not api_key:
        return []
    base_url = (
        base_url
        or FREE_PROVIDERS.get(provider_id, {}).get("base_url")
        or PAID_PROVIDERS.get(provider_id, {}).get("base_url")
    )
    if not base_url:
        return []
    headers = (
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        if api_kind == "anthropic"
        else {"Authorization": f"Bearer {api_key}"}
    )
    try:
        resp = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else (data or [])
        out: list[dict] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or ""
            if not mid:
                continue
            if provider_id == "openai" and not (
                mid.startswith("gpt-") or mid.startswith("o") or mid.startswith("chatgpt-")
            ):
                continue
            if provider_id == "anthropic" and not mid.startswith("claude-"):
                continue
            ctx = m.get("context_length") or m.get("context") or 0
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = 0
            out.append(
                {
                    "id": mid,
                    "name": m.get("display_name") or m.get("name") or mid,
                    "context": ctx,
                    "free": False,
                    "provider": provider_id,
                }
            )
        return out
    except Exception:
        return []


_context_cache: dict[tuple[str, str], int] = {}

# Known context windows for the default free-provider models (failover lanes).
# Used when a provider's live /models list isn't available.
KNOWN_CONTEXT: dict[str, int] = {
    "groq": 131072,
    "cerebras": 131072,
    "google": 1048576,
    "openrouter": 1000000,
    "nvidia": 1000000,
    "mistral": 256000,
    "github": 128000,
    "sambanova": 131072,
    "togetherai": 131072,
    "anthropic": 200000,
    "openai": 128000,
    "deepseek": 128000,
    "xai": 256000,
    "deepinfra": 128000,
    "ollama": 131072,
}


def _known_context(provider_id: str, model_id: str) -> int:
    """Context size from the bundled Zen free-model list or known defaults."""
    if provider_id == "opencode":
        for m in FREE_MODELS:
            if m["id"] == model_id:
                return int(m.get("context") or 0)
    return KNOWN_CONTEXT.get(provider_id, 0)


def model_context_size(provider_id: str, model_id: str) -> int:
    """Best-effort context-window size (in tokens) for a provider/model lane.

    Resolution order: bundled known sizes (no network) -> live provider model
    lists. Returns 0 when unknown (the UI then omits the percentage). Cached
    per (provider, model) so the live model list is only fetched once.
    """
    key = (provider_id, model_id)
    if key in _context_cache:
        return _context_cache[key]
    size = _known_context(provider_id, model_id)
    if size:
        _context_cache[key] = size
        return size
    if provider_id == "opencode":
        for m in fetch_zen_models():
            if m.get("id") == model_id:
                size = int(m.get("context") or 0)
                break
    elif provider_id == "openrouter":
        for m in fetch_openrouter_models():
            if m.get("id") == model_id:
                # fetch_openrouter_models reports context in thousands
                size = int(m.get("context") or 0) * 1000
                break
    _context_cache[key] = size
    return size


def _normalize_models(raw: list[dict]) -> list[dict]:
    fallback = {f["id"]: f for f in FREE_MODELS}
    out = []
    for m in raw:
        cost = m.get("cost") or {}
        limit = m.get("limit") or {}
        fb = fallback.get(m.get("id"), {})
        is_free = (
            (isinstance(cost.get("input"), (int, float)) and cost["input"] == 0)
            or m.get("id") in fallback
            or m.get("free") is True
        )
        out.append(
            {
                "id": m.get("id", ""),
                "name": m.get("name") or fb.get("name") or m.get("id", ""),
                "context": limit.get("context", 0) or m.get("context", 0) or fb.get("context", 0),
                "output": limit.get("output", 0) or m.get("output", 0) or fb.get("output", 0),
                "free": bool(is_free),
                "status": m.get("status", "active"),
                "provider": "opencode",
            }
        )
    return sorted(out, key=lambda x: (not x["free"], x["id"]))


def check_provider(cfg: Config, auth) -> dict[str, Any]:
    """Ping a provider; used by --check. Returns status dict."""
    result: dict[str, Any] = {}
    lanes = cfg.rotation or [{"provider": cfg.provider, "model": cfg.model}]
    for lane in lanes:
        pid = lane.get("provider", "zen")
        model = lane.get("model", FREE_DEFAULT_MODELS.get(pid, "deepseek-v4-flash-free"))
        try:
            provider = build_provider(cfg, pid, model, auth)
            if not provider.api_key and pid != "opencode" and pid != "ollama":
                result[pid] = {"ok": False, "model": model, "error": "no API key (see /connect or env)"}
                continue
            # lightweight models list ping (GET /models) for openai-compat
            try:
                resp = httpx.get(f"{provider.base_url}/models", headers={"Authorization": f"Bearer {provider.api_key}"}, timeout=10)
                ok = resp.status_code == 200
                result[pid] = {"ok": ok, "model": model, "status": resp.status_code}
            except Exception as e:
                result[pid] = {"ok": False, "model": model, "error": str(e)}
        except Exception as e:
            result[pid] = {"ok": False, "model": model, "error": str(e)}
    return result
