"""OpenCode Zen provider (https://opencode.ai/zen/v1) — the free models.

Thin wrapper over OpenAICompatProvider pointing at Zen's OpenAI-compatible
endpoint. With no API key, free (cost==0) models are used and Zen accepts the
literal API key "public" (mirrors opencode's behavior).
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"

# Limited-time free models on Zen ($0). Live-fetched in factory; this is the
# bundled fallback for when the network model list is unavailable (R2 risk).
FREE_MODELS: list[dict] = [
    {"id": "big-pickle", "name": "Big Pickle", "context": 200000, "output": 32000},
    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash Free", "context": 200000, "output": 128000},
    {"id": "mimo-v2.5-free", "name": "MiMo-V2.5 Free", "context": 200000, "output": 32000},
    {"id": "laguna-s-2.1-free", "name": "Laguna S 2.1 Free", "context": 200000, "output": 32000},
    {"id": "ling-3.0-flash-free", "name": "Ling-3.0-flash Free", "context": 200000, "output": 32000},
    {"id": "north-mini-code-free", "name": "North Mini Code Free", "context": 256000, "output": 64000},
    {"id": "nemotron-3-ultra-free", "name": "Nemotron 3 Ultra Free", "context": 1000000, "output": 128000},
]


class ZenProvider(OpenAICompatProvider):
    def __init__(self, *, api_key: str | None = None, model: str = "deepseek-v4-flash-free", **kwargs: Any):
        # If no key given, we still need SOMETHING; Zen accepts "public" for free models.
        effective_key = api_key or "public"
        super().__init__(
            id="opencode",
            name="OpenCode Zen",
            base_url=ZEN_BASE_URL,
            api_key=effective_key,
            model=model,
            is_free=True,
            **kwargs,
        )
        self.has_key = bool(api_key)
