"""Global paths, version, data dir, and config discovery helpers.

Mirrors opencode's Global.Path / config discovery behavior:
  - config:  ~/.config/opencode_py  (or $XDG_CONFIG_HOME/opencode_py)
  - data:    ~/.local/share/opencode_py  (or $XDG_DATA_HOME/opencode_py)  auth.json + sessions
  - cache:   ~/.cache/opencode_py  (or $XDG_CACHE_HOME/opencode_py)  models.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import platformdirs

__all__ = [
    "VERSION",
    "Path",
    "GLOBAL_DIRS",
    "HOME",
    "Platform",
]

VERSION = "0.1.0"

HOME = Path.home()


class Path:
    config = Path(platformdirs.user_config_dir("opencode_py"))
    data = Path(platformdirs.user_data_dir("opencode_py"))
    cache = Path(platformdirs.user_cache_dir("opencode_py"))
    tmp = Path("/tmp")

    @classmethod
    def init(cls) -> None:
        for d in (cls.config, cls.data, cls.cache):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def auth_file(cls) -> Path:
        return cls.data / "auth.json"

    @classmethod
    def sessions_dir(cls) -> Path:
        return cls.data / "sessions"

    @classmethod
    def models_file(cls) -> Path:
        return cls.cache / "models.json"

    @classmethod
    def truncation_dir(cls) -> Path:
        return cls.data / "truncation"


class Platform:
    @staticmethod
    def name() -> str:
        return sys.platform

    @staticmethod
    def is_windows() -> bool:
        return sys.platform == "win32"


GLOBAL_DIRS = [
    Path.config,
]


def resolve_worktree(directory: Path) -> Path:
    """Walk up from directory to find the git worktree root (else the dir itself)."""
    d = directory.resolve()
    if (d / ".git").exists():
        return d
    for parent in d.parents:
        if (parent / ".git").exists():
            return parent
    return d
