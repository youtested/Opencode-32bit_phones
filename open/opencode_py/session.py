"""Session persistence: JSON files under the data dir.

A session stores: id, title, created/completed times, directory, provider,
model, agent, and the OpenAI-style message history. Auto-save after each turn.

Sub-agent sessions (spawned via the `task` tool) are regular sessions with a
`parent_id` pointing at the session that spawned them.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .globals import Path as GPath


class Session:
    def __init__(self, data: dict[str, Any], directory: str | None = None):
        self.id = data.get("id", uuid.uuid4().hex)
        self.title = data.get("title", "")
        self.created = data.get("created", time.time())
        self.completed = data.get("completed")
        self.directory = data.get("directory") or directory or ""
        self.provider = data.get("provider", "")
        self.model = data.get("model", "")
        self.agent = data.get("agent", "build")
        self.parent_id = data.get("parent_id")
        self.messages: list[dict[str, Any]] = data.get("messages", [])
        self.metadata: dict[str, Any] = data.get("metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "completed": self.completed,
            "directory": self.directory,
            "provider": self.provider,
            "model": self.model,
            "agent": self.agent,
            "parent_id": self.parent_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @property
    def path(self) -> Path:
        return session_path(self.id)


def session_path(session_id: str) -> Path:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        safe_id = "invalid"
    return GPath.sessions_dir() / f"{safe_id}.json"


def save_session(session: Session) -> Path:
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    path = session_path(session.id)
    path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    return path


def load_session(session_id: str) -> Session | None:
    path = session_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Session(data)


def list_sessions() -> list[Session]:
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in GPath.sessions_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(Session(data))
        except (OSError, json.JSONDecodeError):
            continue
    sessions.sort(key=lambda s: s.created, reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    path = session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def new_session(
    *,
    directory: str | None = None,
    provider: str = "",
    model: str = "",
    agent: str = "build",
    title: str = "",
    parent_id: str | None = None,
) -> Session:
    return Session(
        {
            "id": uuid.uuid4().hex,
            "title": title,
            "created": time.time(),
            "directory": directory,
            "provider": provider,
            "model": model,
            "agent": agent,
            "parent_id": parent_id,
            "messages": [],
        }
    )
