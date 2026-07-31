from __future__ import annotations

import json
import re
from pathlib import Path

from danycode.config import SESSIONS_DIR

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Session:
    def __init__(self, name: str):
        self.name = name if _SAFE_NAME_RE.match(name) else "default"
        self.path = SESSIONS_DIR / f"{self.name}.json"
        self.messages: list[dict] = []
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.messages = [m for m in data if isinstance(m, dict)]
            except (OSError, ValueError):
                self.messages = []

    def add(self, message: dict) -> None:
        self.messages.append(message)
        self.save()

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def remove_last(self) -> None:
        if self.messages:
            self.messages.pop()
            self.save()

    def clear(self) -> None:
        self.messages = []
        self.save()

    @staticmethod
    def list_sessions() -> list[str]:
        if not SESSIONS_DIR.exists():
            return []
        return [p.stem for p in SESSIONS_DIR.glob("*.json")]
