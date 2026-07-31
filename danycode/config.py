from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".danycode"
CONFIG_FILE = CONFIG_DIR / "config.toml"
SESSIONS_DIR = CONFIG_DIR / "sessions"

DEFAULTS = {
    "host": "http://localhost:11434",
    "model": "",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "min_p": 0.0,
    "num_ctx": 8192,
    "num_predict": 4096,
    "seed": 1,
    "think": "false",
    "keep_alive": "5m",
    "system_prompt": "Coding assistant. Use tools. Be concise.",
    "mode": "ask",
    "tool_result_limit": 500,
}

VALID_MODES = ("yolo", "ask")
VALID_THINK = ("false", "true", "high", "medium", "low", "max")

_FLOAT_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
}

_INT_MIN = {
    "top_k": 0,
    "num_ctx": 1,
    "num_predict": 1,
    "seed": None,
    "tool_result_limit": 1,
}


def _warn(message: str) -> None:
    print(f"[danycode] {message}", file=sys.stderr)


def _validated_value(key: str, raw) -> tuple[bool, object]:
    if key in _FLOAT_RANGES:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return False, None
        value = float(raw)
        lo, hi = _FLOAT_RANGES[key]
        return (lo <= value <= hi), value
    if key in _INT_MIN:
        if not isinstance(raw, int) or isinstance(raw, bool):
            return False, None
        lo = _INT_MIN[key]
        return (lo is None or raw >= lo), raw
    if key == "think":
        return (isinstance(raw, str) and raw in VALID_THINK), raw
    if key == "mode":
        return (isinstance(raw, str) and raw in VALID_MODES), raw
    if key in ("host", "model", "keep_alive", "system_prompt"):
        return isinstance(raw, str), raw
    return False, None


@dataclass
class Config:
    host: str = DEFAULTS["host"]
    model: str = DEFAULTS["model"]
    temperature: float = DEFAULTS["temperature"]
    top_p: float = DEFAULTS["top_p"]
    top_k: int = DEFAULTS["top_k"]
    min_p: float = DEFAULTS["min_p"]
    num_ctx: int = DEFAULTS["num_ctx"]
    num_predict: int = DEFAULTS["num_predict"]
    seed: int = DEFAULTS["seed"]
    think: str = DEFAULTS["think"]
    keep_alive: str = DEFAULTS["keep_alive"]
    system_prompt: str = DEFAULTS["system_prompt"]
    mode: str = DEFAULTS["mode"]
    tool_result_limit: int = DEFAULTS["tool_result_limit"]

    @classmethod
    def load(cls, overrides: dict | None = None) -> Config:
        data = dict(DEFAULTS)
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "rb") as f:
                    file_cfg = tomllib.load(f)
            except Exception as e:
                file_cfg = {}
                _warn(f"Failed to read {CONFIG_FILE}: {e}")
            for key in DEFAULTS:
                if key not in file_cfg:
                    continue
                ok, value = _validated_value(key, file_cfg[key])
                if ok:
                    data[key] = value
                else:
                    _warn(
                        f"Ignoring invalid config value for '{key}': {file_cfg[key]!r}"
                    )
        if overrides:
            for key, val in overrides.items():
                if val is None or key not in DEFAULTS:
                    continue
                ok, value = _validated_value(key, val)
                if ok:
                    data[key] = value
                else:
                    _warn(f"Ignoring invalid override for '{key}': {val!r}")
        return cls(**data)

    def ensure_dirs(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def update(self, key: str, value: str) -> str | None:
        if key not in DEFAULTS:
            return f"Unknown parameter: {key}. Valid: {', '.join(DEFAULTS.keys())}"
        if key == "temperature":
            try:
                v = float(value)
                if not (0.0 <= v <= 2.0):
                    return "temperature must be between 0.0 and 2.0"
                self.temperature = v
            except ValueError:
                return "temperature must be a number"
        elif key == "top_p":
            try:
                v = float(value)
                if not (0.0 <= v <= 1.0):
                    return "top_p must be between 0.0 and 1.0"
                self.top_p = v
            except ValueError:
                return "top_p must be a number"
        elif key == "top_k":
            try:
                v = int(value)
                if v < 0:
                    return "top_k must be >= 0"
                self.top_k = v
            except ValueError:
                return "top_k must be an integer"
        elif key == "min_p":
            try:
                v = float(value)
                if not (0.0 <= v <= 1.0):
                    return "min_p must be between 0.0 and 1.0"
                self.min_p = v
            except ValueError:
                return "min_p must be a number"
        elif key == "num_ctx":
            try:
                v = int(value)
                if v <= 0:
                    return "num_ctx must be positive"
                self.num_ctx = v
            except ValueError:
                return "num_ctx must be an integer"
        elif key == "num_predict":
            try:
                v = int(value)
                if v <= 0:
                    return "num_predict must be positive"
                self.num_predict = v
            except ValueError:
                return "num_predict must be an integer"
        elif key == "seed":
            try:
                v = int(value)
                self.seed = v
            except ValueError:
                return "seed must be an integer (-1 for random)"
        elif key == "think":
            if value not in VALID_THINK:
                return f"think must be one of: {', '.join(VALID_THINK)}"
            self.think = value
        elif key == "keep_alive":
            self.keep_alive = value
        elif key == "mode":
            if value not in VALID_MODES:
                return f"mode must be one of: {', '.join(VALID_MODES)}"
            self.mode = value
        elif key == "model":
            self.model = value
        elif key == "host":
            self.host = value
        elif key == "system_prompt":
            self.system_prompt = value
        elif key == "tool_result_limit":
            try:
                v = int(value)
                if v <= 0:
                    return "tool_result_limit must be positive"
                self.tool_result_limit = v
            except ValueError:
                return "tool_result_limit must be an integer"
        return None

    def save(self) -> None:
        self.ensure_dirs()
        lines = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, str):
                escaped = (
                    val.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                )
                lines.append(f'{f.name} = "{escaped}"')
            elif isinstance(val, float):
                lines.append(f"{f.name} = {val}")
            elif isinstance(val, int):
                lines.append(f"{f.name} = {val}")
        CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def display(self) -> list[tuple[str, str]]:
        sp = self.system_prompt
        if len(sp) > 80:
            sp = sp[:80] + "..."
        return [
            ("host", self.host),
            ("model", self.model or "(auto)"),
            ("temperature", str(self.temperature)),
            ("top_p", str(self.top_p)),
            ("top_k", str(self.top_k)),
            ("min_p", str(self.min_p)),
            ("num_ctx", str(self.num_ctx)),
            ("num_predict", str(self.num_predict)),
            ("seed", str(self.seed)),
            ("think", self.think),
            ("keep_alive", self.keep_alive),
            ("mode", self.mode),
            ("tool_result_limit", str(self.tool_result_limit)),
            ("system_prompt", sp),
        ]
