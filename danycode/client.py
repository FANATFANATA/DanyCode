from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx

from danycode.config import Config
from danycode.tools import TOOL_SCHEMAS


class OllamaClient:
    def __init__(self, config: Config):
        self.config = config

    @property
    def base_url(self) -> str:
        return self.config.host.rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                return resp.status_code == 200
        except Exception:
            return False

    async def version(self) -> str:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{self.base_url}/api/version")
            resp.raise_for_status()
            return resp.json().get("version", "unknown")

    async def list_models(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])

    async def list_models_with_tools(self) -> list[dict]:
        models = await self.list_models()

        async def _check(m: dict) -> dict:
            try:
                info = await self.show_model(m["name"])
                m["capabilities"] = info.get("capabilities", [])
            except Exception:
                m["capabilities"] = []
            return m

        checked = await asyncio.gather(*[_check(m) for m in models])
        return [m for m in checked if "tools" in m.get("capabilities", [])]

    async def list_running(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", [])

    async def show_model(self, name: str) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{self.base_url}/api/show", json={"model": name})
            resp.raise_for_status()
            return resp.json()

    def _build_options(self) -> dict:
        opts: dict = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "num_ctx": self.config.num_ctx,
            "num_predict": self.config.num_predict,
        }
        if self.config.seed >= 0:
            opts["seed"] = self.config.seed
        return opts

    def _build_payload(self, messages: list[dict], stream: bool) -> dict:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": stream,
            "options": self._build_options(),
        }
        if self.config.think == "false":
            payload["think"] = False
        elif self.config.think == "true":
            payload["think"] = True
        else:
            payload["think"] = self.config.think
        if self.config.keep_alive:
            payload["keep_alive"] = self.config.keep_alive
        return payload

    async def chat(self, messages: list[dict]) -> dict:
        payload = self._build_payload(messages, stream=False)
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[dict]:
        payload = self._build_payload(messages, stream=True)
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if stripped:
                        try:
                            yield json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
