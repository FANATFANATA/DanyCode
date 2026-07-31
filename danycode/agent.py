from __future__ import annotations

import json
import os
import platform

from rich import box
from rich.console import Console
from rich.panel import Panel

from danycode.client import OllamaClient
from danycode.config import Config
from danycode.session import Session
from danycode.tools import execute_tool, get_shell_name

console = Console()

MAX_ITERATIONS = 25
REPEAT_THRESHOLD = 3


class Agent:
    def __init__(self, config: Config, session: Session):
        self.config = config
        self.session = session
        self.client = OllamaClient(config)
        self.last_prompt_tokens = 0
        self.last_eval_tokens = 0
        self._ensure_system_prompt()

    def _ensure_system_prompt(self) -> None:
        if not self.session.messages or self.session.messages[0]["role"] != "system":
            shell_name = get_shell_name()
            env = f"{platform.system()} | {shell_name} | {os.getcwd()}"
            full_prompt = f"{env}\n{self.config.system_prompt}"
            self.session.messages.insert(
                0,
                {
                    "role": "system",
                    "content": full_prompt,
                },
            )
            self.session.save()

    def _ask_user(self, question: str) -> str:
        console.print(
            Panel(
                question, title="Assistant asks", border_style="yellow", box=box.SQUARE
            )
        )
        try:
            return input("Your answer: ")
        except (EOFError, KeyboardInterrupt):
            return ""

    def _confirm_inline(self, fn_name: str, fn_args: dict) -> bool:
        args_str = json.dumps(fn_args, ensure_ascii=False)
        if len(args_str) > self.config.tool_result_limit:
            args_str = args_str[: self.config.tool_result_limit] + "..."
        console.print(
            f"[cyan]\\[{fn_name}] {args_str}[/cyan] [bold]\\[Y/n][/bold] ", end=""
        )
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer in ("", "y", "yes")

    def _print_tool_call(self, fn_name: str, fn_args: dict) -> None:
        args_str = json.dumps(fn_args, ensure_ascii=False)
        if len(args_str) > self.config.tool_result_limit:
            args_str = args_str[: self.config.tool_result_limit] + "..."
        console.print(f"[cyan]\\[{fn_name}] {args_str}[/cyan]")

    def _print_tool_result(self, fn_name: str, result: str) -> None:
        display = result
        if len(display) > self.config.tool_result_limit:
            display = display[: self.config.tool_result_limit] + "..."
        console.print(f"[green]\\[{fn_name} ->] {display}[/green]")

    def _print_request(self, text: str) -> None:
        console.print(
            Panel(text, border_style="yellow", box=box.SQUARE, padding=(0, 1))
        )

    def _print_response(
        self, thinking: str | None, content: str | None, tokens: int | None
    ) -> None:
        parts = []
        if thinking:
            parts.append(f"[dim]{thinking}[/dim]")
        if content:
            parts.append(content)
        body = "\n\n".join(parts) if parts else "[dim](empty)[/dim]"
        subtitle = f"{tokens} tok" if tokens is not None else None
        console.print(
            Panel(
                body,
                border_style="green",
                box=box.SQUARE,
                padding=(0, 1),
                subtitle=subtitle,
                subtitle_align="right",
            )
        )

    def _print_error(self, text: str) -> None:
        console.print(Panel(text, border_style="red", box=box.SQUARE, padding=(0, 1)))

    async def run(self, user_input: str) -> None:
        self._print_request(user_input)
        self.session.add({"role": "user", "content": user_input})
        await self._loop()

    async def _loop(self) -> None:
        iterations = 0
        recent_sigs: list[str] = []
        turn_eval_tokens = 0
        last_thinking = None
        last_content = None
        has_response = False

        def stop_on_repeat(sig: str, mark: int) -> bool:
            recent_sigs.append(sig)
            if len(recent_sigs) > REPEAT_THRESHOLD:
                recent_sigs.pop(0)
            if (
                len(recent_sigs) == REPEAT_THRESHOLD
                and len(set(recent_sigs)) == 1
            ):
                del self.session.messages[mark:]
                self.session.save()
                self._print_error("Repeated identical tool call. Stopping.")
                return True
            return False

        while True:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                self._print_error("Max tool iterations reached.")
                break

            messages = self.session.messages

            try:
                assistant_msg = await self._get_response(messages)
            except Exception as e:
                self._print_error(str(e))
                return

            self.last_prompt_tokens = assistant_msg.get("_prompt_tokens", 0)
            self.last_eval_tokens = assistant_msg.get("_eval_tokens", 0)
            turn_eval_tokens += self.last_eval_tokens

            if assistant_msg.get("tool_calls"):
                mark = len(self.session.messages)
                self.session.add(
                    {k: v for k, v in assistant_msg.items() if not k.startswith("_")}
                )

                for tc in assistant_msg["tool_calls"]:
                    func = tc.get("function") if isinstance(tc, dict) else None
                    if not isinstance(func, dict):
                        sig = json.dumps(tc, sort_keys=True, default=str)
                        if stop_on_repeat(sig, mark):
                            return
                        result = json.dumps({"error": f"Malformed tool call: {tc!r}"})
                        self.session.add({"role": "tool", "content": result})
                        continue
                    fn_name = func.get("name", "")
                    fn_args = func.get("arguments", {})
                    if not fn_name:
                        sig = json.dumps(
                            {"n": "", "a": func}, sort_keys=True, default=str
                        )
                        if stop_on_repeat(sig, mark):
                            return
                        result = json.dumps({"error": f"Malformed tool call: {tc!r}"})
                        self.session.add({"role": "tool", "content": result})
                        continue

                    sig = json.dumps({"n": fn_name, "a": fn_args}, sort_keys=True)
                    if stop_on_repeat(sig, mark):
                        return

                    if self.config.mode == "ask" and fn_name != "ask_user":
                        if not self._confirm_inline(fn_name, fn_args):
                            result = json.dumps({"error": "User denied tool execution"})
                            self.session.add(
                                {
                                    "role": "tool",
                                    "content": result,
                                    "tool_name": fn_name,
                                }
                            )
                            continue

                    self._print_tool_call(fn_name, fn_args)

                    try:
                        result = execute_tool(fn_name, fn_args, self._ask_user)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                    self._print_tool_result(fn_name, result)
                    self.session.add(
                        {"role": "tool", "content": result, "tool_name": fn_name}
                    )
            else:
                self.session.add(
                    {k: v for k, v in assistant_msg.items() if not k.startswith("_")}
                )
                last_thinking = assistant_msg.get("thinking")
                last_content = assistant_msg.get("content")
                has_response = True
                break

        if has_response:
            self._print_response(last_thinking, last_content, turn_eval_tokens)

    async def _get_response(self, messages: list[dict]) -> dict:
        try:
            return await self._stream_response(messages)
        except Exception:
            return await self._non_stream(messages)

    async def _stream_response(self, messages: list[dict]) -> dict:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        prompt_tokens = 0
        eval_tokens = 0

        async for chunk in self.client.chat_stream(messages):
            msg = chunk.get("message", {})

            if msg.get("thinking"):
                thinking_parts.append(msg["thinking"])

            if msg.get("content"):
                content_parts.append(msg["content"])

            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                eval_tokens = chunk.get("eval_count", 0)

        if not content_parts and not thinking_parts and not tool_calls:
            return await self._non_stream(messages)

        result: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "_prompt_tokens": prompt_tokens,
            "_eval_tokens": eval_tokens,
        }
        if thinking_parts:
            result["thinking"] = "".join(thinking_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    async def _non_stream(self, messages: list[dict]) -> dict:
        data = await self.client.chat(messages)
        msg = data.get("message", {})
        content = msg.get("content")
        thinking = msg.get("thinking")
        result: dict = {
            "role": "assistant",
            "content": content,
            "_prompt_tokens": data.get("prompt_eval_count", 0),
            "_eval_tokens": data.get("eval_count", 0),
        }
        if thinking:
            result["thinking"] = thinking
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        return result
