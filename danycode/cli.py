from __future__ import annotations

import asyncio
import os
import sys
import time
from urllib.parse import urlparse

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from danycode.agent import Agent
from danycode.client import OllamaClient
from danycode.config import Config
from danycode.session import Session

console = Console()

COMMANDS = [
    "/quit",
    "/clear",
    "/sessions",
    "/models",
    "/model",
    "/ps",
    "/config",
    "/set",
    "/save",
    "/version",
    "/help",
]


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            word = text.split()[0] if text.split() else text
            for cmd in COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.0f} MB"
    return f"{size_bytes} B"


def _build_config(
    model: str | None,
    host: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    num_ctx: int | None,
    num_predict: int | None,
    seed: int | None,
    think: str | None,
    keep_alive: str | None,
    system_prompt: str | None,
    mode: str | None,
    tool_result_limit: int | None,
) -> Config:
    overrides = {
        "model": model,
        "host": host,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "seed": seed,
        "think": think,
        "keep_alive": keep_alive,
        "system_prompt": system_prompt,
        "mode": mode,
        "tool_result_limit": tool_result_limit,
    }
    return Config.load(overrides)


def _normalize_host(raw: str) -> str:
    raw = raw.strip()
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.port:
        return f"{parsed.scheme}://{parsed.hostname}:11434"
    return raw


def _setup_remote(config: Config) -> None:
    try:
        raw = input("Адрес сервера (IP или host, порт опционально): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise typer.Exit(1)
    if not raw:
        raise typer.Exit(1)
    config.host = _normalize_host(raw)
    client = OllamaClient(config)
    if asyncio.run(client.health()):
        console.print(f"[green]Подключено к {config.host}[/green]")
    else:
        console.print(
            Panel(
                f"Не удалось подключиться к {config.host}",
                border_style="red",
                box=box.SQUARE,
            )
        )
        raise typer.Exit(1)


def _wait_for_local(config: Config) -> None:
    console.print("Запустите [bold]ollama serve[/bold] в отдельном терминале.")
    client = OllamaClient(config)
    with console.status("[dim]Ожидание Ollama (до 5 минут)...[/dim]"):
        for _ in range(60):
            if asyncio.run(client.health()):
                console.print("[green]Ollama запущена.[/green]")
                return
            time.sleep(5)
    console.print(
        Panel("Ollama не запустилась за 5 минут.", border_style="red", box=box.SQUARE)
    )
    raise typer.Exit(1)


def _ensure_ollama(config: Config) -> None:
    client = OllamaClient(config)
    if asyncio.run(client.health()):
        return
    console.print(
        Panel(
            f"Ollama не доступна по адресу {config.host}",
            border_style="yellow",
            box=box.SQUARE,
        )
    )
    try:
        choice = input("Сервер локально или удалённо? [local/remote]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise typer.Exit(1)
    if choice.startswith("r"):
        _setup_remote(config)
    else:
        _wait_for_local(config)


def _check_model_tools(config: Config) -> None:
    if not config.model:
        return
    client = OllamaClient(config)
    try:
        info = asyncio.run(client.show_model(config.model))
        caps = info.get("capabilities", [])
        if "tools" not in caps:
            console.print(
                f"[yellow]Warning: {config.model} may not support tool calling.[/yellow]"
            )
    except Exception:
        pass


def _auto_select_model(config: Config) -> None:
    client = OllamaClient(config)
    models = asyncio.run(client.list_models_with_tools())
    if not models:
        console.print(
            Panel(
                "No models with tool calling available.\nPull one: ollama pull qwen3:8b",
                border_style="red",
                box=box.SQUARE,
            )
        )
        raise typer.Exit(1)
    lightest = min(models, key=lambda m: m.get("size", float("inf")))
    config.model = lightest["name"]
    console.print(f"[dim]Auto-selected: {config.model}[/dim]")


def _print_models(config: Config) -> list[dict]:
    client = OllamaClient(config)
    models = asyncio.run(client.list_models_with_tools())
    if not models:
        console.print("[dim]No models with tool calling found.[/dim]")
        return []
    table = Table(title="Models (tool calling)", box=box.SQUARE)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Params", style="green")
    table.add_column("Quant", style="yellow")
    table.add_column("Family", style="dim")
    table.add_column("Size", style="dim")
    for i, m in enumerate(models, 1):
        details = m.get("details", {})
        name = m.get("name", "?")
        params = details.get("parameter_size", "-")
        quant = details.get("quantization_level", "-")
        family = details.get("family", "-")
        size = _format_size(m.get("size", 0))
        table.add_row(str(i), name, params, quant, family, size)
    console.print(table)
    return models


def _print_running(config: Config) -> None:
    client = OllamaClient(config)
    models = asyncio.run(client.list_running())
    if not models:
        console.print("[dim]No running models.[/dim]")
        return
    table = Table(title="Running Models", box=box.SQUARE)
    table.add_column("Name", style="cyan")
    table.add_column("VRAM", style="green")
    table.add_column("Context", style="yellow")
    table.add_column("Expires", style="dim")
    for m in models:
        name = m.get("name", "?")
        vram = m.get("size_vram", 0)
        vram_str = _format_size(vram) if vram else "-"
        ctx = str(m.get("context_length", "-"))
        expires = m.get("expires_at", "")[:19]
        table.add_row(name, vram_str, ctx, expires)
    console.print(table)


def _show_model_info(config: Config, name: str) -> None:
    client = OllamaClient(config)
    try:
        info = asyncio.run(client.show_model(name))
    except Exception:
        return
    details = info.get("details", {})
    caps = info.get("capabilities", [])
    params = info.get("parameters", "")
    console.print(f"  [dim]Family:[/dim] {details.get('family', '-')}")
    console.print(f"  [dim]Params:[/dim] {details.get('parameter_size', '-')}")
    console.print(f"  [dim]Quant:[/dim] {details.get('quantization_level', '-')}")
    console.print(f"  [dim]Format:[/dim] {details.get('format', '-')}")
    if caps:
        console.print(f"  [dim]Capabilities:[/dim] {', '.join(caps)}")
    if params:
        console.print(f"  [dim]Parameters:[/dim] {params.strip()}")


def _select_model(config: Config) -> None:
    models = _print_models(config)
    if not models:
        return
    try:
        choice = input(f"Select model [1-{len(models)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not choice:
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            config.model = models[idx]["name"]
            console.print(f"[green]Model set to:[/green] {config.model}")
            _show_model_info(config, config.model)
        else:
            console.print("[red]Invalid number.[/red]")
    except ValueError:
        console.print("[red]Enter a number.[/red]")


def _handle_set(config: Config, args: str) -> None:
    parts = args.split(maxsplit=1)
    if len(parts) != 2:
        console.print("[dim]Usage: /set <param> <value>[/dim]")
        console.print(
            f"[dim]Params: {', '.join(Config.load().__dataclass_fields__.keys())}[/dim]"
        )
        return
    key, value = parts[0], parts[1]
    err = config.update(key, value)
    if err:
        console.print(f"[red]{err}[/red]")
    else:
        console.print(f"[green]{key}[/green] = {getattr(config, key)}")


def _print_config(config: Config) -> None:
    table = Table(title="Current Config", box=box.SQUARE)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for key, val in config.display():
        table.add_row(key, val)
    console.print(table)


def _print_version(config: Config) -> None:
    client = OllamaClient(config)
    try:
        ver = asyncio.run(client.version())
        console.print(f"Ollama version: [green]{ver}[/green]")
    except Exception:
        console.print("[red]Failed to get version.[/red]")


def _print_help() -> None:
    help_text = (
        "[bold]Commands:[/bold]\n"
        "  /quit              Exit\n"
        "  /clear             Clear session\n"
        "  /sessions          List sessions\n"
        "  /models            List models (tool calling)\n"
        "  /model             Select model\n"
        "  /ps                Running models\n"
        "  /config            Show config\n"
        "  /set <k> <v>       Set parameter\n"
        "  /save              Save config\n"
        "  /version           Ollama version\n"
        "  /help              This help\n\n"
        "[bold]Input:[/bold]\n"
        "  Enter                Send\n"
        "  Ctrl+J / Alt+Enter   Newline\n"
        "  Tab                  Toggle mode (yolo/ask)\n"
        "  //text               Send to model as-is"
    )
    console.print(Panel(help_text, border_style="green", box=box.SQUARE, title="Help"))


def _run_agent(agent: Agent, text: str) -> None:
    try:
        asyncio.run(agent.run(text))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as e:
        console.print(Panel(str(e), border_style="red", box=box.SQUARE))


def _clear_prompt_line(text: str) -> None:
    lines = text.count("\n") + 1
    sys.stdout.write(f"\x1b[{lines}A\x1b[J")
    sys.stdout.flush()


def _build_status_bar(config: Config, agent: Agent | None) -> str:
    mode_color = "ansired" if config.mode == "yolo" else "ansigreen"
    parts = [f"<{mode_color}><b> {config.mode} </b></{mode_color}>"]
    parts.append(f" {config.model} ")

    if agent and agent.last_prompt_tokens > 0:
        pct = agent.last_prompt_tokens / config.num_ctx * 100
        if pct > 95:
            tok_color = "ansired"
        elif pct > 80:
            tok_color = "ansiyellow"
        else:
            tok_color = "ansigreen"
        parts.append(
            f"<{tok_color}> ctx: {agent.last_prompt_tokens}/{config.num_ctx} ({pct:.0f}%) </{tok_color}>"
        )
    else:
        parts.append(f" ctx: 0/{config.num_ctx} (0%) ")

    return " │ ".join(parts)


def _build_rprompt(config: Config, agent: Agent | None) -> HTML:
    return HTML(_build_status_bar(config, agent))


def _build_key_bindings(config: Config) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")
    def _newline_cj(event):
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _newline_alt(event):
        event.current_buffer.insert_text("\n")

    @kb.add("tab")
    def _toggle_mode(event):
        config.mode = "ask" if config.mode == "yolo" else "yolo"

    return kb


def _build_prompt_session(config: Config, agent: Agent) -> PromptSession:
    style = Style.from_dict(
        {
            "prompt": "bold ansigreen",
            "rprompt": "#888888",
        }
    )
    return PromptSession(
        message=HTML("<ansigreen><b>❯</b></ansigreen> "),
        multiline=True,
        key_bindings=_build_key_bindings(config),
        completer=CommandCompleter(),
        complete_while_typing=True,
        rprompt=lambda: _build_rprompt(config, agent),
        style=style,
        prompt_continuation="  ",
    )


app = typer.Typer(
    name="danycode", help="Ollama CLI coding assistant", invoke_without_command=True
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="One-shot prompt. Omit for REPL."),
    session_name: str = typer.Option(
        "default", "--session", "-s", help="Session name."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model name."),
    host: str = typer.Option(None, "--host", help="Ollama API base URL."),
    temperature: float = typer.Option(None, "--temperature", "-t"),
    top_p: float = typer.Option(None, "--top-p"),
    top_k: int = typer.Option(None, "--top-k"),
    min_p: float = typer.Option(None, "--min-p"),
    num_ctx: int = typer.Option(None, "--num-ctx"),
    num_predict: int = typer.Option(None, "--num-predict"),
    seed: int = typer.Option(None, "--seed"),
    think: str = typer.Option(None, "--think", help="false/true/high/medium/low/max."),
    keep_alive: str = typer.Option(None, "--keep-alive"),
    system_prompt: str = typer.Option(None, "--system", help="System prompt."),
    mode: str = typer.Option(None, "--mode", help="yolo or ask."),
    tool_result_limit: int = typer.Option(None, "--tool-result-limit"),
    new: bool = typer.Option(False, "--new", "-n", help="Start fresh session."),
):
    if ctx.invoked_subcommand is not None:
        return

    if prompt == "models":
        models(host=host)
        return
    if prompt == "sessions":
        sessions()
        return

    config = _build_config(
        model,
        host,
        temperature,
        top_p,
        top_k,
        min_p,
        num_ctx,
        num_predict,
        seed,
        think,
        keep_alive,
        system_prompt,
        mode,
        tool_result_limit,
    )
    config.ensure_dirs()
    _ensure_ollama(config)

    if not config.model:
        _auto_select_model(config)
    else:
        _check_model_tools(config)

    session = Session(session_name)
    if new:
        session.clear()

    agent = Agent(config, session)

    if prompt:
        _run_agent(agent, prompt)
        return

    os.system("cls" if os.name == "nt" else "clear")
    console.print("[bold green]DanyCode[/bold green] [dim]| /help - help[/dim]")

    ps = _build_prompt_session(config, agent)

    while True:
        try:
            user_input = ps.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        _clear_prompt_line(user_input)
        stripped = user_input.strip()

        if stripped.startswith("//"):
            _run_agent(agent, stripped[1:])
            continue

        if stripped.startswith("/"):
            cmd = stripped.split()[0]
            if cmd == "/quit":
                break
            elif cmd == "/clear":
                session.clear()
                agent._ensure_system_prompt()
                console.print("[dim]Session cleared.[/dim]")
            elif cmd == "/sessions":
                all_sessions = Session.list_sessions()
                console.print(
                    f"Sessions: {', '.join(all_sessions) if all_sessions else 'none'}"
                )
            elif cmd == "/models":
                _print_models(config)
            elif cmd == "/model":
                _select_model(config)
            elif cmd == "/ps":
                _print_running(config)
            elif cmd == "/config":
                _print_config(config)
            elif cmd == "/set":
                _handle_set(config, stripped[5:])
            elif cmd == "/save":
                config.save()
                console.print("[green]Config saved.[/green]")
            elif cmd == "/version":
                _print_version(config)
            elif cmd == "/help":
                _print_help()
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
            continue

        if not stripped:
            continue

        _run_agent(agent, stripped)


@app.command()
def models(
    host: str = typer.Option(None, "--host", help="Ollama API base URL."),
):
    config = _build_config(host=host)
    _ensure_ollama(config)
    _print_models(config)


@app.command()
def sessions():
    all_sessions = Session.list_sessions()
    if all_sessions:
        for s in all_sessions:
            console.print(f"  {s}")
    else:
        console.print("No sessions found.")


if __name__ == "__main__":
    app()
