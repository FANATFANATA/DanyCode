from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file, creating if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact unique search string with replacement in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "search": {"type": "string", "description": "Exact string to find"},
                    "replace": {"type": "string", "description": "Replacement"},
                },
                "required": ["path", "search", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search regex pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Path to search in"},
                    "include": {"type": "string", "description": "File glob filter"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute shell command via PowerShell (Windows) or sh (Unix). Returns stdout, stderr, returncode, cwd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask user a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question"}
                },
                "required": ["question"],
            },
        },
    },
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_CLIXML_RE = re.compile(r'<S S="Error">(.*?)</S>', re.DOTALL)


def get_shell() -> str:
    if shutil.which("pwsh"):
        return "pwsh"
    if os.name == "nt":
        return "powershell.exe"
    return ""


def get_shell_name() -> str:
    shell = get_shell()
    if shell in ("pwsh", "powershell.exe"):
        return "PowerShell"
    return "sh"


def _ps_args(shell: str, command: str) -> list[str]:
    full = (
        f"[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
        f"$ProgressPreference='SilentlyContinue'; "
        f"{command}"
    )
    encoded = base64.b64encode(full.encode("utf-16-le")).decode("ascii")
    return [shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


def _clean_output(text: str) -> str:
    if "#< CLIXML" in text:
        parts = _CLIXML_RE.findall(text)
        if parts:
            text = "\n".join(parts)
    text = text.replace("_x001B_", "\x1b")
    text = text.replace("_x000D_", "")
    text = text.replace("_x000A_", "\n")
    text = _ANSI_RE.sub("", text)
    return text.strip()


def execute_tool(name: str, arguments, ask_fn=None) -> str:
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "read_file":
        return _read_file(arguments.get("path", ""))
    elif name == "write_file":
        return _write_file(arguments.get("path", ""), arguments.get("content", ""))
    elif name == "edit_file":
        return _edit_file(
            arguments.get("path", ""),
            arguments.get("search", ""),
            arguments.get("replace", ""),
        )
    elif name == "list_dir":
        return _list_dir(arguments.get("path", "."))
    elif name == "search":
        return _search(
            arguments.get("pattern", ""),
            arguments.get("path", "."),
            arguments.get("include"),
        )
    elif name == "run_command":
        return _run_command(arguments.get("command", ""))
    elif name == "ask_user":
        if ask_fn:
            return ask_fn(arguments.get("question", ""))
        return json.dumps({"error": "No ask function provided"})
    return json.dumps({"error": f"Unknown tool: {name}"})


def _read_file(path: str) -> str:
    if not path:
        return json.dumps({"error": "read_file: path is required"})
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps({"error": f"File is not valid UTF-8 text: {path}"})
        return json.dumps({"content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _write_file(path: str, content: str) -> str:
    if not path:
        return json.dumps({"error": "write_file: path is required"})
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(p.resolve())})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _edit_file(path: str, search: str, replace: str) -> str:
    if not path:
        return json.dumps({"error": "edit_file: path is required"})
    if not search:
        return json.dumps({"error": "edit_file: search is required"})
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        content = p.read_text(encoding="utf-8")
        if search not in content:
            return json.dumps({"error": "Search string not found in file"})
        count = content.count(search)
        if count > 1:
            return json.dumps(
                {"error": f"Search string found {count} times, must be unique"}
            )
        new_content = content.replace(search, replace, 1)
        p.write_text(new_content, encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(p.resolve())})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _list_dir(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"Path not found: {path}"})
        entries = []
        for item in sorted(p.iterdir()):
            entry_type = "dir" if item.is_dir() else "file"
            entries.append({"name": item.name, "type": entry_type})
        return json.dumps({"entries": entries, "cwd": os.getcwd()})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _search(pattern: str, path: str, include: str | None) -> str:
    if not pattern:
        return json.dumps({"error": "search: pattern is required"})
    try:
        shell = get_shell()
        if shell:
            ep = pattern.replace("'", "''")
            epath = path.replace("'", "''")
            ps = f"Get-ChildItem -Path '{epath}' -Recurse -File"
            if include:
                ei = include.replace("'", "''")
                ps += f" -Filter '{ei}'"
            ps += f" | Select-String -Pattern '{ep}' | Select-Object -First 100"
            ps += ' | ForEach-Object { "$($_.Path):$($_.LineNumber):$($_.Line)" }'
            args = _ps_args(shell, ps)
        else:
            args = ["grep", "-rn"]
            if include:
                args.extend(["--include", include])
            args.extend([pattern, path])
        result = subprocess.run(args, capture_output=True, timeout=30)
        stdout = _clean_output(result.stdout.decode("utf-8", errors="replace"))
        if not stdout:
            return json.dumps({"results": []})
        lines = stdout.split("\n")[:100]
        return json.dumps({"results": lines})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Search timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _run_command(command: str) -> str:
    if not command.strip():
        return json.dumps({"error": "run_command: command is required"})
    try:
        shell = get_shell()
        if shell:
            args = _ps_args(shell, command)
        else:
            args = ["/bin/sh", "-c", command]
        result = subprocess.run(args, capture_output=True, timeout=120)
        stdout = _clean_output(result.stdout.decode("utf-8", errors="replace"))
        stderr = _clean_output(result.stderr.decode("utf-8", errors="replace"))
        return json.dumps(
            {
                "stdout": stdout[-10000:],
                "stderr": stderr[-5000:],
                "returncode": result.returncode,
                "cwd": os.getcwd(),
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Command timed out (120s)"})
    except Exception as e:
        return json.dumps({"error": str(e)})
