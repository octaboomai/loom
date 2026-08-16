"""
loom.agents.toolkit
----------------------
Builds the concrete ToolSpec objects (JSON schema + Python handler) that
get handed to each agent. Centralized here so every agent that can
"write_file" goes through the exact same guarded implementation in
loom.tools.file_ops — no agent gets its own ad-hoc file access.
"""
from __future__ import annotations

from pathlib import Path

from loom.agents.base import ToolSpec
from loom.config import LoomConfig
from loom.tools import file_ops, git_ops
from loom.tools.shell import run_command


def read_file_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        content = file_ops.read_file(cfg.repo_root, inp["path"], cfg.team.forbidden_paths)
        lines = content.splitlines()
        if len(lines) > 400:
            content = "\n".join(lines[:400]) + f"\n... [truncated, {len(lines)} lines total]"
        return content

    return ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the repository. Path is relative to repo root.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )


def search_repo_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        hits = file_ops.search_repo(cfg.repo_root, inp["pattern"], inp.get("glob", "**/*"))
        if not hits:
            return "(no matches)"
        return "\n".join(f"{path}:{line}: {text}" for path, line, text in hits)

    return ToolSpec(
        name="search_repo",
        description="Regex search across the repository. Returns file:line:text for each match.",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "optional glob filter, default **/*"}},
            "required": ["pattern"],
        },
        handler=handler,
    )


def write_file_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        res = file_ops.write_file(cfg.repo_root, inp["path"], inp["content"], cfg.team.forbidden_paths)
        action = "created" if res.created else "updated"
        return f"{action} {res.path} ({res.bytes_written} bytes)\n{res.diff[:3000]}"

    return ToolSpec(
        name="write_file",
        description="Create or fully overwrite a file with new content. Use for new files or full rewrites.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=handler,
    )


def patch_file_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        res = file_ops.patch_file(cfg.repo_root, inp["path"], inp["find"], inp["replace"], cfg.team.forbidden_paths)
        return f"patched {res.path}\n{res.diff[:3000]}"

    return ToolSpec(
        name="patch_file",
        description=(
            "Replace an exact, unique block of text in an existing file. "
            "`find` must match the current file content exactly (including whitespace) "
            "and appear exactly once — include enough surrounding context to make it unique."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "find": {"type": "string"}, "replace": {"type": "string"}},
            "required": ["path", "find", "replace"],
        },
        handler=handler,
    )


def run_shell_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        res = run_command(cfg.repo_root, inp["command"], timeout=inp.get("timeout", 120))
        status = "TIMED OUT" if res.timed_out else f"exit {res.exit_code}"
        return f"$ {inp['command']}\n[{status}]\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    return ToolSpec(
        name="run_shell",
        description="Run a shell command in the repository root (e.g. install deps, run a script). Requires approval.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"],
        },
        handler=handler,
        needs_approval="shell" in cfg.team.approval_required_for and not cfg.auto_approve,
    )


def git_diff_tool(cfg: LoomConfig) -> ToolSpec:
    def handler(inp: dict) -> str:
        repo = git_ops.get_repo(cfg.repo_root)
        if repo is None:
            return "(not a git repository)"
        return git_ops.diff_full(repo) or "(no changes)"

    return ToolSpec(
        name="git_diff",
        description="Show the current uncommitted git diff for the repository.",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


def readonly_toolset(cfg: LoomConfig) -> list[ToolSpec]:
    return [read_file_tool(cfg), search_repo_tool(cfg), git_diff_tool(cfg)]


def coder_toolset(cfg: LoomConfig) -> list[ToolSpec]:
    return [read_file_tool(cfg), search_repo_tool(cfg), write_file_tool(cfg),
            patch_file_tool(cfg), run_shell_tool(cfg), git_diff_tool(cfg)]


def tester_toolset(cfg: LoomConfig) -> list[ToolSpec]:
    return [read_file_tool(cfg), run_shell_tool(cfg), git_diff_tool(cfg)]
