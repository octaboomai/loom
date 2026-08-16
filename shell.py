"""
loom.tools.shell
-------------------
Shell command execution, always run with cwd pinned inside repo_root and
a timeout. This module does NOT decide whether approval is needed — the
orchestrator checks that against TeamConfig.approval_required_for before
calling run_command. Kept separate so the policy is easy to audit in one
place (orchestrator.py) rather than scattered through tool code.
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Commands that are never allowed regardless of approval — this is a floor,
# not the full guardrail system (team config can add more restrictions).
BLOCKLIST_PREFIXES = [
    "rm -rf /", "rm -rf /*", "mkfs", ":(){ :|:& };:", "dd if=", "> /dev/sda",
    "curl | sh", "wget | sh", "sudo ", "shutdown", "reboot",
]


@dataclass
class ShellResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class BlockedCommandError(Exception):
    pass


def is_blocked(command: str) -> bool:
    normalized = " ".join(command.split())
    return any(bad in normalized for bad in BLOCKLIST_PREFIXES)


def run_command(repo_root: Path, command: str, timeout: int = 120) -> ShellResult:
    if is_blocked(command):
        raise BlockedCommandError(f"Command matches a blocked pattern: {command}")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ShellResult(command=command, exit_code=proc.returncode,
                            stdout=proc.stdout[-8000:], stderr=proc.stderr[-8000:], timed_out=False)
    except subprocess.TimeoutExpired as e:
        return ShellResult(command=command, exit_code=-1,
                            stdout=(e.stdout or "")[-8000:], stderr=(e.stderr or "")[-8000:], timed_out=True)
