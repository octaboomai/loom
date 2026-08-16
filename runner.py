"""
loom.validation.runner
-------------------------
"Validate-then-commit": after every Coder edit, run whatever lint/type/test/
security tools are actually present in the repo. We detect tools by config
file / lockfile presence rather than assuming a stack, so this doesn't
force Python-isms onto a JS repo or vice versa.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from loom.tools.shell import run_command, ShellResult


@dataclass
class CheckResult:
    name: str
    ran: bool
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        ran = [c for c in self.checks if c.ran]
        return all(c.passed for c in ran) if ran else True

    def render(self) -> str:
        lines = []
        for c in self.checks:
            if not c.ran:
                continue
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"[{mark}] {c.name}\n{c.detail.strip()[:1500]}")
        return "\n\n".join(lines) if lines else "(no validation tools detected)"


def _which(tool: str) -> bool:
    return shutil.which(tool) is not None


def _try(repo_root: Path, name: str, command: str, timeout: int = 180) -> CheckResult:
    result: ShellResult = run_command(repo_root, command, timeout=timeout)
    passed = result.exit_code == 0 and not result.timed_out
    detail = (result.stdout + "\n" + result.stderr).strip() or "(no output)"
    if result.timed_out:
        detail = f"TIMED OUT after {timeout}s\n" + detail
    return CheckResult(name=name, ran=True, passed=passed, detail=detail)


def run_validation(repo_root: Path) -> ValidationReport:
    report = ValidationReport()

    has_pyproject = (repo_root / "pyproject.toml").exists()
    has_setup = (repo_root / "setup.py").exists()
    has_requirements = (repo_root / "requirements.txt").exists()
    is_python = has_pyproject or has_setup or has_requirements or any(repo_root.glob("*.py"))

    has_package_json = (repo_root / "package.json").exists()

    has_cargo = (repo_root / "Cargo.toml").exists()

    # --- Python stack ---
    if is_python:
        if _which("ruff"):
            report.checks.append(_try(repo_root, "ruff (lint)", "ruff check ."))
        if _which("mypy") and (has_pyproject or (repo_root / "mypy.ini").exists()):
            report.checks.append(_try(repo_root, "mypy (types)", "mypy . --ignore-missing-imports"))
        if _which("pytest") and (list(repo_root.rglob("test_*.py")) or list(repo_root.rglob("*_test.py"))):
            # No --timeout flag: that needs the pytest-timeout plugin, which
            # isn't guaranteed to be installed. Our own subprocess timeout
            # (passed to _try below) already guards against a hang.
            report.checks.append(_try(repo_root, "pytest (tests)", "pytest -q", timeout=180))
        if _which("bandit"):
            report.checks.append(_try(repo_root, "bandit (security)", "bandit -q -r . -x .venv,venv,.loom"))

    # --- JS/TS stack ---
    if has_package_json:
        if _which("npx"):
            report.checks.append(_try(repo_root, "eslint (lint)", "npx --yes eslint . --max-warnings=0", timeout=180))
            report.checks.append(_try(repo_root, "tsc (types)", "npx --yes tsc --noEmit", timeout=180))
        if _which("npm"):
            report.checks.append(_try(repo_root, "npm test", "npm test --silent", timeout=180))

    # --- Rust stack ---
    if has_cargo and _which("cargo"):
        report.checks.append(_try(repo_root, "cargo check", "cargo check", timeout=240))
        report.checks.append(_try(repo_root, "cargo test", "cargo test", timeout=240))
        if _which("cargo-clippy"):
            report.checks.append(_try(repo_root, "clippy (lint)", "cargo clippy -- -D warnings", timeout=240))

    return report
