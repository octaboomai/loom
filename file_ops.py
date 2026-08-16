"""
loom.tools.file_ops
---------------------
File read/write/patch/search tools. Every path is resolved and checked
to be inside repo_root before any I/O happens — this is the guard against
an agent (accidentally or via prompt injection in file content) writing
outside the project.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path


class PathEscapeError(Exception):
    pass


class ForbiddenPathError(Exception):
    pass


def _safe_path(repo_root: Path, rel_path: str, forbidden_paths: list[str]) -> Path:
    candidate = (repo_root / rel_path).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        raise PathEscapeError(f"Refusing to touch path outside repo: {rel_path}")
    for forbidden in forbidden_paths:
        if candidate.match(forbidden) or str(candidate).endswith(forbidden.lstrip("*")):
            raise ForbiddenPathError(f"Path is guarded by team config: {rel_path}")
    return candidate


@dataclass
class WriteResult:
    path: str
    bytes_written: int
    diff: str
    created: bool


def read_file(repo_root: Path, rel_path: str, forbidden_paths: list[str] | None = None) -> str:
    p = _safe_path(repo_root, rel_path, forbidden_paths or [])
    if not p.exists():
        raise FileNotFoundError(rel_path)
    return p.read_text(encoding="utf-8", errors="ignore")


def write_file(repo_root: Path, rel_path: str, content: str, forbidden_paths: list[str] | None = None) -> WriteResult:
    p = _safe_path(repo_root, rel_path, forbidden_paths or [])
    created = not p.exists()
    old = "" if created else p.read_text(encoding="utf-8", errors="ignore")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    diff = "\n".join(difflib.unified_diff(
        old.splitlines(), content.splitlines(),
        fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", lineterm="",
    ))
    return WriteResult(path=rel_path, bytes_written=len(content.encode("utf-8")), diff=diff, created=created)


def patch_file(repo_root: Path, rel_path: str, find: str, replace: str, forbidden_paths: list[str] | None = None) -> WriteResult:
    """Exact string replace, requires a unique match (same discipline as
    the editing tools Claude itself uses — avoids ambiguous patches)."""
    p = _safe_path(repo_root, rel_path, forbidden_paths or [])
    if not p.exists():
        raise FileNotFoundError(rel_path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"No match for the given text in {rel_path}")
    if count > 1:
        raise ValueError(f"Match is not unique ({count} occurrences) in {rel_path}; widen context")
    new_text = text.replace(find, replace, 1)
    return write_file(repo_root, rel_path, new_text, forbidden_paths)


def search_repo(repo_root: Path, pattern: str, glob: str = "**/*", max_hits: int = 50) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))
    for p in repo_root.glob(glob):
        if len(hits) >= max_hits:
            break
        if not p.is_file() or ".git" in p.parts or ".loom" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append((str(p.relative_to(repo_root)), i, line.strip()[:200]))
                if len(hits) >= max_hits:
                    break
    return hits
