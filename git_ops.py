"""
loom.tools.git_ops
---------------------
Thin wrapper around GitPython for the operations Loom needs: status,
diff, branch, stage+commit. Every commit message includes a short
provenance footer so history stays auditable (which session produced it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from git import Repo, InvalidGitRepositoryError


def get_repo(repo_root: Path) -> Optional[Repo]:
    try:
        return Repo(str(repo_root))
    except InvalidGitRepositoryError:
        return None


def ensure_repo(repo_root: Path) -> Repo:
    repo = get_repo(repo_root)
    if repo is None:
        repo = Repo.init(str(repo_root))
    return repo


def current_branch(repo: Repo) -> str:
    try:
        return repo.active_branch.name
    except TypeError:
        return "(detached HEAD)"


def create_task_branch(repo: Repo, task_slug: str) -> str:
    branch_name = f"loom/{task_slug}"
    existing = [h.name for h in repo.heads]
    if branch_name in existing:
        repo.git.checkout(branch_name)
    else:
        repo.git.checkout("-b", branch_name)
    return branch_name

def diff_stat(repo: Repo) -> str:
    if not repo.head.is_valid():
        return "(no commits yet)"
    return repo.git.diff("--stat")


def diff_full(repo: Repo) -> str:
    if not repo.head.is_valid():
        return repo.git.diff("--cached") or repo.git.diff()
    return repo.git.diff()


def stage_all(repo: Repo) -> None:
    repo.git.add(A=True)


def commit(repo: Repo, message: str, session_id: str) -> str:
    full_message = f"{message}\n\n[loom session {session_id}]"
    repo.git.commit("-m", full_message, allow_empty=False)
    return repo.head.commit.hexsha[:10]
