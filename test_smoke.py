"""
Smoke tests covering the paths verified manually during development:
- repo map extraction (incl. the UTF-8 byte-offset bug regression)
- event store append/replay/resume
- file_ops guardrails (path escape, forbidden paths, unique-match patch)
- shell blocklist
- agent tool-use loop: normal call, denied approval, max_turns cutoff
- orchestrator: approve+commit, retry-then-approve, give-up (no commit)
- validation runner against a real pytest project

Run with: pytest -q  (from the project root, inside the venv)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from loom.agents.base import Agent, ToolSpec
from loom.config import LoomConfig
from loom.context.repo_map import build_repo_map, map_file
from loom.credentials import CredentialStore
from loom.events.store import EventStore
from loom.orchestrator import Orchestrator
from loom.router import ModelRouter, UsageTracker
from loom.tools import file_ops, git_ops, shell
from loom.validation.runner import run_validation


@pytest.fixture()
def tmp_repo(tmp_path: Path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    repo = git_ops.ensure_repo(tmp_path)
    git_ops.stage_all(repo)
    try:
        repo.git.commit("-m", "initial")
    except Exception:
        repo.git.config("user.email", "test@loom.local")
        repo.git.config("user.name", "Loom Test")
        repo.git.commit("-m", "initial")
    return tmp_path, repo


class FakeUsage:
    input_tokens = 10
    output_tokens = 10


def _text(t):
    # Blocks are plain dicts now — this is what a real provider adapter
    # returns (see loom/providers/base.py), and what agents/base.py reads.
    return {"type": "text", "text": t}


def _tool(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


# ---------------------------------------------------------------- repo map

def test_repo_map_handles_multibyte_chars(tmp_path: Path):
    # Regression test for the UTF-8 byte-offset bug: an em-dash before a
    # function def must not corrupt the extracted symbol name.
    f = tmp_path / "mod.py"
    f.write_text('"""A note — with an em dash."""\n\ndef build_thing(x):\n    return x\n')
    fm = map_file(f)
    names = [s.name for s in fm.symbols]
    assert "build_thing" in names


def test_repo_map_builds_over_directory(tmp_repo):
    root, _ = tmp_repo
    maps = build_repo_map(root)
    assert any(m.path.endswith("app.py") for m in maps)


# ---------------------------------------------------------------- event store

def test_event_store_append_replay_resume(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite3")
    sid = store.new_session("task")
    store.append(sid, "tool_call", {"x": 1}, agent="coder")
    store.append(sid, "tool_result", {"ok": True}, agent="coder")
    events = list(store.replay(sid))
    assert [e.kind for e in events] == ["tool_call", "tool_result"]
    assert store.last_open_session() == sid
    store.set_session_status(sid, "completed")
    assert store.last_open_session() is None


# ---------------------------------------------------------------- file ops

def test_write_and_patch_file(tmp_path: Path):
    res = file_ops.write_file(tmp_path, "b.py", "x = 1\n")
    assert res.created
    res2 = file_ops.patch_file(tmp_path, "b.py", "x = 1", "x = 2")
    assert "x = 2" in (tmp_path / "b.py").read_text()


def test_patch_file_requires_unique_match(tmp_path: Path):
    file_ops.write_file(tmp_path, "c.py", "x = 1\nx = 1\n")
    with pytest.raises(ValueError):
        file_ops.patch_file(tmp_path, "c.py", "x = 1", "x = 2")


def test_path_escape_blocked(tmp_path: Path):
    with pytest.raises(file_ops.PathEscapeError):
        file_ops.read_file(tmp_path, "../../etc/passwd")


def test_forbidden_path_blocked(tmp_path: Path):
    with pytest.raises(file_ops.ForbiddenPathError):
        file_ops.write_file(tmp_path, "secrets/key.pem", "x", forbidden_paths=["**/secrets/**"])


# ---------------------------------------------------------------- shell

def test_shell_blocklist(tmp_path: Path):
    with pytest.raises(shell.BlockedCommandError):
        shell.run_command(tmp_path, "sudo rm -rf /")


def test_shell_runs_normal_command(tmp_path: Path):
    res = shell.run_command(tmp_path, "echo hello")
    assert res.exit_code == 0
    assert "hello" in res.stdout


# ---------------------------------------------------------------- agent loop

def test_agent_denied_approval_does_not_run_handler(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite3")
    sid = store.new_session("approval test")
    ran = {"called": False}

    def handler(inp):
        ran["called"] = True
        return "ran it"

    spec = ToolSpec(name="dangerous", description="d", input_schema={"type": "object", "properties": {}},
                     handler=handler, needs_approval=True)

    class FakeRouter:
        def __init__(self):
            self.calls = 0

        def complete(self, role, system, messages, tools=None, max_tokens=4096, on_swap=None):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content=[_tool("t1", "dangerous", {})], stop_reason="tool_use", usage=FakeUsage(), model_label="test-model")
            return SimpleNamespace(content=[_text("done, it was denied")], stop_reason="end_turn", usage=FakeUsage(), model_label="test-model")

    agent = Agent(router=FakeRouter(), system_prompt="test", tools=[spec])
    agent.role = "coder"
    agent.run("do the dangerous thing", store, sid, approval_fn=lambda name, inp: False)
    assert ran["called"] is False


def test_agent_max_turns_cutoff_does_not_hang(tmp_path: Path):
    store = EventStore(tmp_path / "events.sqlite3")
    sid = store.new_session("max turns test")
    spec = ToolSpec(name="loop_tool", description="d", input_schema={"type": "object", "properties": {}},
                     handler=lambda inp: "ok")

    class InfiniteRouter:
        def complete(self, role, system, messages, tools=None, max_tokens=4096, on_swap=None):
            return SimpleNamespace(content=[_tool("t", "loop_tool", {})], stop_reason="tool_use", usage=FakeUsage(), model_label="test-model")

    agent = Agent(router=InfiniteRouter(), system_prompt="test", tools=[spec])
    agent.role = "coder"
    agent.run("loop forever", store, sid, max_turns=5)
    errors = [e for e in store.replay(sid) if e.kind == "error"]
    assert len(errors) == 1 and "max_turns" in errors[0].payload["error"]


# ---------------------------------------------------------------- orchestrator

def _make_orchestrator(tmp_path: Path, script: dict):
    cfg = LoomConfig.load(tmp_path, auto_approve=True)
    store = EventStore(cfg.event_db_path())
    router = ModelRouter.__new__(ModelRouter)
    router.config = cfg
    router.store = CredentialStore(path=tmp_path / "unused_credentials.json")  # deliberately empty/unused
    router.usage = UsageTracker()
    router._provider_cache = {}
    idx = {k: 0 for k in script}

    def fake_complete(role, system, messages, tools=None, max_tokens=4096, on_swap=None):
        r = script[role][idx[role]]
        idx[role] += 1
        router.usage.record(router.model_for(role), r.usage)
        return r

    router.complete = fake_complete
    return Orchestrator(cfg, store, router=router), idx


def test_orchestrator_happy_path_commits(tmp_repo):
    root, repo = tmp_repo
    script = {
        "planner": [_msg(_text("STEPS:\n1. add subtract"))],
        "coder": [
            _msg(_tool("t1", "write_file", {"path": "app.py", "content": "def add(a,b):\n    return a+b\n\ndef subtract(a,b):\n    return a-b\n"}), stop="tool_use"),
            _msg(_text("done")),
        ],
        "tester": [_msg(_text("VERDICT: PASS"))],
        "reviewer": [_msg(_text("DECISION: APPROVE"))],
    }
    orch, _ = _make_orchestrator(root, script)
    result = orch.run("add a subtract function", auto_commit=True)
    assert result.verdict == "approved"
    assert result.committed is True
    assert "subtract" in (root / "app.py").read_text()


def test_orchestrator_gives_up_without_committing(tmp_repo):
    root, repo = tmp_repo
    coder_calls = []
    for i in range(3):
        coder_calls.append(_msg(_tool(f"t{i}", "write_file", {"path": "app.py", "content": f"x={i}\n"}), stop="tool_use"))
        coder_calls.append(_msg(_text(f"try {i}")))
    script = {
        "planner": [_msg(_text("STEPS:\n1. x"))],
        "coder": coder_calls,
        "tester": [_msg(_text("VERDICT: PASS"))] * 3,
        "reviewer": [_msg(_text("DECISION: REQUEST_CHANGES\nISSUES: still bad"))] * 3,
    }
    orch, idx = _make_orchestrator(root, script)
    result = orch.run("a task that never passes review", auto_commit=True, max_repair_loops=2)
    assert result.verdict == "gave_up"
    assert result.committed is False
    assert idx["coder"] == 6  # 3 attempts x (tool_use turn + text turn)


def test_gitignore_excludes_loom_dir(tmp_repo):
    root, repo = tmp_repo
    script = {
        "planner": [_msg(_text("STEPS:\n1. x"))],
        "coder": [_msg(_tool("t1", "write_file", {"path": "app.py", "content": "x=1\n"}), stop="tool_use"), _msg(_text("done"))],
        "tester": [_msg(_text("VERDICT: PASS"))],
        "reviewer": [_msg(_text("DECISION: APPROVE"))],
    }
    orch, _ = _make_orchestrator(root, script)
    orch.run("trivial change", auto_commit=True)
    committed_files = repo.git.show("HEAD", "--stat", "--name-only")
    assert ".loom" not in committed_files


def _msg(block, stop="end_turn"):
    return SimpleNamespace(content=[block], stop_reason=stop, usage=FakeUsage(), model_label="test-model")


# ---------------------------------------------------------------- validation

def test_validation_detects_failing_test(tmp_path: Path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add_broken():\n    assert add(2, 2) == 5\n"
    )
    report = run_validation(tmp_path)
    assert report.all_passed is False
    pytest_check = next(c for c in report.checks if c.name.startswith("pytest"))
    assert not pytest_check.passed
    assert "test_add_broken" in pytest_check.detail


def test_validation_passes_on_correct_code(tmp_path: Path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    report = run_validation(tmp_path)
    assert report.all_passed is True


# ---------------------------------------------------------------- credentials

def test_credential_store_add_list_remove(tmp_path: Path):
    from loom.credentials import CredentialStore

    store = CredentialStore(path=tmp_path / "credentials.json")
    store.add("anthropic", "sk-ant-abcdefghijklmnop", name="work")
    store.add("openai", "sk-openai-xxxxxxxxxxxx", name="personal", models={"core": "gpt-5.1"})

    creds = store.list_all()
    assert len(creds) == 2
    anth = store.for_provider("anthropic")
    assert len(anth) == 1 and anth[0].name == "work"
    assert anth[0].masked_key() != anth[0].api_key  # never exposes the raw key in display form
    assert "sk-ant-abcdefghijklmnop" not in anth[0].masked_key()

    assert store.remove("anthropic", "work") is True
    assert store.for_provider("anthropic") == []
    assert store.remove("anthropic", "nonexistent") is False


def test_credential_store_file_permissions(tmp_path: Path):
    from loom.credentials import CredentialStore
    import stat as statmod
    import platform

    store = CredentialStore(path=tmp_path / "credentials.json")
    store.add("anthropic", "sk-ant-test", name="x")
    if platform.system() != "Windows":
        mode = statmod.S_IMODE((tmp_path / "credentials.json").stat().st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_credential_store_env_fallback(monkeypatch, tmp_path: Path):
    from loom.credentials import CredentialStore

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = CredentialStore(path=tmp_path / "credentials.json")  # empty file
    fallback = store.env_fallback()
    assert len(fallback) == 1
    assert fallback[0].provider == "anthropic"
    assert fallback[0].api_key == "sk-ant-from-env"


# ---------------------------------------------------------------- router failover

def test_router_fails_over_to_second_credential(tmp_path: Path):
    from loom.credentials import CredentialStore
    from loom.providers.base import LLMResponse, ProviderError, QuotaOrAuthError, Usage

    cfg = LoomConfig.load(tmp_path)
    store = CredentialStore(path=tmp_path / "credentials.json")
    store.add("anthropic", "sk-ant-broke", name="broke")
    store.add("anthropic", "sk-ant-works", name="works")

    router = ModelRouter(cfg, credential_store=store)

    class FakeProvider:
        def __init__(self, should_fail):
            self.should_fail = should_fail

        def complete(self, model, system, messages, tools, max_tokens):
            if self.should_fail:
                raise QuotaOrAuthError("credit balance too low")
            return LLMResponse(content=[{"type": "text", "text": "ok"}], stop_reason="end_turn",
                                usage=Usage(input_tokens=5, output_tokens=5), model_label="anthropic:test")

    # patch _get_provider to return a failing provider for "broke", working one for "works"
    def fake_get_provider(cred):
        return FakeProvider(should_fail=(cred.name == "broke"))

    router._get_provider = fake_get_provider
    swaps = []
    resp = router.complete("coder", "sys", [{"role": "user", "content": "hi"}], on_swap=lambda c, e: swaps.append(c.name))
    assert resp.content[0]["text"] == "ok"
    assert swaps == ["broke"]  # confirms it actually tried the broken one first, then moved on


def test_router_all_credentials_failed_raises_clear_error(tmp_path: Path):
    from loom.credentials import CredentialStore
    from loom.providers.base import QuotaOrAuthError
    from loom.router import AllCredentialsFailedError

    cfg = LoomConfig.load(tmp_path)
    store = CredentialStore(path=tmp_path / "credentials.json")
    store.add("anthropic", "sk-ant-dead", name="dead")
    router = ModelRouter(cfg, credential_store=store)

    class AlwaysFails:
        def complete(self, **kw):
            raise QuotaOrAuthError("credit balance too low")

    router._get_provider = lambda cred: AlwaysFails()
    with pytest.raises(AllCredentialsFailedError, match="credit balance too low"):
        router.complete("coder", "sys", [{"role": "user", "content": "hi"}])


def test_router_no_credentials_raises_clear_error(tmp_path: Path, monkeypatch):
    from loom.credentials import CredentialStore
    from loom.router import NoCredentialsError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LoomConfig.load(tmp_path)
    store = CredentialStore(path=tmp_path / "credentials.json")  # nothing registered
    router = ModelRouter(cfg, credential_store=store)
    with pytest.raises(NoCredentialsError):
        router.complete("coder", "sys", [{"role": "user", "content": "hi"}])


def test_router_does_not_failover_on_non_quota_errors(tmp_path: Path):
    """A programming bug (e.g. a real ValueError) must NOT be swallowed as
    a 'try the next key' situation — it should propagate so it's visible."""
    from loom.credentials import CredentialStore

    cfg = LoomConfig.load(tmp_path)
    store = CredentialStore(path=tmp_path / "credentials.json")
    store.add("anthropic", "sk-ant-a", name="a")
    store.add("anthropic", "sk-ant-b", name="b")
    router = ModelRouter(cfg, credential_store=store)

    class BuggyProvider:
        def complete(self, **kw):
            raise ValueError("some unrelated bug")

    router._get_provider = lambda cred: BuggyProvider()
    with pytest.raises(ValueError, match="some unrelated bug"):
        router.complete("coder", "sys", [{"role": "user", "content": "hi"}])


def test_parse_models_cli_helper():
    from loom.cli import _parse_models

    assert _parse_models(None) == {}
    assert _parse_models("gpt-5.1") == {"fast": "gpt-5.1", "core": "gpt-5.1", "deep": "gpt-5.1"}
    assert _parse_models("fast=gpt-5.1-mini,core=gpt-5.1") == {"fast": "gpt-5.1-mini", "core": "gpt-5.1"}


# ---------------------------------------------------------------- licensing

def _sign_test_license(private_key, payload: dict) -> str:
    """Mirrors admin/generate_license.py's encoding, using a throwaway
    keypair generated inside the test — never the real private key."""
    import base64
    import json as _json

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    payload_bytes = _json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = private_key.sign(payload_bytes)
    return f"{b64url(payload_bytes)}.{b64url(signature)}"


@pytest.fixture()
def test_keypair(monkeypatch):
    """Generates a fresh keypair for the test and points loom.licensing at
    its public half, so tests never touch the real production private key
    (which isn't in this codebase at all)."""
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from loom import licensing

    private_key = Ed25519PrivateKey.generate()
    pub_bytes = private_key.public_key().public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    monkeypatch.setattr(licensing, "PUBLIC_KEY_B64", base64.b64encode(pub_bytes).decode())
    return private_key


def test_license_valid_round_trip(test_keypair):
    from loom.licensing import verify_license_string

    lic_str = _sign_test_license(test_keypair, {
        "org": "Acme Inc", "tier": "team", "seats": 10, "issued": "2026-01-01", "expires": "2030-01-01",
    })
    lic = verify_license_string(lic_str)
    assert lic.org == "Acme Inc" and lic.seats == 10 and lic.tier == "team"


def test_license_tampered_payload_rejected(test_keypair):
    from loom.licensing import InvalidLicenseError, verify_license_string

    lic_str = _sign_test_license(test_keypair, {
        "org": "Acme Inc", "tier": "team", "seats": 1, "issued": "2026-01-01", "expires": None,
    })
    payload_part, sig_part = lic_str.split(".")
    forged_str = _sign_test_license(test_keypair, {
        "org": "Acme Inc", "tier": "team", "seats": 99, "issued": "2026-01-01", "expires": None,
    }).split(".")[0] + "." + sig_part  # new payload, OLD signature — must fail
    with pytest.raises(InvalidLicenseError, match="Signature"):
        verify_license_string(forged_str)


def test_license_wrong_signing_key_rejected():
    """A license signed with a DIFFERENT key than the one Loom trusts must
    be rejected — this is the property that makes the business model work."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from loom.licensing import InvalidLicenseError, verify_license_string

    attacker_key = Ed25519PrivateKey.generate()  # NOT the key loom.licensing trusts
    forged = _sign_test_license(attacker_key, {
        "org": "Free Forever Inc", "tier": "team", "seats": 999, "issued": "2026-01-01", "expires": None,
    })
    with pytest.raises(InvalidLicenseError, match="Signature"):
        verify_license_string(forged)


def test_license_expired_rejected(test_keypair):
    from loom.licensing import InvalidLicenseError, verify_license_string

    lic_str = _sign_test_license(test_keypair, {
        "org": "Old Co", "tier": "team", "seats": 1, "issued": "2020-01-01", "expires": "2020-06-01",
    })
    with pytest.raises(InvalidLicenseError, match="expired"):
        verify_license_string(lic_str)


def test_license_store_activate_and_deactivate(test_keypair, tmp_path):
    from loom.licensing import LicenseStore, is_team_licensed

    store = LicenseStore(path=tmp_path / "license.json")
    assert store.current() is None
    assert is_team_licensed(store) is False

    lic_str = _sign_test_license(test_keypair, {
        "org": "Acme Inc", "tier": "team", "seats": 5, "issued": "2026-01-01", "expires": None,
    })
    store.activate(lic_str)
    assert is_team_licensed(store) is True
    assert store.current().org == "Acme Inc"

    store.deactivate()
    assert store.current() is None
    assert is_team_licensed(store) is False


def test_license_hand_edited_file_rejected(test_keypair, tmp_path):
    """Someone hand-editing ~/.loom/license.json to say tier=team must NOT
    work — current() re-verifies the signature every time, not just the
    first time."""
    import json as _json

    from loom.licensing import LicenseStore

    store = LicenseStore(path=tmp_path / "license.json")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(_json.dumps({"raw": "not-a-real-signed-license"}))
    assert store.current() is None


# ---------------------------------------------------------------- usage limiter

def test_usage_limiter_daily_cap(tmp_path):
    from loom.licensing import FREE_TIER_DAILY_RUNS, UsageLimiter

    limiter = UsageLimiter(path=tmp_path / "usage.json")
    assert limiter.runs_in_last_24h() == 0
    for _ in range(FREE_TIER_DAILY_RUNS):
        limiter.record_run()
    assert limiter.runs_in_last_24h() == FREE_TIER_DAILY_RUNS


def test_check_run_allowed_free_tier_blocks_after_limit(tmp_path):
    from loom.licensing import FREE_TIER_DAILY_RUNS, LicenseStore, UsageLimiter, check_run_allowed

    license_store = LicenseStore(path=tmp_path / "license.json")  # no license = free tier
    usage = UsageLimiter(path=tmp_path / "usage.json")

    for _ in range(FREE_TIER_DAILY_RUNS):
        allowed, msg, max_repairs = check_run_allowed(license_store, usage)
        assert allowed is True
        assert max_repairs == 1  # FREE_TIER_MAX_REPAIRS
        usage.record_run()

    allowed, msg, max_repairs = check_run_allowed(license_store, usage)
    assert allowed is False
    assert "Upgrade" in msg


def test_check_run_allowed_team_tier_unlimited(test_keypair, tmp_path):
    from loom.licensing import FREE_TIER_DAILY_RUNS, LicenseStore, UsageLimiter, check_run_allowed

    license_store = LicenseStore(path=tmp_path / "license.json")
    usage = UsageLimiter(path=tmp_path / "usage.json")
    lic_str = _sign_test_license(test_keypair, {
        "org": "Acme Inc", "tier": "team", "seats": 5, "issued": "2026-01-01", "expires": None,
    })
    license_store.activate(lic_str)

    for _ in range(FREE_TIER_DAILY_RUNS + 5):  # well past the free cap
        usage.record_run()

    allowed, msg, max_repairs = check_run_allowed(license_store, usage)
    assert allowed is True
    assert max_repairs > FREE_TIER_DAILY_RUNS  # effectively unlimited repairs too


# ---------------------------------------------------------------- report

def test_report_build_summaries_aggregates_correctly(tmp_path):
    from loom.report import build_summaries, to_csv, to_json

    store = EventStore(tmp_path / "events.sqlite3")
    sid = store.new_session("add feature X")
    store.append(sid, "model_response", {"input_tokens": 100, "output_tokens": 50}, agent="planner")
    store.append(sid, "model_response", {"input_tokens": 200, "output_tokens": 80}, agent="coder")
    store.append(sid, "approval", {"approved": True}, agent="coder")
    store.append(sid, "approval", {"approved": False}, agent="coder")
    store.append(sid, "key_swap", {"failed_credential": "anthropic:broke"}, agent="coder")
    store.append(sid, "commit", {"sha": "abc123"})
    store.set_session_status(sid, "completed")

    summaries = build_summaries(store)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.input_tokens == 300 and s.output_tokens == 130
    assert s.approvals_granted == 1 and s.approvals_denied == 1
    assert s.key_swaps == ["anthropic:broke"]
    assert s.committed is True and s.commit_sha == "abc123"

    csv_out = to_csv(summaries)
    assert "add feature X" in csv_out and "300" in csv_out
    json_out = to_json(summaries)
    parsed = json.loads(json_out)
    assert parsed[0]["input_tokens"] == 300
