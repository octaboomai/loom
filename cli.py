from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from loom.config import LoomConfig
from loom.context.repo_map import build_repo_map, render_map
from loom.credentials import CredentialStore
from loom.events.store import EventStore
from loom.licensing import (
    FREE_TIER_DAILY_RUNS,
    InvalidLicenseError,
    LicenseStore,
    UsageLimiter,
    check_run_allowed,
    is_team_licensed,
)
from loom.orchestrator import Orchestrator
from loom.report import build_summaries, to_csv, to_json
from loom.router import AllCredentialsFailedError, ModelRouter, NoCredentialsError

app = typer.Typer(add_completion=False, help="Loom — a hierarchical multi-agent coding assistant.")
keys_app = typer.Typer(add_completion=False, help="Manage API credentials (stored in ~/.loom/credentials.json, never in a repo).")
license_app = typer.Typer(add_completion=False, help="Manage your Loom Team license.")
app.add_typer(keys_app, name="keys")
app.add_typer(license_app, name="license")
console = Console()


def _load(repo: Optional[str], yes: bool) -> tuple[LoomConfig, EventStore]:
    root = Path(repo).resolve() if repo else Path.cwd()
    if not root.is_dir():
        console.print(f"[red]Not a directory:[/red] {root}")
        raise typer.Exit(1)
    cfg = LoomConfig.load(root, auto_approve=yes)
    store = EventStore(cfg.event_db_path())
    return cfg, store


def _cli_approval(tool_name: str, tool_input: dict) -> bool:
    console.print(Panel.fit(
        json.dumps(tool_input, indent=2)[:1500],
        title=f"[yellow]Approval needed: {tool_name}[/yellow]",
        border_style="yellow",
    ))
    return Confirm.ask("Allow this action?", default=False)


def _has_any_credentials(cfg: LoomConfig) -> bool:
    store = CredentialStore()
    if store.providers_configured():
        return True
    return bool(store.env_fallback())


def _no_credentials_message() -> None:
    console.print(Panel.fit(
        "No API credentials found. Set one up either way:\n\n"
        "  [bold]Quick (single key, this session only):[/bold]\n"
        "    export ANTHROPIC_API_KEY=sk-ant-...\n\n"
        "  [bold]Recommended (saved, supports multiple keys/providers):[/bold]\n"
        "    loom keys add anthropic sk-ant-...\n"
        "    loom keys add openai sk-... --model gpt-5.1\n\n"
        "Then verify it actually works (cheap, ~1 token) with:\n"
        "    loom keys test",
        title="[red]No credentials configured[/red]", border_style="red",
    ))


def _parse_models(model_arg: Optional[str]) -> dict:
    """--model accepts either a single model name (applied to all three
    tiers) or a tier=model,tier=model list, e.g.:
        --model gpt-5.1
        --model "fast=gpt-5.1-mini,core=gpt-5.1,deep=gpt-5.1"
    """
    if not model_arg:
        return {}
    if "=" not in model_arg:
        return {"fast": model_arg, "core": model_arg, "deep": model_arg}
    out = {}
    for part in model_arg.split(","):
        tier, _, model = part.partition("=")
        tier, model = tier.strip(), model.strip()
        if tier and model:
            out[tier] = model
    return out


# ---------------------------------------------------------------- keys

@keys_app.command("add")
def keys_add(
    provider: str = typer.Argument(
        ...,
        help="anthropic | openai | a shortcut (deepseek, kimi, openrouter, groq, together) "
             "— shortcuts are all OpenAI-compatible under the hood, just with the base URL prefilled.",
    ),
    api_key: str = typer.Argument(..., help="The API key / token."),
    name: str = typer.Option(None, "--name", help="Label for this key, e.g. 'work', 'personal'. Defaults to the provider name."),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model name (or tier=model,tier=model list). Required for openai/other providers; "
             "optional for anthropic, which has sensible built-in defaults.",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="Override the API endpoint — needed for a provider/service not in the built-in shortcut list.",
    ),
):
    """Register an API key. Add as many as you like, across providers — Loom tries them
    in order and automatically moves to the next one if a key is out of quota or invalid.

    Examples:
        loom keys add anthropic sk-ant-...
        loom keys add deepseek sk-...   --model deepseek-v4-flash
        loom keys add kimi sk-...       --model kimi-k3
        loom keys add openrouter sk-... --model meta-llama/llama-3.3-70b
    """
    from loom.credentials import PROVIDER_PRESETS

    actual_provider = provider
    resolved_base_url = base_url
    if provider in PROVIDER_PRESETS:
        actual_provider = "openai"  # every preset speaks the OpenAI Chat Completions format
        resolved_base_url = base_url or PROVIDER_PRESETS[provider]
    elif provider not in ("anthropic", "openai"):
        console.print(
            f"[red]Unknown provider:[/red] {provider}. Supported: anthropic, openai, "
            f"or a shortcut ({', '.join(PROVIDER_PRESETS)}). Any other OpenAI-compatible "
            f"service also works: loom keys add openai <key> --base-url <url> --model <model>"
        )
        raise typer.Exit(1)

    models = _parse_models(model)
    if actual_provider == "openai" and not models:
        console.print(f"[red]--model is required for '{provider}'[/red] "
                       "— model naming varies by vendor and can't be guessed safely. "
                       "Example: --model deepseek-v4-flash")
        raise typer.Exit(1)

    store = CredentialStore()
    store.add(actual_provider, api_key, name=name or provider, base_url=resolved_base_url, models=models)
    console.print(f"[green]Added[/green] {provider}:{name or provider} to {store.path}")


@keys_app.command("list")
def keys_list():
    """List registered credentials (keys are masked)."""
    store = CredentialStore()
    creds = store.list_all()
    if not creds:
        console.print("[dim]No keys registered via `loom keys add`.[/dim]")
        env = store.env_fallback()
        if env:
            console.print("[dim]Using environment-variable fallback instead:[/dim]")
            for c in env:
                console.print(f"  [cyan]{c.provider}[/cyan]:{c.name} (from {c.provider.upper()}_API_KEY)")
        return
    table = Table(title=f"Loom credentials ({store.path})")
    for col in ("provider", "name", "key", "base_url", "models"):
        table.add_column(col)
    for c in creds:
        table.add_row(c.provider, c.name, c.masked_key(), c.base_url or "(default)",
                       ", ".join(f"{k}={v}" for k, v in c.models.items()) or "(uses built-in defaults)")
    console.print(table)


@keys_app.command("remove")
def keys_remove(provider: str, name: str):
    """Remove a registered key."""
    store = CredentialStore()
    if store.remove(provider, name):
        console.print(f"[green]Removed[/green] {provider}:{name}")
    else:
        console.print(f"[yellow]Not found:[/yellow] {provider}:{name}")


@keys_app.command("test")
def keys_test():
    """Try every configured credential with a tiny (~1 token) request, so you
    can catch a billing/quota problem before spending a real task's worth of tokens on it."""
    store = CredentialStore()
    creds = store.list_all() or store.env_fallback()
    if not creds:
        console.print("[yellow]No credentials configured.[/yellow] Run: loom keys add anthropic sk-ant-...")
        raise typer.Exit(1)

    for cred in creds:
        model = cred.models.get("fast") or cred.models.get("core")
        if not model and cred.provider == "anthropic":
            from loom.config import DEFAULT_MODELS
            model = DEFAULT_MODELS["anthropic"]["fast"]
        if not model:
            console.print(f"[yellow]SKIP[/yellow] {cred.provider}:{cred.name} — no model configured to test with")
            continue
        try:
            if cred.provider == "anthropic":
                from loom.providers.anthropic_provider import AnthropicProvider
                provider = AnthropicProvider(cred.api_key, cred.base_url)
            else:
                from loom.providers.openai_provider import OpenAIProvider
                provider = OpenAIProvider(cred.api_key, cred.base_url)
            provider.complete(model=model, system="Reply with one word.",
                               messages=[{"role": "user", "content": "Say OK."}], tools=None, max_tokens=8)
            console.print(f"[green]PASS[/green] {cred.provider}:{cred.name} ({model})")
        except Exception as e:
            console.print(f"[red]FAIL[/red] {cred.provider}:{cred.name} ({model}) — {e}")


# ---------------------------------------------------------------- license

@license_app.command("activate")
def license_activate(license_key: str = typer.Argument(..., help="The license string you were given.")):
    """Activate a Team license — lifts the free-tier daily run limit and
    single-repair cap, and unlocks `loom report --export`."""
    store = LicenseStore()
    try:
        lic = store.activate(license_key)
    except InvalidLicenseError as e:
        console.print(f"[red]Could not activate:[/red] {e}")
        raise typer.Exit(1)
    console.print(Panel.fit(
        f"Organization: [bold]{lic.org}[/bold]\nSeats: {lic.seats}\nExpires: {lic.expires or 'never'}",
        title="[green]Team license activated[/green]", border_style="green",
    ))


@license_app.command("status")
def license_status():
    """Show the currently active license, if any."""
    store = LicenseStore()
    lic = store.current()
    if not lic:
        usage = UsageLimiter()
        used = usage.runs_in_last_24h()
        console.print(Panel.fit(
            f"No Team license active — running on the free tier.\n"
            f"Runs used today: {used}/{FREE_TIER_DAILY_RUNS}\n\n"
            f"Upgrade with: loom license activate <key>",
            title="Free tier", border_style="cyan",
        ))
        return
    console.print(Panel.fit(
        f"Organization: [bold]{lic.org}[/bold]\nTier: {lic.tier}\nSeats: {lic.seats}\n"
        f"Issued: {lic.issued}\nExpires: {lic.expires or 'never'}",
        title="[green]Team license active[/green]", border_style="green",
    ))


@license_app.command("deactivate")
def license_deactivate():
    """Remove the active license and drop back to the free tier."""
    LicenseStore().deactivate()
    console.print("Deactivated. Back on the free tier.")


# ---------------------------------------------------------------- report

@app.command()
def report(
    repo: Optional[str] = typer.Option(None, "--repo", "-r"),
    export: Optional[str] = typer.Option(None, "--export", help="csv or json — requires a Team license."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File path to write the export to."),
    limit: int = typer.Option(50, "--limit", help="Max sessions to include."),
):
    """Cost & audit summary across sessions in this project: tokens used,
    approvals granted/denied, credential swaps, and what got committed.
    Free tier: console view. Team tier: --export csv/json for sharing."""
    cfg, store = _load(repo, yes=False)
    summaries = build_summaries(store, limit=limit)

    if not summaries:
        console.print("[dim]No sessions recorded yet in this project.[/dim]")
        return

    if export:
        if not is_team_licensed():
            console.print(Panel.fit(
                "Exporting reports requires a Team license.\n\n"
                "Upgrade with: loom license activate <key>\n"
                "(Console view below is still free.)",
                title="[yellow]Team feature[/yellow]", border_style="yellow",
            ))
        else:
            if export not in ("csv", "json"):
                console.print("[red]--export must be 'csv' or 'json'[/red]")
                raise typer.Exit(1)
            content = to_csv(summaries) if export == "csv" else to_json(summaries)
            if output:
                Path(output).write_text(content)
                console.print(f"[green]Wrote[/green] {output}")
            else:
                print(content)
            return

    table = Table(title=f"Loom report — {cfg.repo_root.name}")
    for col in ("session", "task", "status", "in/out tokens", "approvals", "denied", "swaps", "committed"):
        table.add_column(col)
    total_in = total_out = 0
    for s in summaries:
        table.add_row(s.session_id, s.task[:40], s.status, f"{s.input_tokens}/{s.output_tokens}",
                       str(s.approvals_granted), str(s.approvals_denied), str(len(s.key_swaps)),
                       s.commit_sha or "—")
        total_in += s.input_tokens
        total_out += s.output_tokens
    console.print(table)
    console.print(f"[dim]{len(summaries)} sessions · {total_in} input / {total_out} output tokens total[/dim]")
    if not is_team_licensed():
        console.print("[dim]Tip: `loom report --export csv` (Team license) for a shareable file.[/dim]")


# ---------------------------------------------------------------- run / plan

@app.command()
def run(
    task: str = typer.Argument(..., help="Natural-language description of what to build/fix."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repo root (default: cwd)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve all shell commands and the final commit."),
    no_branch: bool = typer.Option(False, "--no-branch", help="Work on the current branch instead of creating loom/<task>."),
    max_repairs: int = typer.Option(2, "--max-repairs", help="Max coder retry loops if tests fail or review requests changes."),
):
    """Run the full Planner -> Coder -> Tester -> Reviewer pipeline on a task."""
    cfg, store = _load(repo, yes)
    if not _has_any_credentials(cfg):
        _no_credentials_message()
        raise typer.Exit(1)

    allowed, block_message, tier_max_repairs = check_run_allowed()
    if not allowed:
        console.print(Panel.fit(block_message, title="[yellow]Free tier limit reached[/yellow]", border_style="yellow"))
        raise typer.Exit(1)
    effective_max_repairs = min(max_repairs, tier_max_repairs)
    if effective_max_repairs < max_repairs:
        console.print(f"[dim]Free tier caps repair attempts at {tier_max_repairs} "
                       f"(requested {max_repairs}) — loom license activate <key> to lift this.[/dim]")
    UsageLimiter().record_run()

    orch = Orchestrator(cfg, store)
    approval_fn = None if yes else _cli_approval

    def status(msg: str):
        console.print(f"[cyan]›[/cyan] {msg}")

    console.print(Panel.fit(f"[bold]{task}[/bold]", title="Loom run", border_style="cyan"))
    try:
        result = orch.run(task, approval_fn=approval_fn, status=status,
                           max_repair_loops=effective_max_repairs, create_branch=not no_branch, auto_commit=yes)
    except (NoCredentialsError, AllCredentialsFailedError) as e:
        console.print(f"[red]{e}[/red]")
        console.print("[dim]Tip: run `loom keys test` to check which of your keys actually work.[/dim]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print()
    if result.verdict == "approved" and result.committed:
        console.print(Panel.fit(
            f"Committed [bold]{result.commit_sha}[/bold] on branch [bold]{result.branch}[/bold]",
            title="[green]Done[/green]", border_style="green",
        ))
    elif result.verdict == "approved":
        console.print(Panel.fit("Review approved but nothing was committed (declined or no diff).",
                                 title="[yellow]Approved, not committed[/yellow]", border_style="yellow"))
    elif result.verdict == "no_changes":
        console.print(Panel.fit("Approved, but no file changes were detected.", title="[yellow]No changes[/yellow]",
                                 border_style="yellow"))
    else:
        console.print(Panel.fit(
            f"Gave up after repair attempts. Inspect the session for details:\n"
            f"  loom log {result.session_id}",
            title="[red]Did not pass review[/red]", border_style="red",
        ))
    console.print(f"[dim]session: {result.session_id}[/dim]")
    console.print(f"[dim]{orch.router.usage.summary()}[/dim]")


@app.command()
def plan(
    task: str = typer.Argument(..., help="Natural-language description of what you want built/fixed."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repo root (default: cwd)."),
):
    """Read-only dry run: see the Planner's breakdown WITHOUT writing any files,
    running any commands, or committing anything. Costs one model call — the
    cheapest way to sanity-check a task (and your API key) before `loom run`."""
    cfg, store = _load(repo, yes=False)
    if not _has_any_credentials(cfg):
        _no_credentials_message()
        raise typer.Exit(1)

    from loom.agents.planner import build_planner
    from loom.context.repo_map import build_repo_map, rank_relevant_files, render_map

    router = ModelRouter(cfg)
    session_id = store.new_session(f"[plan-only] {task}")
    console.print("[cyan]›[/cyan] Mapping repository...")
    maps = build_repo_map(cfg.repo_root)
    relevant = rank_relevant_files(maps, task)
    console.print("[cyan]›[/cyan] Planning...")
    planner = build_planner(router, cfg)
    try:
        result = planner.run(
            f"TASK: {task}\n\nREPO MAP (most relevant files):\n{render_map(relevant)}\n\n"
            "Produce the implementation plan now.",
            store, session_id,
        )
    except (NoCredentialsError, AllCredentialsFailedError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    store.set_session_status(session_id, "completed")
    console.print()
    console.print(Panel(result, title="Plan (nothing was changed on disk)", border_style="cyan"))
    console.print(f"[dim]{router.usage.summary()}[/dim]")


@app.command()
def sessions(repo: Optional[str] = typer.Option(None, "--repo", "-r")):
    """List recent sessions."""
    cfg, store = _load(repo, yes=False)
    rows = store._conn.execute(
        "SELECT id, task, status, datetime(created_at, 'unixepoch') FROM sessions ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    table = Table(title="Loom sessions")
    for col in ("id", "task", "status", "created"):
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(x) for x in row])
    console.print(table)


@app.command()
def log(session_id: str, repo: Optional[str] = typer.Option(None, "--repo", "-r"), n: int = 40):
    """Show the last N events for a session (for debugging / audit)."""
    cfg, store = _load(repo, yes=False)
    for ev in store.tail(session_id, n=n):
        console.print(f"[dim]{ev.seq:>5}[/dim] [bold]{ev.kind:<14}[/bold] [magenta]{ev.agent or '-':<10}[/magenta] "
                       f"{json.dumps(ev.payload)[:180]}")


@app.command()
def map(
    repo: Optional[str] = typer.Option(None, "--repo", "-r"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Show only files relevant to this query."),
):
    """Print the repository context map Loom builds before planning."""
    cfg, _ = _load(repo, yes=False)
    maps = build_repo_map(cfg.repo_root)
    if query:
        from loom.context.repo_map import rank_relevant_files
        maps = rank_relevant_files(maps, query, top_k=25)
    console.print(render_map(maps))
    console.print(f"\n[dim]{len(maps)} files mapped[/dim]")


@app.command()
def init(repo: Optional[str] = typer.Option(None, "--repo", "-r")):
    """Create a .loom/config.json template for team-shared guardrails."""
    cfg, _ = _load(repo, yes=False)
    path = cfg.loom_dir() / "config.json"
    if path.exists():
        console.print(f"[yellow]Already exists:[/yellow] {path}")
        raise typer.Exit(0)
    template = {
        "guardrails": ["Never weaken authentication checks", "Keep public API responses backward compatible"],
        "forbidden_paths": ["**/migrations/**", "**/*.pem", "**/secrets/**"],
        "approval_required_for": ["shell", "git_push"],
        "model_overrides": {},
        "provider_priority": ["anthropic", "openai"],
    }
    path.write_text(json.dumps(template, indent=2))
    console.print(f"[green]Created[/green] {path}")


if __name__ == "__main__":
    app()
