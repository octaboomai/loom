# Loom

A hierarchical multi-agent coding assistant, run from your terminal against
your own repository, powered by the Anthropic API.

Instead of one model doing everything in a single loop, Loom splits the
work across four role-specialized agents that hand off to each other, with
real validation and a human approval gate in between:

```
Planner (read-only) → Coder (writes code) → automated lint/type/test
   → Tester (verifies) → Reviewer (read-only) → you approve → git commit
```

If the Tester or Reviewer isn't satisfied, Loom sends the Coder specific
feedback and retries (bounded — it won't loop forever).

## Install

```bash
git clone <this repo>
cd loom
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Setting up API access

Loom works with **any number of API keys, across multiple providers** — if
one runs out of quota or credit, it automatically tries the next one and
tells you it swapped, instead of just failing.

**Quick start (single key, matches earlier versions of this README):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Recommended — register keys so they persist and you can add more than one:**

```bash
loom keys add anthropic sk-ant-...                        # primary
loom keys add anthropic sk-ant-...  --name backup          # tried if the first is out of credit
loom keys add deepseek sk-...       --model deepseek-v4-flash
loom keys add kimi sk-...           --model kimi-k3
loom keys add openai sk-...         --model gpt-5.1        # OpenAI itself
```

Keys are stored at `~/.loom/credentials.json` — **outside any git repo**,
so they can never end up committed by accident, no matter what a project's
`.gitignore` says.

`deepseek`, `kimi` (alias `moonshot`), `openrouter`, `groq`, and `together`
are built-in shortcuts — they're all OpenAI-compatible APIs under the hood,
so `loom keys add deepseek ...` is really just `loom keys add openai ...
--base-url https://api.deepseek.com/v1` with the URL filled in. Anything
else that speaks the OpenAI Chat Completions format (a local Ollama/vLLM
server, a vendor not on that shortlist) works the same way with
`--base-url` spelled out:

```bash
loom keys add openai sk-... --model llama-3.3-70b --base-url https://openrouter.ai/api/v1 --name openrouter
```

`--model` is required for every non-Anthropic provider — model naming
varies too much by vendor to guess safely and stay correct over time.

Before running anything real, check your keys actually work — this costs
about one token per key, not a full task's worth:

```bash
loom keys test
```

Other key management:

```bash
loom keys list             # see what's registered (keys are masked)
loom keys remove openai openrouter
```

## Quickstart

```bash
cd /path/to/your/project
loom init          # optional: creates .loom/config.json for team guardrails
loom plan "add input validation to the login endpoint"   # cheap dry run — one model call, no files touched
loom run  "add input validation to the login endpoint"   # the real thing
```

Loom will:
1. Build a compact map of your repo (function/class signatures, not full files)
2. Have the Planner turn your task into concrete steps
3. Let the Coder implement it (asking your approval before running any shell command)
4. Auto-run whatever lint/type-check/test tools it detects in your repo
5. Have the Tester and Reviewer agents sign off (or send it back for a fix)
6. Ask you to approve the final commit

Other commands:

```bash
loom plan "..."                 # Planner-only dry run — see the plan, spend nothing else
loom map                        # see the repo map Loom builds before planning
loom map -q "auth login"        # see just the files relevant to a query
loom sessions                   # list past runs
loom log <session_id>           # inspect what an agent actually did, step by step — includes key_swap events
```

## Team guardrails

`loom init` creates `.loom/config.json`:

```json
{
  "guardrails": ["Never weaken authentication checks"],
  "forbidden_paths": ["**/migrations/**", "**/*.pem", "**/secrets/**"],
  "approval_required_for": ["shell", "git_push"],
  "model_overrides": {},
  "provider_priority": ["anthropic", "openai"]
}
```

Commit this file to your repo — every agent gets these guardrails prepended
to its instructions, `forbidden_paths` is enforced at the file-write layer
(not just requested of the model), and `provider_priority` controls which
provider Loom tries first when you have keys for more than one (this is
separate from your personal keys, which never belong in a committed file —
see "Setting up API access" above).

## Team license

The CLI is free for individual use: 5 runs per 24 hours, single-attempt
auto-repair. A Team license lifts both limits and unlocks exporting
`loom report` (see below) for sharing with a manager or compliance review.

```bash
loom license activate <key>      # unlocks unlimited runs + report export
loom license status
loom license deactivate
```

If you're selling licenses yourself: see `admin/README.md` and
`admin/generate_license.py`. License keys are signed with Ed25519 and
verified offline against a public key baked into this repo — publishing
this repo does not let anyone mint their own free license (there's a test
suite covering exactly that: tampered payloads, wrong signing keys, and
hand-edited license files are all rejected — see `tests/test_smoke.py`).

## Cost & audit report

```bash
loom report                      # console summary — free
loom report --export csv         # full export — requires Team license
```

Aggregates every session in a project: tokens used, approvals granted vs.
denied, credential swaps, and what got committed — useful for a manager or
compliance review, which is why the export half is the paid feature.

## Architecture

- **`loom/agents/`** — Planner, Coder, Tester, Reviewer. All four share one
  generic tool-use loop (`agents/base.py`); what differs is the system
  prompt and which tools each role is allowed to touch (Planner and Reviewer
  are read-only).
- **`loom/providers/`** — one adapter per LLM backend (`anthropic_provider.py`,
  `openai_provider.py`), each translating to/from a normalized response
  format so the agent loop doesn't need to know which backend answered.
  Also classifies errors into "worth trying the next key" (quota/auth/
  connection) vs. "a real bug, don't hide it" (a malformed request).
- **`loom/credentials.py`** — stores any number of API keys, across any
  number of providers, at `~/.loom/credentials.json` (outside any repo).
- **`loom/router.py`** — resolves each agent role to a model tier
  (`fast`/`core`/`deep`), then tries configured credentials in order,
  automatically moving to the next one on a quota/auth/connection failure
  and logging the swap.
- **`loom/context/repo_map.py`** — builds a symbol-level map of the repo
  using tree-sitter where a grammar is available, falling back to a regex
  heuristic otherwise. Ranks files by relevance to the task before handing
  them to an agent, so context stays small.
- **`loom/tools/`** — file read/write/patch (path-restricted to the repo,
  guarded against team-configured forbidden paths), sandboxed shell exec
  (blocklist + timeout), and git operations.
- **`loom/validation/runner.py`** — auto-detects and runs whatever's
  actually present in your repo (ruff/mypy/pytest/bandit for Python,
  eslint/tsc/npm test for JS/TS, cargo check/test/clippy for Rust).
- **`loom/events/store.py`** — append-only SQLite event log. Every model
  call, tool call, approval decision, credential swap, and edit is recorded
  per session, so a run can be inspected (`loom log`).
- **`loom/licensing.py`** / **`loom/report.py`** — the freemium mechanics:
  offline-verified license keys gating the daily run cap and repair-loop
  limit, and the cost/audit report built from the event log.

## What's in this build vs. the original concept

This implements the actual agent pipeline, context engine, validation
layer, event store, guardrail system, and multi-provider credential
failover end-to-end, and it's tested (see `tests/test_smoke.py`, 24 tests
covering the failover logic specifically) — not just a scaffold.

**Not included**, and each would be its own separate project:
- IDE extensions (VS Code / JetBrains / Zed / Neovim)
- A web dashboard / team UI
- A REST API for CI/CD integration
- A dedicated adapter for every LLM vendor's native API — `openai` covers
  OpenAI itself plus anything OpenAI-compatible (OpenRouter, Groq,
  Together, local Ollama/vLLM via `--base-url`), which is most of the
  market, but something with a genuinely different API shape (e.g. Google's
  native Gemini API, not through an OpenAI-compatible proxy) would need its
  own adapter in `loom/providers/`.
- Semantic embeddings / vector search for context ranking — the current
  relevance ranking is a fast token-overlap heuristic, which is cheap and
  dependency-free but cruder than embeddings on a very large repo.

## Running the test suite

```bash
pip install pytest ruff
pytest -q
```
