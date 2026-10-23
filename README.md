# hagent: a small + big model coding agent

A minimal, bash-only coding agent in the style of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
extended with the three things you asked for:

1. **Two models.** A small, fast model drives almost every step. A big model is consulted rarely, with a compact digest instead of the full transcript.
2. **Token discipline.** Output truncation, history compaction, a hard budget on big-model calls, and a cost cap.
3. **No infinite loops.** Hard caps plus a loop detector that escalates in stages and always ends the run.

No runtime dependencies (stdlib only, Python 3.11+). Works with any OpenAI-compatible endpoint.

## Quick start

```bash
cd hybrid-agent
pip install -e .                      # or just: export PYTHONPATH=src
cp examples/config.example.toml config.toml
# edit config.toml: model ids, base_url, api_key_env, prices
export NVIDIA_API_KEY=...             # whatever api_key_env you set

hagent run --repo /path/to/git/repo --task "Fix: add() returns the wrong value" --config config.toml
```

The final patch goes to stdout (or `--patch-out file`), a one-line summary with per-model calls and cost to stderr.
Useful flags: `--mode hybrid|small|big`, `--trajectory out.json`, `--task-file issue.md`, `--init-git`, `-v`.

Try it without any API key: `python examples/demo_offline.py`. From Python:

```python
from hagent import solve, load_settings
result = solve("fix the bug", "/path/to/repo", load_settings("config.toml"), mode="hybrid")
print(result.exit_status, result.total_cost, result.usage)   # result.patch is always filled in
```

## How it works

```
task -> [system, user] -> loop:
          compaction check -> (hybrid: consult big model if needed) -> limits check
          -> model reply -> exactly one ```bash block -> execute -> append output
          -> loop detector sees (command, output, git-diff hash)
        submit = command whose first output line is HAGENT_SUBMIT_FINAL_OUTPUT (then: reviewer, then done)
```

Kept from mini-swe-agent: bash is the only tool, every command is an independent process, history is one linear message list,
the submit-by-magic-first-line convention, exceptions as control flow, trajectory saved every step.

| File | Role |
|---|---|
| `agent.py` | Base loop (single model). Also your **baseline** for benchmarking |
| `hybrid.py` | `HybridAgent`: routing, escalation ladder, final review, explorer hook |
| `detector.py` | Loop / no-progress detection |
| `compaction.py` | Replaces the middle of the history with a summary |
| `helpers.py`, `prompts.py` | Single-call helpers (advise / review / summarize) and all prompts |
| `explorer.py` | Optional read-only subagent loop (off by default) |
| `env.py`, `models.py`, `config.py`, `runner.py`, `cli.py` | Shell env + git helpers, model clients, settings, `solve()`, CLI |

### Who runs when

| Role | Model | Type | When |
|---|---|---|---|
| Main agent | small | full loop | every step |
| Compactor | small | one call | history over `ctx_char_limit` |
| Advisor | big | one call | once at `plan_at_step`, and when stuck |
| Takeover | big | a few loop steps | third and later stuck flags |
| Reviewer | big | one call | when a submission is attempted |
| Explorer | small | tiny isolated loop | only if the main agent calls `ask_explorer "..."` and `enable_explorer = true` |

The big model always sees a digest: task, the agent's notes, the last 6 messages (truncated), the current `git diff`, the last test output, and why it is being asked.

### Loop protection (layers, in order of strictness)

1. **Hard caps, always enforced:** `max_steps`, `max_cost` (all models combined), `max_wall_s`, per-command `cmd_timeout_s` (kills the whole process group), `max_format_errors` in a row.
2. **Loop detector** (`detector.py`). "Repeated" means *same command, same repo state*, so re-running tests after an edit is fine. Signals: `repeated_command`, `repeated_output`, `oscillation` (repo returns to an earlier state), `stalled` (no new diff and no new error for `stall_steps`, only after `stall_warmup` steps).
3. **Escalation ladder** (each flag is a strike; a new repo state or new error resets strikes):
   strike 1 nudge, strike 2 big-model advice, strike 3+ advice plus `takeover_steps` big-model steps, `max_strikes` stops the run with `Stuck`.
4. **Review cannot loop:** at most `max_review_rejections`; an unparseable reviewer reply counts as APPROVE; empty-diff submissions are bounced without a big call.
5. **Failures degrade, not crash:** big model or summarizer errors fall back to nudges / a command-list summary. Every exit still returns the current patch.

### Token control

- Command output is truncated head+tail to `obs_max_chars`; the prompt teaches `grep -n`, `sed -n`, `head`.
- Compaction keeps system prompt, task and the last `keep_last_messages`; the rest becomes one summary inside the task message (roles keep alternating).
- Big-model calls are capped by `max_big_calls`, with one call reserved for the final review.
- `<think>...</think>` blocks in replies are stripped before they enter the history.

### Repair-incident extras

- **Reviewer pass probability.** The reviewer replies `APPROVE`/`REJECT` and then `PASS_PROBABILITY: <0-1>`. The probability is logged in `RunResult.reviews`, the trajectory's `reviews`, and `usage["big"].pass_probabilities`. It never gates anything; a missing or garbled value is logged as `null`.
- **Task templates.** `load_task_template(path).render(IncidentContext(...))` builds the task text from `<<TASK>>`, `<<ROUTER_VERDICT>>`, `<<LOCATION_EVIDENCE>>`, `<<REPRO_COMMAND>>`, `<<ROLLOUT_SUMMARY>>`, `<<REPO_PATH>>`. The default (`examples/task_template.md`) frames the router verdict as a hypothesis to verify first.
- **Custom commands.** `[[commands]]` entries (`name`, `path`, `description`, `max_output_chars`) are listed in the system prompt and put on PATH via shims; output of a command that calls one uses that command's truncation limit.

## Benchmarking (do this before tuning)

Run the same tasks with `--mode small`, `--mode big` and `--mode hybrid`, then compare resolve rate vs cost using the `usage` field / `--trajectory` JSON.
Only keep a component (planning consult, review, explorer) if it moves that curve. Knobs worth sweeping: `plan_at_step`, `repeat_threshold`, `stall_steps`, `max_big_calls`, `final_review`.

## Known limitations and caveats

- **The agent runs arbitrary bash.** Use the container backend for anything you do not fully trust: `[env] backend = "docker"` (or `runtime = "podman"`) runs every command in a container holding a *copy* of the repo, with no network and CPU/memory/pid limits by default. Timeouts kill the whole process tree, including children that left the process group. The image needs bash, git and coreutils (`examples/docker/Dockerfile`). The copy includes `.git`, so history is visible to the agent; hiding it is the caller's job.
- POSIX only (process groups, `/bin/bash`). The working dir should be a git repo with a sensible `.gitignore`; the detector and reviewer rely on `git diff`.
- The explorer's read-only guard is a best-effort filter (it can false-positive on `>` inside quoted code, and cannot see writes made by scripts). It reports it if the repo changed. Real isolation needs a read-only mount.
- Token counts and the cost cap use the provider's `usage` block (`prompt_tokens`/`completion_tokens`, or `input_tokens`/`output_tokens`, or one side plus `total_tokens`). Only when it is missing are tokens estimated at ~4 characters per token; `Usage.estimated_calls` counts those calls. History budgets (`ctx_char_limit`, `obs_max_chars`) are still in characters. Costs are only as accurate as the prices you configure; with prices at 0 the cost cap never triggers.
- Each compaction rewrites the task message, which invalidates provider prefix caching once per compaction.
- Model ids in the example config are placeholders. I have not verified the Nemotron model names, context sizes or pricing; check your provider.
- Tested with scripted models, a fake HTTP server and a local Docker daemon (157 tests; container tests skip without Docker); it has not been run against a live model or on SWE-bench yet.

## Tests

```bash
pip install pytest && pytest -q
```
