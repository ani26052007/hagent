# Decisions

## Step 1A: hagent improvements (plan.md 3.3)

**Token accounting.** `models.py` already read `usage` and fell back to a character estimate, so the change is explicitness, not new plumbing: `Reply.usage_source` (api / partial / estimated), support for `input_tokens`/`output_tokens` and `total_tokens`, per-side fallback, and `Usage.estimated_calls`. An all-zero usage block still counts as missing (kept from the original so caps never silently stop working). Rejected: a tokenizer dependency for exact estimates (breaks stdlib-only; provider counts are the truth anyway).

**Container env.** New `docker_env.DockerEnv` with the same contract as `LocalEnv` (now written down as `env.Env`); git helpers moved unchanged into a shared `GitHelpers` mixin. One long-lived container per run, repo copied in with `docker cp`, one `docker exec` per command (keeps "no state between commands"). Timeout = in-container `timeout -s KILL` (kills the process group) + a sweep by a per-exec env-var id (catches setsid escapes) + a host backstop. Podman uses the same CLI subset. Rejected: bind-mounting the repo read-write (could damage the original); a container per command (slow, and repo state would need copying back and forth); the Docker SDK (non-stdlib).

**Reviewer pass probability.** Second reviewer line `PASS_PROBABILITY: p`, parsed separately from the verdict; the first line alone decides, as before. Logged in `RunResult.reviews`, trajectory `reviews`, and `usage["big"].pass_probabilities` (the user asked for `usage`). Rejected: putting it in the verdict line (would risk breaking the fail-open parse).

**Task template.** `task_template.py`, `<<NAME>>` placeholders (same syntax as prompts.py), single-pass substitution so values cannot inject placeholders, unknown placeholders rejected at load, empty fields rendered as "(not provided)". The default frames the router verdict as a hypothesis to verify first. Rejected: str.format/Jinja (braces in code/logs; dependency).

**Custom commands.** `[[commands]]` config; shims on PATH (host temp dir or in-container dir); system-prompt section that is empty when no commands exist (prompt byte-identical otherwise); per-command output limit via a new `Agent.observation_limit()` hook that defaults to `obs_max_chars`. Rejected: intercepting commands in the agent like `ask_explorer` (would bypass the env and its timeout).

**Not changed:** loop, detector, compaction, caps, review-cap logic. Only hooks (`observation_limit`, `review_log`) and logging were added; defaults reproduce old behaviour.
