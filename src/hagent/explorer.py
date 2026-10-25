"""Optional read-only explorer: a tiny isolated agent loop that answers one question.

Its whole value is context isolation: the explorer's noisy grep/cat steps live in its own
message list; only a short answer returns to the main agent.

Read-only is enforced twice, both best-effort (real isolation needs a read-only container):
  1. a command filter that blocks obvious writes before they run,
  2. a before/after git-diff comparison; if the repo changed anyway the caller is told.
"""

from __future__ import annotations

import dataclasses
import re

from .agent import Agent, Usage
from .config import Settings
from .env import SUBMIT_MARKER, Obs
from .models import Model
from .prompts import EXPLORER_FORMAT_ERROR, build_explorer_prompt
from .textutil import truncate_middle

_REDIRECT_OK = re.compile(r"\d?>&\d|&>\s*/dev/null|\d?>\s*/dev/null")
_WRITE_PATTERNS = re.compile(
    r"(?:^|[;&|(\s])(?:rm|mv|cp|touch|mkdir|rmdir|tee|truncate|chmod|chown|dd|patch|ln|pip3?|npm|yarn|apt|apt-get)\b"
    r"|\bsed\s+(?:-\w*i|--in-place)"
    r"|\bgit\s+(?:checkout|reset|clean|apply|commit|stash|add|rebase|merge|restore|push|rm|mv|cherry-pick)\b"
    r"|>"
)


def is_write_command(command: str) -> bool:
    stripped = command.lstrip()
    if stripped.startswith(f"echo {SUBMIT_MARKER}"):  # the answer-submission command (uses a heredoc)
        return False
    return bool(_WRITE_PATTERNS.search(_REDIRECT_OK.sub("", command)))


class ReadOnlyEnv:
    """Wraps an env and refuses commands that look like they write."""

    def __init__(self, inner):
        self.inner = inner
        self.cwd = inner.cwd

    def execute(self, command: str, timeout_s: int | None = None) -> Obs:
        if is_write_command(command):
            return Obs("Blocked: the explorer is read-only. Use grep/find/cat/sed -n/head/tail/git log/git show.", 1)
        return self.inner.execute(command, timeout_s)

    def git_diff(self) -> str:
        return self.inner.git_diff()

    def git_diff_hash(self) -> str:
        return self.inner.git_diff_hash()


def run_explorer(question: str, env, model: Model, settings: Settings) -> tuple[str, dict[str, Usage], bool]:
    """Returns (answer text, usage, repo_was_modified)."""
    lim = dataclasses.replace(
        settings.limits,
        max_steps=settings.limits.explorer_max_steps,
        max_cost=settings.limits.explorer_max_cost,
        max_wall_s=min(settings.limits.max_wall_s, 300) if settings.limits.max_wall_s else 300,
        plan_at_step=0,
        final_review=False,
    )
    sub_settings = dataclasses.replace(settings, limits=lim)
    before = env.git_diff_hash()
    agent = Agent(model, ReadOnlyEnv(env), sub_settings, role="explorer", system_prompt=build_explorer_prompt(env.cwd))
    agent.format_error_template = EXPLORER_FORMAT_ERROR
    result = agent.run(question)
    modified = env.git_diff_hash() != before

    if result.submitted and result.detail.strip():
        answer = truncate_middle(result.detail.strip(), 2500)
    else:
        answer = (f"(The explorer did not finish: {result.exit_status}. "
                  f"Narrow the question or search the code yourself.)")
    if modified:
        answer += "\n[WARNING: the repository changed while the explorer ran; check `git diff` before continuing.]"
    return answer, result.usage, modified
