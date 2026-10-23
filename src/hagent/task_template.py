"""Task prompt templates for repair incidents (plan.md 3.3).

A template is a text file with <<NAME>> placeholders (same syntax as prompts.py). The rendered
text is the `task` passed to `solve()`.

Placeholders (all optional except TASK):
  <<TASK>>               what failed, in plain words (required)
  <<ROUTER_VERDICT>>     the router's verdict. Templates must frame it as a hypothesis to verify
                         first, not as a fact: the router can be wrong (plan.md 3.2, 9)
  <<LOCATION_EVIDENCE>>  where the fault probably is (files, modules, replay that flipped)
  <<REPRO_COMMAND>>      a command that reproduces the failure
  <<ROLLOUT_SUMMARY>>    failing-rollout summary (success rate, action-trace stats)
  <<REPO_PATH>>          repo path of the controller/wrapper code

Contract:
  - load_task_template(path) raises ValueError if the file uses an unknown placeholder or lacks
    <<TASK>>.
  - TaskTemplate.render(ctx) substitutes in ONE pass, so placeholder-like text inside a value
    (e.g. a log line containing "<<TASK>>") is never re-substituted. Empty values render as
    `missing_text`.
  - Nothing here may carry bug-injection ground truth; callers decide what goes in the context
    (no-leakage rule, plan.md 3.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

_PLACEHOLDER = re.compile(r"<<([A-Z_]+)>>")


@dataclass
class IncidentContext:
    task: str
    router_verdict: str = ""
    location_evidence: str = ""
    repro_command: str = ""
    rollout_summary: str = ""
    repo_path: str = ""

    def values(self) -> dict[str, str]:
        return {f.name.upper(): str(getattr(self, f.name) or "") for f in fields(self)}


PLACEHOLDERS = frozenset(f.name.upper() for f in fields(IncidentContext))

DEFAULT_TEMPLATE = """<<TASK>>

## Router hypothesis (verify this first; it may be wrong)
The automatic router classified this failure as: <<ROUTER_VERDICT>>
Before editing anything, check this hypothesis against the code and the evidence below. If the
evidence contradicts it, say so in one sentence and follow the evidence instead.

## Where to look
<<LOCATION_EVIDENCE>>

## Reproduce
<<REPRO_COMMAND>>

## Failing rollout summary
<<ROLLOUT_SUMMARY>>

## Code location
Controller/wrapper code: <<REPO_PATH>>
"""


@dataclass(frozen=True)
class TaskTemplate:
    text: str
    missing_text: str = "(not provided)"

    def __post_init__(self) -> None:
        validate_template(self.text)

    def render(self, ctx: IncidentContext) -> str:
        values = ctx.values()
        if not values["TASK"].strip():
            raise ValueError("IncidentContext.task must not be empty")

        def sub(m: re.Match) -> str:
            v = values[m.group(1)].strip()
            return v if v else self.missing_text

        return _PLACEHOLDER.sub(sub, self.text).strip() + "\n"


def validate_template(text: str) -> None:
    used = set(_PLACEHOLDER.findall(text))
    unknown = used - PLACEHOLDERS
    if unknown:
        raise ValueError(f"unknown placeholder(s) {sorted(unknown)}; valid: {sorted(PLACEHOLDERS)}")
    if "TASK" not in used:
        raise ValueError("template must contain <<TASK>>")


def load_task_template(path: str | Path | None = None, missing_text: str = "(not provided)") -> TaskTemplate:
    """Load a template file; with path=None, return the built-in DEFAULT_TEMPLATE."""
    text = DEFAULT_TEMPLATE if path is None else Path(path).read_text()
    return TaskTemplate(text, missing_text)
