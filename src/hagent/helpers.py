"""Single-call helpers: no loops, one model call each, always returning a Reply for cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Model, Reply
from .prompts import ADVISOR_SYSTEM, COMPACTOR_SYSTEM, REVIEWER_SYSTEM
from .textutil import truncate_middle


@dataclass
class Verdict:
    ok: bool
    feedback: str = ""


def _chat(model: Model, system: str, user: str) -> Reply:
    return model.query([{"role": "system", "content": system}, {"role": "user", "content": user}])


def build_digest(*, task: str, summary: str, recent: list[dict], diff: str, test_output: str, reason: str) -> str:
    """What the big model sees: a compact picture of the situation, never the raw transcript."""
    recent_txt = "\n\n".join(
        f"[{'AGENT' if m['role'] == 'assistant' else 'OUTPUT'}]\n{truncate_middle(m['content'], 1200)}" for m in recent
    )
    sections = [
        f"## Why you are being consulted\n{reason}",
        f"## Task\n{truncate_middle(task, 3000)}",
    ]
    if summary.strip():
        sections.append(f"## Agent's notes on earlier work\n{truncate_middle(summary, 3000)}")
    sections.append(f"## Most recent steps\n{recent_txt or '(none yet)'}")
    sections.append(f"## Current diff\n{truncate_middle(diff, 6000) if diff.strip() else '(no changes yet)'}")
    sections.append(f"## Last test output\n{truncate_middle(test_output, 3000) if test_output.strip() else '(no tests run yet)'}")
    return "\n\n".join(sections)


def advise(model: Model, digest: str) -> Reply:
    return _chat(model, ADVISOR_SYSTEM, digest)


def review(model: Model, task: str, diff: str, test_output: str) -> tuple[Verdict, Reply]:
    user = (
        f"## Task\n{truncate_middle(task, 3000)}\n\n"
        f"## Diff the agent wants to submit\n{truncate_middle(diff, 12000)}\n\n"
        f"## Last test output\n{truncate_middle(test_output, 3000) if test_output.strip() else '(no tests run)'}"
    )
    reply = _chat(model, REVIEWER_SYSTEM, user)
    return parse_verdict(reply.content), reply


def parse_verdict(text: str) -> Verdict:
    """First line decides. Anything unparseable is treated as APPROVE (fail open) so a flaky
    reviewer can never trap the agent in a review loop."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].strip().upper().startswith("REJECT"):
        first = lines[0].strip()[len("REJECT"):].lstrip(" :-")
        feedback = "\n".join(([first] if first else []) + lines[1:]).strip()
        return Verdict(False, feedback or "The reviewer rejected the patch without details.")
    return Verdict(True)


def summarize(model: Model, transcript: str, previous_summary: str) -> Reply:
    user = ""
    if previous_summary.strip():
        user += f"## Previous notes\n{previous_summary.strip()}\n\n"
    user += f"## History to compress\n{transcript}"
    return _chat(model, COMPACTOR_SYSTEM, user)
