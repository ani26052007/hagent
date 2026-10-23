"""Single-call helpers: no loops, one model call each, always returning a Reply for cost accounting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Model, Reply
from .prompts import ADVISOR_SYSTEM, COMPACTOR_SYSTEM, REVIEWER_SYSTEM
from .textutil import truncate_middle


@dataclass
class Verdict:
    """Reviewer outcome. `ok` alone decides accept/reject. `pass_probability` is the reviewer's
    own estimate (0-1) that the patch passes simulator validation; it is logged for later
    calibration and must never gate anything. None when absent or unparseable."""

    ok: bool
    feedback: str = ""
    pass_probability: float | None = None


_PASS_PROB = re.compile(r"^\s*pass[ _-]?probability\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)\s*$", re.IGNORECASE)


def parse_pass_probability(text: str) -> float | None:
    """First `PASS_PROBABILITY: <x>` line (x in 0-1, or a percentage like 70%), clamped to [0, 1]."""
    for line in (text or "").splitlines():
        m = _PASS_PROB.match(line)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            return None
        if m.group(2) == "%" or value > 1.0:
            value /= 100.0
        return min(1.0, max(0.0, value))
    return None


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
    reviewer can never trap the agent in a review loop. The pass-probability line is parsed
    separately, removed from the feedback, and never changes `ok`."""
    prob = parse_pass_probability(text)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].strip().upper().startswith("REJECT"):
        first = lines[0].strip()[len("REJECT"):].lstrip(" :-")
        rest = [ln for ln in lines[1:] if not _PASS_PROB.match(ln)]
        feedback = "\n".join(([first] if first else []) + rest).strip()
        return Verdict(False, feedback or "The reviewer rejected the patch without details.", prob)
    return Verdict(True, "", prob)


def summarize(model: Model, transcript: str, previous_summary: str) -> Reply:
    user = ""
    if previous_summary.strip():
        user += f"## Previous notes\n{previous_summary.strip()}\n\n"
    user += f"## History to compress\n{transcript}"
    return _chat(model, COMPACTOR_SYSTEM, user)
