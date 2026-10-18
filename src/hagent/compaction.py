"""History compaction.

When the message list grows past a character budget, the middle of the history is replaced by
a short summary. The system prompt, the task, and the most recent messages stay verbatim.

Shape is preserved so any chat API accepts it:  system, user(task + summary), assistant, user, ...
The summary lives inside the task message (so there are never two consecutive user messages),
and the kept tail always starts with an assistant message.
"""

from __future__ import annotations

from typing import Callable

from .textutil import extract_bash_blocks, truncate_middle

SUMMARY_MARK = "\n\n## Summary of earlier work (auto-compacted; the full history was dropped)\n"
_PER_MESSAGE_CHARS = 1500
_TRANSCRIPT_CHARS = 40_000


def total_chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


def split_summary(task_message_content: str) -> tuple[str, str]:
    """Split the task message into (task text, previous summary or '')."""
    if SUMMARY_MARK in task_message_content:
        task, _, summary = task_message_content.partition(SUMMARY_MARK)
        return task, summary
    return task_message_content, ""


def render_transcript(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        label = "AGENT" if m["role"] == "assistant" else "OUTPUT"
        parts.append(f"[{label}]\n{truncate_middle(m['content'], _PER_MESSAGE_CHARS)}")
    return truncate_middle("\n\n".join(parts), _TRANSCRIPT_CHARS)


def fallback_summary(middle: list[dict], previous: str) -> str:
    """Deterministic summary used when the summarizer model fails: just the commands that were run."""
    cmds = []
    for m in middle:
        if m["role"] == "assistant":
            blocks, _ = extract_bash_blocks(m["content"])
            if blocks:
                cmds.append(" ".join(blocks[0].split())[:100])
    lines = [previous.strip()] if previous.strip() else []
    lines.append("(Automatic summary unavailable. Commands run in the dropped part of the history:)")
    lines += [f"- {c}" for c in cmds[-40:]]
    return "\n".join(lines)


def maybe_compact(
    messages: list[dict],
    summarize: Callable[[str, str], str | None],
    ctx_char_limit: int,
    keep_last: int,
) -> bool:
    """Compact `messages` in place if over budget. Returns True if it compacted."""
    if ctx_char_limit <= 0 or total_chars(messages) <= ctx_char_limit:
        return False
    if len(messages) <= 2 + keep_last:
        return False

    start = len(messages) - keep_last
    while start < len(messages) and messages[start]["role"] != "assistant":
        start += 1  # the tail must begin with an assistant message
    if start >= len(messages) - 1:
        return False
    middle = messages[2:start]
    if not middle:
        return False

    task_text, previous = split_summary(messages[1]["content"])
    summary = None
    try:
        summary = summarize(render_transcript(middle), previous)
    except Exception:  # a failing summarizer must never kill the run
        summary = None
    if not summary or not summary.strip():
        summary = fallback_summary(middle, previous)

    new_task = {"role": "user", "content": task_text + SUMMARY_MARK + summary.strip()}
    messages[:] = [messages[0], new_task] + messages[start:]
    return True
