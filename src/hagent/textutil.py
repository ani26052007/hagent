"""Small text helpers: action parsing, truncation, normalization, failure signatures."""

from __future__ import annotations

import hashlib
import re

from .exceptions import FormatError


def extract_bash_blocks(text: str) -> tuple[list[str], bool]:
    """Return (bash blocks, has_unterminated_block).

    Line-based on purpose: a regex mis-pairs the closing fence of a non-bash block
    with the opening fence of the next one. Blocks tagged with another language
    (python, text, ...) are skipped; untagged, `bash` and `sh` blocks count.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    is_bash = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if current is None:
            if stripped.startswith("```"):
                lang = stripped[3:].strip().lower()
                is_bash = lang in ("", "bash", "sh")
                current = []
        elif stripped == "```":
            if is_bash:
                blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    return blocks, current is not None


def parse_action(text: str) -> str:
    """Return the single bash command in a model reply, or raise FormatError."""
    blocks, unterminated = extract_bash_blocks(text)
    if unterminated and not blocks:
        raise FormatError("found a code block that is never closed with ```")
    if len(blocks) != 1:
        raise FormatError(f"expected exactly ONE ```bash block, found {len(blocks)}")
    command = blocks[0].strip()
    if not command:
        raise FormatError("the bash block is empty")
    return command


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of long text, eliding the middle (errors tend to be at the end)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    elided = len(text) - head - tail
    return f"{text[:head]}\n[... {elided} characters elided ...]\n{text[-tail:] if tail else ''}"


def normalize_cmd(command: str) -> str:
    return " ".join(command.split())


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


_FAIL_LINE = re.compile(r"(FAILED|ERROR|Traceback|AssertionError|\bError:|\berror:|\bpanic:|\bFAIL\b)")
_NOISE = re.compile(r"0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b")


def failure_signature(output: str) -> str | None:
    """Stable fingerprint of the failures in a command output, or None if it has none.

    Durations and memory addresses are stripped so that re-running the same failing
    test gives the same signature.
    """
    lines = [ln.strip() for ln in output.splitlines() if _FAIL_LINE.search(ln)]
    if not lines:
        return None
    cleaned = [_NOISE.sub("", ln) for ln in lines[-5:]]
    return short_hash("\n".join(cleaned))
