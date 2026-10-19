"""Loop / no-progress detection.

The key idea: a command is only "repeated" if it ran against the *same repo state*
(same git diff hash). Re-running `pytest` after an edit is normal; re-running it with
nothing changed is a loop. Four signals, checked in this order:

  repeated_command  same command + same repo state, `repeat_threshold` times in the window
  repeated_output   same output + same repo state, `repeat_threshold` times in the window
  oscillation       the repo returned to an earlier state (edit A -> B -> A)
  stalled           no new diff state and no new failure signature for `stall_steps` steps
                    (only counted after `stall_warmup` steps, so exploration is not punished)
"""

from __future__ import annotations

from collections import deque

from .textutil import failure_signature, normalize_cmd, short_hash


class LoopDetector:
    def __init__(self, window: int = 8, repeat_threshold: int = 3, stall_warmup: int = 10, stall_steps: int = 12):
        self.window: deque[tuple[tuple[str, str], tuple[str, str]]] = deque(maxlen=window)
        self.repeat_threshold = repeat_threshold
        self.stall_warmup = stall_warmup
        self.stall_steps = stall_steps
        self.step = 0
        self.last_progress_step = 0
        self.seen_diffs: set[str] = set()
        self.seen_tracked: set[str] = set()  # states of tracked files only (see git_tracked_diff_hash)
        self.last_tracked: str | None = None
        self.seen_sigs: set[str] = set()
        self.last_diff: str | None = None
        self.progress = False  # True if the latest observation was genuinely new

    def observe(self, command: str, output: str, diff_hash: str, tracked_hash: str | None = None) -> str | None:
        """Record one executed command. Returns a human-readable flag reason, or None.

        diff_hash is the full repo state (new files included) and drives repeat/progress detection.
        tracked_hash covers edits to already-tracked files only and drives oscillation detection;
        when omitted it falls back to diff_hash."""
        self.step += 1
        cmd_key = (normalize_cmd(command), diff_hash)
        out_key = (short_hash(output), diff_hash)
        self.window.append((cmd_key, out_key))

        sig = failure_signature(output)
        novel_diff = diff_hash not in self.seen_diffs
        tracked = diff_hash if tracked_hash is None else tracked_hash
        revisit = tracked in self.seen_tracked and tracked != self.last_tracked
        novel_sig = sig is not None and sig not in self.seen_sigs

        self.progress = novel_diff or novel_sig
        if self.progress:
            self.last_progress_step = self.step
        self.seen_diffs.add(diff_hash)
        if sig is not None:
            self.seen_sigs.add(sig)
        self.last_diff = diff_hash
        self.seen_tracked.add(tracked)
        self.last_tracked = tracked

        n_cmd = sum(1 for c, _ in self.window if c == cmd_key)
        if n_cmd >= self.repeat_threshold:
            return f"repeated_command: you ran the same command {n_cmd} times without changing any files"
        n_out = sum(1 for _, o in self.window if o == out_key)
        if n_out >= self.repeat_threshold:
            return f"repeated_output: {n_out} commands produced identical output without any file changes"
        if revisit:
            return "oscillation: the repository is back in a state it was in before (an edit was undone or redone)"
        if self.step > self.stall_warmup and self.step - self.last_progress_step >= self.stall_steps:
            return (f"stalled: no new file changes and no new error output for "
                    f"{self.step - self.last_progress_step} steps")
        return None

    def acknowledge(self) -> None:
        """Call after reacting to a flag, so the same evidence does not immediately re-trigger it."""
        self.window.clear()
        self.last_progress_step = self.step
