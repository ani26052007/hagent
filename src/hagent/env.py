"""Local bash environment. Every command is an independent process (no persistent shell)."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .textutil import truncate_middle

# Same convention as mini-swe-agent: if the *first line* of a successful command's output is
# this marker, the rest of the output is the final submission.
SUBMIT_MARKER = "HAGENT_SUBMIT_FINAL_OUTPUT"

_ENV_DEFAULTS = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "TERM": "dumb",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_PROGRESS_BAR": "off",
    "PYTHONDONTWRITEBYTECODE": "1",  # keep __pycache__ out of the diff
}


@dataclass
class Obs:
    output: str
    returncode: int
    timed_out: bool = False
    submission: str | None = None  # set when the command output started with SUBMIT_MARKER


class LocalEnv:
    """Runs commands with bash in `cwd`. POSIX only (process groups are used for timeouts)."""

    def __init__(self, cwd: str | Path, timeout_s: int = 60, extra_env: dict[str, str] | None = None,
                 max_output_chars: int = 200_000):
        self.cwd = str(Path(cwd).resolve())
        self.timeout_s = timeout_s
        self.extra_env = extra_env or {}
        self.max_output_chars = max_output_chars

    # ----- command execution -----
    def execute(self, command: str, timeout_s: int | None = None) -> Obs:
        timeout = timeout_s or self.timeout_s
        env = os.environ | _ENV_DEFAULTS | self.extra_env
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash" if os.path.exists("/bin/bash") else None,
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,  # a command waiting for input fails fast instead of hanging
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)  # kill children too, so nothing is orphaned
            out, _ = proc.communicate()
        out = truncate_middle(out or "", self.max_output_chars)
        returncode = -1 if timed_out else proc.returncode
        return Obs(out, returncode, timed_out, self._submission(out, returncode))

    @staticmethod
    def _submission(output: str, returncode: int) -> str | None:
        lines = output.lstrip().splitlines(keepends=True)
        if returncode == 0 and lines and lines[0].strip() == SUBMIT_MARKER:
            return "".join(lines[1:])
        return None

    # ----- git helpers (used by the loop detector, reviewer and final result) -----
    def _git(self, *args: str, timeout: int = 30) -> tuple[int, str]:
        try:
            p = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, text=True,
                               errors="replace", timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return 1, ""
        return p.returncode, p.stdout

    def is_git_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree")[0] == 0

    def git_diff(self) -> str:
        """Everything changed relative to HEAD, including new files (via intent-to-add)."""
        if not self.is_git_repo():
            return ""
        self._git("add", "-A", "-N")
        rc, out = self._git("diff", "HEAD")
        if rc != 0:  # repository without commits
            rc, out = self._git("diff")
        return out if rc == 0 else ""

    def git_diff_hash(self) -> str:
        return hashlib.sha1(self.git_diff().encode("utf-8", "replace")).hexdigest()[:12]

    def git_tracked_diff_hash(self) -> str:
        """Hash of edits to files git already tracks (modified/deleted only).

        New untracked files, such as a throwaway repro script, do not change it. The loop detector
        uses this for oscillation, so creating and deleting a scratch file is not mistaken for an
        edit being undone."""
        if not self.is_git_repo():
            return ""
        flags = ["--no-renames", "--diff-filter=MDT"]
        rc, out = self._git("diff", "HEAD", *flags)
        if rc != 0:  # repository without commits
            rc, out = self._git("diff", *flags)
        return hashlib.sha1(out.encode("utf-8", "replace")).hexdigest()[:12] if rc == 0 else ""
