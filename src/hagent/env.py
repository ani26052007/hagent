"""Local bash environment. Every command is an independent process (no persistent shell)."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from .config import CustomCommand

from .exceptions import EnvError
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


def clean_environ(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """`base` (default: os.environ) without any GIT_* variable. An inherited GIT_DIR, GIT_WORK_TREE,
    GIT_INDEX_FILE or GIT_CONFIG_* would point git (the helpers and the agent's own commands) at
    another repository, or change what git prints."""
    return {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith("GIT_")}


@dataclass
class Obs:
    output: str
    returncode: int
    timed_out: bool = False
    submission: str | None = None  # set when the command output started with SUBMIT_MARKER


class Env(Protocol):
    """What the agent needs from an execution environment.

    Contract:
      execute(command, timeout_s)  run `command` with bash in `cwd` as an independent process
                                   (no state survives between calls), stdin closed, stdout+stderr
                                   merged and truncated to the env's max_output_chars; on timeout
                                   the whole process tree is killed and Obs(timed_out=True,
                                   returncode=-1) is returned. Never raises for command failures.
      cwd                          the repo path as seen by the commands.
      git_diff()                   everything changed since the commit the env started at (HEAD when it
                                   was created), including new files and the agent's own commits ("" if
                                   no repo).
      git_diff_hash()              short hash of git_diff().
      git_tracked_diff_hash()      short hash of edits to tracked files only.
      install_commands(cmds)       make each custom command callable by name; raises EnvError if a
                                   command's path is not executable in this env.
      close()                      release resources; idempotent.
    """

    cwd: str

    def execute(self, command: str, timeout_s: int | None = None) -> Obs: ...
    def git_diff(self) -> str: ...
    def git_diff_hash(self) -> str: ...
    def git_tracked_diff_hash(self) -> str: ...
    def install_commands(self, commands: Sequence[CustomCommand]) -> None: ...
    def close(self) -> None: ...


def submission_of(output: str, returncode: int) -> str | None:
    """The submission text if `output` starts with SUBMIT_MARKER and the command succeeded."""
    lines = output.lstrip().splitlines(keepends=True)
    if returncode == 0 and lines and lines[0].strip() == SUBMIT_MARKER:
        return "".join(lines[1:])
    return None


def shim_script(path: str) -> str:
    """A tiny bash wrapper that forwards all arguments to `path`."""
    return f"#!/bin/bash\nexec {shlex.quote(path)} \"$@\"\n"


class GitHelpers:
    """git-based helpers shared by all envs. Subclasses implement `_git(*args, timeout)`
    returning (returncode, stdout) and must not raise.

    The patch is the change since the commit the run started from (``record_start``, called when
    the env is created), not since the current HEAD: an agent that runs ``git commit`` keeps its
    change in the patch (vla-router ISSUES #56)."""

    _start_commit: str | None = None  # commit id recorded at creation; None: no commits yet

    def record_start(self) -> None:
        """Remember the commit the working copy is at now. Called once by the env's constructor."""
        rc, out = self._git("rev-parse", "--verify", "-q", "HEAD^{commit}")
        self._start_commit = out.strip() or None if rc == 0 else None

    def _git(self, *args: str, timeout: int = 30) -> tuple[int, str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def is_git_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree")[0] == 0

    def git_diff(self) -> str:
        """Everything changed since the starting commit, including new files (via intent-to-add)
        and commits the agent made."""
        if not self.is_git_repo():
            return ""
        self._git("add", "-A", "-N")
        rc, out = self._git("diff", self._start_commit or "HEAD")
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
        rc, out = self._git("diff", self._start_commit or "HEAD", *flags)
        if rc != 0:  # repository without commits
            rc, out = self._git("diff", *flags)
        return hashlib.sha1(out.encode("utf-8", "replace")).hexdigest()[:12] if rc == 0 else ""


class LocalEnv(GitHelpers):
    """Runs commands with bash in `cwd` on the host. POSIX only (process groups are used for timeouts)."""

    def __init__(self, cwd: str | Path, timeout_s: int = 60, extra_env: dict[str, str] | None = None,
                 max_output_chars: int = 200_000):
        self.cwd = str(Path(cwd).resolve())
        self.timeout_s = timeout_s
        self.extra_env = extra_env or {}
        self.max_output_chars = max_output_chars
        self._shim_dir: str | None = None
        self.record_start()

    # ----- command execution -----
    def execute(self, command: str, timeout_s: int | None = None) -> Obs:
        timeout = timeout_s or self.timeout_s
        env = clean_environ() | _ENV_DEFAULTS | self.extra_env
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
        return submission_of(output, returncode)

    # ----- custom commands / lifecycle -----
    def install_commands(self, commands: Sequence[CustomCommand]) -> None:
        """Put a shim per command in a private temp dir and prepend it to PATH."""
        if not commands:
            return
        for c in commands:
            if not (os.path.isfile(c.path) and os.access(c.path, os.X_OK)):
                raise EnvError(f"custom command {c.name!r}: {c.path} is not an executable file")
        if self._shim_dir is None:
            self._shim_dir = tempfile.mkdtemp(prefix="hagent-cmds-")
        for c in commands:
            shim = Path(self._shim_dir) / c.name
            shim.write_text(shim_script(c.path))
            shim.chmod(0o755)
        base_path = self.extra_env.get("PATH") or os.environ.get("PATH", "")
        if not base_path.startswith(self._shim_dir + os.pathsep):
            self.extra_env["PATH"] = self._shim_dir + os.pathsep + base_path

    def close(self) -> None:
        if self._shim_dir:
            shutil.rmtree(self._shim_dir, ignore_errors=True)
            self._shim_dir = None

    # ----- git helpers (used by the loop detector, reviewer and final result) -----
    def _git(self, *args: str, timeout: int = 30) -> tuple[int, str]:
        try:
            p = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, text=True,
                               errors="replace", timeout=timeout, env=clean_environ())
        except (OSError, subprocess.TimeoutExpired):
            return 1, ""
        return p.returncode, p.stdout
