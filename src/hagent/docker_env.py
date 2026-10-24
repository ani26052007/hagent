"""Docker/Podman execution environment. Same contract as `env.Env` (see env.py), stdlib only.

Isolation model
  - The repo is COPIED into a fresh container (`docker cp`), never bind-mounted, so the agent
    cannot modify the original checkout. The final patch is read with `git diff` inside the
    container; nothing is copied back.
  - Network, CPU, memory and process-count limits come from `EnvConfig` (defaults: no network,
    2 CPUs, 4 GB, 512 pids, no-new-privileges).
  - One long-lived container per env (`sleep infinity`); every command is a separate
    `docker exec`, so no shell state survives between commands, as with LocalEnv.
  - The container only gets the variables passed with `-e`. Host-side runtime calls run with
    `clean_environ()` (no inherited GIT_* variables), like LocalEnv.

Timeouts kill the whole process tree, in three layers:
  1. In the container, the command runs under `timeout -s KILL <t>`. Without --foreground,
     GNU timeout puts itself in a new process group and kills the whole group.
  2. Every exec carries a unique HAGENT_EXEC_ID environment variable. After a timeout, a sweep
     kills every process in the container whose environment holds that id, which also catches
     children that escaped the process group (setsid, nohup into a new session).
  3. A host-side backstop (`t + kill_grace_s`) kills the `docker exec` client if the container
     does not answer, then runs the same sweep.

Image requirements: bash, git, and coreutils `timeout` and `sleep`. Checked at start-up.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Sequence

from .config import CustomCommand, EnvConfig
from .env import _ENV_DEFAULTS, GitHelpers, Obs, clean_environ, shim_script, submission_of
from .exceptions import EnvError
from .textutil import truncate_middle

_REQUIRED_TOOLS = ("bash", "git", "timeout", "sleep")
# Lets git work on a repo copied in as a different owner than the user running commands.
_GIT_SAFE_ENV = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*"}
_KILLED_RC = 137  # 128 + SIGKILL


def runtime_available(runtime: str = "docker", timeout_s: int = 15) -> bool:
    """True if the runtime CLI exists and its daemon/service answers."""
    if not shutil.which(runtime):
        return False
    try:
        return subprocess.run([runtime, "info"], capture_output=True, timeout=timeout_s).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerEnv(GitHelpers):
    """Runs commands in a container holding a copy of `repo`. Call close() (or use `with`) to
    remove the container. Raises EnvError if the container cannot be prepared."""

    def __init__(self, repo: str | Path, cfg: EnvConfig, timeout_s: int = 60,
                 extra_env: dict[str, str] | None = None, max_output_chars: int = 200_000):
        if not cfg.image:
            raise EnvError("[env] image is required for the docker backend")
        repo_path = Path(repo).resolve()
        if not repo_path.is_dir():
            raise EnvError(f"{repo_path} is not a directory")
        self.cfg = cfg
        self.rt = cfg.runtime
        self.cwd = cfg.workdir
        self.timeout_s = timeout_s
        self.extra_env = dict(extra_env or {})
        self.max_output_chars = max_output_chars
        self.container: str | None = None
        self._commands_installed = False
        if not shutil.which(self.rt):
            raise EnvError(f"container runtime {self.rt!r} not found on PATH")
        self._start(repo_path)
        self.record_start()

    # ------------------------------------------------------------------ lifecycle
    def _run_cli(self, *args: str, timeout: int | None = None, check: bool = True,
                 input_text: str | None = None) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run([self.rt, *args], capture_output=True, text=True, errors="replace",
                               timeout=timeout or self.cfg.startup_timeout_s, input=input_text, env=clean_environ())
        except (OSError, subprocess.TimeoutExpired) as e:
            raise EnvError(f"{self.rt} {args[0]} failed: {e!r}") from e
        if check and p.returncode != 0:
            raise EnvError(f"{self.rt} {args[0]} failed ({p.returncode}): {p.stderr.strip()[:500]}")
        return p

    def _start(self, repo: Path) -> None:
        cfg = self.cfg
        name = f"hagent-{uuid.uuid4().hex[:12]}"
        run = ["run", "-d", "--name", name, "--label", "hagent=1", "--network", cfg.network]
        if cfg.cpus > 0:
            run += ["--cpus", str(cfg.cpus)]
        if cfg.memory:
            run += ["--memory", cfg.memory]
        if cfg.pids_limit > 0:
            run += ["--pids-limit", str(cfg.pids_limit)]
        run += [*cfg.extra_run_args, "--entrypoint", "sleep", cfg.image, "infinity"]
        self.container = self._run_cli(*run).stdout.strip() or name
        try:
            check = " && ".join(f"command -v {t} >/dev/null" for t in _REQUIRED_TOOLS)
            p = self._run_cli("exec", self.container, "bash", "-c", check, check=False)
            if p.returncode != 0:
                raise EnvError(f"image {cfg.image!r} must provide: {', '.join(_REQUIRED_TOOLS)}")
            self._run_cli("exec", "-u", "0", self.container, "mkdir", "-p", cfg.workdir)
            self._run_cli("cp", f"{repo}/.", f"{self.container}:{cfg.workdir}")
            if cfg.user:
                self._run_cli("exec", "-u", "0", self.container, "chown", "-R", cfg.user, cfg.workdir)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Remove the container. Idempotent; never raises."""
        cid, self.container = self.container, None
        if cid:
            try:
                subprocess.run([self.rt, "rm", "-f", cid], capture_output=True, timeout=60, env=clean_environ())
            except (OSError, subprocess.TimeoutExpired):
                pass

    def __enter__(self) -> "DockerEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ execution
    def _exec_argv(self, script: str, env: dict[str, str], user: str | None = None) -> list[str]:
        argv = [self.rt, "exec", "-w", self.cwd]
        u = self.cfg.user if user is None else user
        if u:
            argv += ["-u", u]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        return [*argv, self._require_container(), "bash", "-c", script]

    def _require_container(self) -> str:
        if not self.container:
            raise EnvError("the container has been closed")
        return self.container

    def execute(self, command: str, timeout_s: int | None = None) -> Obs:
        timeout = int(timeout_s or self.timeout_s)
        exec_id = uuid.uuid4().hex
        env = {**_ENV_DEFAULTS, **_GIT_SAFE_ENV, **self.extra_env, "HAGENT_EXEC_ID": exec_id}
        prefix = f"export PATH={shlex.quote(self.cfg.commands_dir)}:$PATH; " if self._commands_installed else ""
        script = f"{prefix}exec timeout -s KILL {timeout} bash -c {shlex.quote(command)}"
        t0 = time.monotonic()
        proc = subprocess.Popen(self._exec_argv(script, env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                start_new_session=True, env=clean_environ())
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=timeout + self.cfg.kill_grace_s)
        except subprocess.TimeoutExpired:  # container did not answer: kill the client, then sweep
            timed_out = True
            proc.kill()
            out, _ = proc.communicate()
        if not timed_out and proc.returncode == _KILLED_RC and time.monotonic() - t0 >= timeout - 0.5:
            timed_out = True
        if timed_out:
            self._sweep(exec_id)
        out = truncate_middle(out or "", self.max_output_chars)
        returncode = -1 if timed_out else proc.returncode
        return Obs(out, returncode, timed_out, submission_of(out, returncode))

    def _sweep(self, exec_id: str) -> None:
        """SIGKILL every process in the container started by exec `exec_id` (pure bash)."""
        script = (
            "for d in /proc/[0-9]*; do "
            "while IFS= read -r -d '' e; do "
            f"if [[ $e == HAGENT_EXEC_ID={exec_id} ]]; then kill -9 \"${{d#/proc/}}\" 2>/dev/null; break; fi; "
            "done < \"$d/environ\" 2>/dev/null; done; true"
        )
        try:
            subprocess.run(self._exec_argv(script, {}, user="0"), capture_output=True, timeout=30,
                           env=clean_environ())
        except (OSError, subprocess.TimeoutExpired, EnvError):
            pass

    # ------------------------------------------------------------------ custom commands
    def install_commands(self, commands: Sequence[CustomCommand]) -> None:
        """Write a shim per command into cfg.commands_dir inside the container; `path` must be an
        executable inside the container."""
        if not commands:
            return
        cid = self._require_container()
        lines = [f"mkdir -p {shlex.quote(self.cfg.commands_dir)}"]
        for c in commands:
            shim = f"{self.cfg.commands_dir}/{c.name}"
            lines.append(f"test -x {shlex.quote(c.path)} || {{ echo 'missing: {c.name}'; exit 3; }}")
            lines.append(f"printf %s {shlex.quote(shim_script(c.path))} > {shlex.quote(shim)}")
            lines.append(f"chmod 755 {shlex.quote(shim)}")
        p = self._run_cli("exec", "-u", "0", cid, "bash", "-c", " && ".join(lines), check=False)
        if p.returncode != 0:
            raise EnvError(f"custom commands could not be installed: {(p.stdout + p.stderr).strip()[:300]} "
                           f"(paths must be executable inside the container)")
        self._commands_installed = True

    # ------------------------------------------------------------------ git
    def _git(self, *args: str, timeout: int = 30) -> tuple[int, str]:
        if not self.container:
            return 1, ""
        script = "exec git " + " ".join(shlex.quote(a) for a in args)
        try:
            p = subprocess.run(self._exec_argv(script, dict(_GIT_SAFE_ENV)), capture_output=True, text=True,
                               errors="replace", timeout=timeout, stdin=subprocess.DEVNULL, env=clean_environ())
        except (OSError, subprocess.TimeoutExpired):
            return 1, ""
        return p.returncode, p.stdout
