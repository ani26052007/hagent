"""The same env contract tests for LocalEnv and DockerEnv, plus container-only isolation tests.

Docker tests need a running daemon and an image with bash, git, coreutils (and python3 for the
network test). Set HAGENT_TEST_IMAGE (default "hagent-sandbox:test"); they skip cleanly otherwise.
"""

import os
import subprocess
import time

import pytest

from hagent.config import CustomCommand, EnvConfig
from hagent.docker_env import DockerEnv, runtime_available
from hagent.env import SUBMIT_MARKER, LocalEnv
from hagent.exceptions import EnvError

IMAGE = os.environ.get("HAGENT_TEST_IMAGE", "hagent-sandbox:test")
RUNTIME = os.environ.get("HAGENT_TEST_RUNTIME", "docker")


def _image_present() -> bool:
    try:
        return subprocess.run([RUNTIME, "image", "inspect", IMAGE], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


DOCKER_OK = runtime_available(RUNTIME) and _image_present()
needs_docker = pytest.mark.skipif(not DOCKER_OK, reason=f"{RUNTIME} daemon or image {IMAGE!r} not available")


def docker_cfg(**kw) -> EnvConfig:
    base = dict(backend="docker", runtime=RUNTIME, image=IMAGE, cpus=1.0, memory="512m", pids_limit=256, kill_grace_s=5)
    base.update(kw)
    return EnvConfig(**base)


@pytest.fixture(params=["local", pytest.param("docker", marks=needs_docker)])
def make_env(request, repo):
    """Factory: make_env(timeout_s=...) -> env of the parametrized backend; closed after the test."""
    made = []

    def make(timeout_s=60):
        if request.param == "local":
            env = LocalEnv(repo, timeout_s=timeout_s)
        else:
            env = DockerEnv(repo, docker_cfg(), timeout_s=timeout_s)
        made.append(env)
        return env

    yield make
    for env in made:
        env.close()


# ================================================================ shared contract
def test_runs_commands_and_returns_code(make_env):
    env = make_env()
    assert env.execute("echo hi").output.strip() == "hi"
    assert env.execute("exit 3").returncode == 3


def test_is_stateless_between_commands(make_env):
    env = make_env()
    env.execute("cd /tmp && export FOO=1")
    assert env.execute("pwd; echo ${FOO:-unset}").output.split() == [env.cwd, "unset"]


def test_timeout_kills_process_group(make_env):
    env = make_env(timeout_s=1)
    t0 = time.time()
    obs = env.execute("sleep 30 & sleep 30")
    assert obs.timed_out and obs.returncode == -1
    assert time.time() - t0 < 8


def test_stdin_is_closed(make_env):
    assert make_env(timeout_s=5).execute("cat").returncode == 0


def test_submission_detection(make_env):
    env = make_env()
    assert env.execute(f"echo {SUBMIT_MARKER}; echo patch-body").submission == "patch-body\n"
    assert env.execute(f"echo not-first; echo {SUBMIT_MARKER}").submission is None
    assert env.execute(f"echo {SUBMIT_MARKER}; exit 1").submission is None


def test_git_diff_includes_new_files_and_hash_changes(make_env):
    env = make_env()
    assert env.git_diff() == ""
    h0 = env.git_diff_hash()
    env.execute("echo x > new.txt && sed -i 's/a - b/a + b/' app.py")
    diff = env.git_diff()
    assert "new.txt" in diff and "+    return a + b" in diff
    assert env.git_diff_hash() != h0


def test_tracked_hash_ignores_new_files_but_sees_edits(make_env):
    env = make_env()
    h0 = env.git_tracked_diff_hash()
    env.execute("echo 'print(1)' > repro.py")
    env.git_diff()
    assert env.git_tracked_diff_hash() == h0
    env.execute("sed -i 's/a - b/a + b/' app.py")
    assert env.git_tracked_diff_hash() != h0
    env.execute("git checkout -- app.py")
    assert env.git_tracked_diff_hash() == h0


def test_output_is_truncated_to_env_limit(make_env):
    env = make_env()
    env.max_output_chars = 500
    out = env.execute("for ((i=0;i<2000;i++)); do echo line$i; done").output
    assert "elided" in out and len(out) < 600


def test_install_commands_puts_them_on_path(make_env, tmp_path_factory):
    env = make_env()
    if isinstance(env, LocalEnv):
        tool = tmp_path_factory.mktemp("tools") / "tool.sh"
        tool.write_text('#!/bin/bash\necho "rung1 args: $*"\n')
        tool.chmod(0o755)
        path = str(tool)
    else:  # the path must exist inside the container
        path = "/usr/local/bin/tool.sh"
        subprocess.run([RUNTIME, "exec", "-u", "0", env.container, "bash", "-c",
                        f"mkdir -p /usr/local/bin && printf '#!/bin/bash\\necho \"rung1 args: $*\"\\n' > {path} "
                        f"&& chmod 755 {path}"], check=True, capture_output=True)
    env.install_commands([CustomCommand("run_rung1", path, "run validation")])
    assert env.execute("run_rung1 --seeds 10").output.strip() == "rung1 args: --seeds 10"
    assert env.execute("echo ok").output.strip() == "ok"  # normal commands unaffected


def test_install_commands_rejects_missing_path(make_env):
    env = make_env()
    with pytest.raises(EnvError):
        env.install_commands([CustomCommand("run_rung1", "/nonexistent/run_rung1", "x")])


def test_close_is_idempotent(make_env):
    env = make_env()
    env.close()
    env.close()


# ================================================================ container-only guarantees
@needs_docker
def test_docker_original_repo_is_never_modified(repo):
    before = (repo / "app.py").read_text()
    with DockerEnv(repo, docker_cfg()) as env:
        env.execute("sed -i 's/a - b/a + b/' app.py && echo new > extra.txt && rm -rf .git/hooks")
        assert "+    return a + b" in env.git_diff()
    assert (repo / "app.py").read_text() == before
    assert not (repo / "extra.txt").exists()
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout == ""


@needs_docker
def test_docker_has_no_network(repo):
    with DockerEnv(repo, docker_cfg()) as env:
        out = env.execute("cat /sys/class/net/*/operstate 2>/dev/null; ls /sys/class/net").output.split()
        assert set(out) <= {"lo", "unknown", "down"}  # loopback only


@needs_docker
def test_docker_resource_limits_are_applied(repo):
    with DockerEnv(repo, docker_cfg(cpus=1.5, memory="300m", pids_limit=100)) as env:
        fmt = "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}} {{.HostConfig.PidsLimit}} {{.HostConfig.NetworkMode}}"
        info = subprocess.run([RUNTIME, "inspect", "-f", fmt, env.container], capture_output=True, text=True).stdout.split()
        assert info == [str(int(1.5e9)), str(300 * 1024 * 1024), "100", "none"]
        mounts = subprocess.run([RUNTIME, "inspect", "-f", "{{len .Mounts}}", env.container],
                                capture_output=True, text=True).stdout.strip()
        assert mounts == "0"  # repo is copied, not mounted


@needs_docker
def test_docker_timeout_kills_escaped_session(repo):
    """A child that leaves the process group (setsid) is still killed by the exec-id sweep."""
    with DockerEnv(repo, docker_cfg(), timeout_s=1) as env:
        obs = env.execute("setsid sleep 300 & sleep 300")
        assert obs.timed_out
        time.sleep(0.5)
        left = env.execute("for d in /proc/[0-9]*; do c=$(tr '\\0' ' ' < $d/cmdline 2>/dev/null); "
                           "[[ $c == 'sleep 300 ' ]] && echo $c; done; true").output
        assert "sleep 300" not in left


@needs_docker
def test_docker_container_removed_on_close(repo):
    env = DockerEnv(repo, docker_cfg())
    cid = env.container
    env.close()
    assert subprocess.run([RUNTIME, "inspect", cid], capture_output=True).returncode != 0
    assert env.git_diff() == ""  # closed env degrades instead of raising


@needs_docker
def test_docker_runs_as_configured_user(repo):
    with DockerEnv(repo, docker_cfg(user="nobody")) as env:
        assert env.execute("id -un").output.strip() == "nobody"
        assert env.execute("echo x >> app.py").returncode == 0  # repo copy is writable by that user
        assert "app.py" in env.git_diff()


@needs_docker
def test_docker_bad_image_raises_and_cleans_up(repo):
    with pytest.raises(EnvError):
        DockerEnv(repo, docker_cfg(image="hagent-no-such-image:missing"))


def test_docker_requires_image(repo):
    with pytest.raises(EnvError, match="image"):
        DockerEnv(repo, EnvConfig(backend="docker", image=""))
