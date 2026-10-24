"""Inherited GIT_* variables (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...) must not redirect the
git helpers or the agent's commands to another repository (vla-router F2a review fixes)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hagent.env import LocalEnv


def _repo(root: Path, text: str) -> Path:
    root.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    (root / "a.txt").write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_clean_environ_drops_every_git_variable():
    from hagent.env import clean_environ
    env = clean_environ({"GIT_DIR": "x", "GIT_CONFIG_COUNT": "1", "GITHUB_TOKEN": "t", "PATH": "/bin"})
    assert env == {"GITHUB_TOKEN": "t", "PATH": "/bin"}


def test_local_env_git_and_commands_ignore_inherited_git_variables(tmp_path, monkeypatch):
    other = _repo(tmp_path / "other", "other\n")
    repo = _repo(tmp_path / "repo", "mine\n")
    before = _files(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    env = LocalEnv(repo)
    (repo / "a.txt").write_text("mine, changed\n")
    assert env.is_git_repo() and "mine, changed" in env.git_diff()
    obs = env.execute("git rev-parse --show-toplevel; env | grep '^GIT_DIR=' || echo no-git-dir")
    lines = obs.output.splitlines()
    assert lines[0] == str(repo.resolve()) and lines[-1] == "no-git-dir"
    env.execute("git add -A")
    assert _files(other) == before


def test_patch_keeps_changes_the_agent_committed(tmp_path):
    """vla-router ISSUES #56: the patch is the change since the starting commit, not since HEAD."""
    repo = _repo(tmp_path / "repo", "one\n")
    env = LocalEnv(repo)
    assert env.git_diff() == "" and env._start_commit
    env.execute("echo two > a.txt && git add -A && git -c user.name=a -c user.email=a@a commit -qm agent "
                "&& echo three > b.txt")
    diff = env.git_diff()
    assert "+two" in diff and "+three" in diff and "b.txt" in diff        # committed edit and new file
    env.execute("git -c user.name=a -c user.email=a@a commit -qam more")
    assert "+two" in env.git_diff()
    assert "+two" not in LocalEnv(repo).git_diff()                         # a new env starts from the new HEAD


def test_tracked_diff_hash_sees_committed_edits(tmp_path):
    repo = _repo(tmp_path / "repo", "one\n")
    env = LocalEnv(repo)
    before = env.git_tracked_diff_hash()
    env.execute("echo two > a.txt && git -c user.name=a -c user.email=a@a commit -qam agent")
    assert env.git_tracked_diff_hash() != before


def test_repo_without_commits_still_diffs(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    env = LocalEnv(root)
    (root / "x.txt").write_text("hello\n")
    assert env._start_commit is None and "+hello" in env.git_diff()
