"""Command line: hagent run --repo ./myrepo --task "Fix the bug in ..." --config config.toml"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import Settings, load_settings, pricing_warnings
from .exceptions import HAgentError
from .runner import MODES, solve


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hagent", description="Small+big model coding agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent on a repository")
    r.add_argument("--repo", required=True, help="path to the git repository to work in")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", help="task description")
    g.add_argument("--task-file", help="file containing the task description (e.g. an issue)")
    r.add_argument("--config", help="TOML config (see examples/config.example.toml)")
    r.add_argument("--mode", choices=MODES, default="hybrid")
    r.add_argument("--trajectory", help="write the full message trajectory (JSON) here")
    r.add_argument("--patch-out", help="write the final patch here instead of stdout")
    r.add_argument("--init-git", action="store_true", help="git init + initial commit if the repo is not a git repo")
    r.add_argument("-v", "--verbose", action="store_true")
    return p


def _init_git(repo: Path) -> None:
    def git(*a: str) -> None:
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=hagent", "-c", "user.email=hagent@localhost", "commit", "-q", "-m", "initial", "--allow-empty")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    repo = Path(args.repo)
    if not repo.is_dir():
        print(f"error: {repo} is not a directory", file=sys.stderr)
        return 2
    if args.init_git and not (repo / ".git").exists():
        _init_git(repo)
    task = Path(args.task_file).read_text() if args.task_file else args.task
    settings = load_settings(args.config) if args.config else Settings()
    for warning in pricing_warnings(settings, args.mode):
        print(f"warning: {warning}", file=sys.stderr)

    try:
        result = solve(task, repo, settings, args.mode, trajectory_path=args.trajectory)
    except HAgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    usage = ", ".join(f"{k}: {v.calls} calls ${v.cost:.4f}" for k, v in result.usage.items()) or "no model calls"
    print(f"[hagent] {result.exit_status} | {result.steps} steps | ${result.total_cost:.4f} | {result.elapsed_s}s | {usage}",
          file=sys.stderr)
    if not result.submitted:
        print(f"[hagent] reason: {result.detail}", file=sys.stderr)
    if args.patch_out:
        Path(args.patch_out).write_text(result.patch)
    else:
        print(result.patch)
    return 0 if result.submitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
