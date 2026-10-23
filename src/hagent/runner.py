"""Public entry point: solve(task, repo, settings, mode)."""

from __future__ import annotations

from pathlib import Path

from .agent import Agent, RunResult
from .config import Settings
from .env import Env, LocalEnv
from .hybrid import HybridAgent
from .models import Model, OpenAICompatModel

MODES = ("hybrid", "small", "big")


def solve(
    task: str,
    repo: str | Path,
    settings: Settings,
    mode: str = "hybrid",
    *,
    small: Model | None = None,
    big: Model | None = None,
    trajectory_path: str | Path | None = None,
    env: Env | None = None,
) -> RunResult:
    """Run the agent on `repo`.

    mode="hybrid"  small model drives, big model advises/reviews   (the point of this project)
    mode="small"   small model alone                               (baseline)
    mode="big"     big model alone                                 (baseline / upper bound)

    Pass `small=` / `big=` to inject your own Model objects (tests, other providers);
    otherwise they are built from `settings`.
    Pass `env=` to use your own environment (you then own its lifecycle); otherwise one is built
    from `settings.env` and closed when the run ends. Custom commands are installed either way.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    owned = env is None
    env = env or make_env(repo, settings)
    try:
        env.install_commands(settings.commands)
        return _run(task, settings, mode, env, small, big, trajectory_path)
    finally:
        if owned:
            env.close()


def make_env(repo: str | Path, settings: Settings) -> Env:
    """Build the execution environment selected by settings.env.backend."""
    if settings.env.backend == "docker":
        from .docker_env import DockerEnv
        return DockerEnv(repo, settings.env, timeout_s=settings.limits.cmd_timeout_s)
    return LocalEnv(repo, timeout_s=settings.limits.cmd_timeout_s)


def _run(task: str, settings: Settings, mode: str, env: Env, small: Model | None, big: Model | None,
         trajectory_path: str | Path | None) -> RunResult:
    if mode == "hybrid":
        small = small or OpenAICompatModel(settings.small, "small")
        big = big or OpenAICompatModel(settings.big, "big")
        agent: Agent = HybridAgent(small, big, env, settings, trajectory_path=trajectory_path)
    elif mode == "small":
        small = small or OpenAICompatModel(settings.small, "small")
        agent = Agent(small, env, settings, role="small", trajectory_path=trajectory_path)
    else:
        big = big or OpenAICompatModel(settings.big, "big")
        agent = Agent(big, env, settings, role="big", trajectory_path=trajectory_path)
    return agent.run(task)
