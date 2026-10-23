"""Settings: two model endpoints plus every limit and threshold in one place."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """One OpenAI-compatible chat endpoint (NVIDIA NIM, vLLM, OpenRouter, ...)."""

    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"  # name of the env var holding the key; empty = no auth header
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: int = 300
    price_in_per_mtok: float = 0.0  # USD per 1M input tokens, used for cost limits
    price_out_per_mtok: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)  # merged into the request JSON


@dataclass
class Limits:
    # --- hard caps: always terminate the run ---
    max_steps: int = 80
    max_cost: float = 3.0  # USD, all models combined. Only works if model prices are set!
    max_total_tokens: int = 3_000_000  # input+output tokens over every call and model; works without prices; 0 disables
    max_wall_s: int = 1800
    cmd_timeout_s: int = 60
    max_format_errors: int = 3  # consecutive

    # --- token control ---
    obs_max_chars: int = 3000  # per command output shown to the model
    ctx_char_limit: int = 60000  # compact history above this (~15k tokens); 0 disables
    keep_last_messages: int = 10  # verbatim tail kept after compaction

    # --- big-model budget (hybrid mode) ---
    max_big_calls: int = 12  # advice + review + takeover steps, combined
    plan_at_step: int = 8  # one planning consult after this many steps; 0 disables
    takeover_steps: int = 3  # steps the big model drives after repeated stuck flags
    final_review: bool = True
    max_review_rejections: int = 2

    # --- loop detection ---
    window: int = 8
    repeat_threshold: int = 3
    stall_warmup: int = 10
    stall_steps: int = 12
    max_strikes: int = 5  # stuck flags without real progress before the run is stopped

    # --- optional explorer helper ---
    enable_explorer: bool = False
    max_explorer_calls: int = 3
    explorer_max_steps: int = 12
    explorer_max_cost: float = 0.10


@dataclass
class EnvConfig:
    """Where commands run. backend="local" runs bash on the host in the repo itself;
    backend="docker" runs them in a container holding a COPY of the repo (see docker_env.py)."""

    backend: str = "local"  # local | docker
    runtime: str = "docker"  # docker | podman (same CLI subset is used)
    image: str = ""  # must provide bash, git, coreutils `timeout` and `sleep`
    network: str = "none"  # passed to --network; "none" = no network
    cpus: float = 2.0  # --cpus; 0 = no limit
    memory: str = "4g"  # --memory; "" = no limit
    pids_limit: int = 512  # --pids-limit; 0 = no limit
    workdir: str = "/workspace/repo"  # where the repo copy lives inside the container
    user: str = ""  # run commands as this user (repo copy is chown'ed to it); "" = image default
    kill_grace_s: int = 10  # host-side backstop after the in-container timeout
    startup_timeout_s: int = 120  # for container start and the repo copy
    extra_run_args: list[str] = field(default_factory=lambda: ["--security-opt", "no-new-privileges"])
    commands_dir: str = "/opt/hagent/bin"  # in-container dir for custom-command shims


_CMD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass
class CustomCommand:
    """An extra shell command offered to the agent (e.g. `run_rung1`).

    `path` is the executable as seen by the execution environment (host path for the local
    env, in-container path for the docker env). The agent calls it by `name`; a shim on PATH
    forwards to `path`. `max_output_chars` is this command's own output-truncation limit
    (0 = use limits.obs_max_chars)."""

    name: str
    path: str
    description: str
    max_output_chars: int = 0

    def __post_init__(self) -> None:
        if not _CMD_NAME.match(self.name or ""):
            raise ValueError(f"custom command name {self.name!r} must match {_CMD_NAME.pattern}")
        if not self.path:
            raise ValueError(f"custom command {self.name!r} has no path")
        if not (self.description or "").strip():
            raise ValueError(f"custom command {self.name!r} needs a description (it is shown to the agent)")
        if self.max_output_chars < 0:
            raise ValueError(f"custom command {self.name!r}: max_output_chars must be >= 0")


@dataclass
class Settings:
    small: ModelConfig = field(default_factory=ModelConfig)
    big: ModelConfig = field(default_factory=ModelConfig)
    limits: Limits = field(default_factory=Limits)
    env: EnvConfig = field(default_factory=EnvConfig)
    commands: list[CustomCommand] = field(default_factory=list)


def _build(cls: type, data: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [{where}]: {sorted(unknown)}. Valid keys: {sorted(known)}")
    return cls(**data)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    unknown = set(data) - {"small", "big", "limits", "env", "commands"}
    if unknown:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown)}")
    env = _build(EnvConfig, data.get("env", {}), "env")
    if env.backend not in ("local", "docker"):
        raise ValueError(f"[env] backend must be 'local' or 'docker', got {env.backend!r}")
    raw_cmds = data.get("commands", [])
    if not isinstance(raw_cmds, list):
        raise ValueError("[[commands]] must be an array of tables")
    commands = [_build(CustomCommand, c, f"commands[{i}]") for i, c in enumerate(raw_cmds)]
    names = [c.name for c in commands]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate custom command name(s): {dupes}")
    return Settings(
        small=_build(ModelConfig, data.get("small", {}), "small"),
        big=_build(ModelConfig, data.get("big", {}), "big"),
        limits=_build(Limits, data.get("limits", {}), "limits"),
        env=env,
        commands=commands,
    )


def pricing_warnings(settings: Settings, mode: str) -> list[str]:
    """max_cost can only trigger if prices are configured; say so when it silently could not."""
    if settings.limits.max_cost <= 0:
        return []
    used = {"hybrid": ("small", "big"), "small": ("small",), "big": ("big",)}[mode]
    unpriced = [n for n in used
                if not (getattr(settings, n).price_in_per_mtok or getattr(settings, n).price_out_per_mtok)]
    if not unpriced:
        return []
    return [f"max_cost=${settings.limits.max_cost} cannot trigger: no prices set for model(s) {', '.join(unpriced)}. "
            f"Set price_in_per_mtok / price_out_per_mtok; until then only max_total_tokens and max_steps protect you."]


def load_settings(path: str | Path) -> Settings:
    return settings_from_dict(tomllib.loads(Path(path).read_text()))
