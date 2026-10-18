"""Settings: two model endpoints plus every limit and threshold in one place."""

from __future__ import annotations

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
    max_cost: float = 3.0  # USD, all models combined
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
class Settings:
    small: ModelConfig = field(default_factory=ModelConfig)
    big: ModelConfig = field(default_factory=ModelConfig)
    limits: Limits = field(default_factory=Limits)


def _build(cls: type, data: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [{where}]: {sorted(unknown)}. Valid keys: {sorted(known)}")
    return cls(**data)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    unknown = set(data) - {"small", "big", "limits"}
    if unknown:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown)}")
    return Settings(
        small=_build(ModelConfig, data.get("small", {}), "small"),
        big=_build(ModelConfig, data.get("big", {}), "big"),
        limits=_build(Limits, data.get("limits", {}), "limits"),
    )


def load_settings(path: str | Path) -> Settings:
    return settings_from_dict(tomllib.loads(Path(path).read_text()))
