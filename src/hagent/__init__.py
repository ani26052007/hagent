"""hagent: a minimal, bash-only coding agent that routes between a small and a big model."""

from .agent import Agent, RunResult, Usage
from .config import Limits, ModelConfig, Settings, load_settings, settings_from_dict
from .env import SUBMIT_MARKER, LocalEnv, Obs
from .hybrid import HybridAgent
from .models import OpenAICompatModel, Reply, ScriptedModel
from .runner import solve

__all__ = [
    "Agent", "HybridAgent", "RunResult", "Usage", "Limits", "ModelConfig", "Settings",
    "load_settings", "settings_from_dict", "LocalEnv", "Obs", "SUBMIT_MARKER",
    "OpenAICompatModel", "ScriptedModel", "Reply", "solve",
]
__version__ = "0.1.0"
