"""hagent: a minimal, bash-only coding agent that routes between a small and a big model."""

from .agent import Agent, RunResult, Usage
from .config import CustomCommand, EnvConfig, Limits, ModelConfig, Settings, load_settings, settings_from_dict
from .env import SUBMIT_MARKER, Env, LocalEnv, Obs
from .hybrid import HybridAgent
from .models import OpenAICompatModel, Reply, ScriptedModel
from .runner import make_env, solve
from .task_template import IncidentContext, TaskTemplate, load_task_template

__all__ = [
    "Agent", "HybridAgent", "RunResult", "Usage", "Limits", "ModelConfig", "Settings",
    "load_settings", "settings_from_dict", "LocalEnv", "Obs", "SUBMIT_MARKER",
    "OpenAICompatModel", "ScriptedModel", "Reply", "solve", "make_env", "Env", "EnvConfig", "CustomCommand",
    "IncidentContext", "TaskTemplate", "load_task_template",
]
__version__ = "0.1.0"
