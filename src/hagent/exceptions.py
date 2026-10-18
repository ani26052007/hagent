"""Exceptions used for control flow, in the spirit of mini-swe-agent."""

from __future__ import annotations


class HAgentError(Exception):
    """Base class for all hagent errors."""


class FormatError(HAgentError):
    """The model reply did not contain exactly one bash block."""


class ModelError(HAgentError):
    """The model API failed (after retries) or returned something unusable."""


class AgentExit(HAgentError):
    """Raised to end the run. `status` becomes RunResult.exit_status.

    Statuses used: Submitted, LimitsExceeded, TimeExceeded, Stuck, ModelError,
    RepeatedFormatError.
    """

    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = status
        self.detail = detail
