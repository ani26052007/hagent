"""The base agent loop. Same skeleton as mini-swe-agent:

    one linear message list -> model -> exactly one bash block -> execute -> append output -> repeat

plus hard limits, observation truncation, history compaction and per-role usage accounting.
It works on its own (single model) and is the baseline you benchmark the hybrid against.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import helpers
from .compaction import maybe_compact
from .config import Settings
from .env import Obs
from .exceptions import AgentExit, FormatError, ModelError
from .models import Model, Reply
from .prompts import FORMAT_ERROR, TASK_TEMPLATE, build_system_prompt
from .textutil import parse_action, truncate_middle

TEST_CMD_RE = re.compile(
    r"\b(pytest|py\.test|unittest|tox|nosetests|npm (?:run )?test|yarn test|go test|cargo test|mvn test|gradle test|make test|ctest)\b"
)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class RunResult:
    exit_status: str  # Submitted | LimitsExceeded | TimeExceeded | Stuck | ModelError | RepeatedFormatError
    detail: str  # submission text on success, otherwise the reason
    patch: str  # `git diff` of the repo at the end, always filled in, even when the run failed
    steps: int
    elapsed_s: float
    total_cost: float
    usage: dict[str, Usage] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)

    @property
    def submitted(self) -> bool:
        return self.exit_status == "Submitted"

    def to_dict(self) -> dict:
        return asdict(self)


def format_observation(obs: Obs, max_chars: int) -> str:
    parts = [f"<returncode>{obs.returncode}</returncode>"]
    if obs.timed_out:
        parts.append("<warning>The command timed out and was killed. Use a faster or narrower command.</warning>")
    body = truncate_middle(obs.output, max_chars)
    parts.append(f"<output>\n{body}\n</output>" if body.strip() else "<output>(no output)</output>")
    return "\n".join(parts)


class Agent:
    def __init__(self, model: Model, env, settings: Settings, *, role: str = "small",
                 system_prompt: str | None = None, trajectory_path: str | Path | None = None):
        self.model = model
        self.env = env
        self.settings = settings
        self.limits = settings.limits
        self.role = role
        self.system_prompt = system_prompt or build_system_prompt(env.cwd)
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None

        self.log = logging.getLogger("hagent")
        self.messages: list[dict] = []
        self.usage: dict[str, Usage] = {}
        self.task = ""
        self.n_steps = 0
        self.n_compactions = 0
        self.format_errors = 0
        self.last_test_output = ""
        self.last_role = role
        self._t0 = time.time()

    # ------------------------------------------------------------------ messages
    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content or "(empty reply)"})

    def inject(self, text: str) -> None:
        """Add text for the model to read next. Merged into the last user message when there is
        one, so roles keep alternating."""
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += "\n\n" + text
        else:
            self.add("user", text)

    # ------------------------------------------------------------------ accounting
    def account(self, role: str, reply: Reply) -> None:
        u = self.usage.setdefault(role, Usage())
        u.calls += 1
        u.input_tokens += reply.input_tokens
        u.output_tokens += reply.output_tokens
        u.cost += reply.cost

    def merge_usage(self, role: str, other: dict[str, Usage]) -> None:
        u = self.usage.setdefault(role, Usage())
        for v in other.values():
            u.calls += v.calls
            u.input_tokens += v.input_tokens
            u.output_tokens += v.output_tokens
            u.cost += v.cost

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage.values())

    # ------------------------------------------------------------------ hooks (HybridAgent overrides)
    def active(self) -> tuple[str, Model]:
        """Which model answers the next step."""
        return self.role, self.model

    def compactor(self) -> tuple[str, Model]:
        return self.role, self.model

    def before_query(self) -> None:
        self._maybe_compact()

    def run_command(self, command: str) -> Obs:
        return self.env.execute(command, self.limits.cmd_timeout_s)

    def on_observation(self, command: str, obs: Obs) -> None:
        if TEST_CMD_RE.search(command):
            self.last_test_output = obs.output

    def handle_submission(self, submission: str) -> str | None:
        """Return None to accept the submission, or feedback text to reject it."""
        return None

    # ------------------------------------------------------------------ the loop
    def run(self, task: str) -> RunResult:
        self.task = task
        self._t0 = time.time()
        self.messages = []
        self.add("system", self.system_prompt)
        self.add("user", TASK_TEMPLATE.replace("<<CWD>>", self.env.cwd).replace("<<TASK>>", task))

        status, detail = "Unknown", ""
        while True:
            try:
                self.step()
                self.format_errors = 0
            except FormatError as e:
                self.format_errors += 1
                cap = self.limits.max_format_errors
                if 0 < cap <= self.format_errors:
                    status, detail = "RepeatedFormatError", str(e)
                    break
                self.inject(FORMAT_ERROR.replace("<<DETAIL>>", str(e)))
            except AgentExit as e:
                status, detail = e.status, e.detail
                break
            finally:
                self._save()
        self.log.info("exit: %s (%s)", status, detail[:200])
        result = RunResult(
            exit_status=status,
            detail=detail,
            patch=self.env.git_diff(),
            steps=self.n_steps,
            elapsed_s=round(time.time() - self._t0, 2),
            total_cost=round(self.total_cost, 6),
            usage=dict(self.usage),
            messages=self.messages,
        )
        self._save(result)
        return result

    def step(self) -> None:
        reply = self.query()
        command = parse_action(reply.content)  # raises FormatError
        obs = self.run_command(command)
        self.add("user", format_observation(obs, self.limits.obs_max_chars))
        if obs.submission is not None:
            feedback = self.handle_submission(obs.submission)
            if feedback is None:
                raise AgentExit("Submitted", obs.submission)
            self.inject(feedback)
            return
        self.on_observation(command, obs)

    def query(self) -> Reply:
        self.before_query()
        self._check_limits()
        role, model = self.active()
        self.n_steps += 1
        try:
            reply = model.query(self.messages)
        except ModelError as e:
            raise AgentExit("ModelError", str(e)) from e
        self.last_role = role
        self.account(role, reply)
        self.add("assistant", reply.content)
        return reply

    # ------------------------------------------------------------------ internals
    def _check_limits(self) -> None:
        lim = self.limits
        if 0 < lim.max_steps <= self.n_steps:
            raise AgentExit("LimitsExceeded", f"step limit {lim.max_steps} reached")
        if 0 < lim.max_cost <= self.total_cost:
            raise AgentExit("LimitsExceeded", f"cost limit ${lim.max_cost} reached")
        if 0 < lim.max_wall_s <= time.time() - self._t0:
            raise AgentExit("TimeExceeded", f"wall-clock limit {lim.max_wall_s}s reached")

    def _maybe_compact(self) -> None:
        lim = self.limits

        def summarize(transcript: str, previous: str) -> str | None:
            role, model = self.compactor()
            try:
                reply = helpers.summarize(model, transcript, previous)
            except ModelError:
                return None
            self.account(role, reply)
            return reply.content

        if maybe_compact(self.messages, summarize, lim.ctx_char_limit, lim.keep_last_messages):
            self.n_compactions += 1
            self.log.info("history compacted (%d so far)", self.n_compactions)

    def _save(self, result: RunResult | None = None) -> None:
        if not self.trajectory_path:
            return
        data = {
            "messages": self.messages,
            "usage": {k: asdict(v) for k, v in self.usage.items()},
            "steps": self.n_steps,
        }
        if result is not None:
            data["exit_status"] = result.exit_status
            data["detail"] = result.detail
            data["patch"] = result.patch
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_path.write_text(json.dumps(data, indent=2))
