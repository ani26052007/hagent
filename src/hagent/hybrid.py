"""HybridAgent: a small model drives, a big model is consulted rarely and with a compact digest.

Who does what
  small model  every normal step, history compaction, the optional explorer
  big model    one planning consult, advice when stuck, a short takeover when still stuck,
               and one final review before a submission is accepted

Escalation ladder when the loop detector fires (each fire is a "strike"; a genuinely new
repo state or a new error resets the count):
  strike 1        nudge text, still the small model
  strike 2        big model advice
  strike 3+       big model advice, then the big model drives for `takeover_steps` steps
  max_strikes     the run ends with status "Stuck" (the best patch so far is still returned)
"""

from __future__ import annotations

import shlex

from . import helpers
from .agent import Agent
from .compaction import split_summary
from .config import Settings
from .detector import LoopDetector
from .env import Obs
from .exceptions import AgentExit, ModelError
from .explorer import run_explorer
from .models import Model
from .prompts import CONSULT_HEADER, NUDGE, NUDGE_FINAL, build_system_prompt


class HybridAgent(Agent):
    def __init__(self, small: Model, big: Model, env, settings: Settings, **kw):
        lim = settings.limits
        prompt = kw.pop("system_prompt", None) or build_system_prompt(
            env.cwd, lim.max_explorer_calls if lim.enable_explorer else 0
        )
        super().__init__(small, env, settings, role="small", system_prompt=prompt, **kw)
        self.big = big
        self.detector = LoopDetector(lim.window, lim.repeat_threshold, lim.stall_warmup, lim.stall_steps)
        self.big_calls = 0
        self.strikes = 0
        self.takeover_left = 0
        self.planned = False
        self.pending_flag: str | None = None
        self.review_rejections = 0
        self.explorer_calls = 0

    # ------------------------------------------------------------------ big-model budget
    def big_available(self, for_review: bool = False) -> bool:
        """Advice and takeover leave one call in reserve for the final review."""
        lim = self.limits
        reserve = 1 if (lim.final_review and not for_review) else 0
        return self.big_calls < lim.max_big_calls - reserve

    # ------------------------------------------------------------------ routing hooks
    def active(self) -> tuple[str, Model]:
        if self.takeover_left > 0 and self.big_available():
            return "big", self.big
        return "small", self.model

    def compactor(self) -> tuple[str, Model]:
        return "small", self.model

    def query(self):
        reply = super().query()
        if self.last_role == "big":  # takeover steps spend big-model budget
            self.big_calls += 1
            self.takeover_left = max(0, self.takeover_left - 1)
        return reply

    def before_query(self) -> None:
        super().before_query()  # compaction first, so digests see the compact history
        self._consult_if_needed()

    # ------------------------------------------------------------------ observations / loop detection
    def on_observation(self, command: str, obs: Obs) -> None:
        super().on_observation(command, obs)
        flag = self.detector.observe(
            command, obs.output, self.env.git_diff_hash(), self.env.git_tracked_diff_hash()
        )
        if flag:
            self.pending_flag = flag
        elif self.detector.progress:
            self.strikes = 0  # real progress: new repo state or a new error

    def _consult_if_needed(self) -> None:
        lim = self.limits
        if self.pending_flag:
            flag, self.pending_flag = self.pending_flag, None
            self.strikes += 1
            self.detector.acknowledge()
            self.log.info("loop flag #%d: %s", self.strikes, flag)
            if self.strikes >= lim.max_strikes:
                raise AgentExit("Stuck", flag)
            if self.strikes == 1:
                self.inject(NUDGE.replace("<<FLAG>>", flag))
                return
            got_advice = self._consult_big(f"the agent appears stuck ({flag})")
            if got_advice and self.strikes >= 3:
                self.takeover_left = lim.takeover_steps
            if not got_advice:
                self.inject(NUDGE_FINAL.replace("<<FLAG>>", flag)
                            .replace("<<N>>", str(self.strikes)).replace("<<MAX>>", str(lim.max_strikes)))
            return
        if lim.plan_at_step and not self.planned and self.n_steps >= lim.plan_at_step:
            self.planned = True
            self._consult_big("planning checkpoint after initial exploration")

    def _consult_big(self, reason: str) -> bool:
        if not self.big_available():
            return False
        _, summary = split_summary(self.messages[1]["content"])
        digest = helpers.build_digest(
            task=split_summary(self.messages[1]["content"])[0],
            summary=summary,
            recent=self.messages[-6:],
            diff=self.env.git_diff(),
            test_output=self.last_test_output,
            reason=reason,
        )
        try:
            reply = helpers.advise(self.big, digest)
        except ModelError as e:
            self.log.warning("big model advice failed: %s", e)
            return False
        self.account("big", reply)
        self.big_calls += 1
        text = reply.content.strip()
        if not text:
            return False
        self.inject(CONSULT_HEADER.replace("<<REASON>>", reason) + "\n" + text)
        return True

    # ------------------------------------------------------------------ final review
    def handle_submission(self, submission: str) -> str | None:
        lim = self.limits
        if not lim.final_review or self.review_rejections >= lim.max_review_rejections:
            return None
        if not self.big_available(for_review=True):
            return None
        diff = self.env.git_diff()
        if not diff.strip():
            self.review_rejections += 1
            return "[Reviewer] Your submission has an empty diff. Make the change, verify it, then submit again."
        try:
            verdict, reply = helpers.review(self.big, self.task, diff, self.last_test_output)
        except ModelError as e:
            self.log.warning("review failed, accepting submission: %s", e)
            return None
        self.account("big", reply)
        self.big_calls += 1
        if verdict.ok:
            return None
        self.review_rejections += 1
        return f"[Reviewer rejected your submission]\n{verdict.feedback}\nFix these issues, re-verify, then submit again."

    # ------------------------------------------------------------------ explorer
    def run_command(self, command: str) -> Obs:
        if self.limits.enable_explorer and command.lstrip().startswith("ask_explorer"):
            return self._ask_explorer(command)
        return super().run_command(command)

    def _ask_explorer(self, command: str) -> Obs:
        rest = command.strip()[len("ask_explorer"):].strip()
        try:
            question = " ".join(shlex.split(rest))
        except ValueError:
            question = rest
        if not question:
            return Obs('Usage: ask_explorer "your question"', 2)
        if self.explorer_calls >= self.limits.max_explorer_calls:
            return Obs("Explorer budget exhausted. Search the code yourself with grep/sed.", 1)
        self.explorer_calls += 1
        answer, usage, _ = run_explorer(question, self.env, self.model, self.settings)
        self.merge_usage("explorer", usage)
        return Obs(answer, 0)
