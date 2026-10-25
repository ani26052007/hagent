"""Model wrappers. No SDK dependency: plain urllib against an OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .config import ModelConfig
from .exceptions import ModelError


# Where a Reply's token counts came from.
USAGE_API = "api"            # both counts read from the provider's `usage` block
USAGE_PARTIAL = "partial"    # one count from the provider, the other estimated
USAGE_ESTIMATED = "estimated"  # no usable usage block: ~4 characters per token


@dataclass
class Reply:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    usage_source: str = USAGE_ESTIMATED
    finish_reason: str = ""  # provider's finish_reason; "length" means the reply was cut off at max_tokens


def _first_int(usage: dict, *keys: str) -> int | None:
    for k in keys:
        v = usage.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            return int(v)
    return None


def read_usage(payload: dict, messages: list[dict], content: str) -> tuple[int, int, str]:
    """Token counts for one call: (input_tokens, output_tokens, usage_source).

    Contract: provider-reported counts always win. Accepted keys are the OpenAI ones
    (`prompt_tokens`, `completion_tokens`) and the `input_tokens` / `output_tokens` spelling some
    gateways use. If only `total_tokens` and one side are given, the other side is total minus that
    side. A side that is still unknown is estimated at ~4 characters per token, and the source says
    so. A block where every count is 0 is treated as absent (some servers send zeros instead of
    omitting it), which keeps token and cost caps working.
    """
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    tin = _first_int(usage, "prompt_tokens", "input_tokens")
    tout = _first_int(usage, "completion_tokens", "output_tokens")
    total = _first_int(usage, "total_tokens")
    if total is not None:
        if tin is None and tout is not None and total >= tout:
            tin = total - tout
        elif tout is None and tin is not None and total >= tin:
            tout = total - tin
    if not (tin or tout or total):
        tin = tout = None
    est_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
    est_out = len(content) // 4
    if tin is not None and tout is not None:
        return tin, tout, USAGE_API
    if tin is None and tout is None:
        return est_in, est_out, USAGE_ESTIMATED
    return (tin if tin is not None else est_in), (tout if tout is not None else est_out), USAGE_PARTIAL


class Model(Protocol):
    name: str

    def query(self, messages: list[dict]) -> Reply: ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatModel:
    """Chat-completions client with retries for 408/409/429/5xx and network errors."""

    max_retries = 3
    backoff_s = 2.0

    def __init__(self, cfg: ModelConfig, name: str = "model"):
        if not cfg.model:
            raise ModelError(f"[{name}] has no `model` set in the config")
        self.cfg = cfg
        self.name = name

    def query(self, messages: list[dict]) -> Reply:
        cfg = self.cfg
        body = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            **cfg.extra_body,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode("utf-8")

        payload = None
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise ModelError(f"[{self.name}] {last_err}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last_err = repr(e)
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_s * 2**attempt)
        if payload is None:
            raise ModelError(f"[{self.name}] failed after {self.max_retries} attempts: {last_err}")

        try:
            choice = payload["choices"][0]
            raw_content = choice["message"].get("content") or ""
            finish_reason = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ModelError(f"[{self.name}] unexpected response shape: {str(payload)[:300]}") from e
        content = _THINK.sub("", raw_content).strip()  # some reasoning models inline their thoughts

        # read the raw content length for estimates before <think> stripping would undercount output
        tin, tout, source = read_usage(payload, messages, raw_content)
        cost = tin * cfg.price_in_per_mtok / 1e6 + tout * cfg.price_out_per_mtok / 1e6
        return Reply(content, tin, tout, cost, source, finish_reason)


ScriptItem = "str | Callable[[list[dict]], str]"


class ScriptExhausted(AssertionError):
    pass


class ScriptedModel:
    """Deterministic fake model for tests and offline experiments.

    `script` items are strings or callables(messages) -> str. With repeat_last=True the
    final item is repeated forever, which is how loop scenarios are simulated.
    `tokens_per_call=(in, out)` makes replies report fixed "API" token counts instead of the
    character estimate (for testing real-usage accounting).
    """

    def __init__(self, script: Sequence, name: str = "scripted", cost_per_call: float = 0.0, repeat_last: bool = False,
                 tokens_per_call: tuple[int, int] | None = None):
        self.script = list(script)
        self.name = name
        self.cost_per_call = cost_per_call
        self.repeat_last = repeat_last
        self.tokens_per_call = tokens_per_call
        self.calls: list[list[dict]] = []  # a snapshot of the messages for every call

    def query(self, messages: list[dict]) -> Reply:
        self.calls.append([dict(m) for m in messages])
        i = len(self.calls) - 1
        if i >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(f"[{self.name}] script exhausted after {len(self.script)} calls")
            i = len(self.script) - 1
        item = self.script[i]
        content = item(messages) if callable(item) else item
        if self.tokens_per_call is not None:
            tin, tout = self.tokens_per_call
            return Reply(content, tin, tout, self.cost_per_call, USAGE_API)
        tin = sum(len(m["content"]) for m in messages) // 4
        return Reply(content, tin, len(content) // 4, self.cost_per_call, USAGE_ESTIMATED)


def bash(command: str, thought: str = "Running a command.") -> str:
    """Build a well-formed scripted reply (test convenience)."""
    return f"{thought}\n\n```bash\n{command}\n```"
