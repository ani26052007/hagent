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


@dataclass
class Reply:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


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
            content = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ModelError(f"[{self.name}] unexpected response shape: {str(payload)[:300]}") from e
        content = _THINK.sub("", content).strip()  # some reasoning models inline their thoughts

        usage = payload.get("usage") or {}
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        if not (tin or tout):  # provider sent no usage block: estimate (~4 chars/token) so token caps still work
            tin = sum(len(str(m.get("content", ""))) for m in messages) // 4
            tout = len(content) // 4
        cost = tin * cfg.price_in_per_mtok / 1e6 + tout * cfg.price_out_per_mtok / 1e6
        return Reply(content, tin, tout, cost)


ScriptItem = "str | Callable[[list[dict]], str]"


class ScriptExhausted(AssertionError):
    pass


class ScriptedModel:
    """Deterministic fake model for tests and offline experiments.

    `script` items are strings or callables(messages) -> str. With repeat_last=True the
    final item is repeated forever, which is how loop scenarios are simulated.
    """

    def __init__(self, script: Sequence, name: str = "scripted", cost_per_call: float = 0.0, repeat_last: bool = False):
        self.script = list(script)
        self.name = name
        self.cost_per_call = cost_per_call
        self.repeat_last = repeat_last
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
        tin = sum(len(m["content"]) for m in messages) // 4
        return Reply(content, tin, len(content) // 4, self.cost_per_call)


def bash(command: str, thought: str = "Running a command.") -> str:
    """Build a well-formed scripted reply (test convenience)."""
    return f"{thought}\n\n```bash\n{command}\n```"
