"""Unit tests: parsing, truncation, env, detector, compaction, config, models."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hagent.compaction import SUMMARY_MARK, maybe_compact, total_chars
from hagent.config import ModelConfig, load_settings, settings_from_dict
from hagent.detector import LoopDetector
from hagent.env import SUBMIT_MARKER, LocalEnv
from hagent.exceptions import FormatError, ModelError
from hagent.helpers import parse_verdict
from hagent.models import OpenAICompatModel, ScriptedModel
from hagent.textutil import failure_signature, parse_action, truncate_middle


# ---------------------------------------------------------------- textutil
def test_parse_action_single_block():
    assert parse_action("thinking\n```bash\nls -la\n```") == "ls -la"


def test_parse_action_untagged_and_sh_blocks():
    assert parse_action("```\necho a\n```") == "echo a"
    assert parse_action("```sh\necho b\n```") == "echo b"


def test_parse_action_ignores_other_languages():
    text = "Example:\n```python\nprint(1)\n```\nNow run:\n```bash\nls\n```"
    assert parse_action(text) == "ls"


@pytest.mark.parametrize("text", ["no code here", "```bash\na\n```\n```bash\nb\n```", "```bash\n\n```"])
def test_parse_action_rejects_bad_replies(text):
    with pytest.raises(FormatError):
        parse_action(text)


def test_parse_action_unterminated_block():
    with pytest.raises(FormatError, match="never closed"):
        parse_action("```bash\nls")


def test_truncate_middle_keeps_head_and_tail():
    text = "H" * 500 + "M" * 5000 + "T" * 500
    out = truncate_middle(text, 1000)
    assert out.startswith("HHH") and out.endswith("TTT")
    assert "elided" in out and len(out) < 1100
    assert truncate_middle("short", 1000) == "short"


def test_failure_signature_ignores_timings():
    a = "FAILED tests/test_x.py::test_a - AssertionError\n== 1 failed in 0.12s =="
    b = "FAILED tests/test_x.py::test_a - AssertionError\n== 1 failed in 3.40s =="
    assert failure_signature(a) == failure_signature(b)
    assert failure_signature("all good") is None


# ---------------------------------------------------------------- env
def test_env_runs_commands_and_returns_code(repo):
    env = LocalEnv(repo)
    assert env.execute("echo hi").output.strip() == "hi"
    assert env.execute("exit 3").returncode == 3


def test_env_is_stateless_between_commands(repo):
    env = LocalEnv(repo)
    env.execute("cd /tmp && export FOO=1")
    out = env.execute("pwd; echo ${FOO:-unset}").output.split()
    assert out == [str(repo.resolve()), "unset"]


def test_env_timeout_kills_process_group(repo):
    env = LocalEnv(repo, timeout_s=1)
    t0 = time.time()
    obs = env.execute("sleep 30 & sleep 30")
    assert obs.timed_out and obs.returncode == -1
    assert time.time() - t0 < 5


def test_env_stdin_is_closed(repo):
    assert LocalEnv(repo, timeout_s=5).execute("cat").returncode == 0  # would hang on an open stdin


def test_env_submission_detection(repo):
    env = LocalEnv(repo)
    assert env.execute(f"echo {SUBMIT_MARKER}; echo patch-body").submission == "patch-body\n"
    assert env.execute(f"echo not-first; echo {SUBMIT_MARKER}").submission is None
    assert env.execute(f"echo {SUBMIT_MARKER}; exit 1").submission is None


def test_env_git_diff_includes_new_files_and_hash_changes(repo):
    env = LocalEnv(repo)
    assert env.git_diff() == ""
    h0 = env.git_diff_hash()
    env.execute("echo x > new.txt && sed -i 's/a - b/a + b/' app.py")
    diff = env.git_diff()
    assert "new.txt" in diff and "+    return a + b" in diff
    assert env.git_diff_hash() != h0


# ---------------------------------------------------------------- detector
def test_detector_repeated_command_same_state():
    d = LoopDetector(repeat_threshold=3)
    assert d.observe("ls", "a", "h1") is None
    assert d.observe("ls", "a", "h1") is None
    assert d.observe("ls", "a", "h1").startswith("repeated_command")


def test_detector_same_command_after_edit_is_not_a_loop():
    d = LoopDetector(repeat_threshold=3)
    flags = [d.observe("pytest -q", f"out{i}", f"h{i}") for i in range(5)]  # repo changes each time
    assert all(f is None for f in flags)


def test_detector_repeated_output_with_different_commands():
    d = LoopDetector(repeat_threshold=3)
    d.observe("cat a", "same", "h")
    d.observe("cat b", "same", "h")
    assert d.observe("cat c", "same", "h").startswith("repeated_output")


def test_detector_oscillation():
    d = LoopDetector()
    assert d.observe("edit1", "ok", "A") is None
    assert d.observe("edit2", "ok", "B") is None
    assert d.observe("edit3", "ok", "A").startswith("oscillation")


def test_detector_stall_only_after_warmup():
    d = LoopDetector(stall_warmup=2, stall_steps=3, repeat_threshold=99)
    flags = [d.observe(f"cmd{i}", f"out{i}", "same") for i in range(1, 6)]
    assert flags[:3] == [None, None, None]  # step 4 is the first with 3 stale steps after warmup
    assert flags[3].startswith("stalled")


def test_detector_new_error_counts_as_progress():
    d = LoopDetector(stall_warmup=0, stall_steps=3, repeat_threshold=99)
    d.observe("t", "FAILED a - AssertionError", "same")
    d.observe("t2", "ok", "same")
    d.observe("t3", "ok2", "same")
    assert d.observe("t4", "FAILED b - KeyError: 'x'", "same") is None
    assert d.progress is True


def test_detector_acknowledge_resets_window():
    d = LoopDetector(repeat_threshold=3)
    for _ in range(3):
        flag = d.observe("ls", "a", "h")
    assert flag
    d.acknowledge()
    assert d.observe("ls", "a", "h") is None


# ---------------------------------------------------------------- compaction
def _history(n_pairs: int, size: int = 400) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK"}]
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "content": f"```bash\necho {i}\n```\n" + "a" * size})
        msgs.append({"role": "user", "content": f"out {i} " + "o" * size})
    return msgs


def _roles_ok(msgs):
    roles = [m["role"] for m in msgs]
    return roles[0] == "system" and all(roles[i] != roles[i + 1] for i in range(1, len(roles) - 1)) and roles[-1] == "user"


def test_compaction_replaces_middle_and_keeps_shape():
    msgs = _history(30)
    before = total_chars(msgs)
    assert maybe_compact(msgs, lambda t, p: "SUMMARY-1", 5000, keep_last=6)
    assert total_chars(msgs) < before
    assert SUMMARY_MARK in msgs[1]["content"] and "SUMMARY-1" in msgs[1]["content"]
    assert msgs[1]["content"].startswith("TASK")
    assert msgs[2]["role"] == "assistant" and _roles_ok(msgs)
    assert len(msgs) <= 2 + 6 + 1


def test_compaction_is_noop_when_small():
    msgs = _history(3, size=10)
    assert not maybe_compact(msgs, lambda t, p: "x", 50_000, keep_last=6)


def test_compaction_merges_previous_summary():
    msgs = _history(30)
    maybe_compact(msgs, lambda t, p: "FIRST", 5000, keep_last=6)
    seen = {}
    msgs += _history(30)[2:]
    maybe_compact(msgs, lambda t, p: seen.setdefault("prev", p) and "SECOND", 5000, keep_last=6)
    assert "FIRST" in seen["prev"]
    assert msgs[1]["content"].count(SUMMARY_MARK) == 1


def test_compaction_survives_summarizer_failure():
    msgs = _history(30)

    def boom(t, p):
        raise RuntimeError("model down")

    assert maybe_compact(msgs, boom, 5000, keep_last=6)
    assert "echo" in msgs[1]["content"] and _roles_ok(msgs)


# ---------------------------------------------------------------- helpers / config
def test_parse_verdict():
    assert parse_verdict("APPROVE\nlooks good").ok
    v = parse_verdict("REJECT: debug print left\nremove it")
    assert not v.ok and "debug print" in v.feedback
    assert parse_verdict("¯\\_(ツ)_/¯ no idea").ok  # fail open


def test_example_config_loads(tmp_path):
    from pathlib import Path

    s = load_settings(Path(__file__).parent.parent / "examples" / "config.example.toml")
    assert s.limits.max_steps == 80 and s.small.base_url.startswith("https://")


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        settings_from_dict({"limits": {"max_stepz": 3}})
    with pytest.raises(ValueError, match="top-level"):
        settings_from_dict({"smal": {}})


# ---------------------------------------------------------------- models (local fake server)
class _Handler(BaseHTTPRequestHandler):
    responses: list = []
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.headers.get("Authorization"), body))
        code, payload = _Handler.responses.pop(0)
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    _Handler.responses, _Handler.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def _ok(text):
    return 200, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}


def test_openai_model_parses_reply_cost_and_strips_think(server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sekret")
    cfg = ModelConfig(model="m", base_url=server, api_key_env="TEST_KEY", price_in_per_mtok=2.0, price_out_per_mtok=10.0,
                      extra_body={"foo": 1})
    _Handler.responses = [_ok("<think>hmm</think>```bash\nls\n```")]
    reply = OpenAICompatModel(cfg).query([{"role": "user", "content": "hi"}])
    assert reply.content == "```bash\nls\n```"
    assert reply.cost == pytest.approx(1000 * 2 / 1e6 + 500 * 10 / 1e6)
    auth, body = _Handler.seen[0]
    assert auth == "Bearer sekret" and body["model"] == "m" and body["foo"] == 1


def test_openai_model_retries_then_succeeds(server):
    m = OpenAICompatModel(ModelConfig(model="m", base_url=server, api_key_env=""))
    m.backoff_s = 0
    _Handler.responses = [(503, {"error": "busy"}), (429, {"error": "slow"}), _ok("fine")]
    assert m.query([{"role": "user", "content": "x"}]).content == "fine"


def test_openai_model_client_error_is_not_retried(server):
    m = OpenAICompatModel(ModelConfig(model="m", base_url=server, api_key_env=""))
    m.backoff_s = 0
    _Handler.responses = [(400, {"error": "bad request"}), _ok("never reached")]
    with pytest.raises(ModelError, match="400"):
        m.query([{"role": "user", "content": "x"}])
    assert len(_Handler.seen) == 1


def test_openai_model_gives_up_after_retries(server):
    m = OpenAICompatModel(ModelConfig(model="m", base_url=server, api_key_env=""))
    m.backoff_s = 0
    _Handler.responses = [(500, {})] * 3
    with pytest.raises(ModelError, match="3 attempts"):
        m.query([{"role": "user", "content": "x"}])


def test_scripted_model_exhaustion_and_repeat():
    m = ScriptedModel(["a"])
    m.query([{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError):
        m.query([{"role": "user", "content": "x"}])
    r = ScriptedModel(["a"], repeat_last=True)
    assert [r.query([{"role": "user", "content": "x"}]).content for _ in range(3)] == ["a"] * 3


# ---------------------------------------------------------------- fixes: oscillation noise, cost cap
def test_detector_scratch_file_create_and_delete_is_not_oscillation():
    d = LoopDetector()
    flags = [
        d.observe("ls", "app.py", "EMPTY", "T0"),
        d.observe("echo 'print(1)' > repro.py", "", "WITH_REPRO", "T0"),
        d.observe("python repro.py", "1", "WITH_REPRO", "T0"),
        d.observe("rm repro.py", "", "EMPTY", "T0"),
    ]
    assert flags == [None, None, None, None]


def test_detector_oscillation_in_tracked_files_is_still_caught():
    d = LoopDetector()
    assert d.observe("edit1", "ok", "F1", "A") is None
    assert d.observe("edit2", "ok", "F2", "B") is None
    assert d.observe("edit3", "ok", "F3", "A").startswith("oscillation")


def test_env_tracked_hash_ignores_new_files_but_sees_edits(repo):
    env = LocalEnv(repo)
    h0 = env.git_tracked_diff_hash()
    env.execute("echo 'print(1)' > repro.py")
    env.git_diff()  # registers the new file (intent-to-add), as the agent does every step
    assert env.git_tracked_diff_hash() == h0
    env.execute("sed -i 's/a - b/a + b/' app.py")
    assert env.git_tracked_diff_hash() != h0
    env.execute("git checkout -- app.py")
    assert env.git_tracked_diff_hash() == h0


def test_pricing_warnings():
    from hagent.config import Settings, pricing_warnings

    s = Settings()  # default max_cost=3.0 and no prices
    assert "model(s) small." in pricing_warnings(s, "small")[0]
    assert "model(s) small, big." in pricing_warnings(s, "hybrid")[0]
    s.small.price_in_per_mtok = s.big.price_out_per_mtok = 1.0
    assert pricing_warnings(s, "hybrid") == []
    s.limits.max_cost = 0
    s.small.price_in_per_mtok = 0
    assert pricing_warnings(s, "hybrid") == []  # no cost cap, nothing to warn about


def test_openai_model_estimates_tokens_when_usage_missing(server):
    m = OpenAICompatModel(ModelConfig(model="m", base_url=server, api_key_env=""))
    _Handler.responses = [(200, {"choices": [{"message": {"content": "x" * 400}}]})]
    r = m.query([{"role": "user", "content": "y" * 4000}])
    assert r.input_tokens == 1000 and r.output_tokens == 100
