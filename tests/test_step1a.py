"""Step 1A: real token accounting, reviewer pass probability, task templates, custom commands,
and the env wiring in solve()."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import FIX, SUBMIT
from hagent import Agent, HybridAgent, LocalEnv, solve
from hagent.config import CustomCommand, Limits, ModelConfig, Settings, settings_from_dict
from hagent.exceptions import EnvError
from hagent.helpers import parse_pass_probability, parse_verdict
from hagent.models import USAGE_API, USAGE_ESTIMATED, USAGE_PARTIAL, OpenAICompatModel, ScriptedModel, bash, read_usage
from hagent.prompts import build_system_prompt
from hagent.task_template import DEFAULT_TEMPLATE, IncidentContext, TaskTemplate, load_task_template

MSGS = [{"role": "user", "content": "y" * 4000}]  # estimate: 1000 input tokens


# ================================================================ 1. token accounting
@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": 7, "completion_tokens": 3}, (7, 3, USAGE_API)),
    ({"input_tokens": 7, "output_tokens": 3}, (7, 3, USAGE_API)),
    ({"prompt_tokens": 7, "total_tokens": 10}, (7, 3, USAGE_API)),
    ({"completion_tokens": 3, "total_tokens": 10}, (7, 3, USAGE_API)),
    ({"prompt_tokens": 7}, (7, 100, USAGE_PARTIAL)),
    ({"prompt_tokens": 0, "completion_tokens": 0}, (1000, 100, USAGE_ESTIMATED)),
    ({"prompt_tokens": "7", "completion_tokens": None}, (1000, 100, USAGE_ESTIMATED)),
    (None, (1000, 100, USAGE_ESTIMATED)),
])
def test_read_usage(usage, expected):
    payload = {} if usage is None else {"usage": usage}
    assert read_usage(payload, MSGS, "x" * 400) == expected


class _H(BaseHTTPRequestHandler):
    payloads: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        raw = json.dumps(_H.payloads.pop(0) if len(_H.payloads) > 1 else _H.payloads[0]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def api():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_cost_cap_uses_provider_token_counts(api, repo, mk_settings):
    """Tiny messages but a large reported usage: the cap must trip on the reported tokens."""
    _H.payloads = [{"choices": [{"message": {"content": bash("ls")}}],
                    "usage": {"prompt_tokens": 400_000, "completion_tokens": 100_000}}]
    model = OpenAICompatModel(ModelConfig(model="m", base_url=api, api_key_env="",
                                          price_in_per_mtok=1.0, price_out_per_mtok=2.0))
    # each call: 0.4 + 0.2 = $0.60 -> cap of $1.00 stops the run after the 2nd call
    result = Agent(model, LocalEnv(repo), mk_settings(max_cost=1.0, max_total_tokens=0)).run("x")
    assert result.exit_status == "LimitsExceeded" and "cost" in result.detail
    u = result.usage["small"]
    assert u.calls == 2 and u.input_tokens == 800_000 and u.output_tokens == 200_000 and u.estimated_calls == 0
    assert result.total_cost == pytest.approx(1.2)


def test_estimated_calls_are_counted(api, repo, mk_settings):
    _H.payloads = [{"choices": [{"message": {"content": FIX}}]},
                   {"choices": [{"message": {"content": SUBMIT}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}}]
    model = OpenAICompatModel(ModelConfig(model="m", base_url=api, api_key_env=""))
    result = Agent(model, LocalEnv(repo), mk_settings()).run("x")
    assert result.submitted and result.usage["small"].estimated_calls == 1


def test_scripted_model_can_report_api_tokens():
    r = ScriptedModel(["a"], tokens_per_call=(11, 22)).query([{"role": "user", "content": "x"}])
    assert (r.input_tokens, r.output_tokens, r.usage_source) == (11, 22, USAGE_API)
    assert ScriptedModel(["a"]).query([{"role": "user", "content": "x"}]).usage_source == USAGE_ESTIMATED


# ================================================================ 3. reviewer pass probability
@pytest.mark.parametrize("text, prob", [
    ("APPROVE\nPASS_PROBABILITY: 0.8", 0.8),
    ("APPROVE\npass probability = 75%", 0.75),
    ("REJECT\nPASS_PROBABILITY: 42", 0.42),
    ("APPROVE\nPASS_PROBABILITY: 7", 0.07),
    ("APPROVE\nPASS_PROBABILITY: 250", 1.0),
    ("APPROVE\nPASS_PROBABILITY: high", None),
    ("APPROVE", None),
    ("", None),
])
def test_parse_pass_probability(text, prob):
    got = parse_pass_probability(text)
    assert got == (pytest.approx(prob) if prob is not None else None)


def test_pass_probability_never_changes_the_verdict():
    v = parse_verdict("REJECT: wrong file\nPASS_PROBABILITY: 0.99\nfix utils.py")
    assert not v.ok and v.pass_probability == pytest.approx(0.99)
    assert "PASS_PROBABILITY" not in v.feedback and "wrong file" in v.feedback and "utils.py" in v.feedback
    v = parse_verdict("APPROVE\nPASS_PROBABILITY: 0.0")
    assert v.ok and v.pass_probability == 0.0
    v = parse_verdict("PASS_PROBABILITY: 0.1\nREJECT")  # unparseable first line still fails open
    assert v.ok and v.pass_probability == pytest.approx(0.1)
    assert parse_verdict("garbage reply").ok


def _reviewer(*replies):
    it = iter(replies)
    return ScriptedModel([lambda m: next(it)] * len(replies))


def test_hybrid_logs_pass_probability_in_result_usage_and_trajectory(repo, mk_settings, tmp_path):
    path = tmp_path / "t.json"
    small = ScriptedModel([FIX, SUBMIT, SUBMIT])
    big = _reviewer("REJECT\nPASS_PROBABILITY: 0.9\nadd a test", "APPROVE\nPASS_PROBABILITY: 0.3")
    result = HybridAgent(small, big, LocalEnv(repo), mk_settings(final_review=True),
                         trajectory_path=path).run("x")
    assert result.submitted  # outcome follows APPROVE/REJECT, not the probabilities
    assert [r["verdict"] for r in result.reviews] == ["REJECT", "APPROVE"]
    assert [r["pass_probability"] for r in result.reviews] == [0.9, 0.3]
    assert result.usage["big"].pass_probabilities == [0.9, 0.3]
    data = json.loads(path.read_text())
    assert [r["pass_probability"] for r in data["reviews"]] == [0.9, 0.3]
    assert data["usage"]["big"]["pass_probabilities"] == [0.9, 0.3]


def test_unparseable_review_with_probability_still_approves(repo, mk_settings):
    big = _reviewer("I think it is fine?? PASS_PROBABILITY: n/a")
    result = HybridAgent(ScriptedModel([FIX, SUBMIT]), big, LocalEnv(repo), mk_settings(final_review=True)).run("x")
    assert result.submitted and result.reviews[0]["pass_probability"] is None


def test_review_cap_still_holds_with_probabilities(repo, mk_settings):
    big = ScriptedModel(["REJECT\nPASS_PROBABILITY: 0.01\nno"], repeat_last=True)
    small = ScriptedModel([FIX] + [SUBMIT] * 5)
    result = HybridAgent(small, big, LocalEnv(repo), mk_settings(final_review=True, max_review_rejections=2)).run("x")
    assert result.submitted and len(result.reviews) == 2


def test_reviewer_prompt_asks_for_probability():
    from hagent.prompts import REVIEWER_SYSTEM
    assert "PASS_PROBABILITY" in REVIEWER_SYSTEM and REVIEWER_SYSTEM.index("APPROVE") < REVIEWER_SYSTEM.index("PASS_")


# ================================================================ 4. task template
def test_default_template_frames_verdict_as_hypothesis():
    text = load_task_template().render(IncidentContext(task="Gripper never closes.", router_verdict="code bug"))
    assert text.startswith("Gripper never closes.")
    assert "hypothesis" in text.lower() and "verify" in text.lower() and "code bug" in text
    assert "(not provided)" in text  # empty fields are explicit, not silently blank


def test_template_file_roundtrip(tmp_path):
    f = tmp_path / "t.md"
    f.write_text("T: <<TASK>>\nV: <<ROUTER_VERDICT>>\nE: <<LOCATION_EVIDENCE>>\nR: <<REPRO_COMMAND>>\n"
                 "S: <<ROLLOUT_SUMMARY>>\nP: <<REPO_PATH>>\n")
    ctx = IncidentContext("fail", "code bug", "wrapper.py:40", "run_rung1 --seeds 10", "12% success", "/repo/ctrl")
    assert load_task_template(f).render(ctx) == (
        "T: fail\nV: code bug\nE: wrapper.py:40\nR: run_rung1 --seeds 10\nS: 12% success\nP: /repo/ctrl\n")


def test_template_values_are_not_resubstituted():
    out = TaskTemplate("<<TASK>> | <<REPO_PATH>>").render(IncidentContext("log says <<REPO_PATH>>", repo_path="/r"))
    assert out == "log says <<REPO_PATH>> | /r\n"


@pytest.mark.parametrize("text, msg", [("<<TASK>> <<SECRET_ANSWER>>", "unknown"), ("no task here", "TASK")])
def test_template_validation(text, msg):
    with pytest.raises(ValueError, match=msg):
        TaskTemplate(text)


def test_template_requires_task():
    with pytest.raises(ValueError):
        TaskTemplate(DEFAULT_TEMPLATE).render(IncidentContext(task="  "))


def test_example_template_file_is_valid():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "examples" / "task_template.md"
    assert load_task_template(path).text == DEFAULT_TEMPLATE


# ================================================================ 5. custom commands
CMD = CustomCommand("run_rung1", "/opt/run_rung1", "Run staged validation. Usage: run_rung1 --seeds N", 5000)


def test_commands_config_parsing():
    s = settings_from_dict({"commands": [{"name": "run_rung1", "path": "/x", "description": "d", "max_output_chars": 9}]})
    assert s.commands == [CustomCommand("run_rung1", "/x", "d", 9)]
    with pytest.raises(ValueError, match="duplicate"):
        settings_from_dict({"commands": [{"name": "a", "path": "/x", "description": "d"}] * 2})
    for bad in ({"name": "bad name", "path": "/x", "description": "d"},
                {"name": "a", "path": "", "description": "d"},
                {"name": "a", "path": "/x", "description": " "},
                {"name": "a", "path": "/x", "description": "d", "max_output_chars": -1},
                {"name": "a", "path": "/x", "description": "d", "oops": 1}):
        with pytest.raises((ValueError, TypeError)):
            settings_from_dict({"commands": [bad]})
    with pytest.raises(ValueError, match="backend"):
        settings_from_dict({"env": {"backend": "vm"}})


def test_system_prompt_lists_commands_and_is_unchanged_without_them():
    assert build_system_prompt("/r", commands=()) == build_system_prompt("/r")
    assert "Extra commands" not in build_system_prompt("/r")
    p = build_system_prompt("/r", commands=[CMD])
    assert "`run_rung1`: Run staged validation. Usage: run_rung1 --seeds N" in p


def test_agent_system_prompt_includes_commands(repo):
    s = Settings(commands=[CMD])
    assert "run_rung1" in Agent(ScriptedModel([]), LocalEnv(repo), s).system_prompt
    assert "run_rung1" in HybridAgent(ScriptedModel([]), ScriptedModel([]), LocalEnv(repo), s).system_prompt


@pytest.mark.parametrize("command, limit", [
    ("run_rung1 --seeds 10", 5000),
    ("cd /x && run_rung1 | tail -5", 5000),
    ("echo run_rung1_extra", 3000),
    ("grep -rn foo .", 3000),
])
def test_observation_limit_per_command(repo, command, limit):
    s = Settings(limits=Limits(obs_max_chars=3000), commands=[CMD, CustomCommand("other", "/y", "d", 0)])
    assert Agent(ScriptedModel([]), LocalEnv(repo), s).observation_limit(command) == limit


def test_custom_command_runs_with_its_own_truncation(repo, mk_settings, tmp_path_factory):
    tool = tmp_path_factory.mktemp("tools") / "rung1"
    tool.write_text("#!/bin/bash\nfor ((i=0;i<400;i++)); do echo \"seed $i ok\"; done\necho PASS_RATE=0.9\n")
    tool.chmod(0o755)
    s = mk_settings(obs_max_chars=200)
    s.commands = [CustomCommand("run_rung1", str(tool), "validate", 8000)]
    small = ScriptedModel([bash("run_rung1 --seeds 400"), bash("cat app.py; for ((i=0;i<400;i++)); do echo x; done"),
                           FIX, SUBMIT])
    result = solve("x", repo, s, "small", small=small)
    assert result.submitted
    rung_obs, plain_obs = small.calls[1][-1]["content"], small.calls[2][-1]["content"]
    assert "PASS_RATE=0.9" in rung_obs and "seed 0 ok" in rung_obs and "elided" not in rung_obs
    assert len(plain_obs) < 400  # ordinary commands keep obs_max_chars


def test_solve_rejects_missing_command_path(repo, mk_settings):
    s = mk_settings()
    s.commands = [CustomCommand("run_rung1", "/nonexistent/rung1", "x")]
    with pytest.raises(EnvError):
        solve("x", repo, s, "small", small=ScriptedModel([SUBMIT]))


# ================================================================ env wiring in solve()
def test_solve_uses_injected_env_and_leaves_it_open(repo, mk_settings):
    class Spy(LocalEnv):
        closed = False

        def close(self):
            Spy.closed = True
            super().close()

    env = Spy(repo)
    result = solve("x", "/ignored", mk_settings(), "small", small=ScriptedModel([FIX, SUBMIT]), env=env)
    assert result.submitted and "a + b" in result.patch and not Spy.closed


def test_solve_docker_backend_without_image_fails_cleanly(repo, mk_settings):
    s = mk_settings()
    s.env.backend = "docker"
    with pytest.raises(EnvError, match="image"):
        solve("x", repo, s, "small", small=ScriptedModel([SUBMIT]))


def test_solve_end_to_end_in_docker(repo, mk_settings):
    from test_env_parity import DOCKER_OK, IMAGE, RUNTIME
    if not DOCKER_OK:
        pytest.skip("docker or test image not available")
    s = mk_settings()
    s.env.backend, s.env.image, s.env.runtime, s.env.cpus, s.env.memory = "docker", IMAGE, RUNTIME, 1.0, "512m"
    big = ScriptedModel(["APPROVE\nPASS_PROBABILITY: 0.7"], repeat_last=True)
    s.limits.final_review = True
    result = solve("fix add", repo, s, "hybrid", small=ScriptedModel([FIX, SUBMIT]), big=big)
    assert result.submitted and "+    return a + b" in result.patch
    assert (repo / "app.py").read_text() == "def add(a, b):\n    return a - b\n"  # original untouched
    assert result.reviews[0]["pass_probability"] == 0.7


def test_system_prompt_workflow_rules():
    p = build_system_prompt("/r")
    for needle in ("Reproduce first", "hypotheses", "Make only the change the task asks for", "Never run pip or conda install"):
        assert needle in p
    assert p.count("```bash") >= 2 and "<<" not in p.replace("<<'PY'", "").replace("<<'EOF'", "")
