"""Agent-level tests with scripted models: base loop, hybrid routing, loop ladder, review, explorer."""

import json

import pytest

from conftest import FIX, SUBMIT
from hagent import Agent, HybridAgent, LocalEnv, solve
from hagent.env import SUBMIT_MARKER
from hagent.exceptions import ModelError
from hagent.explorer import ReadOnlyEnv, is_write_command
from hagent.models import ScriptedModel, bash


def roles_alternate(messages):
    roles = [m["role"] for m in messages]
    return roles[0] == "system" and all(roles[i] != roles[i + 1] for i in range(1, len(roles) - 1))


def advisor_or_bash(command="ls", advice="ADVICE: try something different"):
    """Big-model stub: answers advisor/reviewer prompts with text and takeover steps with a command."""

    def fn(messages):
        sysmsg = messages[0]["content"]
        if "senior engineer advising" in sysmsg:
            return advice
        if "strict code reviewer" in sysmsg:
            return "APPROVE"
        return bash(command)

    return fn


# ================================================================ base agent
def test_base_agent_fixes_bug_and_submits(repo, mk_settings):
    model = ScriptedModel([FIX, bash("python -c 'import app; assert app.add(1, 2) == 3'"), SUBMIT])
    r = Agent(model, LocalEnv(repo), mk_settings()).run("fix add()")
    assert r.submitted and r.steps == 3
    assert "+    return a + b" in r.patch and "+    return a + b" in r.detail
    assert roles_alternate(r.messages)


def test_format_error_is_recovered(repo, mk_settings):
    model = ScriptedModel(["I will just think about it.", FIX, SUBMIT])
    r = Agent(model, LocalEnv(repo), mk_settings()).run("fix")
    assert r.submitted
    assert "Format error" in model.calls[1][-1]["content"]
    assert roles_alternate(r.messages)


def test_repeated_format_errors_end_run(repo, mk_settings):
    r = Agent(ScriptedModel(["nope"], repeat_last=True), LocalEnv(repo), mk_settings(max_format_errors=3)).run("x")
    assert r.exit_status == "RepeatedFormatError" and r.steps == 3


def test_format_error_counter_resets_on_good_step(repo, mk_settings):
    script = ["bad", bash("echo ok"), "bad", bash("echo ok"), "bad", SUBMIT]
    r = Agent(ScriptedModel(script), LocalEnv(repo), mk_settings(max_format_errors=2)).run("x")
    assert r.submitted


def test_step_limit(repo, mk_settings):
    r = Agent(ScriptedModel([bash("echo x")], repeat_last=True), LocalEnv(repo), mk_settings(max_steps=5)).run("x")
    assert r.exit_status == "LimitsExceeded" and r.steps == 5


def test_cost_limit_counts_every_call(repo, mk_settings):
    model = ScriptedModel([bash("echo x")], repeat_last=True, cost_per_call=1.0)
    r = Agent(model, LocalEnv(repo), mk_settings(max_cost=2.5)).run("x")
    assert r.exit_status == "LimitsExceeded" and r.steps == 3 and r.total_cost == 3.0


def test_wall_time_limit(repo, mk_settings):
    r = Agent(ScriptedModel([bash("sleep 1.3")], repeat_last=True), LocalEnv(repo), mk_settings(max_wall_s=1)).run("x")
    assert r.exit_status == "TimeExceeded"


def test_model_error_becomes_exit_status_and_patch_is_kept(repo, mk_settings):
    def boom(messages):
        raise ModelError("endpoint down")

    r = Agent(ScriptedModel([FIX, boom]), LocalEnv(repo), mk_settings()).run("x")
    assert r.exit_status == "ModelError" and "endpoint down" in r.detail
    assert "+    return a + b" in r.patch  # work done before the failure is not lost


def test_observation_is_truncated(repo, mk_settings):
    model = ScriptedModel([bash("python -c \"print('z' * 20000)\""), SUBMIT])
    Agent(model, LocalEnv(repo), mk_settings(obs_max_chars=1000)).run("x")
    obs = model.calls[1][-1]["content"]
    assert "elided" in obs and len(obs) < 1400


def test_trajectory_is_saved(repo, mk_settings, tmp_path):
    path = tmp_path / "out" / "traj.json"
    Agent(ScriptedModel([FIX, SUBMIT]), LocalEnv(repo), mk_settings(), trajectory_path=path).run("x")
    data = json.loads(path.read_text())
    assert data["exit_status"] == "Submitted" and data["steps"] == 2 and data["messages"][0]["role"] == "system"


def test_compaction_inside_agent_loop(repo, mk_settings):
    def small(messages):
        if "compress the working history" in messages[0]["content"]:
            return "NOTES: looked around a lot"
        return bash("python -c \"print('o' * 500)\"")

    model = ScriptedModel([small], repeat_last=True)
    agent = Agent(model, LocalEnv(repo), mk_settings(max_steps=25, ctx_char_limit=4000, keep_last_messages=4))
    r = agent.run("x")
    assert agent.n_compactions >= 1 and r.exit_status == "LimitsExceeded"
    assert "NOTES: looked around a lot" in agent.messages[1]["content"]
    assert roles_alternate(agent.messages)
    peak = max(sum(len(m["content"]) for m in call) for call in model.calls)
    assert peak < 4000 + 1500  # the history never grows far beyond the budget


# ================================================================ hybrid agent
def test_hybrid_planning_consult_then_review_approves(repo, mk_settings):
    small = ScriptedModel([FIX, bash("echo two"), bash("echo three"), SUBMIT])
    big = ScriptedModel([advisor_or_bash(advice="ADVICE: check app.py"), advisor_or_bash()])
    s = mk_settings(plan_at_step=2, final_review=True)
    agent = HybridAgent(small, big, LocalEnv(repo), s)
    r = agent.run("fix")
    assert r.submitted
    assert "ADVICE: check app.py" in small.calls[2][-1]["content"]  # delivered before small's 3rd call
    assert agent.big_calls == 2 and r.usage["big"].calls == 2  # one advice + one review
    assert roles_alternate(r.messages)


def test_hybrid_review_rejects_then_accepts(repo, mk_settings):
    small = ScriptedModel([FIX, SUBMIT, bash("git diff --stat"), SUBMIT])
    verdicts = iter(["REJECT: you left no test evidence", "APPROVE"])
    big = ScriptedModel([lambda m: next(verdicts)] * 2)
    agent = HybridAgent(small, big, LocalEnv(repo), mk_settings(final_review=True))
    r = agent.run("fix")
    assert r.submitted and agent.review_rejections == 1
    assert "no test evidence" in small.calls[2][-1]["content"]


def test_hybrid_review_cannot_block_forever(repo, mk_settings):
    small = ScriptedModel([FIX, SUBMIT], repeat_last=True)
    big = ScriptedModel(["REJECT: still not happy"], repeat_last=True)
    agent = HybridAgent(small, big, LocalEnv(repo), mk_settings(final_review=True, max_review_rejections=2))
    r = agent.run("fix")
    assert r.submitted and len(big.calls) == 2  # third submission is accepted without another big call


def test_hybrid_empty_diff_submission_is_bounced_without_big_call(repo, mk_settings):
    small = ScriptedModel([SUBMIT, FIX, SUBMIT])
    big = ScriptedModel([advisor_or_bash()])
    agent = HybridAgent(small, big, LocalEnv(repo), mk_settings(final_review=True))
    r = agent.run("fix")
    assert r.submitted and "empty diff" in small.calls[1][-1]["content"] and len(big.calls) == 1


def test_hybrid_stuck_ladder_always_terminates(repo, mk_settings):
    small = ScriptedModel([bash("ls")], repeat_last=True)
    big = ScriptedModel([advisor_or_bash("ls")], repeat_last=True)
    s = mk_settings(max_steps=100, repeat_threshold=3, max_strikes=5, takeover_steps=2, max_big_calls=10)
    agent = HybridAgent(small, big, LocalEnv(repo), s)
    r = agent.run("do something impossible")

    assert r.exit_status == "Stuck" and agent.strikes == 5
    assert r.steps < 100 and agent.big_calls <= 10
    all_small_text = "\n".join(m["content"] for call in small.calls for m in call)
    assert "[Loop guard]" in all_small_text                      # strike 1: nudge
    assert "ADVICE: try something different" in all_small_text   # strike 2+: big model advice
    assert r.usage["big"].calls > 2                              # advice plus takeover steps
    assert roles_alternate(r.messages)


def test_hybrid_big_budget_exhaustion_still_terminates(repo, mk_settings):
    small = ScriptedModel([bash("ls")], repeat_last=True)
    big = ScriptedModel([advisor_or_bash("ls")], repeat_last=True)
    s = mk_settings(max_steps=200, max_strikes=6, max_big_calls=1, final_review=False)
    agent = HybridAgent(small, big, LocalEnv(repo), s)
    r = agent.run("x")
    assert r.exit_status == "Stuck" and agent.big_calls <= 1
    assert "Loop guard - warning" in "\n".join(m["content"] for m in small.calls[-1])


def test_hybrid_survives_big_model_failure(repo, mk_settings):
    def boom(messages):
        raise ModelError("big model down")

    small = ScriptedModel([bash("ls")], repeat_last=True)
    big = ScriptedModel([boom], repeat_last=True)
    r = HybridAgent(small, big, LocalEnv(repo), mk_settings(max_steps=100, max_strikes=4)).run("x")
    assert r.exit_status == "Stuck"  # no crash; fell back to nudges and ended cleanly


def test_hybrid_progress_resets_strikes(repo, mk_settings):
    """Real progress (a new diff state) between loops must reset the strike counter."""
    small = ScriptedModel([bash("ls")] * 3 + [bash("echo 1 >> notes.txt")] + [bash("ls")] * 3 + [SUBMIT])
    big = ScriptedModel([advisor_or_bash()], repeat_last=True)
    agent = HybridAgent(small, big, LocalEnv(repo), mk_settings(max_strikes=2, repeat_threshold=3))
    r = agent.run("x")
    assert r.submitted  # without the reset, the second loop would have hit max_strikes=2


def test_solve_modes_and_injection(repo, mk_settings):
    s = mk_settings()
    r = solve("fix", repo, s, "small", small=ScriptedModel([FIX, SUBMIT]))
    assert r.submitted and set(r.usage) == {"small"}
    with pytest.raises(ValueError):
        solve("fix", repo, s, "nope")


# ================================================================ explorer
@pytest.mark.parametrize("cmd,blocked", [
    ("rm -rf build", True), ("cp a b", True), ("sed -i 's/a/b/' f", True), ("echo x > f", True),
    ("git checkout -- .", True), ("pip install foo", True), ("touch x", True),
    ("ls -la 2>&1", False), ("grep -rn foo . 2>/dev/null", False), ("sed -n '1,5p' f", False),
    ("git log --oneline | head", False), ("cat f | grep x", False), ("git diff", False),
    (f"echo {SUBMIT_MARKER}\ncat <<'EOF'\na -> b\nEOF", False),
])
def test_explorer_write_filter(cmd, blocked):
    assert is_write_command(cmd) is blocked


def test_read_only_env_blocks_writes(repo):
    env = ReadOnlyEnv(LocalEnv(repo))
    obs = env.execute("rm app.py")
    assert obs.returncode == 1 and "read-only" in obs.output and (repo / "app.py").exists()


def test_hybrid_explorer_returns_short_answer(repo, mk_settings):
    answer = bash(f"echo {SUBMIT_MARKER}\ncat <<'EOF'\napp.py:1 defines add()\nEOF", "Found it.")
    # one shared scripted model: main asks, explorer greps and answers (2 calls), main continues
    small = ScriptedModel([bash('ask_explorer "where is add defined?"'), bash("grep -rn 'def add' ."), answer, FIX, SUBMIT])
    big = ScriptedModel([advisor_or_bash()])
    s = mk_settings(enable_explorer=True)
    agent = HybridAgent(small, big, LocalEnv(repo), s)
    r = agent.run("fix add")
    assert r.submitted
    assert "app.py:1 defines add()" in small.calls[3][-1]["content"]  # main agent saw only the answer
    assert "def add" not in small.calls[3][-1]["content"].split("app.py:1 defines add()")[0]  # not the explorer's grep noise
    assert r.usage["explorer"].calls == 2 and agent.explorer_calls == 1


def test_explorer_budget(repo, mk_settings):
    small = ScriptedModel([bash('ask_explorer "q"')] * 3 + [SUBMIT])
    s = mk_settings(enable_explorer=True, max_explorer_calls=0)
    agent = HybridAgent(small, ScriptedModel([advisor_or_bash()]), LocalEnv(repo), s)
    agent.run("x")
    assert "budget exhausted" in small.calls[1][-1]["content"]


# ================================================================ fixes: oscillation noise, token cap
def test_hybrid_scratch_repro_file_workflow_is_not_flagged(repo, mk_settings):
    small = ScriptedModel([bash("ls"), bash("echo 'print(1)' > repro.py"), bash("python repro.py"), bash("rm repro.py"),
                           bash("echo done"), SUBMIT])
    agent = HybridAgent(small, ScriptedModel([advisor_or_bash()]), LocalEnv(repo), mk_settings())
    r = agent.run("x")
    assert r.submitted and agent.strikes == 0
    assert not any("Loop guard" in m["content"] for call in small.calls for m in call)


def test_token_limit_stops_run(repo, mk_settings):
    model = ScriptedModel([bash("echo x")], repeat_last=True)
    r = Agent(model, LocalEnv(repo), mk_settings(max_steps=100, max_total_tokens=1500)).run("x")
    assert r.exit_status == "LimitsExceeded" and "token limit" in r.detail and r.steps < 100
