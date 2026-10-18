import subprocess

from hagent import cli
from hagent.agent import RunResult


def _fake_result(status="Submitted", patch="diff --git a/x b/x\n"):
    return RunResult(exit_status=status, detail="", patch=patch, steps=3, elapsed_s=0.1, total_cost=0.0)


def test_cli_prints_patch_and_returns_zero(repo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "solve", lambda *a, **k: _fake_result())
    assert cli.main(["run", "--repo", str(repo), "--task", "fix it"]) == 0
    captured = capsys.readouterr()
    assert "diff --git" in captured.out and "Submitted" in captured.err


def test_cli_nonzero_when_not_submitted(repo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "solve", lambda *a, **k: _fake_result("Stuck", ""))
    assert cli.main(["run", "--repo", str(repo), "--task", "x"]) == 1


def test_cli_missing_repo(tmp_path, capsys):
    assert cli.main(["run", "--repo", str(tmp_path / "nope"), "--task", "x"]) == 2


def test_cli_init_git_and_patch_out(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("hi")
    monkeypatch.setattr(cli, "solve", lambda *a, **k: _fake_result())
    out = tmp_path / "p.diff"
    assert cli.main(["run", "--repo", str(tmp_path), "--task", "x", "--init-git", "--patch-out", str(out)]) == 0
    assert (tmp_path / ".git").exists() and out.read_text().startswith("diff --git")
    log = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "initial" in log


def test_cli_missing_model_gives_clean_error(repo, capsys):
    # default Settings has no model ids -> a clear error, not a traceback
    assert cli.main(["run", "--repo", str(repo), "--task", "x"]) == 2
    assert "no `model` set" in capsys.readouterr().err
