import subprocess

import pytest

from hagent.config import Limits, Settings
from hagent.env import SUBMIT_MARKER
from hagent.models import bash

SUBMIT = bash(f"echo {SUBMIT_MARKER} && git add -A && git diff --cached", "Done, submitting.")
FIX = bash("sed -i 's/a - b/a + b/' app.py && git diff --stat", "Fixing the operator.")


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return tmp_path


@pytest.fixture
def mk_settings():
    """Settings factory with test-friendly defaults; override any Limits field by keyword."""

    def make(**limits):
        base = dict(
            max_steps=60, max_cost=0.0, max_wall_s=0, cmd_timeout_s=10, plan_at_step=0,
            final_review=False, ctx_char_limit=0, enable_explorer=False,
        )
        base.update(limits)
        return Settings(limits=Limits(**base))

    return make
