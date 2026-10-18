"""Offline demo: runs the hybrid agent on a throwaway repo with scripted models (no API key needed).

    python examples/demo_offline.py

It shows the three behaviours that matter: the small model doing the work, the loop guard firing
when it repeats itself, and the big model advising and then reviewing the final patch.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hagent import Limits, Settings, solve  # noqa: E402
from hagent.env import SUBMIT_MARKER  # noqa: E402
from hagent.models import ScriptedModel, bash  # noqa: E402

repo = Path(tempfile.mkdtemp(prefix="hagent-demo-"))
run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)  # noqa: E731
run("git", "init", "-q")
(repo / "app.py").write_text("def add(a, b):\n    return a - b\n")
run("git", "add", "-A")
run("git", "-c", "user.name=demo", "-c", "user.email=d@d", "commit", "-q", "-m", "init")

# The small model first loops on `ls` (a typical weak-model failure), then follows the big model's advice.
small = ScriptedModel([
    bash("ls"), bash("ls"), bash("ls"),                       # loop -> strike 1 (nudge)
    bash("grep -n 'return' app.py"),
    bash("sed -i 's/a - b/a + b/' app.py"),
    bash("python -c 'import app; assert app.add(2, 3) == 5'"),
    bash(f"echo {SUBMIT_MARKER} && git add -A && git diff --cached", "Verified, submitting."),
])


def big_fn(messages):
    system = messages[0]["content"]
    return "APPROVE\nMinimal and correct." if "strict code reviewer" in system else "Look at the operator in add()."


big = ScriptedModel([big_fn] * 3)
settings = Settings(limits=Limits(plan_at_step=5, final_review=True, repeat_threshold=3, max_cost=0))
result = solve("add() returns the wrong value", repo, settings, "hybrid", small=small, big=big)

print(f"status : {result.exit_status}")
print(f"steps  : {result.steps}   small calls: {result.usage['small'].calls}   big calls: {result.usage['big'].calls}")
print("patch  :\n" + result.patch)
