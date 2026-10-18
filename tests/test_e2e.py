"""End to end through the real HTTP client: CLI -> TOML config -> OpenAI-compatible API -> agent -> patch."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import FIX, SUBMIT
from hagent import cli

SCRIPTS = {}
SEEN = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        content = SCRIPTS[body["model"]].pop(0)
        raw = json.dumps({"choices": [{"message": {"content": content}}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def api():
    SCRIPTS.clear()
    SEEN.clear()
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_cli_hybrid_end_to_end(repo, tmp_path, api, capsys):
    SCRIPTS["small-m"] = [FIX, SUBMIT]
    SCRIPTS["big-m"] = ["APPROVE\nLooks right."]
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""
[small]
model = "small-m"
base_url = "{api}"
api_key_env = ""
price_in_per_mtok = 1.0

[big]
model = "big-m"
base_url = "{api}"
api_key_env = ""
price_in_per_mtok = 5.0

[limits]
plan_at_step = 0
final_review = true
""")
    traj = tmp_path / "traj.json"
    code = cli.main(["run", "--repo", str(repo), "--task", "add() subtracts instead of adding",
                     "--config", str(cfg), "--trajectory", str(traj)])
    captured = capsys.readouterr()

    assert code == 0
    assert "+    return a + b" in captured.out
    assert "Submitted" in captured.err and "big: 1 calls" in captured.err and "small: 2 calls" in captured.err
    assert [b["model"] for b in SEEN] == ["small-m", "small-m", "big-m"]  # big model called once, at the end
    assert "strict code reviewer" in SEEN[-1]["messages"][0]["content"]
    assert json.loads(traj.read_text())["exit_status"] == "Submitted"
