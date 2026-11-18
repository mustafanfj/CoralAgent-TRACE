"""Unattended stage runner for the lab PC, steered through control/CONTROL.json in git.

Loop: git pull -> for stages in run_order that are allowed and not finished, run one stage in a fresh
subprocess (so pulled code fixes take effect) -> push reports/results -> sleep. A stage that stopped or
crashed is only retried after its 'token' in CONTROL.json is increased (a deliberate decision).

Start (inside tmux):  .venv/bin/python -m coralagent.daemon
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import config

STATE = config.RUN / "daemon_state.json"
LOG = config.REPO / "logs"


def sh(cmd, cwd=config.REPO, check=False, timeout=600):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=check, timeout=timeout)


def git_sync(message: str, paths=("reports", "results")) -> None:
    sh(["git", "add", "-A", "--"] + [p for p in paths if (config.REPO / p).exists()])
    if sh(["git", "diff", "--cached", "--quiet"]).returncode != 0:
        sh(["git", "commit", "-q", "-m", message])
    for _ in range(2):
        sh(["git", "pull", "--rebase", "--autostash", "-q"])
        if sh(["git", "push", "-q"]).returncode == 0:
            return
        time.sleep(10)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    config.ensure_dirs()
    STATE.write_text(json.dumps(st, indent=2))
    (config.REPORTS / "daemon_state.json").write_text(json.dumps(st, indent=2))


def run_stage(stage: str) -> dict:
    LOG.mkdir(exist_ok=True)
    log_path = LOG / f"{stage}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {stage} =====\n"); f.flush()
        p = subprocess.run([sys.executable, "-m", "coralagent.run", "stage", stage], cwd=str(config.REPO),
                           stdout=f, stderr=subprocess.STDOUT, text=True, env={**__import__("os").environ, "CORAL_AUTOPUSH": "1"})
    summ_path = config.REPORTS / f"{stage}_summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    if p.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        (config.REPORTS / f"ERROR_{stage}.md").write_text(f"# {stage} crashed (exit {p.returncode})\n\n```\n{tail}\n```\n")
        return {"status": "error", "exit": p.returncode}
    if summary.get("stopped"):
        return {"status": "stopped", "reason": summary["stopped"]}
    return {"status": "completed"}


def main() -> None:
    print("daemon started", flush=True)
    while True:
        try:
            sh(["git", "pull", "--rebase", "--autostash", "-q"])
            ctrl = config.load_control()
            st = load_state()
            ran = False
            for stage in ctrl.get("run_order", []):
                c = (ctrl.get("stages") or {}).get(stage) or {}
                s = st.get(stage) or {}
                if not c.get("allow"):
                    break                                   # stages run strictly in order
                if s.get("status") == "completed":
                    continue
                if s.get("status") in {"stopped", "error"} and s.get("token") == c.get("token", 0):
                    break                                   # wait for a deliberate re-allow
                st[stage] = {"status": "running", "token": c.get("token", 0), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_state(st); git_sync(f"run: {stage} started")
                res = run_stage(stage)
                st[stage].update(res); st[stage]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_state(st); git_sync(f"run: {stage} {res['status']}")
                ran = True
                break
            if not ran:
                sh([sys.executable, "-m", "coralagent.run", "export"])
                git_sync("run: heartbeat", paths=("reports",))
        except Exception:
            (config.REPORTS / "ERROR_daemon.md").write_text("```\n" + traceback.format_exc() + "\n```\n")
            try:
                git_sync("run: daemon error", paths=("reports",))
            except Exception:
                pass
        time.sleep(300)


if __name__ == "__main__":
    main()
