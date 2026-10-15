"""The coding tool. Only the executor receives it.

It does two things:
  * write  - write a Python file into the working directory chosen for this run
  * run    - run that file with a time limit of five minutes and return the
             exit code, the standard output and the standard error.

Every call is logged, so the graph can tell which numbers were really printed
by a script that ran (see `parse_results`).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

TIME_LIMIT_S = 300
MAX_CHARS = 6000          # stdout / stderr returned to the model are truncated to the tail

RESULT_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]{2,})\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M)


def parse_results(stdout: str) -> dict:
    """Numbers the script itself printed on lines 'NAME=value', e.g. 'READ_NOISE_E=5.97'. Last occurrence wins."""
    return {k: float(v) for k, v in RESULT_RE.findall(stdout or "")}


def _tail(s: str, n: int = MAX_CHARS) -> str:
    s = s or ""
    return s if len(s) <= n else "...[truncated]...\n" + s[-n:]


@dataclass
class CodingTool:
    workdir: str
    max_writes_per_turn: int = 1
    max_runs_per_turn: int = 1
    log: list = field(default_factory=list)       # every call, across the whole run
    _writes: int = 0
    _runs: int = 0

    def new_turn(self):
        self._writes = 0
        self._runs = 0

    # -- action: write -------------------------------------------------------
    def write(self, filename: str, code: str) -> dict:
        name = os.path.basename(filename or "")
        if not name.endswith(".py") or name != filename:
            return self._record("write", filename, {"ok": False, "error": "filename must be a bare *.py name, no directories"})
        path = os.path.join(self.workdir, name)
        if os.path.exists(path):
            return self._record("write", name, {"ok": False, "error": f"{name} already exists; every implementation goes into a NEW file (e.g. a new version suffix)"})
        if self._writes >= self.max_writes_per_turn:
            return self._record("write", name, {"ok": False, "error": "write budget for this turn is used; run the file and report"})
        if not code or not code.strip():
            return self._record("write", name, {"ok": False, "error": "empty code"})
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        self._writes += 1
        return self._record("write", name, {"ok": True, "path": path, "bytes": len(code.encode())})

    # -- action: run ---------------------------------------------------------
    def run(self, filename: str) -> dict:
        name = os.path.basename(filename or "")
        path = os.path.join(self.workdir, name)
        if not os.path.exists(path):
            return self._record("run", name, {"ok": False, "error": f"{name} does not exist in the working directory"})
        if self._runs >= self.max_runs_per_turn:
            return self._record("run", name, {"ok": False, "error": "run budget for this turn is used; report the result"})
        self._runs += 1
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, name], cwd=self.workdir, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=TIME_LIMIT_S, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
            res = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "timed_out": False}
        except subprocess.TimeoutExpired as e:
            res = {"exit_code": None, "timed_out": True,
                   "stdout": (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or "",
                   "stderr": ((e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or "")
                   + f"\nTIMEOUT: killed after {TIME_LIMIT_S} s"}
        res["seconds"] = round(time.time() - t0, 1)
        res["parsed"] = parse_results(res["stdout"]) if res["exit_code"] == 0 else {}
        res["ok"] = res["exit_code"] == 0
        with open(path[:-3] + ".log", "w", encoding="utf-8") as fh:                      # full output kept on disk
            fh.write(f"exit_code={res['exit_code']} seconds={res['seconds']}\n--- stdout ---\n{res['stdout']}\n--- stderr ---\n{res['stderr']}")
        return self._record("run", name, res)

    # -- single entry point used as the LLM tool -------------------------------
    def __call__(self, action: str, filename: str, code: str = "") -> dict:
        if action == "write":
            return self.write(filename, code)
        if action == "run":
            return self.run(filename)
        return {"ok": False, "error": "action must be 'write' or 'run'"}

    def _record(self, action, filename, result):
        self.log.append({"action": action, "filename": filename, "time": time.time(), **result})
        shown = dict(result)
        for k in ("stdout", "stderr"):
            if k in shown:
                shown[k] = _tail(shown[k])
        return shown
