"""Offline wiring test: NO language model is called.

A scripted stand-in replaces the model behind each role so that the graph, the
routing, the 8-decision budget, the 'finish only after a verified RMSE' rule and
the coding tool (real file writes, real runs under the 5-minute limit) can be
tested without an API key.  This is a test of the plumbing, not the Task 6
validation run - that one is run_task5.py with a real model.

    python tests/offline_smoke_test.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agents import Decision, build_graph, verified_rmse   # noqa: E402
from coding_tool import CodingTool                          # noqa: E402

BROKEN = "import numpy as np\nimport scipy.io as sio\nd = sio.loadmat('SNLS80mV.mat')\nu = d['V3']\n"
with open(os.path.join(HERE, "stub_solution.py")) as fh:
    WORKING = fh.read()


class ScriptedBackend:
    def __init__(self):
        self.decisions = [
            Decision(next_role="scientist", instruction="Write the ODE, the network and a numbered plan.", reason="A plan is needed first."),
            Decision(next_role="general", instruction="What are V1 and V2 in SNLS80mV.mat?", reason="Clarify the record."),
            Decision(next_role="finish", instruction="done", reason="(deliberately invalid: nothing ran yet)"),
            Decision(next_role="executor", instruction="Implement the plan and run it.", reason="The plan must be executed."),
            Decision(next_role="executor", instruction="Repair the script from the traceback.", reason="The run failed."),
            Decision(next_role="finish", instruction="Report the verified RMSE.", reason="A verified RMSE exists."),
        ]
        self.exec_calls = 0
        self.seen_prompts = []

    def decide(self, system, prompt):
        self.seen_prompts.append(prompt)
        return self.decisions.pop(0)

    def chat(self, role, system, prompt):
        return {"scientist": "1. y'' + a y' + b y + c y^3 = g u ... (scripted plan)",
                "general": "V1 is the input voltage, V2 the output voltage (scripted)."}[role]

    def tool_loop(self, system, prompt, tool, max_iters=6):
        self.exec_calls += 1
        name, code = ("solution_v1.py", BROKEN) if self.exec_calls == 1 else ("solution_v2.py", WORKING)
        w = tool("write", name, code)
        second = tool("write", "extra.py", "print(1)")          # must be refused: one write per turn
        assert not second["ok"], second
        r = tool("run", name)
        if r["exit_code"] == 0:
            return f"Wrote and ran {name}; exit 0.\n" + "\n".join(l for l in r["stdout"].splitlines() if "RMSE" in l)
        return f"Wrote and ran {name}; exit {r['exit_code']}. Traceback:\n{r['stderr'][-800:]}"


def main():
    work = tempfile.mkdtemp(prefix="agent_smoke_")
    shutil.copy(os.path.join(HERE, "..", "..", "data", "SNLS80mV.mat"), work)
    tool = CodingTool(workdir=work)
    backend = ScriptedBackend()
    graph = build_graph(backend, tool, on_turn=lambda t: print(f"--- {t['role']}: {t['content'][:400]}\n", flush=True))
    state = graph.invoke({"dialogue": [{"role": "user", "content": "Solve Task 5.", "meta": {}}]},
                         config={"recursion_limit": 60})
    d = state["dialogue"]
    roles = [t["role"] for t in d]
    print("roles:", roles)
    rec = verified_rmse(d)
    # checks
    assert roles[0] == "user" and roles[-1] == "orchestrator"
    assert d[-1]["meta"]["decision"]["next_role"] == "finish"
    guard = [t for t in d if t["role"] == "orchestrator" and t["meta"].get("guard_notes")]
    assert guard and guard[0]["meta"]["decision"]["next_role"] == "executor", "premature finish was not blocked"
    ex = [t for t in d if t["role"] == "executor"]
    assert ex[0]["meta"]["verified"] is None and "exit code 1" in ex[0]["content"]
    assert rec is not None and rec["script"] == "solution_v2.py"
    print("VERIFIED:", rec)
    out = {"model": "scripted stand-in (offline smoke test, no LLM)", "dialogue": d, "tool_log": tool.log, "verified": rec}
    os.makedirs(os.path.join(HERE, "smoke_output"), exist_ok=True)
    with open(os.path.join(HERE, "smoke_output", "transcript.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for f in ("test_segment.png", "solution_v1.log", "solution_v2.log"):
        if os.path.exists(os.path.join(work, f)):
            shutil.copy(os.path.join(work, f), os.path.join(HERE, "smoke_output", f))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
