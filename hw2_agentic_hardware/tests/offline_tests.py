"""Offline tests, NO language model: a scripted stand-in plays the model behind every role, so the
graph, the LangGraph interrupt/checkpoint/resume mechanics and the tools are tested without an API key.
These are plumbing tests, not the runs the homework asks for (those use run_task1.py / run_task2.py).

    python tests/offline_tests.py
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_v2 import (DecisionAuto, DecisionHITL, build_graph, coding_spec, focusing_spec,  # noqa: E402
                      pending_interrupts, verified)


class Scripted:
    """decisions: list of (Decision, check) where check(prompt) asserts what the orchestrator must have read."""

    def __init__(self, decisions, executor_actions):
        self.decisions, self.actions = list(decisions), list(executor_actions)

    def decide(self, system, prompt, schema):
        dec, check = self.decisions.pop(0)
        if check:
            check(prompt)
        assert isinstance(dec, schema)
        return dec

    def chat(self, role, system, prompt):
        return f"({role}, scripted) answer"

    def tool_loop(self, system, prompt, tools, max_iters=8):
        act = self.actions.pop(0)
        return act({t.name: t.func for t in tools})


def show(t):
    print(f"--- {t['role']}: {t['content'][:220]}".replace("\n", " | "), flush=True)


def test_ambiguous():
    print("\n=== test (a): ambiguous query ===")
    work = tempfile.mkdtemp()
    reply = "the free-run test RMSE of the kept PINN network"

    def must_read_reply(prompt):
        assert reply in prompt, "the orchestrator did not see the human reply"

    be = Scripted([
        (DecisionHITL(next_role="human", stop_kind="ambiguous", reason="'error' has several readings",
                      question="Which error: (1) free-run test RMSE of the kept network, (2) its train RMSE, (3) the Duffing reference?"), None),
        (DecisionHITL(next_role="general", instruction="Explain what the free-run test RMSE of the kept network measures.",
                      reason="The human chose reading (1)."), must_read_reply),
    ], [])
    g = build_graph(be, [coding_spec(work)], work, allow_human=True, max_decisions=2, checkpointer=InMemorySaver(), on_turn=show)
    cfg = {"configurable": {"thread_id": "a"}}
    g.invoke({"dialogue": [{"role": "user", "content": "What is the error of the Silverbox model?", "meta": {}}]}, cfg)
    intr = pending_interrupts(g, cfg)
    assert intr and intr[0].value["kind"] == "ambiguous", "graph did not pause"
    assert g.get_state(cfg).next == ("human",)
    print("PAUSED, question:", intr[0].value["question"])
    g.invoke(Command(resume=reply), cfg)
    d = g.get_state(cfg).values["dialogue"]
    roles = [t["role"] for t in d]
    print("roles:", roles)
    assert roles == ["user", "orchestrator", "human", "orchestrator", "general", "orchestrator"]
    assert d[2]["content"] == reply and d[3]["meta"]["decision"]["next_role"] == "general"
    print("OK (a)")


def test_hard_decision_with_restart():
    print("\n=== test (b): hard decision, refuse then accept, resumed from the sqlite checkpoint ===")
    work = tempfile.mkdtemp()
    db = os.path.join(work, "ckpt.sqlite")
    script = "print('TRAIN_RMSE=0.000409')\nprint('TEST_RMSE=0.000425')\n"

    def run_script(tools):
        tools["coding_tool"]("write", "evaluate.py", script)
        r = tools["coding_tool"]("run", "evaluate.py")
        return f"ran evaluate.py, exit {r['exit_code']}: {r['stdout'].strip()}"

    def must_see_refusal(prompt):
        assert "REFUSED" in prompt and "did NOT run" in prompt

    be = Scripted([
        (DecisionHITL(next_role="human", stop_kind="hard_decision", pending_role="executor",
                      pending_instruction="Write and run evaluate.py that overwrites silverbox/results/kept.json.",
                      question="Run the coding tool and overwrite kept.json?", reason="Running code and overwriting a result."), None),
        (DecisionHITL(next_role="human", stop_kind="hard_decision", pending_role="executor",
                      pending_instruction="Write and run evaluate.py, writing to a NEW file results_v2.json.",
                      question="Run the coding tool, writing a new file instead?", reason="Changed after the refusal."), must_see_refusal),
        (DecisionHITL(next_role="human", stop_kind="hard_decision", pending_role="finish",
                      pending_instruction="Free-run test RMSE = 0.000425 V (TRAIN 0.000409 V).",
                      question="Finish and report TEST_RMSE=0.000425 V?", reason="Ending reports a measurement."), None),
    ], [run_script])

    def build():                                     # a fresh graph object on the same checkpoint = a new process
        return build_graph(be, [coding_spec(work)], work, allow_human=True, max_decisions=8,
                           checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)), on_turn=show)

    cfg = {"configurable": {"thread_id": "b"}}
    build().invoke({"dialogue": [{"role": "user", "content": "Re-evaluate the kept network and save it.", "meta": {}}]}, cfg)
    g = build()                                      # "restart"
    assert pending_interrupts(g, cfg)[0].value["kind"] == "hard_decision"
    g.invoke(Command(resume="no, do not overwrite kept.json"), cfg)
    assert not os.path.exists(os.path.join(work, "evaluate.py")), "refused step ran!"
    d = g.get_state(cfg).values["dialogue"]
    assert d[-1]["meta"].get("refused_record") and "did NOT run" in d[-1]["content"]
    print("refused: no file written, record ->", d[-1]["content"].splitlines()[0][:160])
    g = build()
    g.invoke(Command(resume="yes"), cfg)             # accept: the executor runs the proposed step
    assert os.path.exists(os.path.join(work, "evaluate.py"))
    assert pending_interrupts(g, cfg)[0].value["proposed_step"]["role"] == "finish"
    g.invoke(Command(resume="yes, finish"), cfg)
    d = g.get_state(cfg).values["dialogue"]
    roles = [t["role"] for t in d]
    print("roles:", roles)
    assert roles == ["user", "orchestrator", "human", "orchestrator", "human", "executor", "orchestrator", "human", "orchestrator"]
    assert verified(d)["test"] == 0.000425 and d[-1]["meta"]["stop"] == "finished"
    assert not pending_interrupts(g, cfg) and g.get_state(cfg).next == ()
    print("OK (b)")
    return d


def test_focus_agent():
    print("\n=== test task 2: executor calls the focusing tool, autonomous ===")
    work = tempfile.mkdtemp()

    def two_experiments(tools):
        f = tools["focusing_tool"]
        r1 = f(macropixel=8, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=300, flips=4)
        r2 = f(macropixel=4, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=1500, flips=8)
        bad = f(macropixel=3, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=10, flips=1)
        assert not bad["ok"]
        return f"enhancements {r1['enhancement']:.1f} and {r2['enhancement']:.1f}"

    be = Scripted([
        (DecisionAuto(next_role="scientist", instruction="Plan the focusing settings.", reason="Need a plan."), None),
        (DecisionAuto(next_role="executor", instruction="Run the planned experiments.", reason="Execute."), None),
        (DecisionAuto(next_role="finish", instruction="Kept enhancement reported.", reason="Verified."), None),
    ], [two_experiments])
    tools = [coding_spec(work), focusing_spec(os.path.join(work, "bench"))]
    g = build_graph(be, tools, work, allow_human=False, checkpointer=InMemorySaver(), on_turn=show)
    cfg = {"configurable": {"thread_id": "f"}}
    g.invoke({"dialogue": [{"role": "user", "content": "Focus the beam.", "meta": {}}]}, cfg)
    d = g.get_state(cfg).values["dialogue"]
    lines = open(os.path.join(work, "bench", "experiments.jsonl")).read().splitlines()
    assert len(lines) == 2 and verified(d)["tool"] == "focusing_tool"
    print("experiments:", [round(json.loads(x)["enhancement"], 1) for x in lines], "kept:", verified(d)["enhancement"])
    print("OK (task 2)")


if __name__ == "__main__":
    test_ambiguous()
    test_hard_decision_with_restart()
    test_focus_agent()
    print("\nALL OFFLINE TESTS PASSED")
