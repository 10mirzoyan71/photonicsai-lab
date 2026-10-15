"""Homework 2, Task 2: ask the agent to focus the beam; from then on it works on its own.

    export ANTHROPIC_API_KEY=...
    python run_task2.py                         # request: "Focus the beam."
    python run_task2.py --request "Focus the beam onto the camera centre."

The executor holds the coding tool and the focusing tool (bench.py).  No settings are typed in: the agent
chooses macropixel size, region sizes, iterations, flips... Everything is printed live (good for the screen
recording) and saved in runs_task2/<timestamp>/ (transcript.json, work/bench/experiments.jsonl, experiment_XX.png).
Afterwards:  python render_hw2.py task2 runs_task2/<timestamp> --tex ../report/hw2_task2_run.tex
"""
import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent_v2 import build_graph, make_backend, coding_spec, focusing_spec, verified  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", default="Focus the beam.")
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5"))
    p.add_argument("--max-decisions", type=int, default=8)
    a = p.parse_args()
    run_dir = os.path.join(HERE, "runs_task2", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    work = os.path.join(run_dir, "work")
    os.makedirs(work, exist_ok=True)
    tools = [coding_spec(work), focusing_spec(os.path.join(work, "bench"))]

    def show(t):
        print(f"\n{'=' * 22} {t['role'].upper()} {'=' * 22}\n{t['content'][:3000]}", flush=True)

    graph = build_graph(make_backend(a.model), tools, work, allow_human=False, max_decisions=a.max_decisions,
                        checkpointer=InMemorySaver(), on_turn=show)
    config = {"configurable": {"thread_id": "task2"}, "recursion_limit": 80}
    print(f"USER: {a.request}", flush=True)
    graph.invoke({"dialogue": [{"role": "user", "content": a.request, "meta": {}}]}, config)
    d = graph.get_state(config).values["dialogue"]
    out = {"model": a.model, "request": a.request, "dialogue": d, "verified": verified(d),
           "experiments": tools[1].log, "coding_log": tools[0].log}
    json.dump(out, open(os.path.join(run_dir, "transcript.json"), "w"), indent=1, default=str)
    print("\nKEPT:", verified(d), "\nsaved to", run_dir)


if __name__ == "__main__":
    main()
