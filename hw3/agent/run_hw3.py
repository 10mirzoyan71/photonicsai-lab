"""Homework 3: run the agent system on one request, with the camera behind the MCP server.

    python run_hw3.py --thread uc1_exposure --request "Set the exposure so that ... "              # autonomous
    python run_hw3.py --thread uc5_hitl --human --request "..."             # the graph may stop and ask you
    python run_hw3.py --thread uc5_hitl --human --reply "yes"               # resume a paused thread (also later)

Options: --model (default $AGENT_MODEL or anthropic:claude-sonnet-5-5; claude-cli:<model> uses the Claude Code CLI),
--camera sim|hikrobot, --seed (simulated camera), --limits '{"max_exposure_us": 500000}', --no-wait (exit at a pause).
Everything is saved in runs/<thread>/: transcript.json, events.json, work/ (analysis scripts) and work/camera/ (frames,
previews and calls.jsonl, the record of every instrument call). Threads are checkpointed in runs/checkpoints.sqlite.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_v3 import build_graph, make_backend, pending_interrupts, verified  # noqa: E402
from mcp_bridge import MCPBridge  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True)
    p.add_argument("--request")
    p.add_argument("--reply")
    p.add_argument("--human", action="store_true", help="the orchestrator may stop and ask the person")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5"))
    p.add_argument("--camera", default=os.environ.get("CAMERA_KIND", "sim"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limits", default='{"max_exposure_us": 500000, "max_gain_db": 24, "max_frames": 16}')
    p.add_argument("--max-decisions", type=int, default=10)
    p.add_argument("--no-bench", action="store_true", help="do not give the executor the HW2 simulated-bench focusing tool")
    a = p.parse_args()

    runs = os.path.join(HERE, "runs")
    run_dir = os.path.join(runs, a.thread)
    work = os.path.join(run_dir, "work")
    meta_path = os.path.join(run_dir, "run.json")
    if a.request:
        if os.path.exists(run_dir):
            sys.exit(f"thread {a.thread} exists; resume it with --reply or pick another --thread")
        os.makedirs(work)
        json.dump({"thread": a.thread, "model": a.model, "camera": a.camera, "seed": a.seed, "limits": json.loads(a.limits),
                   "human": a.human, "bench": not a.no_bench, "request": a.request},
                  open(meta_path, "w", encoding="utf-8"), indent=1)
    elif not os.path.exists(meta_path):
        sys.exit("unknown thread")
    meta = json.load(open(meta_path, encoding="utf-8"))
    events_path = os.path.join(run_dir, "events.json")
    events = json.load(open(events_path, encoding="utf-8")) if os.path.exists(events_path) else []

    def show(t):
        tag = t["role"].upper() + (f" [{t['meta']['hw']}]" if t.get("meta", {}).get("hw") else "")
        print(f"\n{'=' * 18} {tag} {'=' * 18}\n{t['content'][:2500]}", flush=True)

    bridge = MCPBridge(os.path.join(work, "camera"), kind=meta["camera"], seed=meta["seed"], limits=meta["limits"])
    conn = sqlite3.connect(os.path.join(runs, "checkpoints.sqlite"), check_same_thread=False)
    graph, coding = build_graph(make_backend(a.model), bridge, work, allow_human=meta["human"],
                                max_decisions=a.max_decisions, checkpointer=SqliteSaver(conn), on_turn=show,
                                bench_dir=os.path.join(work, "bench") if meta.get("bench", True) else None)
    config = {"configurable": {"thread_id": a.thread}, "recursion_limit": 200}

    def save():
        d = graph.get_state(config).values.get("dialogue", [])
        json.dump({**meta, "model": a.model, "dialogue": d, "events": events, "verified": verified(d)},
                  open(os.path.join(run_dir, "transcript.json"), "w", encoding="utf-8"), indent=1, default=str)
        json.dump(events, open(events_path, "w", encoding="utf-8"), indent=1, default=str)

    try:
        if a.request:
            events.append({"type": "request", "text": a.request, "time": time.time()})
            print(f"USER: {a.request}", flush=True)
            graph.invoke({"dialogue": [{"role": "user", "content": a.request, "meta": {}}]}, config)
        else:
            if not pending_interrupts(graph, config):
                sys.exit("this thread is not paused")
            events.append({"type": "reply", "text": a.reply, "time": time.time()})
            graph.invoke(Command(resume=a.reply), config)
        while True:
            save()
            intr = pending_interrupts(graph, config)
            if not intr:
                break
            q = intr[0].value
            events.append({"type": "interrupt", "payload": q, "time": time.time()})
            save()
            print("\n" + "#" * 70 + f"\n# GRAPH PAUSED on thread '{a.thread}' ({q['kind']})")
            if q.get("proposed_step"):
                print(f"# proposed step: {q['proposed_step']['role']}: {q['proposed_step']['instruction']}")
            print(f"# question: {q['question']}\n" + "#" * 70, flush=True)
            if a.no_wait:
                print(f"\nPaused. Resume with:  python run_hw3.py --thread {a.thread} --reply \"<your reply>\"")
                return
            reply = input("your reply > ").strip()
            events.append({"type": "reply", "text": reply, "time": time.time()})
            graph.invoke(Command(resume=reply), config)
        d = graph.get_state(config).values["dialogue"]
        print("\nThread finished. VERIFIED:", json.dumps(verified(d), default=str)[:600])
        print("Saved in", run_dir)
    finally:
        save()
        bridge.close()


if __name__ == "__main__":
    main()
