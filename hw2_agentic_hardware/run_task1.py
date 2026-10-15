"""Homework 2, Task 1: run the human-in-the-loop system on one request.

    export ANTHROPIC_API_KEY=...
    python run_task1.py --thread run_a --request "How good is the Silverbox model?"
        -> when the graph pauses it prints the question and waits for your reply (typed in the terminal),
           then resumes the SAME thread with Command(resume=<your reply>).

    python run_task1.py --thread run_b --request "..." --no-wait     # stop at the first pause and exit
    python run_task1.py --thread run_b --reply "no"                  # later: resume that paused thread

The graph is compiled with a SqliteSaver checkpointer (runs/checkpoints.sqlite); the thread id is --thread, so a
paused thread survives the process and is resumed from its checkpoint.  Everything is saved in runs/<thread>/.
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_v2 import build_graph, make_backend, coding_spec, pending_interrupts  # noqa: E402

ROOT = os.path.join(HERE, "..")
CONTEXT = ("(Context: the working directory holds my Homework 1 Silverbox work: the record data/SNLS80mV.mat, the code "
           "silverbox/silverbox_pinn.py (load_excerpt, load_model, free_run, half_sample_input), and its results in "
           "silverbox/results/: A2_h5.pt + A2_h5.json = the kept PINN network, R_duffing.pt + R_duffing.json = the "
           "4-coefficient Duffing reference, kept.json, summary.md = all 26 runs with free-run train/test RMSE.)")


def prepare_workdir(work):
    os.makedirs(os.path.join(work, "data"), exist_ok=True)
    os.makedirs(os.path.join(work, "silverbox", "results"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "data", "SNLS80mV.mat"), os.path.join(work, "data"))
    t5 = os.path.join(ROOT, "task5_pinn_silverbox")
    shutil.copy(os.path.join(t5, "silverbox_pinn.py"), os.path.join(work, "silverbox"))
    for f in ("A2_h5.pt", "A2_h5.json", "R_duffing.pt", "R_duffing.json", "kept.json", "summary.md"):
        shutil.copy(os.path.join(t5, "results", f), os.path.join(work, "silverbox", "results"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True, help="thread id (also the run folder name)")
    p.add_argument("--request", help="start a new thread with this request")
    p.add_argument("--reply", help="resume a paused thread with this reply")
    p.add_argument("--no-wait", action="store_true", help="exit at the first pause instead of asking in the terminal")
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5"))
    p.add_argument("--max-decisions", type=int, default=8)
    a = p.parse_args()

    runs = os.path.join(HERE, "runs")
    run_dir = os.path.join(runs, a.thread)
    work = os.path.join(run_dir, "work")
    os.makedirs(runs, exist_ok=True)
    if a.request:
        if os.path.exists(run_dir):
            sys.exit(f"thread {a.thread} exists; resume it with --reply or pick another --thread")
        prepare_workdir(work)
    elif not os.path.exists(run_dir):
        sys.exit("unknown thread")

    coding = coding_spec(work)
    log_path = os.path.join(run_dir, "tool_log.json")
    if os.path.exists(log_path):                                  # tool calls of earlier segments of this thread
        coding.log.extend(json.load(open(log_path)))
    events_path = os.path.join(run_dir, "events.json")
    events = json.load(open(events_path)) if os.path.exists(events_path) else []

    def show(t):
        print(f"\n{'=' * 22} {t['role'].upper()} {'=' * 22}\n{t['content'][:2500]}", flush=True)

    conn = sqlite3.connect(os.path.join(runs, "checkpoints.sqlite"), check_same_thread=False)
    graph = build_graph(make_backend(a.model), [coding], work, allow_human=True, max_decisions=a.max_decisions,
                        checkpointer=SqliteSaver(conn), on_turn=show)
    config = {"configurable": {"thread_id": a.thread}, "recursion_limit": 80}

    def save():
        st = graph.get_state(config).values
        json.dump({"thread": a.thread, "model": a.model, "dialogue": st.get("dialogue", []), "events": events},
                  open(os.path.join(run_dir, "transcript.json"), "w"), indent=1, default=str)
        json.dump(coding.log, open(log_path, "w"), indent=1, default=str)
        json.dump(events, open(events_path, "w"), indent=1, default=str)

    if a.request:
        events.append({"type": "request", "text": a.request, "time": time.time()})
        graph.invoke({"dialogue": [{"role": "user", "content": a.request + "\n\n" + CONTEXT, "meta": {}}]}, config)
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
        print(f"# question: {q['question']}\n" + "#" * 70)
        if a.no_wait:
            print(f"\nPaused. Resume later with:  python run_task1.py --thread {a.thread} --reply \"<your reply>\"")
            return
        reply = input("your reply > ").strip()
        events.append({"type": "reply", "text": reply, "time": time.time()})
        graph.invoke(Command(resume=reply), config)
    print("\nThread finished. Transcript:", os.path.join(run_dir, "transcript.json"))


if __name__ == "__main__":
    main()
