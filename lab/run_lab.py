"""Run the lab assistant (Homework 4/5) in the terminal: you speak to the director, the teams do the work.

    python lab/run_lab.py --thread t1 --request "Measure the camera's noise and predict the focusing enhancement with it."
    python lab/run_lab.py --thread t1 --reply "yes"            # resume a paused thread (also from a new terminal)

Options: --camera sim-room (default) | sim | hikrobot (the lab camera), --auto (the director does not ask the person;
team questions such as "block the light" still pause), --director-model / --team-model (default: Claude Sonnet for the
director and Claude Haiku for the teams, through the Claude Code CLI, with low effort: model@low), --no-wait (exit at a pause instead of asking).
Runs are saved in lab/runs/<thread>/ (transcript.json with the token usage, the teams' folders and checkpoints).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lab_assistant import LabRun  # noqa: E402

RUNS = os.path.join(HERE, "runs")


def show(source, turn):
    if source == "user":
        return
    who = "DIRECTOR" if source == "director" and turn["role"] == "director" else \
          f"{source.upper()} / {turn['role']}" if source != "director" else turn["role"].upper()
    hw = turn.get("meta", {}).get("hw")
    print(f"\n{'=' * 14} {who}{f' [{hw}]' if hw else ''} {'=' * 14}\n{turn['content'][:1500]}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True)
    p.add_argument("--request")
    p.add_argument("--reply")
    p.add_argument("--camera", default="sim-room")
    p.add_argument("--auto", action="store_true", help="the director decides without asking the person")
    p.add_argument("--director-model", default="claude-cli:claude-sonnet-5-5@low")
    p.add_argument("--team-model", default="claude-cli:claude-haiku-5-5@low")
    p.add_argument("--no-wait", action="store_true")
    a = p.parse_args()

    meta_path = os.path.join(RUNS, a.thread, "run.json")
    if a.request:
        if os.path.exists(meta_path):
            sys.exit(f"thread {a.thread} exists; resume it with --reply or pick another --thread")
        cfg = {"camera": a.camera, "director_model": a.director_model, "team_model": a.team_model,
               "allow_human": not a.auto}
    elif os.path.exists(meta_path):
        m = json.load(open(meta_path, encoding="utf-8"))
        cfg = {k: m[k] for k in ("camera", "director_model", "team_model", "allow_human")}
    else:
        sys.exit("unknown thread: give --request to start it")
    run = LabRun(a.thread, RUNS, emit=show, **cfg)
    try:
        if a.request:
            print(f"USER: {a.request}", flush=True)
            run.start(a.request)
        else:
            if run.pending() is None:
                sys.exit("this thread is not paused")
            run.resume(a.reply)
        while True:
            q = run.pending()
            if q is None:
                break
            print("\n" + "#" * 70 + f"\n# PAUSED ({q.get('source')}, {q.get('kind')})")
            step = q.get("proposed_step")
            if step:
                print(f"# proposed step: {json.dumps(step)[:400]}")
            print(f"# question: {q.get('question')}\n" + "#" * 70, flush=True)
            if a.no_wait:
                print(f'\nPaused. Resume with:  python lab/run_lab.py --thread {a.thread} --reply "<your reply>"')
                return
            run.resume(input("your reply > ").strip())
        u = run.usage()
        print(f"\nFinished. Model calls: {u['calls']}, output tokens: {u['output']}, input tokens: {u['input']} "
              f"(+{u['cache_read']} cached), estimated cost ${u['cost_usd']:.2f}. Saved in {run.dir}")
    finally:
        run.save()
        run.close()


if __name__ == "__main__":
    main()
