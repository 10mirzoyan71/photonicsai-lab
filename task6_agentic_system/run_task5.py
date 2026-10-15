"""Give the four-role system Task 5 and let it try to solve it.

    export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY with --model openai:<model>
    python run_task5.py --model anthropic:claude-sonnet-5-5

Creates runs/<timestamp>/ with
    work/                 the executor's working directory (scripts, logs, plot, data copy)
    transcript.json       the dialogue (role + text + machine-readable meta) and every tool call
    transcript.md         the same, readable
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agents import build_graph, make_backend, verified_rmse  # noqa: E402
from coding_tool import CodingTool                                # noqa: E402
from render_transcript import write_markdown                      # noqa: E402

DATA = os.path.join(HERE, "..", "data", "SNLS80mV.mat")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5"),
                   help="LangChain model id, e.g. anthropic:claude-sonnet-5-5 or openai:gpt-5, or claude-cli:<model> for the Claude Code CLI")
    p.add_argument("--run-dir", default=None)
    a = p.parse_args()

    run_dir = a.run_dir or os.path.join(HERE, "runs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    work = os.path.join(run_dir, "work")
    os.makedirs(work, exist_ok=True)
    if not os.path.exists(DATA):
        sys.exit("data/SNLS80mV.mat missing: run  python ../data/get_data.py")
    shutil.copy(DATA, os.path.join(work, "SNLS80mV.mat"))

    with open(os.path.join(HERE, "task5_statement.md")) as fh:
        task = fh.read()
    user_msg = "Please solve the following task with your team and report the verified free-run errors.\n\n" + task

    tool = CodingTool(workdir=work)
    backend = make_backend(a.model)

    def show(turn):
        print(f"\n{'=' * 20} {turn['role'].upper()} {'=' * 20}\n{turn['content'][:3000]}", flush=True)

    graph = build_graph(backend, tool, on_turn=show)
    state = graph.invoke({"dialogue": [{"role": "user", "content": user_msg, "meta": {}}]},
                         config={"recursion_limit": 60})
    dialogue = state["dialogue"]
    rec = verified_rmse(dialogue)
    out = {"model": a.model, "run_dir": run_dir, "dialogue": dialogue, "tool_log": tool.log, "verified": rec}
    with open(os.path.join(run_dir, "transcript.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    write_markdown(out, os.path.join(run_dir, "transcript.md"))
    print("\nRESULT:", rec if rec else "no verified TEST_RMSE")
    print("saved to", run_dir)


if __name__ == "__main__":
    main()
