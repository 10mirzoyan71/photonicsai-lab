"""Turn saved runs into LaTeX for report/homework2.tex.

    python render_hw2.py task1 runs/run_a --tex ../report/hw2_task1_a.tex
    python render_hw2.py task1 runs/run_b --tex ../report/hw2_task1_b.tex
    python render_hw2.py task2 runs_task2/<timestamp> --tex ../report/hw2_task2_run.tex
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "task6_agentic_system"))
from render_transcript import ascii, first_line, tex_escape  # noqa: E402

REPORT = os.path.join(HERE, "..", "report")


def lst(text, maxc=2500):
    t = ascii(text)
    if len(t) > maxc:
        t = t[: maxc - 300] + "\n[... abridged ...]\n" + t[-280:]
    return "\\begin{lstlisting}\n" + t.replace("\\end{lstlisting}", "") + "\n\\end{lstlisting}\n"


def turns_table(d):
    L = [r"\begin{longtable}{@{}r l p{0.74\linewidth}@{}}", r"\toprule \# & role & what it did\\ \midrule \endhead"]
    for i, t in enumerate(d, 1):
        m = t.get("meta", {})
        if t["role"] == "orchestrator" and "decision" in m:
            dec = m["decision"]
            s = f"-> {dec['next_role']}" + (f" ({dec.get('stop_kind')})" if dec["next_role"] == "human" else "") + f": {dec['reason']}"
            if m.get("refused_record"):
                s = "[refused step recorded as not run] " + s
        elif t["role"] == "human":
            s = f"reply: {t['content']}" + ("" if m.get("approved") is None else (" [accepted]" if m["approved"] else " [refused]"))
        elif t["role"] == "executor":
            s = t["content"].split("[tool record]")[-1]
        else:
            s = t["content"]
        L.append(f"{i} & {t['role']} & {tex_escape(first_line(s, 170))}\\\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(L)


def task1(run_dir, tex, title, max_stops=None):
    """The first max_stops stops of the run: question, reply, next message (each message printed once); then the
    table of all turns."""
    T = json.load(open(os.path.join(run_dir, "transcript.json"), encoding="utf-8"))
    d = T["dialogue"]
    req = next(e["text"] for e in T["events"] if e["type"] == "request")
    L = [f"Thread \\texttt{{{tex_escape(T['thread'])}}}. \\textbf{{Request typed to the orchestrator:}} "
         f"\\emph{{{tex_escape(req)}}}"]
    stops = [i for i, t in enumerate(d) if t["role"] == "human"]
    shown = stops[:max_stops] if max_stops else stops
    printed = set()
    for n, i in enumerate(shown, 1):
        m, q = d[i]["meta"], i - 1
        head = f"Stop {n} (turn {q + 1}, {tex_escape(m['kind'].replace('_', ' '))}): the question the graph posed"
        if q in printed:
            L.append(f"\\textbf{{{head}}} is the message above (turn {q + 1}).")
        else:
            L += [f"\\textbf{{{head}:}}", lst(d[q]["content"], 1600)]
            printed.add(q)
        L += [f"\\textbf{{Reply typed (turn {i + 1}, role \\texttt{{human}}):}}"
              + ("" if m.get("approved") is None else (" accepted." if m["approved"] else " \\textbf{refused}.")),
              lst(d[i]["content"])]
        if i + 1 < len(d):
            L += [f"\\textbf{{Next message in the dialogue (turn {i + 2}, {d[i + 1]['role']}):}}", lst(d[i + 1]["content"], 1600)]
            printed.add(i + 1)
    if len(shown) < len(stops):
        L.append(f"The run continued with {len(stops) - len(shown)} more stop(s) until the end (turn {len(d)}); "
                 "all turns are in the table.")
    L += ["\\textbf{All turns of this run:}", turns_table(d)]
    open(tex, "w", encoding="utf-8").write("\n".join(L) + "\n")


def task2(run_dir, tex):
    T = json.load(open(os.path.join(run_dir, "transcript.json"), encoding="utf-8"))
    d, ex = T["dialogue"], T["experiments"]
    figdir = os.path.join(REPORT, "hw2_figs", "agent_run")
    os.makedirs(figdir, exist_ok=True)
    L = [r"\subsection*{The agent's run}",
         f"Request: \\emph{{{tex_escape(T['request'])}}} --- nothing else was typed. Model: \\texttt{{{tex_escape(T['model'])}}}.",
         "", r"\begin{longtable}{@{}r r r r c r r r c r r@{}}",
         r"\toprule exp & macropixel & segments & focus & ring $r$ [px] & regions & iterations & flips & remeasure & $\eta$ & $\eta_\text{far}$\\\midrule\endhead"]
    best = max(ex, key=lambda e: e["enhancement"]) if ex else None
    for e in ex:
        s = e["settings"]
        row = (f"{e['experiment']} & {s['macropixel']} & {e['segments']} & {s['focus_size']} & {s['bg_inner']}--{s['bg_outer']} & "
               f"{s['bg_regions']} & {s['iterations']} & {s['flips']}$\\to${s['flips_final']} & {'yes' if s['remeasure'] else 'no'} & "
               f"{e['enhancement']:.1f}$\\pm${e['enhancement_std']:.1f} & {e['enhancement_far']:.1f}\\\\")
        L.append((r"\rowcolor{yellow!25}" if e is best else "") + row)
        shutil.copy(e["images"]["figure"], os.path.join(figdir, os.path.basename(e["images"]["figure"])))
    L += [r"\bottomrule", r"\end{longtable}"]
    if best:                                       # numbers used by the comparison section of homework2.tex
        L.insert(0, f"\\newcommand{{\\agentsegments}}{{{best['segments']}}}\\newcommand{{\\agenteta}}{{{best['enhancement']:.1f}}}"
                    f"\\newcommand{{\\agentetafar}}{{{best['enhancement_far']:.1f}}}"
                    f"\\newcommand{{\\agentfarpct}}{{{100 * best['enhancement_far'] / best['segments']:.1f}}}"
                    f"\\newcommand{{\\agentring}}{{{best['settings']['bg_inner']}--{best['settings']['bg_outer']}}}")
        dg = best["diagnostics"]
        L.append(f"Kept by the agent: experiment {best['experiment']}, enhancement {best['enhancement']:.1f} "
                 f"(far field {best['enhancement_far']:.1f}). Diagnostics only the simulation knows: noise-free "
                 f"{dg['enhancement_noise_free']:.1f}, known-$T$ binary pattern {dg['enhancement_oracle_TM_far']:.1f} (far field), "
                 f"theory $N_\\text{{eff}}/2\\pi={dg['enhancement_theory_Neff_over_2pi']:.1f}$.")
        for e in ex:
            L.append(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{hw2_figs/agent_run/" +
                     os.path.basename(e["images"]["figure"]) + "}\\caption{Agent experiment " + str(e["experiment"]) + ".}\\end{figure}")
    L += [r"\paragraph{Who did what.}", turns_table(d)]
    open(tex, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["task1", "task2"])
    p.add_argument("run_dir")
    p.add_argument("--tex", required=True)
    p.add_argument("--title", default="Run")
    p.add_argument("--stops", type=int, default=None, help="task1: show only the first N stops in detail")
    a = p.parse_args()
    task1(a.run_dir, a.tex, a.title, a.stops) if a.task == "task1" else task2(a.run_dir, a.tex)
    print("wrote", a.tex)
