"""Turn a run's transcript.json into transcript.md and a LaTeX section for the report.

    python render_transcript.py runs/<timestamp>            # writes runs/<timestamp>/task6_run.tex
    python render_transcript.py runs/<timestamp> --tex ../report/task6_run.tex
"""
import argparse
import json
import os
import re

UNICODE_ASCII = {"\u2192": "->", "\u2190": "<-", "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
                 "\u201c": '"', "\u201d": '"', "\u2248": "~", "\u00d7": "x", "\u2264": "<=", "\u2265": ">=",
                 "\u2212": "-", "\u00b2": "^2", "\u00b3": "^3", "\u03b8": "theta", "\u03b1": "alpha",
                 "\u0394": "Delta", "\u03c9": "omega", "\u221a": "sqrt", "\u2026": "...", "\u00b7": "*",
                 "\u2022": "*", "\u2032": "'", "\u00b1": "+/-", "\u2211": "sum", "\u2202": "d", "\u03c0": "pi"}


def ascii(s: str) -> str:
    for k, v in UNICODE_ASCII.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode().replace("?", "?")


def tex_escape(s: str) -> str:
    s = ascii(s)
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "<": r"\textless{}", ">": r"\textgreater{}"}
    return re.sub(r"[\\&%$#_{}~^<>]", lambda m: rep[m.group()], s)


def first_line(s: str, n=150):
    s = " ".join(s.strip().split())
    return s if len(s) <= n else s[: n - 3] + "..."


def summary_of(turn):
    m = turn.get("meta", {})
    if turn["role"] == "orchestrator" and "decision" in m:
        d = m["decision"]
        return f"-> {d['next_role']}: {d['reason']}"
    if turn["role"] == "executor":
        rec = turn["content"].split("[tool record]")[-1]
        return first_line(rec, 160)
    return first_line(turn["content"])


def write_markdown(out: dict, path: str):
    lines = [f"# Task 6 run ({out.get('model', '')})", ""]
    rec = out.get("verified")
    lines.append(f"**Verified free-run result:** {rec}" if rec else "**No verified TEST_RMSE.**")
    lines += ["", "| # | role | summary |", "|---|---|---|"]
    for i, t in enumerate(out["dialogue"], 1):
        lines.append(f"| {i} | {t['role']} | {summary_of(t).replace('|', '/')} |")
    for i, t in enumerate(out["dialogue"], 1):
        lines += ["", f"## Turn {i}: {t['role']}", "", "```text", t["content"], "```"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_tex(out: dict, path: str, max_chars: int = 1800, run_dir: str | None = None):
    rec = out.get("verified")
    L = [r"\subsection*{Validation run on Task 5}",
         f"Model behind every role: \\texttt{{{tex_escape(out.get('model', ''))}}}. "
         f"Turns: {len(out['dialogue'])}; orchestrator decisions: "
         f"{sum(1 for t in out['dialogue'] if t['role'] == 'orchestrator' and 'decision' in t.get('meta', {}))}.",
         ""]
    if rec:
        L.append(f"\\textbf{{Free-run error printed by the script that ran}} (\\texttt{{{tex_escape(rec['script'])}}}): "
                 f"test RMSE $= {rec['test']:.6g}$\\,V" + (f", train RMSE $= {rec['train']:.6g}$\\,V." if rec.get("train") is not None else "."))
    else:
        L.append(r"\textbf{The run produced no verified test RMSE.}")
    L += ["", r"\begin{longtable}{@{}r l p{0.72\linewidth}@{}}", r"\toprule \# & role & what it did\\ \midrule \endhead"]
    for i, t in enumerate(out["dialogue"], 1):
        L.append(f"{i} & {t['role']} & {tex_escape(summary_of(t))}\\\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    png = os.path.join(run_dir, "work", "test_segment.png") if run_dir else None
    if png and os.path.exists(png):
        rel = os.path.relpath(png, os.path.dirname(os.path.abspath(path))).replace("\\", "/")
        L += [r"\begin{figure}[H]\centering", f"\\includegraphics[width=0.85\\linewidth]{{{rel}}}",
              r"\caption{The plot saved by the agents' script: measured output and free-run prediction on the test segment.}",
              r"\end{figure}"]
    L += ["", r"\paragraph{Transcript (abridged).}"]
    for i, t in enumerate(out["dialogue"], 1):
        body = ascii(t["content"])
        if t["role"] == "user":
            body = first_line(body, 90) + "\n[the full Task 5 statement follows: task6_agentic_system/task5_statement.md]"
        if len(body) > max_chars:
            body = body[: max_chars - 400] + "\n[... abridged ...]\n" + body[-380:]
        L += [f"\\noindent\\textbf{{Turn {i} --- {t['role']}}}", r"\begin{lstlisting}", body.replace(r"\end{lstlisting}", ""), r"\end{lstlisting}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--tex", default=None)
    a = p.parse_args()
    with open(os.path.join(a.run_dir, "transcript.json"), encoding="utf-8") as fh:
        out = json.load(fh)
    write_markdown(out, os.path.join(a.run_dir, "transcript.md"))
    tex = a.tex or os.path.join(a.run_dir, "task6_run.tex")
    write_tex(out, tex, run_dir=a.run_dir)
    print("wrote", tex)
