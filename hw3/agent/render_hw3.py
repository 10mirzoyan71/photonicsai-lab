"""Turn a saved run into LaTeX for report/homework3.tex: the request, the instrument record (every MCP tool call and what
the camera did, from calls.jsonl), a sequence diagram of the run (TikZ), the last camera image and the final report.

    python render_hw3.py runs/<thread> --tex ../report/runs/<thread>.tex --title "Use case 1: auto-exposure"
"""
import argparse
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "report")

UNI = {"—": "--", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "→": "->",
       "µ": "u", "μ": "u", "×": "x", "≈": "~", "≤": "<=", "≥": ">=", "±": "+/-",
       "σ": "sigma", "…": "...", "²": "^2", "−": "-", "·": "*", "Δ": "Delta"}


def ascii(s):
    s = str(s)
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode()


def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
    return re.sub(r"[\\&%$#_{}~^<>]", lambda m: rep[m.group()], ascii(s))


def short(s, n):
    s = " ".join(ascii(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def short_block(s, n):
    s = ascii(s)
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " [...]"


def fmt_args(a):
    a = {k: (os.path.basename(v) if k == "frame_file" else v) for k, v in (a or {}).items()}
    return ", ".join(f"{k}={v}" for k, v in a.items() if v not in (None, "", []) and k != "regions") or "-"


# -- what the instrument did, per MCP call ----------------------------------------------------------------
def did(tool, args, res):
    if tool == "server_restart_restore":
        return f"server restarted; settings restored {res.get('restored')}"
    if not res.get("ok"):
        return "REFUSED: " + str(res.get("error") or res.get("text", ""))
    if tool == "camera_info":
        return (f"{res.get('model')}, {res.get('sensor_width')}x{res.get('sensor_height')}, {res.get('bit_depth')}-bit; "
                f"exposure {res.get('exposure_us')} us, gain {res.get('gain_db')} dB")
    if tool == "set_exposure":
        return f"exposure set to {res['exposure_us']:g} us"
    if tool == "set_gain":
        return f"gain set to {res['gain_db']:g} dB"
    if tool in ("set_roi", "reset_roi"):
        r = res["roi"]
        return f"ROI {r['width']}x{r['height']} at ({r['x']}, {r['y']})"
    if tool == "set_illumination":
        return f"light level set to {res['illumination']:g}"
    if tool == "capture":
        s, st = res["settings"], res["stats"]
        t = (f"{res['n_frames']} frame(s) at {s['exposure_us']:g} us, {s['gain_db']:g} dB"
             + (f", light {s['illumination']:g}" if "illumination" in s else "")
             + f": mean {st['mean_dn']}, max {st['max_dn']} DN, {st['saturated_pixels']} saturated px")
        tn = res.get("temporal_noise") or {}
        if tn.get("temporal_std_dn") is not None:
            t += f"; temporal noise {tn['temporal_std_dn']} DN"
        return t
    if tool == "capture_series":
        sm = res["summary"]
        return (f"{res['n_frames']} frames over {res['duration_s']} s: mean {sm['mean_dn']} DN, rel. std "
                f"{sm['rel_std_percent']}%, drift {sm['drift_percent_first_to_last']}%")
    if tool == "analyze_frame":
        sp, bm = res.get("speckle", {}), res.get("beam", {})
        return (f"speckle contrast {sp.get('contrast')}, grain {sp.get('grain_fwhm_x_px')} x {sp.get('grain_fwhm_y_px')} px; "
                f"beam centroid ({bm.get('centroid_x_px')}, {bm.get('centroid_y_px')}) px")
    return short(json.dumps(res), 120)


INSTRUMENT_ACTION = {"camera_info": "read features", "set_exposure": "ExposureTime", "set_gain": "Gain",
                     "set_roi": "Width/Height/Offset", "reset_roi": "full sensor", "set_illumination": "light (sim.)",
                     "capture": "trigger + grab", "capture_series": "timed grabs"}


# -- sequence diagram -------------------------------------------------------------------------------------
def sequence(d, camera_label, max_h=21.0, collapse_at=6):
    roles_used = {t["role"] for t in d}
    actors = ["User", "Orchestrator"] + [r for r, k in (("Scientist", "scientist"), ("General", "general")) if k in roles_used]
    actors += ["Executor"] + (["Human"] if "human" in roles_used else []) + ["MCP server", camera_label]
    X = {a: 2.3 * i for i, a in enumerate(actors)}
    ev = []                                                   # (kind, a, b, label): kind arrow/ret/box

    def arrow(a, b, lab, ret=False):
        ev.append(("ret" if ret else "arrow", a, b, short(lab, 30)))

    target = {"scientist": "Scientist", "general": "General", "executor": "Executor", "hardware": "Executor", "human": "Human"}
    total, turn = {}, -1                                      # steps per hardware turn, to shorten long turns
    for t in d:
        hw = t.get("meta", {}).get("hw")
        if hw == "start":
            turn += 1
        elif hw == "decide":
            total[turn] = t["meta"]["step"]
    turn, step, gap_done = -1, 0, set()
    for t in d:
        m, r = t.get("meta", {}), t["role"]
        hw = m.get("hw")
        if hw == "start":
            turn, step = turn + 1, 0
        elif hw == "decide":
            step = m["step"]
        if hw in ("decide", "command", "measure", "person") and total.get(turn, 0) >= collapse_at and 3 < step < total[turn]:
            if turn not in gap_done:                          # steps 4 .. n-1 of a long turn: one marker
                gap_done.add(turn)
                ev.append(("gap", "Executor", camera_label,
                           f"... {total[turn] - 4} more steps of the same kind (decide, command, measure): see the table ..."))
            continue
        if r == "user":
            arrow("User", "Orchestrator", "request")
        elif r == "orchestrator":
            dec = m.get("decision")
            if dec and dec["next_role"] != "finish":
                nr = dec["next_role"]
                lab = {"hardware": "hardware turn", "executor": "analysis (coding tool)", "human": "question"}.get(nr, nr)
                arrow("Orchestrator", target[nr], f"{m.get('index', '')}: {lab}")
            else:
                arrow("Orchestrator", "User", "final report")
        elif r in ("scientist", "general"):
            arrow(r.capitalize(), "Orchestrator", "plan" if r == "scientist" else "answer", ret=True)
        elif r == "human":
            if hw == "person":
                arrow("Human", "Executor", "reply: " + t["content"], ret=True)
            else:
                ok = m.get("approved")
                arrow("Human", "Orchestrator", "reply" + ("" if ok is None else (" (accepted)" if ok else " (refused)")), ret=True)
        elif r == "executor" and hw == "decide":
            dec = m["decision"]
            lab = {"command": f"decide: {dec['command']}", "measure": "decide: measure", "ask_person": "decide: ask person",
                   "finish": "decide: finish"}[dec["action"]]
            ev.append(("box", "Executor", None, short(lab, 28)))
            if dec["action"] == "ask_person":
                arrow("Executor", "Human", "question")
        elif r == "executor" and hw == "report":
            arrow("Executor", "Orchestrator", "hardware report", ret=True)
        elif r == "executor":
            ev.append(("box", "Executor", None, "coding/focusing tool"))
            arrow("Executor", "Orchestrator", "result", ret=True)
        elif r == "instrument":
            if hw == "start":
                calls = [("camera_info", {}, m["result"])]
            elif hw == "command":
                calls = [(m["tool"], m["args"], m["result"])]
            else:
                calls = [("capture", {}, m["capture"])] + ([("analyze_frame", {}, m["analysis"])] if "analysis" in m else [])
            for tool, args, res in calls:
                arrow("Executor", "MCP server", tool + (f"({fmt_args(args)})" if args else ""))
                if tool == "analyze_frame" or not res.get("ok"):
                    arrow("MCP server", "Executor", "result" if res.get("ok") else "refused", ret=True)
                else:
                    arrow("MCP server", camera_label, INSTRUMENT_ACTION.get(tool, tool))
                    arrow(camera_label, "Executor", "frame stats" if tool.startswith("capture") else
                          ("features" if tool == "camera_info" else "applied"), ret=True)
    n = len(ev)
    dy = min(0.55, max_h / max(n, 1))
    L = [r"\begin{center}\begin{tikzpicture}[x=1cm,y=-%.3fcm,font=\scriptsize,>=Stealth]" % dy]
    for a in actors:
        L.append(f"\\node[draw,fill=gray!12,rounded corners,minimum width=17mm,align=center] at ({X[a]:.2f},-1.4) {{{tex(a)}}};")
        L.append(f"\\draw[gray!60,dashed] ({X[a]:.2f},-0.4) -- ({X[a]:.2f},{n + 0.5});")
    for i, (k, a, b, lab) in enumerate(ev):
        y = i + 0.5
        if k == "box":
            L.append(f"\\node[draw,fill=yellow!25,inner sep=0.6pt,font=\\tiny,anchor=east] at ({X[a] - 0.05:.2f},{y}) {{{tex(lab)}}};")
        elif k == "gap":
            L.append(f"\\node[draw,dotted,fill=gray!8,font=\\tiny,minimum width={X[b] - X[a] + 0.4:.2f}cm] at "
                     f"({(X[a] + X[b]) / 2:.2f},{y}) {{{tex(lab)}}};")
        else:
            style = "->,dashed" if k == "ret" else "->"
            pos = "above" if dy > 0.3 else "above,inner sep=0.5pt"
            L.append(f"\\draw[{style}] ({X[a]:.2f},{y}) -- node[{pos},font=\\tiny] {{{tex(lab)}}} ({X[b]:.2f},{y});")
    L.append(r"\end{tikzpicture}\end{center}")
    return "\n".join(L)


def render(run_dir, out_tex, title, mode="full"):
    """mode: full | lean (no logs or frame, smaller diagram) | compact (request, tool calls in one paragraph, result)."""
    T = json.load(open(os.path.join(run_dir, "transcript.json"), encoding="utf-8"))
    d = T["dialogue"]
    cam_dir = os.path.join(run_dir, "work", "camera")
    calls_path = os.path.join(cam_dir, "calls.jsonl")
    calls = [json.loads(line) for line in open(calls_path, encoding="utf-8")] if os.path.exists(calls_path) else []
    sim = T.get("camera", "sim").startswith("sim")
    room = T.get("camera") == "sim-room"
    camera_label = "Sim. camera" if sim else "Camera"
    L = [f"\\subsection*{{{tex(title)}}}",
         f"\\textbf{{Request:}} \\emph{{{tex(T['request'])}}}\\\\",
         (f"Camera: {('simulated camera, room-light scene (dry run)' if room else 'simulated camera (dry run)') if sim else 'Hikrobot MV-CS028-10UM (real hardware)'}; "
          if calls else "Camera: not used in this run (simulated bench only); ")
         + f"{sum(1 for t in d if t['role'] == 'orchestrator' and 'decision' in t.get('meta', {}))} orchestrator decisions, "
         f"{sum(1 for t in d if t.get('meta', {}).get('hw') == 'decide')} hardware decisions, {len(calls)} MCP tool calls."]
    if mode == "compact":                        # the same record in one paragraph, a small diagram; no table, frame or logs
        if calls:
            L.append(r"\\{\small\textbf{Tool calls and what the instrument did:} " + "; ".join(
                f"({i})~\\texttt{{{tex(c['tool'])}}}" + (f" {tex(short(fmt_args(c['args']), 40))}" if c["args"] else "")
                + f" $\\to$ {tex(short(did(c['tool'], c['args'], c['result']), 70))}" for i, c in enumerate(calls, 1)) + ".}")
        tools = [ln for t in d if t["role"] == "executor" and not t.get("meta", {}).get("hw")
                 for ln in t["content"].split("[tool record]")[-1].strip().splitlines() if ln.startswith("experiment ")]
        if tools:
            L.append(r"\\{\small\textbf{Focusing tool calls:} " + "; ".join(tex(short(x, 140)) for x in tools) + ".}")
        L.append(f"\\\\\\textbf{{Verified result:}} {tex(result_line(T.get('verified')))}.")
        L += [r"\par\smallskip\textbf{Sequence diagram} (as above; long hardware turns shortened).",
              sequence(d, camera_label, 7.5, 4)]
        os.makedirs(os.path.dirname(os.path.abspath(out_tex)), exist_ok=True)
        open(out_tex, "w", encoding="utf-8").write("\n".join(L) + "\n")
        return
    lean = mode == "lean"
    if calls:
        L += [r"\paragraph{Tool calls and what the instrument did.}",
              r"{\footnotesize\begin{longtable}{@{}r l p{0.27\linewidth} p{0.52\linewidth}@{}}",
              r"\toprule \# & tool & arguments & what the instrument did\\ \midrule \endhead"]
        for i, c in enumerate(calls, 1):
            L.append(f"{i} & \\texttt{{{tex(c['tool'])}}} & {tex(short(fmt_args(c['args']), 60))} & "
                     f"{tex(short(did(c['tool'], c['args'], c['result']), 150))}\\\\")
        L += [r"\bottomrule", r"\end{longtable}}"]
    scripts = [t for t in d if t["role"] == "executor" and not t.get("meta", {}).get("hw")]
    if scripts and not lean:
        L.append(r"\par\medskip\textbf{Executor turns with its software tools (coding tool, focusing tool).}\par")
        for t in scripts:
            rec = t["content"].split("[tool record]")[-1].strip()
            L.append(r"\begin{lstlisting}" + "\n" + ascii(rec)[:900] + "\n" + r"\end{lstlisting}")
        if sim and calls:
            import sys
            sys.path.insert(0, os.path.join(HERE, ".."))
            from camera.sim_camera import SimCamera
            tr = SimCamera(seed=T.get("seed", 1), scene="room" if room else "speckle").truth()
            L.append("True parameters of the simulated camera, unknown to the agent: conversion gain "
                     f"{tr['conversion_gain_e_per_dn_at_0dB']:g}\\,e$^-$/DN at 0\\,dB, read noise {tr['read_noise_e']:g}\\,e$^-$ "
                     f"({tr['read_noise_dn_at_0dB']:.2f}\\,DN), full well {tr['full_well_e']:g}\\,e$^-$, black level "
                     f"{tr['black_level_dn']:g}\\,DN, dark current {tr['dark_current_e_s']:g}\\,e$^-$/s"
                     + (f"; room-light flicker {tr['flicker_hz']:g}\\,Hz, depth {tr['flicker_depth']:g}." if room else "."))
    L += [r"\paragraph{Sequence diagram.} Solid arrows: messages and commands; dashed: replies and measurements; boxes: the "
          r"executor's decisions (model calls). Numbers are the orchestrator's decision index.",
          sequence(d, camera_label, *((11.0, 4) if lean else ()))]
    pngs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".png")) if os.path.isdir(cam_dir) else []
    if pngs and not lean:
        figdir = os.path.join(os.path.dirname(os.path.abspath(out_tex)), "img")
        os.makedirs(figdir, exist_ok=True)
        name = f"{os.path.basename(os.path.normpath(run_dir))}_{pngs[-1]}"
        shutil.copy(os.path.join(cam_dir, pngs[-1]), os.path.join(figdir, name))
        L += [r"\begin{figure}[H]\centering", f"\\includegraphics[width=0.48\\linewidth]{{runs/img/{name}}}",
              f"\\caption{{Last frame captured in this run ({tex(pngs[-1])}).}}", r"\end{figure}"]
    fin = d[-1]
    if fin["role"] == "orchestrator" and not lean:          # lean: the report text summarises the outcome
        txt = fin["content"]
        if fin.get("meta", {}).get("decision"):
            txt = fin["meta"]["decision"].get("instruction", txt)
        L += [r"\par\medskip\textbf{Final report of the run.}\par", r"\begin{lstlisting}" + "\n" + short_block(txt, 600 if lean else 2600) + "\n" + r"\end{lstlisting}"]
    os.makedirs(os.path.dirname(os.path.abspath(out_tex)), exist_ok=True)
    open(out_tex, "w", encoding="utf-8").write("\n".join(L) + "\n")


KEY_RESULTS = ["K_E_PER_DN", "READ_NOISE_E$", "CONTRAST_CORR", "CONTRAST_MEAN", "^CONTRAST$", "GRAIN_FWHM_X", "GRAIN_FWHM_Y",
               "DRIFT_PERCENT_PER_MIN$", "SLOPE_PCT_PER_MIN$", "TOTAL_CHANGE_PCT$", "STD_DETRENDED_PERCENT", "REL_STD", "FLICKER", "FULL_WELL", "ENHANCEMENT"]


def result_line(v, n=4):
    if not v:
        return "no verified result"
    if v.get("tool") == "coding_tool":
        res = v["results"]
        keys = [k for pat in KEY_RESULTS for k in res if re.search(pat, k)]
        keys = list(dict.fromkeys(keys + list(res)))[:n]         # key results first, then the rest in printed order
        return f"script {v['script']}: " + ", ".join(f"{k}={res[k]:.4g}" for k in keys)
    if v.get("tool") == "focusing_tool":
        return f"focusing tool (simulated bench), experiment {v['experiment']}: enhancement {v['enhancement']:.4g}"
    if v.get("tool") == "camera":
        s, st = v["settings"], v["stats"]
        return f"{s['exposure_us']:g} us, {s['gain_db']:g} dB: max {st['max_dn']} DN, {st['saturated_pixels']} saturated px"
    return short(json.dumps(v), 120)


def summary(run_dirs, out_tex):
    """One table row per run: request, decisions, hardware steps, MCP calls, verified result."""
    L = [r"{\footnotesize\begin{longtable}{@{}p{0.3\linewidth} r r r p{0.4\linewidth}@{}}",
         r"\toprule request & decisions & hw steps & MCP calls & verified result\\ \midrule \endhead"]
    for rd in run_dirs:
        T = json.load(open(os.path.join(rd, "transcript.json"), encoding="utf-8"))
        d = T["dialogue"]
        cp = os.path.join(rd, "work", "camera", "calls.jsonl")
        n_calls = sum(1 for _ in open(cp, encoding="utf-8")) if os.path.exists(cp) else 0
        L.append(f"{tex(T['request'])} & {sum(1 for t in d if t['role'] == 'orchestrator' and 'decision' in t.get('meta', {}))} & "
                 f"{sum(1 for t in d if t.get('meta', {}).get('hw') == 'decide')} & {n_calls} & {tex(result_line(T.get('verified')))}\\\\")
    L += [r"\bottomrule", r"\end{longtable}}"]
    open(out_tex, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="+")
    p.add_argument("--tex", required=True)
    p.add_argument("--title", default="Run")
    p.add_argument("--summary", action="store_true", help="one summary table over several runs")
    p.add_argument("--mode", default="full", choices=("full", "lean", "compact"),
                   help="full; lean: no logs or frame, smaller diagram; compact: one paragraph of tool calls")
    a = p.parse_args()
    summary(a.run_dir, a.tex) if a.summary else render(a.run_dir[0], a.tex, a.title, a.mode)
    print("wrote", a.tex)
