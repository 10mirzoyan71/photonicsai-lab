"""Writes ../report/task5_discussion.tex: kept settings and the discussion, with numbers read from results/."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "report", "task5_discussion.tex")

R = {}
for p in glob.glob(os.path.join(RES, "*.json")):
    if os.path.basename(p) == "kept.json":
        continue
    with open(p) as fh:
        r = json.load(fh)
    if "config" in r:
        R[r["config"]["name"]] = r
with open(os.path.join(RES, "kept.json")) as fh:
    K = json.load(fh)


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return f"{float(m):.2f}\\cdot10^{{{int(e)}}}"


def te(n):
    return f"{R[n]['rmse_test'] * 1e3:.3f}" if n in R else "--"


def tr(n):
    return f"{R[n]['rmse_train'] * 1e3:.3f}" if n in R else "--"


c = K["config"]
arch = "$" + "\\times".join(map(str, c["hidden"])) + "$ " + c["act"]
settings = (f"Settings of the kept version: input scaling by the training standard deviations, {arch} hidden layers, "
            f"linear output scaled by $\\sigma_a$; RK4 at $\\Delta t$ with sinc-interpolated midpoint input; stage B 500 steps; "
            f"stage C {c['optimizer'].replace('adam', 'Adam').replace('rmsprop', 'RMSprop')}, learning rate {c['lr']:g} with cosine decay, "
            f"{c['steps']} steps, batch {c['batch']} windows, rollout growing from {c['rollout_start']} to {c['rollout']} samples, "
            f"PINN weight $\\lambda_\\text{{phys}}={c['physics_weight']:g}$ with the identified coefficients, gradient clip {c['clip']:g}; "
            f"checkpoint at step {R[K['name']]['best_step']} (lowest training free-run RMSE).")

archs = {"A1": "$2\\times32$ tanh", "A2": "$2\\times64$ tanh", "A3": "$3\\times64$ tanh", "A4": "$2\\times64$ SiLU"}
best_per_arch = []
for a, lab in archs.items():
    runs = [n for n in R if n.startswith(a + "_")]
    if runs:
        b = min(runs, key=lambda n: R[n]["rmse_test"])
        best_per_arch.append(f"{lab}: {te(b)}\\,mV ({b.replace('_', chr(92) + '_')})")

def worst(a):
    runs = [n for n in R if n.startswith(a + "_")]
    return max(runs, key=lambda n: R[n]["rmse_test"]) if runs else None


tanh = [a for a in ("A1", "A2", "A3") if f"{a}_h2" in R and f"{a}_h3" in R]
h3_beats_h2 = all(R[f"{a}_h3"]["rmse_test"] < R[f"{a}_h2"]["rmse_test"] for a in tanh)
rms_worst = [a for a in archs if worst(a) and worst(a).endswith("h4")]
jc = R.get("L_joint", {}).get("pinn_coeffs", {}).get("c")
ic = R["R_duffing"]["duffing_coeffs"]["c"]
hist = [e for e in R[K["name"]]["history"] if e["stage"] == "rollout"]
still_down = hist[-1]["free_train"] < hist[len(hist) // 2]["free_train"]

h5_best = all(min((n for n in R if n.startswith(a + "_")), key=lambda n: R[n]["rmse_test"]).endswith("h5")
              for a in archs if any(n.startswith(a + "_") for n in R))

disc = f"""\\paragraph{{What the sweep shows.}}
\\begin{{itemize}}[leftmargin=*,itemsep=1pt]
\\item \\textbf{{The physics loss is what makes the network extrapolate.}} On A2\\_h2, without it ($\\lambda_\\text{{phys}}=0$, run
L\\_pw0) the network fits the training record ({tr('L_pw0')}\\,mV) but reaches {te('L_pw0')}\\,mV on the test segment; almost all of
that error sits on the 0.205\\,V peak, where the network trained on $|y|\\le0.174$\\,V under-stiffens (force-curve figure). With the
residual to the identified equation the same network gets {te('A2_h2')}\\,mV ($\\lambda=1$); $\\lambda=0.1$ gives {te('L_pw01')}\\,mV and
$\\lambda=10$ gives {te('L_pw10')}\\,mV. Learning $a,b,c,g$ jointly with the network instead of identifying them first
(L\\_joint) gives {te('L_joint')}\\,mV: the coefficients drift towards whatever the network does outside the data (in that run the
cubic coefficient ended at """ + (f"$c={sci(jc)}$ against the identified ${sci(ic)}$" if jc else "a different value") + f"""), so the residual stops carrying information there.
\\item \\textbf{{Training length matters most.}} {'In every architecture the best setting is h5 (1000 steps)' if h5_best else 'The longest setting (h5, 1000 steps) is among the best settings'};
""" + ("with 500 steps a rollout of 256 (h3) beats 128 (h2) for every tanh network" if h3_beats_h2 else "a rollout of 256 (h3) against 128 (h2) helps for some networks and not for others") + f""", and RMSprop (h4) is the worst setting for """ + (", ".join(rms_worst) if rms_worst else "none of the architectures") + f""". """ + ("The training free-run error of the kept run was still going down at the end, so more steps would likely lower it further." if still_down else "") + f"""
\\item \\textbf{{Architecture.}} Best test RMSE per architecture: {'; '.join(best_per_arch)}. The tanh networks with 64 units are
better than the 32-unit one at equal settings; SiLU is clearly worse here.
\\item \\textbf{{Comparison with the bare equation.}} Eq.~(1) with its four identified coefficients, integrated the same way, scores
{tr('R_duffing')}\\,mV train / {te('R_duffing')}\\,mV test; the linear equation ($c=0$) scores {tr('R_linear')} / {te('R_linear')}\\,mV.
So the cubic term is essential, and on this excerpt the circuit is described by (1) so well that the kept network
({K['rmse_test'] * 1e3:.3f}\\,mV) matches the identified equation to within
{abs(K['rmse_test'] - R['R_duffing']['rmse_test']) * 1e3:.2f}\\,mV {'without beating it' if K['rmse_test'] > R['R_duffing']['rmse_test'] else 'and slightly improves on it'}.
The network's force is free to depart from (1) where the training data ask for it, and the physics term keeps it on (1) where
there are no data; the physics read-out below shows the force it settled on.
\\item \\textbf{{Train/test gap.}} {K['rmse_train'] * 1e3:.3f}\\,mV vs {K['rmse_test'] * 1e3:.3f}\\,mV for the kept version: no sign of a
drift on the test segment. The version is chosen by test score, as the task asks, so the reported test number is slightly
optimistic; inside every run the checkpoint was chosen by the training free-run error only.
\\end{{itemize}}"""

with open(OUT, "w") as fh:
    fh.write("% generated by task5_pinn_silverbox/make_discussion.py\n")
    fh.write(f"\\newcommand{{\\keptarch}}{{{arch}}}\n")
    fh.write("\\newcommand{\\keptsettings}{" + settings + "}\n")
    fh.write("\\newcommand{\\keptdiscussion}{" + disc + "}\n")
print(open(OUT).read())
