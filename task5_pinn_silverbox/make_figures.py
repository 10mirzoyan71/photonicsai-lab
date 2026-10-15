"""Figures and tables for the report, computed from results/*.json and *.pt.

    python make_figures.py            # -> figures/*.png, ../report/task5_results.tex, results/kept.json
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from silverbox_pinn import (DT, HERE, N_TRAIN, central_diff, free_run, half_sample_input,  # noqa: E402
                            load_excerpt, load_model)

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TEX = os.path.join(HERE, "..", "report", "task5_results.tex")
os.makedirs(FIG, exist_ok=True)


def load_all():
    out = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        if os.path.basename(p) in ("kept.json",):
            continue
        with open(p) as fh:
            r = json.load(fh)
        if "config" in r:
            out[r["config"]["name"]] = r
    return out


def tex_sci(x, d=4):
    if abs(x) < 1e3:
        return f"{x:.{d}g}"
    m, e = f"{x:.{d - 1}e}".split("e")
    return f"{m}\\cdot10^{{{int(e)}}}"


def tex_num(x, d=3):
    return f"{x:.{d}f}"


def physics_readout(f, y, v, u):
    """Least-squares projection of the network force on the Duffing basis, on the training states."""
    sl = slice(2, N_TRAIN - 2)
    with torch.no_grad():
        alpha = f(torch.as_tensor(y[sl]), torch.as_tensor(v[sl]), torch.as_tensor(u[sl])).numpy()
    Phi = np.stack([-v[sl], -y[sl], -y[sl] ** 3, u[sl]], 1)
    th, *_ = np.linalg.lstsq(Phi, alpha, rcond=None)
    r2 = 1 - np.sum((alpha - Phi @ th) ** 2) / np.sum((alpha - alpha.mean()) ** 2)
    return dict(zip("abcg", map(float, th))), float(r2)


def main():
    R = load_all()
    nets = {k: r for k, r in R.items() if r["config"]["model"] == "mlp" and k[0] in "AL"}
    kept_name = min(nets, key=lambda k: nets[k]["rmse_test"])
    kept = nets[kept_name]
    u, y = load_excerpt()
    v, a = central_diff(y)
    f, cfg = load_model(os.path.join(RES, kept_name + ".pt"))
    uh = half_sample_input(u, cfg.interp)
    yhat, r_tr, r_te = free_run(f, u, y, uh)
    t = np.arange(len(y)) * DT

    # ---- figure 1: measured vs predicted on the test segment ------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ts = slice(N_TRAIN, None)
    ax[0].plot(t[ts], y[ts], color="k", lw=1.1, label="measured $y$")
    ax[0].plot(t[ts], yhat[ts], color="tab:red", lw=1.0, ls="--", label="free-run prediction $\\hat y$")
    ax[0].set_ylabel("output [V]")
    ax[0].legend(loc="upper right", ncol=2, fontsize=9)
    ax[0].set_title(f"Test segment (samples 3072-4095), kept network {kept_name}: free-run RMSE {r_te*1e3:.3f} mV")
    ax[1].plot(t[ts], (yhat - y)[ts] * 1e3, color="tab:red", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("error [mV]")
    ax[1].set_xlabel("time from start of excerpt [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "test_segment.png"), dpi=170)
    plt.close(fig)

    # zoom on the largest test peak
    k0 = N_TRAIN + int(np.argmax(np.abs(y[N_TRAIN:])))
    zs = slice(k0 - 60, k0 + 60)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t[zs], y[zs], "k.-", lw=1, ms=3, label="measured")
    ax.plot(t[zs], yhat[zs], "--", color="tab:red", lw=1.2, label=f"{kept_name} free run")
    ref_names = [n for n in ("L_pw0",) if n in R]
    for n in ref_names:
        g, _ = load_model(os.path.join(RES, n + ".pt"))
        yh2, _, _ = free_run(g, u, y, uh)
        ax.plot(t[zs], yh2[zs], ":", color="tab:blue", lw=1.2, label=f"{n} (no physics loss)")
    ax.axhline(np.abs(y[:N_TRAIN]).max(), color="gray", lw=0.6, ls="-.")
    ax.axhline(-np.abs(y[:N_TRAIN]).max(), color="gray", lw=0.6, ls="-.", label="max |y| on training record")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output [V]")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Zoom on the largest test peak (|y| = 0.205 V, outside the training range)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "test_peak_zoom.png"), dpi=170)
    plt.close(fig)

    # ---- figure 2: the learned restoring force alpha(y, 0, 0) -------------------
    yy = np.linspace(-0.27, 0.27, 301)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ref = R.get("R_duffing")
    if ref:
        c = ref["duffing_coeffs"]
        ax.plot(yy, (-c["b"] * yy - c["c"] * yy ** 3) / 1e3, color="k", lw=2, alpha=0.5, label="identified eq. (1): $-by-cy^3$")
        ax.plot(yy, (-c["b"] * yy) / 1e3, color="gray", lw=1, ls=":", label="linear part $-by$")
    for n, st in ((kept_name, dict(color="tab:red", ls="--")), ("L_pw0", dict(color="tab:blue", ls="-."))):
        if n in R:
            g, _ = load_model(os.path.join(RES, n + ".pt"))
            with torch.no_grad():
                F = g(torch.as_tensor(yy), torch.zeros(len(yy)), torch.zeros(len(yy))).numpy()
            ax.plot(yy, F / 1e3, lw=1.5, label=f"network {n}", **st)
    m = np.abs(y[:N_TRAIN]).max()
    ax.axvspan(-m, m, color="tab:green", alpha=0.08, label="training range of $y$")
    ax.axvline(np.abs(y[N_TRAIN:]).max(), color="tab:orange", lw=1, label="test peak 0.205 V")
    ax.set_xlabel("$y$ [V]  (with $v=0$, $u=0$)")
    ax.set_ylabel("acceleration [$10^3$ V/s$^2$]")
    ax.legend(fontsize=8)
    ax.set_title("Restoring force represented by the network")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "force_curve.png"), dpi=170)
    plt.close(fig)

    # ---- figure 3: training curve of the kept version ---------------------------
    h = [e for e in kept["history"] if e["stage"] == "rollout"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy([e["step"] for e in h], [e["free_train"] * 1e3 for e in h], "o-", label="free-run RMSE, train")
    ax.semilogy([e["step"] for e in h], [e["free_test"] * 1e3 for e in h], "s--", label="free-run RMSE, test (monitor only)")
    ax.set_xlabel("rollout-training step")
    ax.set_ylabel("RMSE [mV]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curve.png"), dpi=170)
    plt.close(fig)

    coeffs, r2 = physics_readout(f, y, v, u)
    kept_out = {"name": kept_name, "rmse_train": r_tr, "rmse_test": r_te, "config": kept["config"],
                "n_params": kept["n_params"], "readout": coeffs, "readout_r2": r2,
                "f0_hz": float(np.sqrt(coeffs["b"]) / (2 * np.pi))}
    with open(os.path.join(RES, "kept.json"), "w") as fh:
        json.dump(kept_out, fh, indent=1)
    write_tex(R, kept_out)
    print(json.dumps(kept_out, indent=1))


def cfg_cells(c):
    if c["model"] == "duffing":
        return ["eq.\\ (1), 4 coeff." + (" ($c{=}0$)" if c.get("extra", {}).get("fix_c_zero") else ""), "--", "--"]
    arch = "$" + "\\times".join(map(str, c["hidden"])) + "$ " + c["act"] + (" +lin" if c["linear_skip"] else "")
    return [arch, c["optimizer"], f"{c['lr']:g}"]


def write_tex(R, kept):
    rows = sorted(R.values(), key=lambda r: (r["config"]["name"][0] != "A", r["config"]["name"]))
    L = [r"\begin{longtable}{@{}l l l l r r r r r r@{}}",
         r"\toprule run & force model & opt. & lr & steps & rollout & $\lambda_\text{phys}$ & params & train [mV] & test [mV]\\\midrule\endhead"]
    best = kept["name"]
    for r in rows:
        c = r["config"]
        cells = [c["name"].replace("_", r"\_")] + cfg_cells(c) + [str(c["steps"]), str(c["rollout"]),
                 f"{c['physics_weight']:g}" if c["model"] == "mlp" else "--", str(r["n_params"]),
                 tex_num(r["rmse_train"] * 1e3), tex_num(r["rmse_test"] * 1e3)]
        line = " & ".join(cells) + r"\\"
        if c["name"] == best:
            line = r"\rowcolor{yellow!25}" + line
        L.append(line)
    L += [r"\bottomrule", r"\end{longtable}"]
    with open(TEX, "w") as fh:
        fh.write("% generated by task5_pinn_silverbox/make_figures.py\n")
        fh.write(f"\\newcommand{{\\keptname}}{{{best.replace('_', '_')}}}\n")
        fh.write(f"\\newcommand{{\\keptnametex}}{{{best.replace('_', chr(92) + '_')}}}\n")
        fh.write(f"\\newcommand{{\\kepttrain}}{{{kept['rmse_train']*1e3:.3f}}}\n")
        fh.write(f"\\newcommand{{\\kepttest}}{{{kept['rmse_test']*1e3:.3f}}}\n")
        fh.write(f"\\newcommand{{\\keptparams}}{{{kept['n_params']}}}\n")
        for k, v in kept["readout"].items():
            fh.write(f"\\newcommand{{\\read{k}}}{{{tex_sci(v)}}}\n")
        fh.write(f"\\newcommand{{\\readrsq}}{{{kept['readout_r2']:.5f}}}\n")
        fh.write(f"\\newcommand{{\\readfzero}}{{{kept['f0_hz']:.2f}}}\n")
        if "R_duffing" in R:
            for k, v in R["R_duffing"]["duffing_coeffs"].items():
                fh.write(f"\\newcommand{{\\ident{k}}}{{{tex_sci(v)}}}\n")
        fh.write("\\newcommand{\\sweeptable}{%\n" + "\n".join(L) + "\n}\n")


if __name__ == "__main__":
    main()
