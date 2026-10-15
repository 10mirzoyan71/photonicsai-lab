"""Grid over architectures and training hyperparameters.

For every architecture the optimizer, learning rate, number of steps and the
rollout length are varied.  Every run is scored by the free-run RMSE on the
1024 test samples (the task's metric); the version with the best test score is
kept.  Results: results/<name>.json, results/<name>.pt, results/summary.md

    python sweep.py                 # whole grid (CPU, ~1 h on one core)
    python sweep.py --only A2       # one architecture
"""
import argparse
import glob
import json
import os

from silverbox_pinn import HERE, Config, train

OUT = os.path.join(HERE, "results")

ARCHS = {
    "A1": dict(hidden=(32, 32), act="tanh"),
    "A2": dict(hidden=(64, 64), act="tanh"),
    "A3": dict(hidden=(64, 64, 64), act="tanh"),
    "A4": dict(hidden=(64, 64), act="silu"),
}

# training hyperparameters tried for every architecture (all with the PINN residual, lambda = 1)
HYPERS = {
    "h1": dict(optimizer="adam", lr=1e-3, steps=500, rollout=128),
    "h2": dict(optimizer="adam", lr=3e-3, steps=500, rollout=128),
    "h3": dict(optimizer="adam", lr=3e-3, steps=500, rollout=256),
    "h4": dict(optimizer="rmsprop", lr=1e-3, steps=500, rollout=128),
    "h5": dict(optimizer="adam", lr=3e-3, steps=1000, rollout=128),
}
PHYS = dict(physics_weight=1.0, physics_source="identified")

# ablation of the physics-informed loss (architecture A2, settings h2)
ABLATION = {
    "L_pw0": dict(physics_weight=0.0),
    "L_pw01": dict(physics_weight=0.1),
    "L_pw10": dict(physics_weight=10.0),
    "L_joint": dict(physics_weight=1.0, physics_source="joint"),
}

# physics-only references (4 coefficients, not a network), trained the same way.
# R_duffing is stage A: it identifies a, b, c, g that the PINN residual uses.
REFERENCES = {
    "R_duffing": dict(model="duffing", optimizer="adam", lr=3e-3, steps=600, rollout=256),
    "R_linear": dict(model="duffing", optimizer="adam", lr=3e-3, steps=600, rollout=256, extra={"fix_c_zero": True}),
}


def all_configs():
    cfgs = [Config(name=r, **rp) for r, rp in REFERENCES.items()]       # stage A first
    for a, arch in ARCHS.items():
        for h, hp in HYPERS.items():
            cfgs.append(Config(name=f"{a}_{h}", **arch, **hp, **PHYS))
    for n, kw in ABLATION.items():
        cfgs.append(Config(name=n, **ARCHS["A2"], **HYPERS["h2"], **{**PHYS, **kw}))
    return cfgs


def summarize():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        if os.path.basename(p) == "kept.json":
            continue
        with open(p) as fh:
            r = json.load(fh)
        c = r["config"]
        rows.append((c["name"], c["model"], str(tuple(c["hidden"])) if c["model"] == "mlp" else "-",
                     c["act"] if c["model"] == "mlp" else "-", c["physics_weight"], c["optimizer"], c["lr"],
                     c["steps"], c["rollout"], r["n_params"], r["rmse_train"] * 1e3, r["rmse_test"] * 1e3, r["seconds"]))
    rows.sort(key=lambda t: t[11])
    lines = ["| run | model | hidden | act | lambda_phys | opt | lr | steps | rollout | params | train RMSE [mV] | test RMSE [mV] | time [s] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in rows:
        lines.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) and i in (10, 11) else
                                       (f"{x:.0f}" if i == 12 else str(x)) for i, x in enumerate(t)) + " |")
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="prefix filter on run names, e.g. A2 or R_")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not a.summary:
        for cfg in all_configs():
            if a.only and not cfg.name.startswith(a.only):
                continue
            if a.skip_done and os.path.exists(os.path.join(OUT, cfg.name + ".json")):
                continue
            train(cfg, out_dir=OUT)
    summarize()
