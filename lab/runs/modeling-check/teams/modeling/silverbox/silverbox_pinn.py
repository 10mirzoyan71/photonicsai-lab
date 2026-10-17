"""Physics-informed neural ODE for the Silverbox (Duffing) oscillator.

    y'' + a y' + b y + c y^3 = g u(t)                                  (1)

The state is (y, v) with v = y'.  The acceleration is a fully connected
network,  v' = alpha_theta(y, v, u),  and the state is integrated with
classical RK4 at the sampling step dt = 1/fs, driven by the measured input.

Everything needed for one training run and for the free-run metric is here;
sweep.py runs the grid of architectures / hyperparameters and
make_figures.py produces the plots and the physics read-out.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

FS = 610.35
DT = 1.0 / FS
START = 42650          # first index of the excerpt in SNLS80mV.mat
N_TOTAL = 4096         # excerpt length -> slice [42650 : 46746]
N_TRAIN = 3072         # first 3072 samples train, last 1024 test

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAT = os.path.join(HERE, "..", "data", "SNLS80mV.mat")

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_excerpt(path: str = DEFAULT_MAT):
    """Return centred u, y of the excerpt (numpy, length 4096)."""
    d = sio.loadmat(path)
    u = d["V1"].ravel().astype(np.float64)[START:START + N_TOTAL]   # input voltage
    y = d["V2"].ravel().astype(np.float64)[START:START + N_TOTAL]   # output voltage
    assert len(u) == N_TOTAL
    return u - u.mean(), y - y.mean()


def central_diff(y: np.ndarray, dt: float = DT):
    """4th-order central differences for velocity and acceleration.
    The two samples at each edge fall back to np.gradient (2nd order)."""
    v = np.gradient(y, dt)
    a = np.gradient(v, dt)
    v[2:-2] = (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * dt)
    a[2:-2] = (-y[4:] + 16 * y[3:-1] - 30 * y[2:-2] + 16 * y[1:-3] - y[:-4]) / (12 * dt ** 2)
    return v, a


def start_velocity(y: np.ndarray, dt: float = DT) -> float:
    """Finite-difference velocity at the first sample (one-sided, 2nd order).
    Only samples 0..2 of the training record are used."""
    return (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dt)


def half_sample_input(u: np.ndarray, kind: str = "sinc", M: int = 8) -> np.ndarray:
    """u at t_k + dt/2 for k = 0..N-2, needed by the RK4 midpoint stages.

    'linear': (u_k + u_{k+1}) / 2.
    'sinc'  : 2M-tap Hann-windowed sinc (band-limited) interpolator; the input
              is a multisine up to 200 Hz, where linear interpolation loses
              almost half of the amplitude (cos(pi*200/610) ~ 0.51).
    Only u is used, which is known on the whole excerpt (train and test)."""
    if kind == "linear":
        return 0.5 * (u[:-1] + u[1:])
    m = np.arange(-M + 1, M + 1)
    t = m - 0.5
    h = np.sinc(t) * (0.5 + 0.5 * np.cos(np.pi * t / M))
    h /= h.sum()
    up = np.pad(u, (M - 1, M), mode="reflect")
    return np.correlate(up, h, mode="valid")[:-1]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU, "elu": nn.ELU, "relu": nn.ReLU, "softplus": nn.Softplus}


class AccelNet(nn.Module):
    """alpha_theta(y, v, u): fully connected network with fixed input/output scaling.

    inputs are divided by (y_s, v_s, u_s) so that each is O(1); the raw network
    output is multiplied by a_s, the scale of the acceleration (~ 1e4 V/s^2).
    Optional `linear_skip`: a bias-free linear layer from the scaled inputs to
    the output, added to the MLP (the linear part of a spring-damper force)."""

    def __init__(self, hidden=(64, 64), act="tanh", linear_skip=False, scales=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        layers, n_in = [], 3
        for h in hidden:
            layers += [nn.Linear(n_in, h), ACTS[act]()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.mlp = nn.Sequential(*layers)
        self.skip = nn.Linear(3, 1, bias=False) if linear_skip else None
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float64))

    def forward(self, y, v, u):
        ys, vs, us, a_s = self.scales
        x = torch.stack((y / ys, v / vs, u / us), dim=-1)
        out = self.mlp(x).squeeze(-1)
        if self.skip is not None:
            out = out + self.skip(x).squeeze(-1)
        return a_s * out


class DuffingForce(nn.Module):
    """Reference (not a network): alpha = -a v - b y - c y^3 + g u, 4 parameters."""

    def __init__(self, a=30.0, b=1.9e5, c=0.0, g=1.0e4, cubic=True):
        super().__init__()
        self.cubic = cubic
        self.p = nn.Parameter(torch.tensor([a, b, c, g], dtype=torch.float64))
        self.register_buffer("pscale", torch.tensor([10.0, 1e5, 1e6, 1e4]))
        with torch.no_grad():
            self.p.div_(self.pscale)

    def coeffs(self):
        c = (self.p * self.pscale).detach().numpy().copy()
        if not self.cubic:
            c[2] = 0.0
        return c

    def forward(self, y, v, u):
        a, b, c, g = self.p * self.pscale
        if not self.cubic:
            c = c * 0.0
        return -a * v - b * y - c * y ** 3 + g * u


def rk4_step(f, y, v, u0, uh, u1, dt=DT):
    """One classical RK4 step of  y' = v,  v' = f(y, v, u)."""
    k1y, k1v = v, f(y, v, u0)
    y2, v2 = y + 0.5 * dt * k1y, v + 0.5 * dt * k1v
    k2y, k2v = v2, f(y2, v2, uh)
    y3, v3 = y + 0.5 * dt * k2y, v + 0.5 * dt * k2v
    k3y, k3v = v3, f(y3, v3, uh)
    y4, v4 = y + dt * k3y, v + dt * k3v
    k4y, k4v = v4, f(y4, v4, u1)
    y_new = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_new = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_new, v_new


def rollout(f, y0, v0, U, UH, dt=DT):
    """Integrate from (y0, v0) driven by U[..., k] (at t_k) and UH[..., k] (at t_k + dt/2).
    U has shape (..., L), UH has shape (..., L-1).  Returns y of shape (..., L)."""
    ys, y, v = [y0], y0, v0
    for k in range(U.shape[-1] - 1):
        y, v = rk4_step(f, y, v, U[..., k], UH[..., k], U[..., k + 1], dt)
        ys.append(y)
    return torch.stack(ys, dim=-1)


# ----------------------------------------------------------------------------
# the metric: one free run of the whole excerpt
# ----------------------------------------------------------------------------
@torch.no_grad()
def free_run(f, u, y, uh):
    """Start at sample 0 with measured y[0] and a finite-difference velocity,
    integrate through the 3072 training samples and on through the 1024 test
    samples with the same state.  Only u is read after the start; y is never
    read again.  Returns prediction, train RMSE, test RMSE."""
    U = torch.as_tensor(u)
    UH = torch.as_tensor(uh)
    y0 = torch.tensor(y[0])
    v0 = torch.tensor(start_velocity(y))
    yhat = rollout(f, y0, v0, U, UH).numpy()
    err = yhat - y
    rmse_train = float(np.sqrt(np.mean(err[:N_TRAIN] ** 2)))
    rmse_test = float(np.sqrt(np.mean(err[N_TRAIN:] ** 2)))
    if not np.all(np.isfinite(yhat)):
        rmse_train = rmse_test = float("inf")
    return yhat, rmse_train, rmse_test


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "run"
    model: str = "mlp"                 # "mlp" or "duffing" (4-parameter reference)
    hidden: tuple = (64, 64)
    act: str = "tanh"
    linear_skip: bool = False
    input_scale: float = 1.0           # inputs divided by input_scale * std (training record)
    interp: str = "sinc"               # input at RK4 midpoints: "sinc" or "linear"
    pretrain_steps: int = 500          # equation-residual (collocation) pre-fit
    optimizer: str = "adam"            # "adam", "adamw", "rmsprop"
    lr: float = 3e-3
    weight_decay: float = 0.0
    steps: int = 1500                  # rollout-training steps
    rollout: int = 128                 # final rollout length (samples)
    rollout_start: int = 16            # curriculum: first rollout length
    curriculum_frac: float = 0.4       # fraction of steps used to grow the rollout
    batch: int = 32                    # windows per step
    clip: float = 1.0
    physics_weight: float = 0.0        # weight of the PINN residual loss on collocation points
    colloc: int = 512                  # collocation points per step
    colloc_box: float = 1.5            # collocation box = colloc_box * max|.| of the training record
    physics_source: str = "identified" # "identified": a,b,c,g fixed from the rollout fit of eq. (1)
                                       #   (stage A, file results/R_duffing.json);  "joint": learned with the net
    eval_every: int = 100
    seed: int = 0
    extra: dict = field(default_factory=dict)


def make_scales(y, v, u, a, input_scale=1.0):
    t = slice(0, N_TRAIN)                    # scales from the training record only
    k = float(input_scale)
    return (k * float(np.std(y[t])), k * float(np.std(v[t])), k * float(np.std(u[t])), float(np.std(a[t])))


def ls_duffing(y, v, a, u):
    """Least-squares a, b, c, g from finite-difference derivatives on the training record."""
    sl = slice(2, N_TRAIN - 2)
    Phi = np.stack([-v[sl], -y[sl], -y[sl] ** 3, u[sl]], 1)
    th, *_ = np.linalg.lstsq(Phi, a[sl], rcond=None)
    return th


def build(cfg: Config, scales):
    if cfg.model == "duffing":
        return DuffingForce(cubic=not cfg.extra.get("fix_c_zero", False))
    return AccelNet(tuple(cfg.hidden), cfg.act, cfg.linear_skip, scales)


def make_optimizer(cfg, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay or 1e-4)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr)
    raise ValueError(cfg.optimizer)


def train(cfg: Config, mat_path: str = DEFAULT_MAT, out_dir: str | None = None, verbose: bool = True):
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    u, y = load_excerpt(mat_path)
    v_fd, a_fd = central_diff(y)
    uh = half_sample_input(u, cfg.interp)
    scales = make_scales(y, v_fd, u, a_fd, cfg.input_scale)
    f = build(cfg, scales)
    n_params = sum(p.numel() for p in f.parameters())
    th_ls = ls_duffing(y, v_fd, a_fd, u)

    # PINN residual of eq. (1) with the network's acceleration:
    #   r = alpha_theta(y, v, u) + a v + b y + c y^3 - g u
    # on collocation points drawn uniformly in a box that extends beyond the measured
    # range (the test segment reaches |y| = 0.205 V, the training record only 0.174 V).
    # a, b, c, g are separate physical parameters (initialised by least squares),
    # learned together with the network; they are not weights of the network.
    phys = None
    if cfg.physics_weight > 0 and cfg.model == "mlp":
        if cfg.physics_source == "identified":
            with open(os.path.join(HERE, "results", "R_duffing.json")) as fh:
                ref = json.load(fh)["duffing_coeffs"]
            phys = DuffingForce(*[ref[k] for k in "abcg"])
            phys.p.requires_grad_(False)                 # identified in stage A, held fixed here
        else:
            phys = DuffingForce(*th_ls)
    if cfg.model == "duffing":
        with torch.no_grad():
            f.p.copy_(torch.as_tensor(th_ls) / f.pscale)
    box = torch.tensor([cfg.colloc_box * np.abs(y[:N_TRAIN]).max(), cfg.colloc_box * np.abs(v_fd[:N_TRAIN]).max(),
                        cfg.colloc_box * np.abs(u[:N_TRAIN]).max()])
    trainable = list(f.parameters()) + ([q for q in phys.parameters() if q.requires_grad] if phys is not None else [])

    def physics_loss():
        if phys is None:
            return torch.zeros(())
        X = (2 * torch.rand(cfg.colloc, 3) - 1) * box
        r = (f(X[:, 0], X[:, 1], X[:, 2]) - phys(X[:, 0], X[:, 1], X[:, 2])) / a_s
        return (r ** 2).mean()

    # training tensors: ONLY the first 3072 samples are used in any loss
    Ytr = torch.as_tensor(y[:N_TRAIN])
    Vtr = torch.as_tensor(v_fd[:N_TRAIN])
    Atr = torch.as_tensor(a_fd[:N_TRAIN])
    Utr = torch.as_tensor(u[:N_TRAIN])
    UHtr = torch.as_tensor(uh[:N_TRAIN - 1])
    ys, a_s = float(np.std(y[:N_TRAIN])), scales[3]
    history = []

    # ---- stage 1: equation-residual pre-fit at collocation points --------------
    # minimise || y''_fd - alpha_theta(y, y'_fd, u) ||^2 on the training samples:
    # the residual of (2) with finite-difference derivatives.  Gives the network
    # the right sign and size of the restoring force before any rollout.
    if cfg.pretrain_steps > 0:
        opt = torch.optim.Adam(trainable, lr=1e-3 if cfg.model == "duffing" else 3e-3)
        sl = slice(2, N_TRAIN - 2)
        for it in range(cfg.pretrain_steps):
            opt.zero_grad()
            res = (f(Ytr[sl], Vtr[sl], Utr[sl]) - Atr[sl]) / a_s
            loss = (res ** 2).mean() + cfg.physics_weight * physics_loss()
            loss.backward()
            opt.step()
        _, r_tr, r_te = free_run(f, u, y, uh)
        history.append({"stage": "pretrain", "step": 0, "loss": float(loss.detach()), "free_train": r_tr, "free_test": r_te})
        if verbose:
            print(f"[{cfg.name}] pretrain residual {float(loss.detach()):.3e} | free-run train {r_tr*1e3:.2f} mV test {r_te*1e3:.2f} mV", flush=True)

    # ---- stage 2: rollout (simulation-error) training on windows ---------------
    opt = make_optimizer(cfg, trainable)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1), eta_min=cfg.lr * 0.05)
    best = {"free_train": float("inf"), "state": None, "step": -1}
    n_grow = max(1, int(cfg.curriculum_frac * cfg.steps))
    for it in range(1, cfg.steps + 1):
        frac = min(1.0, it / n_grow)
        L = int(round(cfg.rollout_start * (cfg.rollout / cfg.rollout_start) ** frac)) if cfg.rollout > cfg.rollout_start else cfg.rollout
        idx = torch.as_tensor(rng.integers(2, N_TRAIN - L, size=cfg.batch))
        win = idx[:, None] + torch.arange(L)[None, :]
        U, UH, Yt = Utr[win], UHtr[win[:, :-1]], Ytr[win]
        yhat = rollout(f, Ytr[idx], Vtr[idx], U, UH)
        loss = ((yhat - Yt) ** 2).mean() / ys ** 2 + cfg.physics_weight * physics_loss()
        opt.zero_grad()
        loss.backward()
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
        opt.step()
        sched.step()
        if not torch.isfinite(loss):
            if verbose:
                print(f"[{cfg.name}] non-finite loss at step {it}; stopping", flush=True)
            break
        if it % cfg.eval_every == 0 or it == cfg.steps:
            _, r_tr, r_te = free_run(f, u, y, uh)
            history.append({"stage": "rollout", "step": it, "L": L, "loss": float(loss), "free_train": r_tr, "free_test": r_te})
            # checkpoint selection uses the TRAINING free-run error only
            if r_tr < best["free_train"]:
                best = {"free_train": r_tr, "state": {k: t.clone() for k, t in f.state_dict().items()}, "step": it,
                        "phys": phys.coeffs() if phys is not None else None}
            if verbose:
                print(f"[{cfg.name}] step {it:5d} L={L:4d} loss {float(loss):.3e} | free-run train {r_tr*1e3:7.3f} mV  test {r_te*1e3:7.3f} mV  ({time.time()-t0:.0f}s)", flush=True)

    if best["state"] is not None:
        f.load_state_dict(best["state"])
    yhat, r_tr, r_te = free_run(f, u, y, uh)
    result = {
        "config": asdict(cfg), "n_params": n_params, "rmse_train": r_tr, "rmse_test": r_te,
        "best_step": best["step"], "seconds": time.time() - t0, "history": history,
        "baseline_zero_test": float(np.sqrt(np.mean(y[N_TRAIN:] ** 2))),
    }
    result["ls_coeffs"] = dict(zip("abcg", map(float, th_ls)))
    if cfg.model == "duffing":
        result["duffing_coeffs"] = dict(zip("abcg", map(float, f.coeffs())))
    if best.get("phys") is not None:
        result["pinn_coeffs"] = dict(zip("abcg", map(float, best["phys"])))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{cfg.name}.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        torch.save({"config": asdict(cfg), "scales": scales, "state_dict": f.state_dict(),
                    "pinn_coeffs": result.get("pinn_coeffs")},
                   os.path.join(out_dir, f"{cfg.name}.pt"))
    if verbose:
        print(f"[{cfg.name}] DONE  params={n_params}  free-run RMSE train {r_tr*1e3:.3f} mV  test {r_te*1e3:.3f} mV  ({result['seconds']:.0f}s)", flush=True)
    return f, result


def load_model(pt_path: str):
    ck = torch.load(pt_path, weights_only=False)
    cfg = Config(**ck["config"])
    f = build(cfg, ck["scales"])
    f.load_state_dict(ck["state_dict"])
    return f, cfg


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="train one configuration")
    p.add_argument("--json", help="config as a JSON string", default="{}")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    train(Config(**json.loads(a.json)), out_dir=a.out)
