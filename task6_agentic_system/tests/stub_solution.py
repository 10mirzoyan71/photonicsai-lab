"""Compact self-contained Task 5 script (used by the offline smoke test's scripted executor).
Neural acceleration alpha(y, v, u), RK4 at dt, rollout training with a wall-clock guard, free-run metric."""
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

T0, BUDGET = time.time(), 150.0
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
fs = 610.35
dt = 1 / fs
d = sio.loadmat("SNLS80mV.mat")
u = d["V1"].ravel()[42650:46746].astype(float)
y = d["V2"].ravel()[42650:46746].astype(float)
u, y = u - u.mean(), y - y.mean()
NT = 3072
v = np.gradient(y, dt)
a = np.gradient(v, dt)
uh = 0.5 * (u[:-1] + u[1:])
S = torch.tensor([y[:NT].std(), v[:NT].std(), u[:NT].std(), a[:NT].std()])

net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))


def f(yy, vv, uu):
    return S[3] * net(torch.stack((yy / S[0], vv / S[1], uu / S[2]), -1)).squeeze(-1)


def step(yy, vv, u0, um, u1):
    k1y, k1v = vv, f(yy, vv, u0)
    k2y, k2v = vv + dt / 2 * k1v, f(yy + dt / 2 * k1y, vv + dt / 2 * k1v, um)
    k3y, k3v = vv + dt / 2 * k2v, f(yy + dt / 2 * k2y, vv + dt / 2 * k2v, um)
    k4y, k4v = vv + dt * k3v, f(yy + dt * k3y, vv + dt * k3v, u1)
    return yy + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y), vv + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


U, UH, Y, V, A = map(torch.tensor, (u, uh, y, v, a))
opt = torch.optim.Adam(net.parameters(), 3e-3)
for it in range(300):                                    # equation-residual pre-fit
    opt.zero_grad()
    loss = ((f(Y[:NT], V[:NT], U[:NT]) - A[:NT]) / S[3]).pow(2).mean()
    loss.backward()
    opt.step()
L, B = 64, 32
it = 0
while time.time() - T0 < BUDGET:                         # rollout training, wall-clock guard
    idx = torch.randint(0, NT - L, (B,))
    yy, vv, ys = Y[idx], V[idx], [Y[idx]]
    for k in range(L - 1):
        yy, vv = step(yy, vv, U[idx + k], UH[idx + k], U[idx + k + 1])
        ys.append(yy)
    loss = ((torch.stack(ys, 1) - Y[idx[:, None] + torch.arange(L)]) / S[0]).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    it += 1
print("rollout steps", it)
with torch.no_grad():                                    # free run of the whole excerpt
    yy, vv = Y[0], torch.tensor((-3 * y[0] + 4 * y[1] - y[2]) / (2 * dt))
    out = [yy]
    for k in range(len(y) - 1):
        yy, vv = step(yy, vv, U[k], UH[k], U[k + 1])
        out.append(yy)
yhat = torch.stack(out).numpy()
print(f"TRAIN_RMSE={np.sqrt(np.mean((yhat[:NT] - y[:NT]) ** 2)):.6f}")
print(f"TEST_RMSE={np.sqrt(np.mean((yhat[NT:] - y[NT:]) ** 2)):.6f}")
import matplotlib.pyplot as plt  # noqa: E402
t = np.arange(NT, len(y)) * dt
plt.figure(figsize=(9, 3))
plt.plot(t, y[NT:], label="measured")
plt.plot(t, yhat[NT:], "--", label="free run")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("y [V]")
plt.tight_layout()
plt.savefig("test_segment.png", dpi=120)
