# Task 5 — PINN for the Silverbox oscillator

Model: `y' = v`, `v' = alpha_theta(y, v, u)`, alpha_theta a fully connected network with fixed input/output scaling,
integrated by RK4 at dt = 1/610.35 s, input at the RK4 midpoints from a windowed-sinc half-sample interpolator of u.

Training (training record only, samples 0..3071 of the excerpt [42650:46746]):
* stage A (`R_duffing`): identify a, b, c, g of `y'' + a y' + b y + c y^3 = g u` by the same rollout loss;
* stage B: equation-residual pre-fit of the network with finite-difference derivatives;
* stage C: rollout training on random windows (curriculum on the rollout length) plus the PINN residual
  `alpha_theta + a v + b y + c y^3 - g u` on collocation points in a box 1.5x beyond the measured range
  (the test segment reaches |y| = 0.205 V, the training record only 0.174 V).

Metric: one free run of the whole excerpt from measured y[0] and a finite-difference velocity; RMSE on the
1024 test samples (and on the 3072 training samples). `results/summary.md` lists every run.
