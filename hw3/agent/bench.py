"""Simulated wavefront-shaping bench and the focusing tool (Homework 2, Task 2).

Imaging path:  laser -> DMD (binary amplitude) -> scattering medium -> camera

    E_out = |T E_in|^2  (+ camera noise)

* DMD: dmd_side x dmd_side mirrors, each on (1) or off (0); it changes only the amplitude of E_in.
  The expanded laser beam illuminates it with a Gaussian amplitude profile.
* T: transmission matrix from every mirror to every camera pixel.  Each column (the field one
  mirror produces on the camera) is an independent fully developed speckle field: circular
  Gaussian white noise low-passed by a circular pupil, which sets the speckle grain on the camera.
* Camera: shot noise (Poisson), read noise (Gaussian), laser intensity noise per shot, full-well
  clipping.  Two shots of the same pattern are therefore never identical.

Focusing search (the algorithm of the task):
  1. divide the DMD into macropixels (m x m mirrors switched together), start from a random pattern;
  2. after each transmission read the intensity in the focus region and in the proximal background;
  3. split the background ring into several regions; enhancement = mean focus intensity / mean of
     the region means;
  4. flip some macropixels, keep the flip if the enhancement went up, repeat.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import scipy.fft as sfft


# ----------------------------------------------------------------------------
# the bench
# ----------------------------------------------------------------------------
@dataclass
class BenchConfig:
    dmd_side: int = 64            # mirrors per side of the used DMD area
    cam_side: int = 64            # camera pixels per side
    speckle_grain: float = 2.5    # speckle grain on the camera (FWHM of the intensity correlation) [px]
    beam_waist: float = 0.8       # Gaussian illumination waist, in units of half the DMD side
    mean_counts: float = 400.0    # mean camera signal of a random 50% pattern [photo-electrons / px]
    read_noise: float = 5.0       # [e- rms]
    laser_rin: float = 0.01       # relative laser intensity noise per shot
    full_well: float = 2.6e5      # saturation [e-]
    seed: int = 2026              # the scattering medium


class Bench:
    def __init__(self, cfg: BenchConfig = BenchConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        n, c = cfg.dmd_side, cfg.cam_side
        self.n_mirrors, self.n_pix = n * n, c * c
        # pupil radius in FFT-index units: intensity correlation FWHM ~ 0.514 / rho  (rho in cycles/px)
        k = sfft.fftfreq(c) * c
        kx, ky = np.meshgrid(k, k, indexing="xy")
        pupil = (np.hypot(kx, ky) <= 0.514 * c / cfg.speckle_grain).astype(np.complex64)
        T = np.empty((self.n_mirrors, self.n_pix), dtype=np.complex64)
        for s in range(0, self.n_mirrors, 512):                      # chunks keep the memory low
            m = min(512, self.n_mirrors - s)
            w = (rng.standard_normal((m, c, c)) + 1j * rng.standard_normal((m, c, c))).astype(np.complex64)
            T[s:s + m] = sfft.ifft2(sfft.fft2(w, workers=-1) * pupil, workers=-1).reshape(m, -1)
        T /= np.sqrt(np.mean(np.abs(T) ** 2))                        # <|T|^2> = 1
        self.T = T                                                   # (mirrors, pixels)
        x = (np.arange(n) - (n - 1) / 2) / (n / 2)
        X, Y = np.meshgrid(x, x, indexing="xy")
        self.illum = np.exp(-(X ** 2 + Y ** 2) / cfg.beam_waist ** 2).ravel().astype(np.float32)
        self.kappa = cfg.mean_counts / (0.5 * float(np.sum(self.illum ** 2)))   # e- per unit |E|^2
        self.noise_rng = np.random.default_rng(cfg.seed + 1)
        self.shots = 0

    # -- macropixels -------------------------------------------------------------
    def macro_matrix(self, m: int, pixels=None) -> np.ndarray:
        """T for macropixels of m x m mirrors, illumination included: shape (pixels, N)."""
        n = self.cfg.dmd_side
        if n % m:
            raise ValueError(f"macropixel {m} does not divide the DMD side {n}")
        T = self.T if pixels is None else self.T[:, pixels]
        A = (T * self.illum[:, None]).reshape(n // m, m, n // m, m, -1).sum(axis=(1, 3))
        return np.ascontiguousarray(A.reshape((n // m) ** 2, -1).T)

    def macro_amplitudes(self, m: int) -> np.ndarray:
        n = self.cfg.dmd_side
        return np.sqrt((self.illum ** 2).reshape(n // m, m, n // m, m).sum(axis=(1, 3))).ravel()

    def mirror_pattern(self, s_macro: np.ndarray, m: int) -> np.ndarray:
        k = self.cfg.dmd_side // m
        return np.kron(s_macro.reshape(k, k).astype(np.uint8), np.ones((m, m), np.uint8))

    # -- the camera --------------------------------------------------------------
    def detect(self, field: np.ndarray) -> np.ndarray:
        """Noisy camera counts for a (noise-free) complex field on some pixels."""
        cfg = self.cfg
        self.shots += 1
        lam = self.kappa * np.abs(field) ** 2 * max(0.0, 1.0 + cfg.laser_rin * self.noise_rng.standard_normal())
        counts = self.noise_rng.poisson(lam).astype(np.float64) + cfg.read_noise * self.noise_rng.standard_normal(lam.shape)
        return np.clip(counts, 0.0, cfg.full_well)


# ----------------------------------------------------------------------------
# focus region and background regions
# ----------------------------------------------------------------------------
class Regions:
    def __init__(self, cam_side, focus_size, bg_inner, bg_outer, bg_regions, center=None):
        c = cam_side
        cx, cy = center if center is not None else (c // 2, c // 2)
        y, x = np.mgrid[0:c, 0:c]
        h = focus_size // 2
        self.focus = ((np.abs(x - cx) <= h) & (np.abs(y - cy) <= h)).ravel()
        r = np.hypot(x - cx, y - cy).ravel()
        ring = (r > bg_inner) & (r <= bg_outer) & ~self.focus
        ang = (np.arctan2(y - cy, x - cx).ravel() + np.pi) / (2 * np.pi)
        self.sector = np.where(ring, np.minimum((ang * bg_regions).astype(int), bg_regions - 1), -1)
        self.k = bg_regions
        self.center, self.focus_size, self.bg_inner, self.bg_outer = (cx, cy), focus_size, bg_inner, bg_outer
        self.far = (r > bg_outer) & ~self.focus                       # the rest of the camera (not used by the search)
        self.roi = np.flatnonzero(self.focus | ring)                 # pixels read during the search
        self.f_roi = self.focus[self.roi]
        self.s_roi = self.sector[self.roi]

    def enhancement(self, I_roi: np.ndarray) -> float:
        """mean focus intensity / mean over background regions of the region mean."""
        bg = [I_roi[self.s_roi == j].mean() for j in range(self.k)]
        return float(I_roi[self.f_roi].mean() / max(np.mean(bg), 1e-9))

    def enhancement_full(self, I: np.ndarray) -> float:
        return self.enhancement(I.ravel()[self.roi])

    def enhancement_far(self, I: np.ndarray) -> float:
        """focus against the mean of the camera outside the background ring (the standard definition:
        focus over the average speckle, which the search does not see)."""
        I = I.ravel()
        return float(I[self.focus].mean() / max(I[self.far].mean(), 1e-9))


# ----------------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------------
def theory_enhancement(amps: np.ndarray) -> tuple[float, float]:
    """Binary-amplitude optimum for a single speckle grain: eta ~ N_eff / (2 pi),
    N_eff = (sum A^2)^2 / sum A^4 (equal amplitudes: N_eff = N)."""
    n_eff = float(np.sum(amps ** 2) ** 2 / np.sum(amps ** 4))
    return n_eff, n_eff / (2 * np.pi)


def focus_search(bench: Bench, macropixel=4, focus_size=3, bg_inner=6, bg_outer=14, bg_regions=8,
                 iterations=3000, flips=8, flips_final=1, remeasure=True, final_shots=5,
                 focus_x=None, focus_y=None, seed=0, progress=None):
    cfg = bench.cfg
    t0 = time.time()
    center = (focus_x if focus_x is not None else cfg.cam_side // 2, focus_y if focus_y is not None else cfg.cam_side // 2)
    reg = Regions(cfg.cam_side, focus_size, bg_inner, bg_outer, bg_regions, center)
    A = bench.macro_matrix(macropixel, reg.roi)                       # (roi pixels, N)
    N = A.shape[1]
    rng = np.random.default_rng(seed)
    s = rng.random(N) < 0.5                                           # random start
    s_start = s.copy()
    E = A @ s.astype(np.complex64)
    measure = lambda field: reg.enhancement(bench.detect(field))      # one camera shot -> enhancement
    eta_cur = measure(E)
    trace, accepted, shots0 = [eta_cur], 0, bench.shots
    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        k = max(1, int(round(flips + (flips_final - flips) * frac)))  # flips per iteration, annealed
        idx = rng.choice(N, size=min(k, N), replace=False)
        E_try = E + A[:, idx] @ np.where(s[idx], -1.0, 1.0).astype(np.complex64)
        eta_try = measure(E_try)
        eta_ref = measure(E) if remeasure else eta_cur                # fresh shot of the current pattern
        if eta_try > eta_ref:
            s[idx] = ~s[idx]
            E, eta_cur = E_try, eta_try
            accepted += 1
        else:
            eta_cur = eta_ref
        trace.append(eta_cur)
        if progress and (it + 1) % max(1, iterations // 10) == 0:
            progress(it + 1, iterations, eta_cur)

    # final images: full camera, fresh shots
    Afull = bench.macro_matrix(macropixel)
    E_final, E_start = Afull @ s.astype(np.complex64), Afull @ s_start.astype(np.complex64)
    finals = [bench.detect(E_final) for _ in range(final_shots)]
    etas = [reg.enhancement_full(I) for I in finals]
    etas_far = [reg.enhancement_far(I) for I in finals]
    img_start = bench.detect(E_start).reshape(cfg.cam_side, -1)
    img_final = np.mean(finals, axis=0).reshape(cfg.cam_side, -1)

    # diagnostics a real bench cannot measure (kept for the report, not given to the agent)
    eta_noise_free = reg.enhancement_full(bench.kappa * np.abs(E_final) ** 2)
    eta_far_noise_free = reg.enhancement_far(bench.kappa * np.abs(E_final) ** 2)
    amps = bench.macro_amplitudes(macropixel)
    n_eff, eta_theory = theory_enhancement(amps)
    t_c = Afull[reg.center[1] * cfg.cam_side + reg.center[0]]       # TM row of the focus centre
    best, best_far = 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.abs(t_c) ** 2)))
    for phi in np.linspace(0, 2 * np.pi, 32, endpoint=False):         # binary-amplitude pattern built from the known T:
        proj = np.real(t_c * np.exp(-1j * phi))                       # keep segments in phase with the focus,
        for thr in (0.0, 0.25, 0.5, 0.75, 1.0):                       # above a threshold (0 = Akbulut's rule)
            so = (proj > thr * rms).astype(np.complex64)
            if not so.any():
                continue
            Io = bench.kappa * np.abs(Afull @ so) ** 2
            if reg.enhancement_far(Io) > best_far:
                best, best_far = reg.enhancement_full(Io), reg.enhancement_far(Io)
    return {
        "segments": int(N),
        "enhancement": float(np.mean(etas)), "enhancement_std": float(np.std(etas)),
        "enhancement_initial": float(reg.enhancement_full(img_start)),
        "enhancement_far": float(np.mean(etas_far)),
        "accepted_flips": int(accepted), "camera_shots": int(bench.shots - shots0 + final_shots + 1),
        "seconds": round(time.time() - t0, 2),
        "trace": [float(x) for x in np.array(trace)[np.linspace(0, len(trace) - 1, 11).astype(int)]],
        "_trace_full": np.array(trace), "_img_start": img_start, "_img_final": img_final,
        "_pattern": bench.mirror_pattern(s, macropixel), "_regions": reg,
        "diagnostics": {"enhancement_noise_free": float(eta_noise_free), "enhancement_far_noise_free": float(eta_far_noise_free),
                        "enhancement_oracle_TM": float(best), "enhancement_oracle_TM_far": float(best_far),
                        "n_eff": n_eff, "enhancement_theory_Neff_over_2pi": float(eta_theory)},
    }


def save_figure(res, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    reg = res["_regions"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    vmax = np.percentile(res["_img_final"], 99.95)
    for a, img, lab in ((ax[0], res["_img_start"], f"camera, random start (eta={res['enhancement_initial']:.1f})"),
                        (ax[1], res["_img_final"], f"after search (eta={res['enhancement']:.1f}, vs far field {res['enhancement_far']:.1f})")):
        im = a.imshow(img, cmap="inferno", vmin=0, vmax=vmax if a is ax[1] else np.percentile(img, 99.9))
        a.set_title(lab, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        h = reg.focus_size / 2
        a.add_patch(Rectangle((reg.center[0] - h, reg.center[1] - h), reg.focus_size, reg.focus_size, fill=False, ec="cyan", lw=1))
        for r in (reg.bg_inner, reg.bg_outer):
            a.add_patch(Circle(reg.center, r, fill=False, ec="w", lw=0.6, ls="--"))
        for j in range(reg.k):
            th = -np.pi + 2 * np.pi * j / reg.k
            a.plot([reg.center[0] + reg.bg_inner * np.cos(th), reg.center[0] + reg.bg_outer * np.cos(th)],
                   [reg.center[1] + reg.bg_inner * np.sin(th), reg.center[1] + reg.bg_outer * np.sin(th)], "w-", lw=0.5)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    ax[2].imshow(res["_pattern"], cmap="gray", interpolation="nearest")
    ax[2].set_title(f"DMD pattern ({res['segments']} macropixels)", fontsize=9)
    ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[3].plot(res["_trace_full"], lw=0.8)
    ax[3].set_xlabel("iteration"); ax[3].set_ylabel("enhancement (current pattern)")
    ax[3].grid(alpha=0.3)
    ax[3].set_title("search progress", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# the tool the executor calls
# ----------------------------------------------------------------------------
LIMITS = {"macropixel": (1, 32), "focus_size": (1, 15), "bg_regions": (2, 16), "iterations": (1, 40000), "flips": (1, 4096),
          "final_shots": (1, 20)}
CAMERA_LIMITS = {"read_noise_e": (0.0, 100.0), "full_well_e": (1e3, 1e6), "mean_signal_e": (1.0, 1e5),
                 "intensity_noise": (0.0, 0.5)}


class FocusingTool:
    """Runs one focusing experiment on the simulated bench and returns the enhancement and the camera images.
    The bench (the scattering medium) is fixed for the whole run, like a real sample on the table."""

    name = "focusing_tool"

    def __init__(self, out_dir: str, bench_cfg: BenchConfig = BenchConfig(), max_calls_per_turn: int = 3, verbose=True):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.bench = Bench(bench_cfg)
        self.default_cfg = bench_cfg
        self.max_calls_per_turn = max_calls_per_turn
        self.verbose = verbose
        self.log = []
        self._calls = 0

    def new_turn(self):
        self._calls = 0

    def describe_bench(self) -> str:
        c = self.bench.cfg
        return (f"DMD area {c.dmd_side}x{c.dmd_side} mirrors (binary amplitude), camera {c.cam_side}x{c.cam_side} px, "
                f"speckle grain about {c.speckle_grain} px, camera noise: shot + {c.read_noise} e- read noise, "
                f"{c.laser_rin * 100:.0f}% laser intensity noise per shot, full well {c.full_well:g} e-, mean signal of a "
                f"random pattern {c.mean_counts:g} e-/px (the camera model can be set per experiment from a real camera's "
                f"measured numbers). Focus target default: camera centre "
                f"({c.cam_side // 2}, {c.cam_side // 2}).")

    def __call__(self, macropixel: int, focus_size: int, bg_inner: int, bg_outer: int, bg_regions: int,
                 iterations: int, flips: int, flips_final: int = 1, remeasure: bool = True, final_shots: int = 5,
                 focus_x: int | None = None, focus_y: int | None = None, label: str = "",
                 read_noise_e: float | None = None, full_well_e: float | None = None, mean_signal_e: float | None = None,
                 intensity_noise: float | None = None) -> dict:
        settings = dict(macropixel=macropixel, focus_size=focus_size, bg_inner=bg_inner, bg_outer=bg_outer,
                        bg_regions=bg_regions, iterations=iterations, flips=flips, flips_final=flips_final,
                        remeasure=bool(remeasure), final_shots=final_shots, focus_x=focus_x, focus_y=focus_y)
        # the bench's camera model for this experiment: the Homework 2 defaults, or a real camera's measured numbers
        d = self.default_cfg
        cam = {"read_noise_e": d.read_noise if read_noise_e is None else float(read_noise_e),
               "full_well_e": d.full_well if full_well_e is None else float(full_well_e),
               "mean_signal_e": d.mean_counts if mean_signal_e is None else float(mean_signal_e),
               "intensity_noise": d.laser_rin if intensity_noise is None else float(intensity_noise)}
        err = self._validate(settings) or self._validate_camera(cam)
        if err:
            return {"ok": False, "error": err, "settings": settings, "camera_model": cam}
        self.bench.cfg = replace(d, read_noise=cam["read_noise_e"], full_well=cam["full_well_e"],
                                 mean_counts=cam["mean_signal_e"], laser_rin=cam["intensity_noise"])
        self.bench.kappa = cam["mean_signal_e"] / (0.5 * float(np.sum(self.bench.illum ** 2)))
        c = self.bench.cfg
        if self._calls >= self.max_calls_per_turn:
            return {"ok": False, "error": f"at most {self.max_calls_per_turn} experiments per executor turn; report now"}
        self._calls += 1
        k = len(self.log) + 1
        if self.verbose:
            print(f"  [bench] experiment {k}: {json.dumps(settings)}, camera model {json.dumps(cam)}", flush=True)
        prog = (lambda i, n, e: print(f"  [bench] experiment {k}: iteration {i}/{n}, enhancement {e:.1f}", flush=True)) if self.verbose else None
        res = focus_search(self.bench, seed=1000 + k, progress=prog, **settings)
        fig = os.path.join(self.out_dir, f"experiment_{k:02d}.png")
        save_figure(res, fig, title=f"experiment {k}: macropixel {macropixel} ({res['segments']} segments), focus {focus_size}px, "
                                   f"background {bg_regions} regions r={bg_inner}-{bg_outer}px, {iterations} iterations")
        np.savez_compressed(os.path.join(self.out_dir, f"experiment_{k:02d}.npz"), camera_start=res["_img_start"],
                            camera_final=res["_img_final"], dmd_pattern=res["_pattern"], trace=res["_trace_full"])
        public = {k2: v for k2, v in res.items() if not k2.startswith("_") and k2 != "diagnostics"}
        out = {"ok": True, "experiment": k, "label": label, "settings": settings, "camera_model": cam, **public,
               "saturated_pixels_final": int(np.sum(res["_img_final"] >= 0.999 * c.full_well)),
               "images": {"figure": fig, "camera_npz": fig[:-4] + ".npz"}}
        self.log.append({**out, "diagnostics": res["diagnostics"], "time": time.time()})
        with open(os.path.join(self.out_dir, "experiments.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.log[-1]) + "\n")
        if self.verbose:
            print(f"  [bench] experiment {k} done: enhancement {out['enhancement']:.1f} +/- {out['enhancement_std']:.1f} "
                  f"(vs far field {out['enhancement_far']:.1f}) "
                  f"({res['segments']} segments, {out['seconds']} s) -> {fig}", flush=True)
        return out                                                   # what the agent sees (no diagnostics)

    @staticmethod
    def _validate_camera(cam):
        for key, (lo, hi) in CAMERA_LIMITS.items():
            if not (lo <= cam[key] <= hi):
                return f"{key} must be in [{lo:g}, {hi:g}]"
        return None

    def _validate(self, s):
        c = self.bench.cfg
        for key, (lo, hi) in LIMITS.items():
            if not (lo <= int(s[key]) <= hi):
                return f"{key} must be in [{lo}, {hi}]"
        if c.dmd_side % s["macropixel"]:
            return f"macropixel must divide the DMD side {c.dmd_side} (1, 2, 4, 8, 16, 32)"
        if s["focus_size"] % 2 == 0:
            return "focus_size must be odd"
        if not (s["focus_size"] // 2 + 1 <= s["bg_inner"] < s["bg_outer"] <= c.cam_side // 2 - 1):
            return f"need focus_size//2 + 1 <= bg_inner < bg_outer <= {c.cam_side // 2 - 1}"
        if s["flips"] > (c.dmd_side // s["macropixel"]) ** 2:
            return "flips exceeds the number of macropixels"
        n_seg = (c.dmd_side // s["macropixel"]) ** 2
        if s["iterations"] * (2 if s["remeasure"] else 1) > 80000:
            return "too many camera shots for one experiment (iterations x shots per iteration must be <= 80000)"
        if n_seg > 1024 and s["iterations"] > 20000:
            return "with more than 1024 segments use at most 20000 iterations"
        return None
