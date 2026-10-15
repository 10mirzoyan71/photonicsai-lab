"""Image statistics returned to the agent: level, saturation, temporal noise, regions, speckle and beam measures."""
from __future__ import annotations

import numpy as np


def _r(x, n=4):
    return float(round(float(x), n))


def frame_stats(frame: np.ndarray, max_dn: int, regions: list[dict] | None = None) -> dict:
    """Level and saturation of one frame, plus mean/std/max inside named rectangular regions (ROI coordinates)."""
    f = frame.astype(np.float64)
    sat = frame >= max_dn
    out = {"height": int(f.shape[0]), "width": int(f.shape[1]), "mean_dn": _r(f.mean(), 2), "std_dn": _r(f.std(), 2),
           "min_dn": int(frame.min()), "max_dn": int(frame.max()), "p99_9_dn": _r(np.percentile(f, 99.9), 1),
           "saturated_pixels": int(sat.sum()), "saturated_fraction": _r(sat.mean(), 6),
           "fraction_of_full_scale": {"mean": _r(f.mean() / max_dn), "p99_9": _r(np.percentile(f, 99.9) / max_dn),
                                      "max": _r(frame.max() / max_dn)}}
    if regions:
        out["regions"] = {}
        for i, r in enumerate(regions):
            name = str(r.get("name", f"region_{i + 1}"))
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
            sub = f[max(y, 0):y + h, max(x, 0):x + w]
            out["regions"][name] = ({"mean_dn": _r(sub.mean(), 2), "std_dn": _r(sub.std(), 2), "max_dn": int(sub.max()),
                                     "pixels": int(sub.size)} if sub.size else {"error": "region outside the frame"})
    return out


def temporal_noise(frames: np.ndarray, max_dn: int) -> dict:
    """Temporal noise from repeated frames of an unchanged scene: variance of the pair difference / 2 (removes the
    fixed pattern), averaged over consecutive pairs. Pixels saturated in any frame are excluded."""
    stack = frames.astype(np.float64)
    if stack.shape[0] < 2:
        return {}
    ok = (frames < max_dn).all(axis=0)
    if not ok.any():
        return {"error": "every pixel is saturated in at least one frame"}
    v = [np.var((stack[i] - stack[i + 1])[ok]) / 2.0 for i in range(stack.shape[0] - 1)]
    return {"pairs": len(v), "mean_dn": _r(stack[:, ok].mean(), 3), "temporal_var_dn2": _r(np.mean(v), 4),
            "temporal_std_dn": _r(np.sqrt(np.mean(v)), 4), "frame_mean_spread_dn": _r(stack.mean(axis=(1, 2)).std(), 3)}


def speckle_stats(frame: np.ndarray, black_level_dn: float = 0.0, region: dict | None = None) -> dict:
    """Speckle contrast C = std/mean of the black-subtracted intensity, and the grain size: FWHM of the normalised
    intensity autocorrelation along x and y (pixels). Use a region where the envelope is roughly flat."""
    f = frame.astype(np.float64) - black_level_dn
    if region:
        f = f[int(region["y"]):int(region["y"]) + int(region["height"]), int(region["x"]):int(region["x"]) + int(region["width"])]
    if f.size < 256 or f.mean() <= 0:
        return {"error": "region too small or no signal above the black level"}
    d = f / f.mean() - 1.0
    ac = np.real(np.fft.ifft2(np.abs(np.fft.fft2(d)) ** 2))
    ac = np.fft.fftshift(ac / ac.flat[0])
    cy, cx = np.array(ac.shape) // 2

    def fwhm(profile, c):
        right = profile[c:]
        k = np.argmax(right < 0.5) if (right < 0.5).any() else len(right) - 1
        if k == 0:
            return float("nan")
        x0, x1, y0, y1 = k - 1, k, right[k - 1], right[k]            # linear interpolation of the half-max crossing
        return 2.0 * (x0 + (y0 - 0.5) / (y0 - y1))

    return {"contrast": _r(f.std() / f.mean()), "mean_above_black_dn": _r(f.mean(), 2),
            "grain_fwhm_x_px": _r(fwhm(ac[cy, :], cx), 3), "grain_fwhm_y_px": _r(fwhm(ac[:, cx], cy), 3),
            "pixels": int(f.size)}


def beam_stats(frame: np.ndarray, black_level_dn: float = 0.0) -> dict:
    """Centroid and second-moment (D4sigma) widths of the black-subtracted frame (ROI coordinates)."""
    f = np.clip(frame.astype(np.float64) - black_level_dn, 0, None)
    tot = f.sum()
    if tot <= 0:
        return {"error": "no signal above the black level"}
    y, x = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    cx, cy = (f * x).sum() / tot, (f * y).sum() / tot
    sx = np.sqrt((f * (x - cx) ** 2).sum() / tot)
    sy = np.sqrt((f * (y - cy) ** 2).sum() / tot)
    return {"centroid_x_px": _r(cx, 2), "centroid_y_px": _r(cy, 2), "d4sigma_x_px": _r(4 * sx, 2),
            "d4sigma_y_px": _r(4 * sy, 2), "total_dn": _r(tot, 1)}
