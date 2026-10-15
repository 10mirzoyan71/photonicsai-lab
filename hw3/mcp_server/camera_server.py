"""MCP server that exposes the lab camera's controls as tools (stdio transport).

    python mcp_server/camera_server.py            # normally started by the agent's MCP client, not by hand

Environment:
    CAMERA_KIND    sim (default) | sim-room | hikrobot   which camera camera/connector.py opens
    CAMERA_OUT     folder for frames, previews and calls.jsonl (default: ./camera_out)
    CAMERA_SEED    seed of the simulated camera (default 1)
    CAMERA_LIMITS  JSON with lab limits, e.g. {"max_exposure_us": 500000, "max_gain_db": 12, "max_frames": 32}

Every tool returns a JSON object with ok=true and the result, or ok=false and the error. Every call is appended to
CAMERA_OUT/calls.jsonl (time, tool, arguments, result) - the record of what the instrument did.
Nothing may be printed to stdout: it carries the MCP protocol.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from camera import CameraError, open_camera  # noqa: E402
from camera.stats import beam_stats, frame_stats, speckle_stats, temporal_noise  # noqa: E402

OUT = os.path.abspath(os.environ.get("CAMERA_OUT", "camera_out"))
os.makedirs(OUT, exist_ok=True)
LIMITS = {"max_exposure_us": None, "max_gain_db": None, "max_frames": 32}
LIMITS.update(json.loads(os.environ.get("CAMERA_LIMITS", "{}") or "{}"))
KIND = os.environ.get("CAMERA_KIND", "sim")
CAM = open_camera(KIND, **({"seed": int(os.environ.get("CAMERA_SEED", "1"))} if KIND.startswith("sim") else {}))
STATE = {"captures": 0}


def _restore():
    """A restarted server (a paused run resumed in a new process) re-applies the settings recorded in calls.jsonl and
    continues the frame numbering, so nothing is overwritten and the camera is where the run left it."""
    path = os.path.join(OUT, "calls.jsonl")
    if not os.path.exists(path):
        return
    last, caps = {}, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            res = rec.get("result", {})
            if not res.get("ok"):
                continue
            if rec["tool"] in ("set_exposure", "set_gain", "set_illumination", "set_black_level", "set_adc_bit_depth"):
                last[rec["tool"]] = res
            elif rec["tool"] in ("set_roi", "reset_roi"):
                last["roi"] = res["roi"]
            elif rec["tool"] == "capture":
                caps += 1
    STATE["captures"] = caps
    applied = {}
    if "set_exposure" in last:
        applied["exposure_us"] = CAM.set_exposure(last["set_exposure"]["exposure_us"])
    if "set_gain" in last:
        applied["gain_db"] = CAM.set_gain(last["set_gain"]["gain_db"])
    if "roi" in last:
        r = last["roi"]
        applied["roi"] = CAM.set_roi(r["x"], r["y"], r["width"], r["height"])
    if "set_illumination" in last:
        applied["illumination"] = CAM.set_illumination(last["set_illumination"]["illumination"])
    if "set_black_level" in last:
        applied["black_level"] = CAM.set_black_level(last["set_black_level"]["black_level"])
    if "set_adc_bit_depth" in last:
        applied["adc_bit_depth"] = CAM.set_adc_bit_depth(last["set_adc_bit_depth"]["adc_bit_depth"])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": round(time.time(), 3), "seconds": 0, "tool": "server_restart_restore", "args": {},
                             "result": {"ok": True, "restored": applied, "captures_so_far": caps}}, default=str) + "\n")


_restore()
mcp = FastMCP("lab-camera", log_level="WARNING")


class Region(BaseModel):
    name: str = Field(description="label of the region")
    x: int = Field(description="left edge in ROI pixels")
    y: int = Field(description="top edge in ROI pixels")
    width: int
    height: int


def _log(tool, args, result, t0):
    rec = {"time": round(time.time(), 3), "seconds": round(time.time() - t0, 3), "tool": tool, "args": args, "result": result}
    with open(os.path.join(OUT, "calls.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return result


def _settings():
    i = CAM.info()
    return {"exposure_us": i["exposure_us"], "gain_db": i["gain_db"], "roi": i["roi"],
            **({"illumination": i["illumination"]} if i.get("illumination_control") else {}),
            **{k: i[k] for k in ("adc_bit_depth", "black_level_setting") if i.get(k) is not None}}


def _call(tool, args, fn):
    t0 = time.time()
    try:
        res = {"ok": True, **fn()}
    except CameraError as e:
        res = {"ok": False, "error": str(e)}
    return _log(tool, args, res, t0)


@mcp.tool()
def camera_info() -> dict:
    """Describe the camera: vendor, model, sensor size, bit depth (max DN), exposure and gain ranges, the lab limits,
    and the current settings (exposure, gain, ROI). Call this first."""
    return _call("camera_info", {}, lambda: {**CAM.info(), "lab_limits": LIMITS})


@mcp.tool()
def set_exposure(exposure_us: float) -> dict:
    """Set the exposure time in microseconds. Returns the value the camera applied."""
    def f():
        lim = LIMITS.get("max_exposure_us")
        if lim is not None and exposure_us > lim:
            raise CameraError(f"exposure {exposure_us} us is above the lab limit {lim} us")
        return {"exposure_us": CAM.set_exposure(exposure_us)}
    return _call("set_exposure", {"exposure_us": exposure_us}, f)


@mcp.tool()
def set_gain(gain_db: float) -> dict:
    """Set the analog gain in dB. Returns the value the camera applied. Higher gain = fewer electrons per DN."""
    def f():
        lim = LIMITS.get("max_gain_db")
        if lim is not None and gain_db > lim:
            raise CameraError(f"gain {gain_db} dB is above the lab limit {lim} dB")
        return {"gain_db": CAM.set_gain(gain_db)}
    return _call("set_gain", {"gain_db": gain_db}, f)


@mcp.tool()
def set_roi(x: int, y: int, width: int, height: int) -> dict:
    """Set the region of interest: offset (x, y) and size in sensor pixels. Frames then contain only this region."""
    return _call("set_roi", {"x": x, "y": y, "width": width, "height": height},
                 lambda: {"roi": dict(zip(("x", "y", "width", "height"), CAM.set_roi(x, y, width, height)))})


@mcp.tool()
def reset_roi() -> dict:
    """Use the full sensor again."""
    i = CAM.info()
    return _call("reset_roi", {}, lambda: {"roi": dict(zip(("x", "y", "width", "height"),
                                                            CAM.set_roi(0, 0, i["sensor_width"], i["sensor_height"])))})


@mcp.tool()
def set_illumination(level: float) -> dict:
    """Light source level 0..1 on the simulated camera only (0 = dark, for dark frames). On the lab camera it returns an
    error, and a person has to block or unblock the light."""
    return _call("set_illumination", {"level": level}, lambda: {"illumination": CAM.set_illumination(level)})


@mcp.tool()
def set_black_level(value: int) -> dict:
    """Set the black level: the offset added before digitisation, in the camera's own units (on the Hikrobot camera a
    LARGER value gives a LOWER dark level; on the simulated camera it is the dark level in DN). Measure the dark level
    with dark frames afterwards: it must stay well above 0 DN, or the dark noise is clipped and under-estimated."""
    return _call("set_black_level", {"value": value}, lambda: {"black_level": CAM.set_black_level(value)})


@mcp.tool()
def set_adc_bit_depth(bits: int) -> dict:
    """Set the bits of the analogue-to-digital converter (8 or 12). Frames keep the same DN scale, but an 8-bit ADC
    gives only 256 levels (steps of about 16 DN), which hides a read noise of a few DN. Use 12 bits for noise
    measurements."""
    return _call("set_adc_bit_depth", {"bits": bits}, lambda: {"adc_bit_depth": CAM.set_adc_bit_depth(bits)})


@mcp.tool()
def read_temperature() -> dict:
    """The camera's device temperature in degrees C (dark current roughly doubles every 6-8 C, so long series and
    dark frames should report it)."""
    return _call("read_temperature", {}, lambda: {"temperature_c": CAM.temperature_c()})


@mcp.tool()
def save_tiff(frame_file: str) -> dict:
    """Save a captured frame stack (.npy) as a 16-bit TIFF, one page per frame, for ImageJ/Fiji or other software.
    Returns the TIFF file."""
    def f():
        from PIL import Image
        path = frame_file if os.path.isabs(frame_file) else os.path.join(OUT, frame_file)
        if not os.path.exists(path):
            raise CameraError(f"no such frame file: {frame_file}")
        stack = np.load(path)
        pages = [Image.fromarray(fr.astype(np.uint16)) for fr in stack]
        tif = os.path.splitext(path)[0] + ".tif"
        pages[0].save(tif, save_all=True, append_images=pages[1:])
        return {"tiff_file": tif, "pages": len(pages), "shape": list(stack.shape[1:]), "dtype": "uint16"}
    return _call("save_tiff", {"frame_file": frame_file}, f)


@mcp.tool()
def capture(n_frames: int = 1, label: str = "", regions: Optional[list[Region]] = None) -> dict:
    """Acquire n_frames frames with the current settings and save them. Returns the statistics of the first frame
    (mean, std, max, 99.9th percentile, saturated pixels, fraction of full scale, optional region statistics), the
    temporal noise of the series when n_frames >= 2 (pair-difference variance / 2), the frame file (.npy, uint16 stack)
    and a preview image (.png)."""
    args = {"n_frames": n_frames, "label": label, "regions": [r.model_dump() for r in regions] if regions else None}

    def f():
        if not 1 <= n_frames <= LIMITS["max_frames"]:
            raise CameraError(f"n_frames must be 1..{LIMITS['max_frames']}")
        stack = np.stack([CAM.grab() for _ in range(n_frames)])
        npy, png = _save_stack(stack, label)
        out = {"settings": _settings(), "stats": frame_stats(stack[0], CAM.max_dn, args["regions"]), "frame_file": npy,
               "preview_png": png, "n_frames": n_frames}
        if n_frames >= 2:
            out["temporal_noise"] = temporal_noise(stack, CAM.max_dn)
        return out
    return _call("capture", args, f)


def _save_stack(stack, label, title_extra=""):
    """Save a uint16 stack as capNNN_label.npy and a preview of its first frame as capNNN_label.png."""
    STATE["captures"] += 1
    tag = f"cap{STATE['captures']:03d}" + (f"_{''.join(c if c.isalnum() else '_' for c in label)[:40]}" if label else "")
    npy, png = os.path.join(OUT, tag + ".npy"), os.path.join(OUT, tag + ".png")
    np.save(npy, stack)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(stack[0], cmap="inferno", vmin=0, vmax=CAM.max_dn)
    fig.colorbar(im, ax=ax, label="DN")
    s = _settings()
    ax.set_title(f"{tag}: {s['exposure_us']:g} us, {s['gain_db']:g} dB{title_extra}", fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return npy, png


@mcp.tool()
def capture_series(n_frames: int, interval_s: float, label: str = "", regions: Optional[list[Region]] = None) -> dict:
    """Stability measurement: n_frames frames, one every interval_s seconds, with the current settings. Returns the time
    series of the frame mean (and of each region's mean), its relative standard deviation and drift, and saves the
    first and last frames (.npy, .png) and the series (.json). The whole series may last at most the lab limit
    max_series_s (default 300 s)."""
    args = {"n_frames": n_frames, "interval_s": interval_s, "label": label,
            "regions": [r.model_dump() for r in regions] if regions else None}

    def f():
        if not 2 <= n_frames <= 600 or interval_s < 0:
            raise CameraError("n_frames must be 2..600 and interval_s >= 0")
        if (n_frames - 1) * interval_s > LIMITS.get("max_series_s", 300):
            raise CameraError(f"the series would last {(n_frames - 1) * interval_s:g} s, above the limit "
                              f"{LIMITS.get('max_series_s', 300)} s")
        t0, ts, means, maxes, sats, reg = time.time(), [], [], [], [], {}
        first = last = None
        for i in range(n_frames):
            if i:
                time.sleep(max(0.0, t0 + i * interval_s - time.time()))
            fr = CAM.grab()
            ts.append(round(time.time() - t0, 3))
            st = frame_stats(fr, CAM.max_dn, args["regions"])
            means.append(st["mean_dn"]); maxes.append(st["max_dn"]); sats.append(st["saturated_pixels"])
            for name, r in st.get("regions", {}).items():
                reg.setdefault(name, []).append(r.get("mean_dn"))
            first = fr if first is None else first
            last = fr
        npy, png = _save_stack(np.stack([first, last]), label or "series", " (first frame of the series)")
        m = np.array(means)
        series = {"t_s": ts, "mean_dn": means, "max_dn": maxes, "saturated_pixels": sats, "region_mean_dn": reg}
        jpath = npy[:-4] + "_series.json"
        json.dump(series, open(jpath, "w", encoding="utf-8"), indent=1)
        slope = float(np.polyfit(ts, m, 1)[0]) if len(ts) > 1 and ts[-1] > 0 else 0.0
        return {"settings": _settings(), "n_frames": n_frames, "duration_s": ts[-1],
                "summary": {"mean_dn": round(float(m.mean()), 3), "rel_std_percent": round(float(100 * m.std() / m.mean()), 4),
                            "drift_percent_first_to_last": round(float(100 * (m[-1] - m[0]) / m[0]), 4),
                            "slope_percent_per_min": round(float(100 * slope * 60 / m.mean()), 4),
                            "min_dn": round(float(m.min()), 3), "max_dn": round(float(m.max()), 3)},
                "series": series if n_frames <= 40 else {"note": f"{n_frames} points, see series_file"},
                "frame_file": npy, "preview_png": png, "series_file": jpath}
    return _call("capture_series", args, f)


@mcp.tool()
def analyze_frame(frame_file: str, black_level_dn: float = 0.0, region: Optional[Region] = None) -> dict:
    """Speckle and beam measures of a saved frame (first frame of the file): speckle contrast and grain size (FWHM of
    the intensity autocorrelation) inside `region` (or the whole frame), and the beam centroid and D4sigma widths.
    Subtract the black level measured from dark frames."""
    args = {"frame_file": frame_file, "black_level_dn": black_level_dn, "region": region.model_dump() if region else None}

    def f():
        path = frame_file if os.path.isabs(frame_file) else os.path.join(OUT, frame_file)
        if not os.path.exists(path):
            raise CameraError(f"no such frame file: {frame_file}")
        fr = np.load(path)[0]
        return {"speckle": speckle_stats(fr, black_level_dn, args["region"]), "beam": beam_stats(fr, black_level_dn)}
    return _call("analyze_frame", args, f)


if __name__ == "__main__":
    mcp.run("stdio")
