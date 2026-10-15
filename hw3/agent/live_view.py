"""Live window for the demo: shows the newest frame the camera captured during a run, with its settings, and the last
instrument call. Start it next to the run's terminal; it follows the newest camera folder of any run (Homework 3 runs
and the lab assistant's instrument team) unless --thread is given.
    python live_view.py [--thread uc1_autoexposure]
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")                                                     # Homework 3 runs
LAB_RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "lab", "runs")         # lab assistant (HW4/5)


def camera_dirs(thread=None):
    """Camera folders of the Homework 3 runs and of the lab assistant's instrument team (one thread, or all)."""
    t = thread or "*"
    return (glob.glob(os.path.join(RUNS, t, "work", "camera"))
            + glob.glob(os.path.join(LAB_RUNS, t, "teams", "instrument", "camera")))


def newest_camera_dir(thread=None):
    dirs = [d for d in camera_dirs(thread) if glob.glob(os.path.join(d, "cap*.npy"))]
    return max(dirs, key=lambda d: max(os.path.getmtime(f) for f in glob.glob(os.path.join(d, "cap*.npy")))) if dirs else None


def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def last_call(cam_dir):
    path = os.path.join(cam_dir, "calls.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thread")
    p.add_argument("--every", type=float, default=0.5, help="refresh interval in seconds")
    a = p.parse_args()
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.canvas.manager.set_window_title("Camera - live view of the agent's captures")
    shown, im = None, None
    while plt.fignum_exists(fig.number):
        cam_dir = newest_camera_dir(a.thread)
        npy = newest(os.path.join(cam_dir, "cap*.npy")) if cam_dir else None
        call = last_call(cam_dir) if cam_dir else None
        if npy and npy != shown:
            try:
                fr = np.load(npy)[0]
            except (OSError, ValueError):            # file still being written
                plt.pause(a.every)
                continue
            if im is None or im.get_array().shape != fr.shape:
                ax.clear()
                im = ax.imshow(fr, cmap="inferno")
                if not getattr(fig, "_cbar", None):
                    fig._cbar = fig.colorbar(im, ax=ax, label="DN")
            im.set_data(fr)
            im.set_clim(0, max(int(fr.max()), 1))
            shown = npy
        if call:
            res = call.get("result", {})
            s = res.get("settings", {})
            head = f"{os.path.basename(shown or '')}" + (f"   {s.get('exposure_us', '')} us, {s.get('gain_db', '')} dB" if s else "")
            ax.set_title(f"{head}\nlast MCP call: {call['tool']} {json.dumps(call.get('args', {}))[:70]} -> "
                         f"{'ok' if res.get('ok') else 'refused'}", fontsize=9)
        plt.pause(a.every)


if __name__ == "__main__":
    main()
