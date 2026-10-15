"""The web app's engine, without Streamlit (so it can be tested on its own): a worker thread per run, the saved runs,
and the camera frames and figures a run produced."""
from __future__ import annotations

import glob
import json
import os
import queue
import threading

import numpy as np

from lab_assistant import TEAMS, LabRun

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("LAB_RUNS") or os.path.join(HERE, "runs")      # LAB_RUNS: another runs folder (tests)


class Worker:
    """Owns one LabRun on its own thread: the model calls take minutes, and the page keeps refreshing meanwhile.
    The page sends jobs (start, resume, retry) and reads busy / pending / error / finished and the run's events."""

    def __init__(self, thread: str, runs: str = RUNS, **cfg):
        self.thread, self.runs, self.cfg = thread, runs, cfg
        self.jobs = queue.Queue()
        self.error, self.pending, self.finished, self.closed = None, None, False, False
        self._loading, self._open_jobs = True, 0            # busy = loading the run, or a job sent and not yet done
        self.run: LabRun | None = None
        self.th = threading.Thread(target=self._loop, daemon=True, name=f"lab-{thread}")
        self.th.start()

    def _loop(self):
        try:
            self.run = LabRun(self.thread, self.runs, **self.cfg)
            self.pending = self.run.pending()
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.error, self.closed, self._loading = f"{type(e).__name__}: {e}", True, False
            return
        self._loading = False
        while True:
            kind, text = self.jobs.get()
            if kind == "close":
                break
            try:
                {"start": self.run.start, "resume": self.run.resume}.get(kind, lambda _t: self.run.retry())(text)
                self.pending = self.run.pending()
                self.finished = self.pending is None
            except Exception as e:  # noqa: BLE001 - shown on the page, with a Retry button
                self.error = f"{type(e).__name__}: {e}"
                try:
                    self.run.save()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._open_jobs -= 1
            if self.finished:
                break
        self.run.close()                                   # releases the camera for the next run
        self.closed = True

    @property
    def busy(self) -> bool:
        return self._loading or self._open_jobs > 0

    def send(self, kind: str, text: str | None = None):
        self.error = None
        self._open_jobs += 1
        self.jobs.put((kind, text))

    def close(self):
        if not self.closed and not self.busy:
            self.jobs.put(("close", None))

    def events(self) -> list:
        if self.run is None:
            return []
        with self.run.lock:
            return list(self.run.events)

    def usage(self) -> dict:
        return self.run.usage() if self.run is not None else {}


# -- saved runs ------------------------------------------------------------------------------------------------------
def load_transcript(run_dir: str) -> dict:
    path = os.path.join(run_dir, "transcript.json")
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):                          # being written
        return {}


def run_config(run_dir: str) -> dict:
    m = json.load(open(os.path.join(run_dir, "run.json"), encoding="utf-8"))
    return {k: m[k] for k in ("camera", "director_model", "team_model", "allow_human")}


def status(transcript: dict) -> str:
    d = transcript.get("dialogue") or []
    if not d:
        return "new"
    return "finished" if d[-1].get("meta", {}).get("stop") else "paused"


def list_runs(runs: str = RUNS) -> list[dict]:
    out = []
    for f in glob.glob(os.path.join(runs, "*", "run.json")):
        rd = os.path.dirname(f)
        tr = load_transcript(rd)
        first = next((e["content"] for e in tr.get("events", []) if e.get("role") == "user"), "")
        meta = json.load(open(f, encoding="utf-8"))
        out.append({"thread": os.path.basename(rd), "dir": rd, "status": status(tr), "request": first,
                    "camera": meta.get("camera"), "mtime": os.path.getmtime(f)})
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def records(events: list) -> list[tuple[str, dict]]:
    """The VERIFIED records the teams returned to the director: [(team, record), ...]."""
    return [(e["meta"]["team"], e["meta"]["verified"]) for e in events
            if e.get("source") == "director" and e.get("role") == "team" and e.get("meta", {}).get("verified")]


# -- camera frames and figures ---------------------------------------------------------------------------------------
def camera_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "teams", "instrument", "camera")


def newest_frame(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(camera_dir(run_dir), "cap*.npy")) or glob.glob(os.path.join(camera_dir(run_dir), "cap*.png"))
    return max(files, key=os.path.getmtime) if files else None


def frame_at(run_dir: str, events: list) -> str | None:
    """The frame of the last capture among these events (for the replay): the saved .npy, or its .png preview."""
    for e in reversed(events):
        cap = e.get("meta", {}).get("capture") or {}
        if cap.get("ok") and cap.get("frame_file"):
            base = os.path.splitext(os.path.basename(cap["frame_file"]))[0]
            for ext in (".npy", ".png"):
                p = os.path.join(camera_dir(run_dir), base + ext)
                if os.path.exists(p):
                    return p
    return None


def frame_image(path: str, max_side: int = 900):
    """(grey image, caption numbers) of a saved frame: the first frame of a .npy stack shown from 0 DN (black) to its
    99.9th percentile (white), like a camera viewer with auto-brightness (the caption gives the true levels), or a .png
    preview. Stretching from the frame's own minimum instead would turn the pixel noise of an evenly lit frame into
    full-contrast 'TV static'."""
    if path.endswith(".png"):
        return path, {}
    fr = np.load(path)[0].astype(float)
    step = max(1, int(np.ceil(max(fr.shape) / max_side)))
    info = {"shape": fr.shape, "mean_dn": float(fr.mean()), "max_dn": int(fr.max())}
    small = fr[::step, ::step]
    hi = max(float(np.percentile(small, 99.9)), 1.0)
    return (np.clip(small / hi, 0, 1) * 255).astype(np.uint8), info


def last_call(run_dir: str) -> dict | None:
    path = os.path.join(camera_dir(run_dir), "calls.jsonl")
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


def figures(run_dir: str) -> list[str]:
    """Figures the teams wrote (focusing results, analysis plots), newest first; camera previews are left out."""
    files = [f for f in glob.glob(os.path.join(run_dir, "teams", "**", "*.png"), recursive=True)
             if os.path.dirname(f) != camera_dir(run_dir)]
    return sorted(files, key=os.path.getmtime, reverse=True)


__all__ = ["TEAMS", "RUNS", "Worker", "load_transcript", "run_config", "status", "list_runs", "records", "camera_dir",
           "newest_frame", "frame_at", "frame_image", "last_call", "figures"]
