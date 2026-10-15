"""The web app without model calls:  python lab/tests/test_app_offline.py
1. The app's worker runs the lab hierarchy on its own thread with the scripted director and teams of
   test_lab_offline.py and the real camera MCP server on the simulated camera: start -> approval -> "block the light"
   -> finished, with the events, VERIFIED records and frames the page shows.
2. The page itself (Streamlit's AppTest, no browser): the start page, a finished run, the replay controls, and a
   paused run continued in the page, which shows the approval card with its buttons."""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..")
ROOT = tempfile.mkdtemp(prefix="lab_app_test_")
os.environ["LAB_RUNS"] = ROOT                          # the app and the worker use this runs folder
sys.path.insert(0, HERE)
sys.path.insert(0, LAB)
import app_core as core  # noqa: E402
from test_lab_offline import director_script, team_script  # noqa: E402


def wait(w, timeout=120):
    t0 = time.time()
    while w.busy:
        assert time.time() - t0 < timeout, "worker did not finish"
        time.sleep(0.1)
    assert w.error is None, w.error


def worker_flow():
    w = core.Worker("t1", director_backend=director_script(), team_backend=team_script(), camera="sim-room")
    w.send("start", "Measure the camera's dark level and predict the focusing with it.")   # sent while the run
    assert w.busy                                                                         # still loads, as the page
    wait(w)                                                                               # does: busy until done
    assert w.pending["kind"] == "hard_decision", w.pending
    w.send("resume", "yes")
    wait(w)
    assert w.pending["kind"] == "person_action" and "block the light" in w.pending["question"], w.pending
    w.send("resume", "done")
    wait(w)
    t0 = time.time()
    while not w.closed and time.time() - t0 < 30:
        time.sleep(0.1)
    assert w.finished and w.closed and w.pending is None
    ev = w.events()
    assert [t for t, _ in core.records(ev)] == ["instrument", "optics"]
    assert any(e["source"] == "instrument" and e["role"] == "instrument" for e in ev)
    run_dir = os.path.join(ROOT, "t1")
    assert core.newest_frame(run_dir) and core.frame_at(run_dir, ev)
    img, info = core.frame_image(core.newest_frame(run_dir))
    assert img.ndim == 2 and img.dtype.name == "uint8" and info["max_dn"] > 0
    assert any("experiment" in os.path.basename(f) for f in core.figures(run_dir))
    assert core.list_runs(ROOT)[0]["status"] == "finished"
    print("worker: start -> approval -> block the light -> finished;", len(ev), "events")


def paused_run():
    """A run stopped at the director's approval question, as if started in the terminal and left there."""
    from lab_assistant import LabRun
    run = LabRun("t2", ROOT, camera="sim-room", director_backend=director_script(), team_backend=team_script())
    run.start("Measure the camera's dark level and predict the focusing with it.")
    assert run.pending()["kind"] == "hard_decision"
    run.close()


def page():
    from streamlit.testing.v1 import AppTest
    app = os.path.join(LAB, "app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.button) >= 3 and "Find an exposure" in at.button[0].label
    print("page: start page with", len(at.button), "example buttons")

    at.sidebar.selectbox[1].set_value("t1").run()
    assert not at.exception, at.exception
    assert any("Final report" in s.value for s in at.success), [s.value for s in at.success]
    assert any("No model calls" in c.value for c in at.caption)   # scripted backends make no model calls
    print("page: finished run shows the final report and", len(at.chat_message), "chat messages")

    at.sidebar.radio(key="mode").set_value("Replay").run()
    assert not at.exception, at.exception
    before = len(at.chat_message)
    at.button[2].click().run()                          # next step
    assert not at.exception, at.exception
    assert "step 2 of" in at.get("progress")[0].proto.text, at.get("progress")[0].proto.text
    print("page: replay steps forward;", before, "->", len(at.chat_message), "messages")

    at.sidebar.radio(key="mode").set_value("Live").run()
    at.sidebar.selectbox[1].set_value("t2").run()
    assert not at.exception, at.exception
    cont = [b for b in at.button if b.label.startswith("▶ Continue")]
    assert cont, [b.label for b in at.button]
    cont[0].click().run()
    for _ in range(50):                                 # the worker loads the paused run on its thread
        at.run()
        if any(b.label == "✅ Approve" for b in at.button):
            break
        time.sleep(0.2)
    labels = [b.label for b in at.button]
    assert "✅ Approve" in labels and "❌ Refuse" in labels, labels
    print("page: paused run continued in the page shows the approval card (Approve and Refuse buttons)")


if __name__ == "__main__":
    worker_flow()
    paused_run()
    page()
    print("ALL WEB APP CHECKS PASSED")
