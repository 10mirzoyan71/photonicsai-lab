"""The three-level hierarchy with scripted stand-ins for the models (no model calls, no tokens) and the REAL camera MCP
server on the simulated camera:  python lab/tests/test_lab_offline.py
Checks: the director's hard decision is answered by the person; the instrument team runs a hardware turn whose
person-action question ("block the light") is passed up through the director and answered there; the team's VERIFIED
record reaches the director; the optics team runs the Homework 2 focusing tool with the measured camera model; the
director finishes with both records; and the same flow survives a restart of the process between the replies."""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lab_assistant import DirectorDecision, LabRun, team_records  # noqa: E402
from agent_v3 import DecisionAuto, DecisionAutoNoHW, HWDecision  # noqa: E402


class Scripted:
    """Answers decide() from a queue (checking the schema) and runs the executor's tools in tool_loop()."""

    def __init__(self, decisions):
        self.q = list(decisions)

    def decide(self, system, prompt, schema):
        d = self.q.pop(0)
        assert isinstance(d, schema), f"expected {schema.__name__}, got {type(d).__name__}"
        return d

    def chat(self, role, system, prompt):
        return f"[{role}] plan"

    def tool_loop(self, system, prompt, tools, max_iters=6):
        focus = [t for t in tools if t.name == "focusing_tool"][0]
        r = focus.func(macropixel=16, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=60, flips=2,
                       read_noise_e=5.8, full_well_e=24766, mean_signal_e=400, intensity_noise=0.005)
        return f"focusing experiment: enhancement {r['enhancement']:.2f}"


def director_script():
    return Scripted([
        DirectorDecision(next="human", stop_kind="hard_decision", pending="instrument",
                         pending_task="Measure the dark level of the camera at 1 ms.", question="Use the camera?",
                         reason="the instrument team would change the camera settings"),
        DirectorDecision(next="optics", task="Run one focusing experiment with read_noise_e=5.8, full_well_e=24766.",
                         reason="predict the focusing with the measured camera"),
        DirectorDecision(next="finish", task="Final report: dark frames measured; focusing predicted.", reason="done"),
    ])


def team_script():
    return Scripted([
        DecisionAuto(next_role="hardware", instruction="Take dark frames at 1 ms.", reason="needs the camera"),
        HWDecision(action="command", command="set_exposure", command_args={"exposure_us": 1000}, reason="1 ms"),
        HWDecision(action="ask_person", question="Please block the light and reply when done.", reason="dark frames"),
        HWDecision(action="measure", n_frames=2, label="dark", reason="dark frames"),
        HWDecision(action="finish", summary="dark frames at 1 ms taken", reason="done"),
        DecisionAuto(next_role="finish", instruction="Dark frames at 1 ms were taken.", reason="verified"),
        DecisionAutoNoHW(next_role="executor", instruction="Run one focusing experiment.", reason="tool"),
        DecisionAutoNoHW(next_role="finish", instruction="Focusing experiment done.", reason="verified"),
    ])


def main():
    root = tempfile.mkdtemp(prefix="lab_test_")
    d_script, t_script = director_script(), team_script()
    seen = []

    def new_run():
        return LabRun("t1", root, camera="sim-room", director_backend=d_script, team_backend=t_script,
                      emit=lambda source, turn: seen.append((source, turn["role"])))

    run = new_run()
    run.start("Measure the camera's dark level and predict the focusing with it.")
    q = run.pending()
    assert q and q["source"] == "director" and q["kind"] == "hard_decision", q
    run.resume("yes")
    q = run.pending()
    assert q and q["source"] == "instrument team" and "block the light" in q["question"], q
    run.close()

    run = new_run()                                   # a new process resumes the paused thread
    run.resume("done, the light is blocked")
    assert run.pending() is None, run.pending()
    d = run.dialogue()
    roles = [t["role"] for t in d]
    recs = team_records(d)
    assert [r[0] for r in recs] == ["instrument", "optics"], recs
    assert recs[1][1]["camera_model"]["full_well_e"] == 24766, recs
    assert d[-1]["meta"].get("stop") == "finished", d[-1]
    calls = open(os.path.join(root, "t1", "teams", "instrument", "camera", "calls.jsonl"), encoding="utf-8").read()
    assert '"tool": "capture"' in calls and '"label": "dark"' in calls
    assert ("instrument", "human") in seen and ("optics", "executor") in seen
    run.close()
    print("director dialogue:", roles)
    print("team records:", [(n, r.get("tool")) for n, r in recs])
    print("ALL LAB HIERARCHY CHECKS PASSED")


if __name__ == "__main__":
    main()
