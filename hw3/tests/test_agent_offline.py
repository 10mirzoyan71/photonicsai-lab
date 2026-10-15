"""Graph mechanics with a scripted stand-in for the model and the REAL camera MCP server (simulated camera):
    python tests/test_agent_offline.py
Checks: (1) orchestrator -> scientist -> hardware turn (decide -> command -> measure -> decide, a command refused by
the lab limit, analyze_frame) -> report -> executor script on the saved frames -> finish;
(2) human stops: a refused hard decision does not run (no camera call), an accepted one starts the hardware turn,
and a person-action stop inside the hardware turn pauses and resumes on the same thread."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
from langgraph.types import Command  # noqa: E402

from agent_v3 import DecisionAuto, DecisionHITL, HWDecision, build_graph, pending_interrupts, verified  # noqa: E402
from mcp_bridge import MCPBridge  # noqa: E402

SCRIPT = """import json, numpy as np
calls = [json.loads(l) for l in open('camera/calls.jsonl')]
cap = [c for c in calls if c['tool'] == 'capture' and c['result'].get('ok')][-1]
f = np.load(cap['result']['frame_file'])
print('LAST_MEAN_DN=%.3f' % f[0].mean())
print('N_CAPTURES=%d' % sum(1 for c in calls if c['tool'] == 'capture'))
"""


class Scripted:
    def __init__(self, decisions):
        self.q = list(decisions)
        self.prompts = []

    def decide(self, system, prompt, schema):
        self.prompts.append(prompt)
        d = self.q.pop(0)
        assert isinstance(d, schema), f"expected {schema.__name__}, got {type(d).__name__}"
        return d

    def chat(self, role, system, prompt):
        return f"[{role}] plan: set the exposure, capture, analyse."

    def tool_loop(self, system, prompt, tools, max_iters=6):
        t = tools[0]
        t.func("write", "analyse.py", SCRIPT)
        r = t.func("run", "analyse.py")
        return f"ran analyse.py: exit {r['exit_code']}\n{r['stdout']}"


def run(graph, config, inp):
    graph.invoke(inp, config)
    return graph.get_state(config).values["dialogue"]


def test_autonomous(tmp):
    work = os.path.join(tmp, "auto")
    os.makedirs(work)
    br = MCPBridge(os.path.join(work, "camera"), seed=4, limits={"max_exposure_us": 5000})
    try:
        S = Scripted([
            DecisionAuto(next_role="scientist", instruction="Plan an exposure without saturation.", reason="plan first"),
            DecisionAuto(next_role="hardware", instruction="Set the exposure and capture.", reason="measure"),
            HWDecision(action="command", command="set_exposure", command_args={"exposure_us": 50000}, reason="too long on purpose"),
            HWDecision(action="command", command="set_exposure", command_args={"exposure_us": 300}, n_frames=2,
                       label="t300", reason="inside the limit"),
            HWDecision(action="measure", analyze=True, black_level_dn=40,
                       analysis_region={"name": "c", "x": 270, "y": 206, "width": 100, "height": 100}, reason="speckle"),
            HWDecision(action="finish", summary="exposure 300 us, no saturation", reason="done"),
            DecisionAuto(next_role="executor", instruction="Analyse the last frame.", reason="analysis"),
            DecisionAuto(next_role="finish", instruction="Report.", reason="verified"),
        ])
        g, coding = build_graph(S, br, work, allow_human=False)
        d = run(g, {"configurable": {"thread_id": "a"}},
                {"dialogue": [{"role": "user", "content": "Find a good exposure.", "meta": {}}]})
        roles = [t["role"] + (":" + t["meta"]["hw"] if t["meta"].get("hw") else "") for t in d]
        print("roles:", roles)
        hw = [t["meta"].get("hw") for t in d if t["meta"].get("hw")]
        assert hw == ["start", "decide", "command", "decide", "command", "measure", "decide", "measure", "decide", "report"], hw
        refused = next(t for t in d if t["meta"].get("hw") == "command")
        assert "REFUSED" in refused["content"] and "lab limit" in refused["content"]
        report = next(t for t in d if t["meta"].get("hw") == "report")
        v = report["meta"]["verified"]
        assert v["settings"]["exposure_us"] == 300 and "speckle" in v["analysis"], v
        ex = verified(d)
        assert ex["tool"] == "coding_tool" and ex["results"]["N_CAPTURES"] == 2.0, ex
        calls = [json.loads(line) for line in open(os.path.join(work, "camera", "calls.jsonl"), encoding="utf-8")]
        print("instrument calls:", [c["tool"] for c in calls])
        assert [c["tool"] for c in calls] == ["camera_info", "set_exposure", "set_exposure", "capture", "capture", "analyze_frame"]
        assert not S.q
    finally:
        br.close()
    print("OK (autonomous hardware turn + analysis)")


def test_human(tmp):
    work = os.path.join(tmp, "hitl")
    os.makedirs(work)
    br = MCPBridge(os.path.join(work, "camera"), seed=5)
    try:
        S = Scripted([
            DecisionHITL(next_role="human", question="Run a hardware turn that sets the exposure to 2000 us?",
                         reason="setting change", stop_kind="hard_decision", pending_role="hardware",
                         pending_instruction="Set 2000 us and capture."),
            DecisionHITL(next_role="human", question="Then 500 us instead?", reason="adapted", stop_kind="hard_decision",
                         pending_role="hardware", pending_instruction="Set 500 us, take dark frames, capture."),
            HWDecision(action="command", command="set_exposure", command_args={"exposure_us": 500}, reason="as approved"),
            HWDecision(action="ask_person", question="Please block the laser and reply when done.", reason="dark frames"),
            HWDecision(action="measure", n_frames=2, label="dark", reason="dark frames"),
            HWDecision(action="finish", summary="500 us frames and dark frames taken", reason="done"),
            DecisionHITL(next_role="finish", instruction="Report.", reason="verified"),
        ])
        g, _ = build_graph(S, br, work, allow_human=True)
        cfg = {"configurable": {"thread_id": "h"}}
        run(g, cfg, {"dialogue": [{"role": "user", "content": "Set up the camera.", "meta": {}}]})
        assert pending_interrupts(g, cfg) and len(br.log) == 0
        d = run(g, cfg, Command(resume="no, not 2000 us"))
        assert "REFUSED by the human and did NOT run" in d[-1]["content"] and len(br.log) == 0
        d = run(g, cfg, Command(resume="yes"))
        intr = pending_interrupts(g, cfg)
        assert intr and intr[0].value["kind"] == "person_action", intr
        assert [c["tool"] for c in br.log] == ["camera_info", "set_exposure", "capture"], [c["tool"] for c in br.log]
        d = run(g, cfg, Command(resume="done, the laser is blocked"))
        assert any(t["role"] == "human" and t["meta"].get("hw") == "person" for t in d)
        assert not pending_interrupts(g, cfg) and d[-1]["meta"].get("decision", {}).get("next_role") == "finish"
        print("roles:", [t["role"] + (":" + t["meta"]["hw"] if t["meta"].get("hw") else "") for t in d])
    finally:
        br.close()
    print("OK (human stops: refused / accepted hard decision, person action inside the hardware turn)")


class BenchScripted(Scripted):
    def tool_loop(self, system, prompt, tools, max_iters=6):
        assert [t.name for t in tools] == ["coding_tool", "focusing_tool"] and "SIMULATED bench" in system
        r = tools[1].func(macropixel=16, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=60, flips=2)
        bad = tools[1].func(macropixel=16, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=60, flips=2,
                            full_well_e=10.0)
        assert not bad["ok"] and "full_well_e" in bad["error"], bad
        # the same experiment with a measured camera model (small full well): the focus saturates the camera
        r2 = tools[1].func(macropixel=16, focus_size=3, bg_inner=4, bg_outer=30, bg_regions=8, iterations=60, flips=2,
                           read_noise_e=2.0, full_well_e=2000.0, mean_signal_e=1500.0, intensity_noise=0.005)
        assert r2["camera_model"] == {"read_noise_e": 2.0, "full_well_e": 2000.0, "mean_signal_e": 1500.0,
                                      "intensity_noise": 0.005} and r2["saturated_pixels_final"] > 0, r2
        assert r["camera_model"]["full_well_e"] == 2.6e5 and r["saturated_pixels_final"] == 0
        from agent_v3 import lc_tools
        args = lc_tools(tools)["focusing_tool"].args
        assert {"read_noise_e", "full_well_e", "mean_signal_e", "intensity_noise"} <= set(args), args
        return f"enhancement {r['enhancement']:.2f}, with the measured camera {r2['enhancement']:.2f}"


def test_bench(tmp):
    work = os.path.join(tmp, "bench")
    os.makedirs(work)
    br = MCPBridge(os.path.join(work, "camera"), seed=6)
    try:
        S = BenchScripted([DecisionAuto(next_role="executor", instruction="Run one focusing experiment.", reason="sim"),
                           DecisionAuto(next_role="finish", instruction="Report.", reason="verified")])
        g, _ = build_graph(S, br, work, allow_human=False, bench_dir=os.path.join(work, "bench"))
        d = run(g, {"configurable": {"thread_id": "b"}}, {"dialogue": [{"role": "user", "content": "Focus.", "meta": {}}]})
        v = verified(d)
        assert v["tool"] == "focusing_tool" and v["enhancement"] > 1, v
        assert v["camera_model"]["full_well_e"] == 2000.0, v        # kept: the latest camera model's experiment
    finally:
        br.close()
    print(f"OK (executor holds the HW2 simulated-bench focusing tool: enhancement {v['enhancement']:.2f} with the "
          f"measured camera model {v['camera_model']})")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="hw3_agent_")
    test_autonomous(tmp)
    test_human(tmp)
    test_bench(tmp)
    print("ALL OFFLINE AGENT TESTS PASSED")
