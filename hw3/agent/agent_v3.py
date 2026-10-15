"""Homework 3 agent: the Homework 2 graph with a real instrument behind an MCP server.

Kept from HW1/HW2: the roles orchestrator, scientist, general and executor; the shared dialogue as the graph state; the
orchestrator's choice of the next role as a model call (structured output) with a decision budget; the coding tool
given only to the executor; the human stop (LangGraph interrupt on a checkpointed thread) for ambiguous queries and
hard decisions.

New: the executor holds the camera's MCP tools (mcp_bridge.MCPBridge, schemas from the server) and drives the camera in
a HARDWARE TURN, a loop of graph nodes:
    hw_start   -> camera_info through MCP (what the instrument is and how it is set)
    hw_decide  -> the executor's model call: the next instrument command and what to measure, ask the person, or finish
    hw_command -> the chosen MCP command tool (set_exposure, set_gain, set_roi, ...)
    hw_measure -> MCP capture (+ analyze_frame when asked); the measurement goes back to hw_decide
    hw_person  -> interrupt: a person does what software cannot (block the light, change the scene) and replies
    hw_report  -> the executor's report of the turn (settings and measurements the instrument returned)
i.e. decide -> command -> measure -> decide, until the executor finishes or the step budget is used.
"""
from __future__ import annotations

import json
import operator
import os
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from coding_tool import CodingTool

MAX_DECISIONS = 10
MAX_HW_STEPS = 12
MEASURE_TOOLS = ("capture", "analyze_frame")
SELF_MEASURING = ("capture_series",)      # commands that are themselves a measurement: no capture follows them
NO_CAPTURE = SELF_MEASURING + ("read_temperature", "save_tiff")   # commands after which no capture follows
INFO_TOOL = "camera_info"


# ----------------------------------------------------------------------------
# state and decisions
# ----------------------------------------------------------------------------
class Turn(TypedDict, total=False):
    role: str          # user | orchestrator | scientist | general | executor | instrument | human
    content: str
    meta: dict


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]


class DecisionHITL(BaseModel):
    """Orchestrator decision when a person can be asked."""
    next_role: Literal["scientist", "general", "executor", "hardware", "human", "finish"] = Field(
        description="who gets the turn; 'hardware' = the executor drives the camera; 'human' pauses the graph")
    instruction: str = Field("", description="instruction for that role; for finish: the final report; empty for human")
    question: str = Field("", description="only for human: the question the person is asked")
    reason: str = Field(description="one sentence: why this destination now")
    stop_kind: Literal["none", "ambiguous", "hard_decision"] = Field("none", description="only for human: why the graph stops")
    pending_role: Literal["none", "executor", "hardware", "finish"] = Field(
        "none", description="only for a hard decision: the step that waits for approval")
    pending_instruction: str = Field("", description="only for a hard decision: the instruction (or final report) that runs if accepted")


class DecisionAuto(BaseModel):
    """Orchestrator decision when the system works on its own."""
    next_role: Literal["scientist", "general", "executor", "hardware", "finish"]
    instruction: str = Field(description="instruction for that role; for finish: the final report")
    reason: str = Field(description="one sentence: why this role now")


class DecisionHITLNoHW(DecisionHITL):
    """Orchestrator decision of a team without an instrument, when a person can be asked."""
    next_role: Literal["scientist", "general", "executor", "human", "finish"] = Field(
        description="who gets the turn; 'human' pauses the graph")
    pending_role: Literal["none", "executor", "finish"] = Field(
        "none", description="only for a hard decision: the step that waits for approval")


class DecisionAutoNoHW(DecisionAuto):
    """Orchestrator decision of a team without an instrument, working on its own."""
    next_role: Literal["scientist", "general", "executor", "finish"]


class HWDecision(BaseModel):
    """The executor's decision at one step of a hardware turn."""
    action: Literal["command", "measure", "ask_person", "finish"] = Field(
        description="command = run one instrument command, then measure; measure = measure with the current settings; "
                    "ask_person = something only a person can do; finish = end the hardware turn")
    command: str = Field("none", description="for action=command: the name of one instrument command tool")
    command_args: dict[str, Any] = Field(default_factory=dict, description="arguments of that command")
    n_frames: int = Field(1, description="frames to capture in the measurement (>= 2 gives the temporal noise)")
    label: str = Field("", description="short label for the saved frames")
    regions: list[dict[str, Any]] = Field(default_factory=list,
                                          description="optional regions {name, x, y, width, height} for region statistics")
    analyze: bool = Field(False, description="also run analyze_frame (speckle contrast, grain, beam) on the new frame")
    black_level_dn: float = Field(0.0, description="black level to subtract in analyze_frame")
    analysis_region: Optional[dict[str, Any]] = Field(None, description="region {name, x, y, width, height} for analyze_frame")
    question: str = Field("", description="for ask_person: what the person should do or answer")
    summary: str = Field("", description="for finish: what the hardware turn achieved, quoting only instrument numbers")
    reason: str = Field(description="one sentence: why this step")


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
ORCH_BASE = """You are the ORCHESTRATOR of a team that works with a laboratory camera. The user speaks only to you. You
read the dialogue and decide who speaks next. You never write theory, plans or code, and you have no tools.
Roles:
- scientist: owns the measurement theory and the plan (what to measure, which settings, which frames, the analysis and
  its formulas, numbered steps). Plan questions go here.
- general: answers a high-level side question: what a camera term means, what a reported number measures.
- hardware: the EXECUTOR drives the camera through its MCP tools in a hardware turn (decide -> command -> measure ->
  decide), following your instruction; it returns the settings and measurements the camera reported.
- executor: the EXECUTOR with its software tools: {executor_tools}.
- finish: end the run; the instruction is your final report, quoting only numbers that the instrument or a script
  returned (VERIFIED records). Finishing is allowed only after a VERIFIED measurement exists.
You have at most {max_dec} decisions in total; budget them."""
ORCH_FIRST_LINE = "You are the ORCHESTRATOR of a team that works with a laboratory camera."
HW_ROLE_LINE = """- hardware: the EXECUTOR drives the camera through its MCP tools in a hardware turn (decide -> command -> measure ->
  decide), following your instruction; it returns the settings and measurements the camera reported.
"""

ORCH_HUMAN = """
- human: pause the graph and ask the person. Use it in exactly two situations, judging from the dialogue:
  1. AMBIGUOUS QUERY (stop_kind="ambiguous"): the request has more than one reading and the readings would lead to
     different measurements or reported numbers. Ask which reading to follow; list the readings.
  2. HARD DECISION (stop_kind="hard_decision"): the next step would change the camera's settings or run a long
     acquisition (a hardware turn), run the coding tool (executor), or end the dialogue and report a measurement
     (finish). Put the step in pending_role ("hardware", "executor" or "finish") and pending_instruction, state it in the
     question, and ask whether to proceed. Accepted -> the step runs as written. Refused -> the turn comes back to you and
     the step did not run: adapt; do not repeat the refused step unchanged without asking.
Do not ask anything else, and do not ask what the dialogue already answers."""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own the measurement theory and the plan for a laboratory camera; you have no
tools and write no code. For the request, write: the quantity to measure and its definition; the physics and formulas
(e.g. photon transfer: temporal variance = signal/K + read_noise^2 in DN, so K [e-/DN] = 1/slope, read noise from dark
frames, full well from the onset of saturation; speckle contrast C = std/mean; grain = FWHM of the intensity
autocorrelation; exposure chosen so the brightest region stays below saturation; light from mains-powered lamps flickers
at twice the mains frequency, so the frame mean fluctuates unless the exposure spans whole flicker periods); the camera
settings and frames to take,
within the camera's ranges and the lab limits; and a NUMBERED list of steps, each marked [hardware] (camera commands and
captures, done in a hardware turn) or [analysis] (one Python script on the saved frames that prints NAME=value lines).
Never claim a number was measured unless the instrument or a script already returned it."""

GENERAL_SYSTEM = """You are the GENERAL agent. You answer one high-level question beside the measurement: what a camera term
or setting means, what a reported number measures. Be accurate and brief. No plan, no code, no invented numbers."""

EXECUTOR_CODING = """You are the EXECUTOR, here with the coding tool:
  coding_tool(action="write", filename="name.py", code="...")   writes a NEW Python file into the working directory
  coding_tool(action="run",   filename="name.py")               runs it (300 s limit) -> exit code, stdout, stderr
One write and one run per turn. Environment: Python 3 with numpy, scipy, matplotlib. Saved frames are in ./camera/ as
.npy files (uint16 stacks: frames x height x width; the first frame is the one the statistics describe), with preview
.png files; ./camera/calls.jsonl lists every instrument call (tool, args, result incl. the settings of each capture).
Print every result as a line NAME=value (e.g. READ_NOISE_E=5.9); save any plot as .png. Then reply with what the script
printed - or the traceback. Never state a number the script did not print."""

FOCUS_TEXT = """
Tool focusing_tool(...): runs ONE focusing experiment on the SIMULATED bench of Homework 2 (laser -> DMD -> scattering
medium -> camera) and returns the enhancement and the camera images. Bench: {bench}
Algorithm inside the tool: the DMD is divided into macropixels of macropixel x macropixel mirrors, starting from a random
binary pattern; each iteration flips `flips` random macropixels (annealed linearly to `flips_final`), takes a camera shot,
reads the mean intensity in the focus square (focus_size x focus_size px) and in bg_regions sectors of the ring
bg_inner < r <= bg_outer px around it, enhancement = focus mean / mean of the sector means; the flip is kept if the
enhancement increased (with remeasure=True the current pattern is re-measured each iteration against noise).
It returns: enhancement (+ std over final_shots fresh shots), enhancement_far (focus against the camera outside the
ring), enhancement_initial, the progress trace, segments, camera_shots, seconds, and image paths. Limits: macropixel
divides {dmd} (1..32), focus_size odd, focus_size//2+1 <= bg_inner < bg_outer <= {rmax}, 2..16 regions,
iterations*(2 if remeasure else 1) <= 80000. Up to {ncalls} experiments per executor turn.
Camera model (optional, per experiment; omitted = the Homework 2 bench camera): read_noise_e [e- rms], full_well_e [e-],
mean_signal_e (mean signal of a random DMD pattern [e-/px]; the focus is about `enhancement` times brighter, so it
saturates when enhancement * mean_signal_e approaches full_well_e), intensity_noise (relative intensity noise of the
light source per shot). Put a real camera's MEASURED numbers here to predict the focusing result with that camera. The
result also gives camera_model and saturated_pixels_final."""

HW_SYSTEM = """You are the EXECUTOR, driving a laboratory camera in a HARDWARE TURN. The camera sits behind an MCP server;
you decide each step, the graph runs it and shows you the result:
  action="command"    : run ONE instrument command tool (name + args), then a measurement (capture) with the new setting;
                        capture_series is itself a measurement (frames spaced in time) and is not followed by a capture
  action="measure": a measurement with the current settings (capture n_frames; optional region stats; analyze=true
                        adds speckle contrast, grain size and beam centroid/width of the new frame)
  action="ask_person" : something only a person can do or answer (e.g. block or unblock the light when the light is not
                        under software control); the graph pauses for the reply
  action="finish"     : the instruction is done; summary = what was achieved, quoting only numbers the camera returned
Follow the orchestrator's instruction and the scientist's latest plan. Stay inside the camera's ranges and the lab
limits; a refused command comes back as an error - adapt. Use n_frames >= 2 when the temporal noise is needed. You have at
most {max_steps} steps in this turn. Never state a number the camera did not return.
Instrument command tools (from the MCP server):
{commands}"""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def render_dialogue(d: list[Turn]) -> str:
    return "\n\n".join(f"=== turn {i} | {t['role'].upper()} ===\n{t['content']}" for i, t in enumerate(d, 1))


def n_decisions(d):
    return sum(1 for t in d if t["role"] == "orchestrator" and "decision" in t.get("meta", {}))


def last_decision(d):
    for t in reversed(d):
        if t["role"] == "orchestrator" and "decision" in t.get("meta", {}):
            return t["meta"]["decision"]
    raise RuntimeError("no decision")


def verified(d):
    for t in reversed(d):
        if t["role"] == "executor" and t.get("meta", {}).get("verified"):
            return t["meta"]["verified"]
    return None


APPROVE = re.compile(r"^\s*(yes|y|ok|okay|sure|proceed|accept|accepted|approve|approved|go( ahead)?|run it|do it|confirm)\b", re.I)


def is_approval(reply: str) -> bool:
    return bool(APPROVE.match(reply or ""))


def hw_turn_start(d) -> int:
    """Index of the hw_start turn of the current hardware turn."""
    for i in range(len(d) - 1, -1, -1):
        if d[i].get("meta", {}).get("hw") == "start":
            return i
    raise RuntimeError("not in a hardware turn")


def short_capture(res: dict) -> str:
    if not res.get("ok"):
        return f"ERROR {res.get('error') or res.get('text')}"
    s, st = res["settings"], res["stats"]
    txt = (f"{res['n_frames']} frame(s) at {s['exposure_us']:g} us, {s['gain_db']:g} dB"
           + (f", illumination {s['illumination']:g}" if "illumination" in s else "")
           + f", ROI {s['roi']['width']}x{s['roi']['height']}+{s['roi']['x']}+{s['roi']['y']}: mean {st['mean_dn']} DN, "
           f"std {st['std_dn']}, max {st['max_dn']}, p99.9 {st['p99_9_dn']} ({st['fraction_of_full_scale']['p99_9']} of full "
           f"scale), saturated {st['saturated_pixels']} px ({st['saturated_fraction']})")
    if st.get("regions"):
        txt += "; regions " + json.dumps(st["regions"])
    tn = res.get("temporal_noise")
    if tn:
        txt += f"; temporal noise {tn.get('temporal_std_dn')} DN rms (var {tn.get('temporal_var_dn2')}, mean {tn.get('mean_dn')})"
    txt += f"; file camera/{os.path.basename(res['frame_file'])}"
    return txt


# ----------------------------------------------------------------------------
# model back-ends (same three methods as HW2)
# ----------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    func: Callable[..., dict]
    new_turn: Callable[[], None]
    log: list


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def lc_tools(tools: list[ToolSpec]) -> dict:
    from langchain_core.tools import StructuredTool
    lc = {}
    for spec in tools:
        if spec.name == "coding_tool":
            def coding_tool(action: Literal["write", "run"], filename: str, code: str = "", _f=spec.func) -> str:
                """Write a NEW Python file (action='write', with code) or run an existing file (action='run', 300 s limit)."""
                return json.dumps(_f(action, filename, code))
            lc[spec.name] = StructuredTool.from_function(coding_tool)
        elif spec.name == "focusing_tool":
            def focusing_tool(macropixel: int, focus_size: int, bg_inner: int, bg_outer: int, bg_regions: int,
                              iterations: int, flips: int, flips_final: int = 1, remeasure: bool = True,
                              final_shots: int = 5, label: str = "", read_noise_e: Optional[float] = None,
                              full_well_e: Optional[float] = None, mean_signal_e: Optional[float] = None,
                              intensity_noise: Optional[float] = None, _f=spec.func) -> str:
                """Run one focusing experiment on the simulated bench (DMD macropixel search); returns the enhancement and images.
                The optional camera-model arguments replace the bench camera's defaults with a real camera's measured numbers."""
                return json.dumps(_f(macropixel=macropixel, focus_size=focus_size, bg_inner=bg_inner, bg_outer=bg_outer,
                                     bg_regions=bg_regions, iterations=iterations, flips=flips, flips_final=flips_final,
                                     remeasure=remeasure, final_shots=final_shots, label=label, read_noise_e=read_noise_e,
                                     full_well_e=full_well_e, mean_signal_e=mean_signal_e,
                                     intensity_noise=intensity_noise), default=str)
            lc[spec.name] = StructuredTool.from_function(focusing_tool)
    return lc


class LangChainBackend:
    def __init__(self, model: str, temperature: float = 0.2):
        from langchain.chat_models import init_chat_model
        self.llm = init_chat_model(model, temperature=temperature, max_tokens=4000)
        self.llm_long = init_chat_model(model, temperature=temperature, max_tokens=16000)

    def decide(self, system, prompt, schema):
        from langchain_core.messages import HumanMessage, SystemMessage
        return self.llm.with_structured_output(schema).invoke([SystemMessage(system), HumanMessage(prompt)])

    def chat(self, role, system, prompt):
        from langchain_core.messages import HumanMessage, SystemMessage
        return _text(self.llm.invoke([SystemMessage(system), HumanMessage(prompt)]))

    def tool_loop(self, system, prompt, tools: list[ToolSpec], max_iters: int = 6):
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        lc = lc_tools(tools)
        llm = self.llm_long.bind_tools(list(lc.values()))
        msgs = [SystemMessage(system), HumanMessage(prompt)]
        ai = None
        for _ in range(max_iters):
            ai = llm.invoke(msgs)
            msgs.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                tool = lc.get(tc["name"])
                out = tool.invoke(tc["args"]) if tool else json.dumps({"ok": False, "error": "unknown tool"})
                msgs.append(ToolMessage(content=out, tool_call_id=tc["id"]))
        if ai is not None and ai.tool_calls:
            msgs.append(HumanMessage("Tool budget for this turn is used. Report now what the tools returned."))
            ai = self.llm_long.invoke(msgs)
        return _text(ai) if ai is not None else ""


class ClaudeCLIBackend:
    """The same three methods through the Claude Code CLI (claude_cli.py): --model claude-cli:<model>, no API key."""

    def __init__(self, model: str):
        from claude_cli import ClaudeCLI
        self.cli = ClaudeCLI(model)

    def decide(self, system, prompt, schema):
        return schema.model_validate(self.cli.structured(system, prompt, schema.model_json_schema()))

    def chat(self, role, system, prompt):
        return self.cli.text(system, prompt)

    def tool_loop(self, system, prompt, tools: list[ToolSpec], max_iters: int = 6):
        return self.cli.tool_loop(system, prompt, list(lc_tools(tools).values()), max_iters,
                                  "Tool budget for this turn is used. Report now what the tools returned.")


def make_backend(model: str):
    if model.startswith("claude-cli:"):
        return ClaudeCLIBackend(model.split(":", 1)[1])
    return LangChainBackend(model)


def coding_spec(workdir) -> ToolSpec:
    tool = CodingTool(workdir=workdir)
    return ToolSpec("coding_tool", tool, tool.new_turn, tool.log)


def focusing_spec(out_dir) -> ToolSpec:
    from bench import FocusingTool                      # the Homework 2 simulated bench
    tool = FocusingTool(out_dir)
    return ToolSpec("focusing_tool", tool, tool.new_turn, tool.log)


# ----------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------
def build_graph(backend, bridge, workdir: str, allow_human: bool = True, max_decisions: int = MAX_DECISIONS,
                max_hw_steps: int = MAX_HW_STEPS, checkpointer=None, on_turn=None, bench_dir: str | None = None,
                prompts: dict | None = None):
    """bridge: mcp_bridge.MCPBridge (or a stand-in with call(name, args), tool_defs), or None for a team without an
    instrument (no hardware turn). workdir: the coding tool's working directory; the camera server writes its frames to
    workdir/camera. bench_dir: if given, the executor also holds the Homework 2 focusing tool (simulated bench), writing
    to that folder. prompts: optional overrides {"team": first line of the orchestrator prompt, "scientist": ...,
    "general": ..., "executor": extra executor instructions} for the teams of the lab assistant (Homework 4/5)."""
    prompts = prompts or {}
    emit = on_turn or (lambda t: None)
    coding = coding_spec(workdir)
    tools = [coding] + ([focusing_spec(bench_dir)] if bench_dir else [])
    exec_sys = EXECUTOR_CODING
    ex_tools = "the coding tool (one Python script that analyses saved frames and prints NAME=value lines)"
    if bench_dir:
        ft = tools[1].func
        b = ft.bench.cfg
        exec_sys += FOCUS_TEXT.format(bench=ft.describe_bench(), dmd=b.dmd_side, rmax=b.cam_side // 2 - 1,
                                      ncalls=ft.max_calls_per_turn)
        ex_tools += ("; and the focusing tool of the Homework 2 SIMULATED bench (laser -> DMD -> scattering medium -> "
                     "camera), for focusing experiments in simulation")
    exec_sys += prompts.get("executor", "")
    if bridge is None:
        schema = DecisionHITLNoHW if allow_human else DecisionAutoNoHW
        base = ORCH_BASE.replace(HW_ROLE_LINE, "")
    else:
        schema = DecisionHITL if allow_human else DecisionAuto
        base = ORCH_BASE
    if prompts.get("team"):
        base = base.replace(ORCH_FIRST_LINE, prompts["team"])
    orch_sys = base.format(max_dec=max_decisions, executor_tools=ex_tools) + (
        ORCH_HUMAN if allow_human else "\nThe person is not available during this run: decide on your own.")
    if bridge is None:
        orch_sys = orch_sys.replace(' ("hardware", "executor" or "finish")', ' ("executor" or "finish")').replace(
            "change the camera's settings or run a long\n     acquisition (a hardware turn), run", "run")
    scientist_sys = prompts.get("scientist", SCIENTIST_SYSTEM)
    general_sys = prompts.get("general", GENERAL_SYSTEM)
    command_defs = [t for t in bridge.tool_defs if t.name not in MEASURE_TOOLS + (INFO_TOOL,)] if bridge else []
    command_names = [t.name for t in command_defs]
    hw_sys = HW_SYSTEM.format(max_steps=max_hw_steps, commands="\n".join(
        f"- {t.name}: {t.description}\n  args: {json.dumps(t.inputSchema.get('properties', {}))}" for t in command_defs))

    def n_tool_calls():
        return sum(len(t.log) for t in tools) + (len(bridge.log) if bridge else 0)

    def files_on_disk():
        out = []
        for root, _, files in os.walk(workdir):
            for f in files:
                if f.endswith((".npy", ".png", ".py", ".json", ".jsonl", ".csv", ".txt", ".log")):
                    out.append(os.path.relpath(os.path.join(root, f), workdir).replace("\\", "/"))
        return sorted(out)[:80]

    def say(role, content, **meta):
        turn = {"role": role, "content": content, "meta": meta}
        emit(turn)
        return {"dialogue": [turn]}

    # -- orchestrator ---------------------------------------------------------------
    def orchestrator(state):
        d = state["dialogue"]
        last = d[-1]
        lm = last.get("meta", {})
        if last["role"] == "human" and lm.get("approved") and (lm.get("pending") or {}).get("role") == "finish":
            return say("orchestrator", "Final report (accepted by the human):\n" + lm["pending"]["instruction"],
                       stop="finished", verified=verified(d))
        k = n_decisions(d)
        rec = verified(d)
        if k >= max_decisions:
            return say("orchestrator", f"Decision budget ({max_decisions}) is used; stopping. Last verified result: {rec}.",
                       stop="budget", verified=rec)
        record = ""
        if last["role"] == "human" and lm.get("kind") == "hard_decision" and not lm.get("approved"):
            p = lm["pending"]
            record = (f"Record: the proposed step ({p['role']}: {p['instruction'][:300]}) was REFUSED by the human and did "
                      f"NOT run. Tool calls in this run: {lm['tool_calls_at_question']} when it was proposed, "
                      f"{n_tool_calls()} now.\n")
        status = (f"[status] decision {k + 1} of {max_decisions}. "
                  + (f"VERIFIED measurement exists: {json.dumps(rec)[:1500]}. " if rec else
                     "No VERIFIED measurement yet, so finish is not allowed. ")
                  + f"Files in the working directory: {files_on_disk()}.")
        prompt = render_dialogue(d) + "\n\n" + record + status + "\nDecide the next turn."
        dec = backend.decide(orch_sys, prompt, schema)
        notes = []

        def wants_finish(x):
            return x.next_role == "finish" or (x.next_role == "human" and getattr(x, "pending_role", "none") == "finish")

        if wants_finish(dec) and rec is None:
            notes.append(f"finish proposed without a verified measurement ({dec.reason}); asked again")
            dec = backend.decide(orch_sys, prompt + "\n\nRejected: no VERIFIED measurement exists, so the run cannot "
                                 "finish yet. Choose another destination.", schema)
            if wants_finish(dec):
                notes.append("finish again; routed to the scientist")
                dec = schema(next_role="scientist", instruction="Give a plan that obtains the requested measurement.",
                             reason="A verified measurement is needed before finishing.")
        if dec.next_role == "human":
            if not dec.question.strip():
                dec.question = dec.reason
            if dec.stop_kind == "none":
                dec.stop_kind = "hard_decision" if dec.pending_role != "none" else "ambiguous"
            body = (f"Decision {k + 1}/{max_decisions} -> human ({dec.stop_kind.replace('_', ' ')})\n"
                    + (f"Proposed step: {dec.pending_role}: {dec.pending_instruction}\n" if dec.stop_kind == "hard_decision" else "")
                    + f"Question: {dec.question}\nReason: {dec.reason}")
        else:
            body = f"Decision {k + 1}/{max_decisions} -> {dec.next_role}\nInstruction: {dec.instruction}\nReason: {dec.reason}"
        if notes:
            body += "\nGuard: " + "; ".join(notes)
        return say("orchestrator", record + body, decision=dec.model_dump(), index=k + 1, guard_notes=notes,
                   verified=rec, refused_record=bool(record))

    # -- human (orchestrator-level stop) --------------------------------------------------
    def human(state):
        d = state["dialogue"]
        dec = last_decision(d)
        hard = dec.get("stop_kind") == "hard_decision"
        payload = {"kind": dec.get("stop_kind"), "question": dec.get("question"),
                   "proposed_step": {"role": dec.get("pending_role"), "instruction": dec.get("pending_instruction")} if hard else None}
        reply = interrupt(payload)
        reply = reply if isinstance(reply, str) else str(reply.get("reply", reply))
        return say("human", reply, kind=payload["kind"], question=payload["question"],
                   approved=is_approval(reply) if hard else None,
                   pending={"role": dec.get("pending_role"), "instruction": dec.get("pending_instruction")} if hard else None,
                   tool_calls_at_question=n_tool_calls())

    def instruction_for(d):
        dec = last_decision(d)
        return dec.get("pending_instruction", "") if dec.get("next_role") == "human" else dec.get("instruction", "")

    def specialist_prompt(d, role):
        note = "\n(The human accepted this step.)" if d[-1]["role"] == "human" and d[-1]["meta"].get("approved") else ""
        return (render_dialogue(d) + f"\n\n=== instruction from the orchestrator to the {role.upper()} ===\n"
                + instruction_for(d) + note)

    # -- scientist / general -------------------------------------------------------------
    def scientist(state):
        return say("scientist", backend.chat("scientist", scientist_sys, specialist_prompt(state["dialogue"], "scientist")))

    def general(state):
        return say("general", backend.chat("general", general_sys, specialist_prompt(state["dialogue"], "general")))

    # -- executor with its software tools (coding tool; focusing tool of the simulated bench) --------
    def executor(state):
        d = state["dialogue"]
        before = {t.name: len(t.log) for t in tools}
        for t in tools:
            t.new_turn()
        text = backend.tool_loop(exec_sys, specialist_prompt(d, "executor")
                                 + f"\n\nFiles in the working directory: {files_on_disk()}", tools)
        lines, ver = [], None
        for t in tools:
            for c in t.log[before[t.name]:]:
                if t.name == "coding_tool":
                    if c["action"] == "write":
                        lines.append(f"write {c['filename']}: " + ("ok" if c.get("ok") else c.get("error", "failed")))
                    elif "exit_code" in c:
                        lines.append(f"run {c['filename']}: exit code {c['exit_code']}, {c['seconds']} s")
                        if c["exit_code"] == 0 and c.get("parsed"):
                            ver = {"tool": "coding_tool", "script": c["filename"], "results": c["parsed"]}
                            lines.append(f"VERIFIED printed by the script: {c['parsed']}")
                        elif c["exit_code"] != 0:
                            lines.append("stderr tail:\n" + "\n".join((c.get("stderr") or "").strip().splitlines()[-10:]))
                    else:
                        lines.append(f"{c['action']} {c['filename']}: {c.get('error', '')}")
                elif t.name == "focusing_tool" and c.get("ok"):
                    lines.append(f"experiment {c['experiment']}: {json.dumps(c['settings'])} -> enhancement "
                                 f"{c['enhancement']:.2f} +/- {c['enhancement_std']:.2f} (far field {c['enhancement_far']:.2f}), "
                                 f"{c['segments']} segments, camera model {json.dumps(c['camera_model'])}, "
                                 f"{c['saturated_pixels_final']} saturated px in the final image")
                    # kept: the best experiment with the latest camera model (default bench camera or measured camera)
                    if (ver is None or ver.get("tool") != "focusing_tool" or ver.get("camera_model") != c["camera_model"]
                            or c["enhancement"] > ver["enhancement"]):
                        ver = {"tool": "focusing_tool", "experiment": c["experiment"], "enhancement": c["enhancement"],
                               "enhancement_far": c["enhancement_far"], "settings": c["settings"],
                               "camera_model": c["camera_model"]}
        return say("executor", text.strip() + "\n\n[tool record]\n" + ("\n".join(lines) or "no tool was called"),
                   verified=ver)

    # -- the hardware turn -------------------------------------------------------------------
    def hw_start(state):
        info = bridge.call(INFO_TOOL, {})
        keys = ("vendor", "model", "sensor_width", "sensor_height", "bit_depth", "max_dn", "exposure_range_us",
                "gain_range_db", "exposure_us", "gain_db", "roi", "illumination_control", "lab_limits")
        brief = {k: info[k] for k in keys if k in info}
        return say("instrument", f"camera_info -> {json.dumps(brief)}", hw="start", tool=INFO_TOOL, result=info)

    def hw_steps(d):
        return d[hw_turn_start(d):]

    def hw_decide(state):
        d = state["dialogue"]
        turn = hw_steps(d)
        k = sum(1 for t in turn if t["meta"].get("hw") == "decide")
        start = hw_turn_start(d)
        prompt = (render_dialogue(d[:start]) + "\n\n=== instruction for this hardware turn ===\n" + instruction_for(d[:start])
                  + "\n\n=== this hardware turn so far ===\n" + render_dialogue(turn)
                  + f"\n\n[status] step {k + 1} of {max_hw_steps}. Decide the next step.")
        if k >= max_hw_steps:
            dec = HWDecision(action="finish", reason="step budget of this hardware turn is used",
                             summary="Step budget used; the measurements above are what the camera returned.")
        else:
            dec = backend.decide(hw_sys, prompt, HWDecision)
        if dec.action == "command" and dec.command not in command_names:
            dec = HWDecision(action="measure", n_frames=dec.n_frames, label=dec.label, regions=dec.regions,
                             reason=f"'{dec.command}' is not an instrument command tool ({command_names}); measuring instead")
        what = {"command": f"{dec.command} {json.dumps(dec.command_args)}"
                           + ("" if dec.command in NO_CAPTURE else f", then capture {dec.n_frames} frame(s)"),
                "measure": f"capture {dec.n_frames} frame(s) with the current settings" + (" + analyze_frame" if dec.analyze else ""),
                "ask_person": f"ask the person: {dec.question}",
                "finish": f"finish: {dec.summary}"}[dec.action]
        return say("executor", f"Hardware step {k + 1}: {what}\nReason: {dec.reason}", hw="decide", step=k + 1,
                   decision=dec.model_dump())

    def last_hw_decision(d):
        for t in reversed(d):
            if t["meta"].get("hw") == "decide":
                return t["meta"]["decision"]
        raise RuntimeError("no hardware decision")

    def hw_command(state):
        dec = last_hw_decision(state["dialogue"])
        res = bridge.call(dec["command"], dec["command_args"])
        shown = {k: v for k, v in res.items() if k not in ("ok", "series")}
        return say("instrument", f"{dec['command']} {json.dumps(dec['command_args'])} -> "
                   + ("ok " + json.dumps(shown) if res.get("ok") else f"REFUSED: {res.get('error')}"),
                   hw="command", tool=dec["command"], args=dec["command_args"], result=res)

    def hw_measure(state):
        dec = last_hw_decision(state["dialogue"])
        args = {"n_frames": max(1, int(dec["n_frames"])), "label": dec["label"]}
        if dec["regions"]:
            args["regions"] = dec["regions"]
        res = bridge.call("capture", args)
        text, meta = "capture -> " + short_capture(res), {"capture": res}
        if res.get("ok") and dec["analyze"]:
            aargs = {"frame_file": res["frame_file"], "black_level_dn": dec["black_level_dn"]}
            if dec.get("analysis_region"):
                aargs["region"] = dec["analysis_region"]
            an = bridge.call("analyze_frame", aargs)
            meta["analysis"] = an
            text += "\nanalyze_frame -> " + (json.dumps({k: an[k] for k in ("speckle", "beam")}) if an.get("ok") else
                                             f"ERROR {an.get('error')}")
        return say("instrument", text, hw="measure", **meta)

    def hw_person(state):
        dec = last_hw_decision(state["dialogue"])
        reply = interrupt({"kind": "person_action", "question": dec["question"], "proposed_step": None})
        reply = reply if isinstance(reply, str) else str(reply.get("reply", reply))
        return say("human", reply, hw="person", kind="person_action", question=dec["question"])

    def hw_report(state):
        d = state["dialogue"]
        turn = hw_steps(d)
        dec = last_hw_decision(d)
        caps = [t["meta"]["capture"] for t in turn if t["meta"].get("hw") == "measure" and t["meta"]["capture"].get("ok")]
        cmds = [t for t in turn if t["meta"].get("hw") == "command"]
        lines = [f"commands: {len(cmds)} ({sum(1 for t in cmds if t['meta']['result'].get('ok'))} applied), "
                 f"captures: {len(caps)}"]
        for t in turn:
            m = t["meta"]
            if m.get("hw") == "command":
                lines.append(f"  {m['tool']} {json.dumps(m['args'])} -> " + ("ok" if m["result"].get("ok") else "refused"))
            elif m.get("hw") == "measure":
                lines.append("  capture " + short_capture(m["capture"])[:300])
        ver = None
        if caps:
            last = caps[-1]
            ver = {"tool": "camera", "settings": last["settings"], "stats": {k: last["stats"][k] for k in
                   ("mean_dn", "max_dn", "p99_9_dn", "saturated_pixels", "saturated_fraction", "fraction_of_full_scale")},
                   "temporal_noise": last.get("temporal_noise"), "frames": [os.path.basename(c["frame_file"]) for c in caps]}
            ans = [t["meta"]["analysis"] for t in turn if t["meta"].get("hw") == "measure" and t["meta"].get("analysis", {}).get("ok")]
            if ans:
                ver["analysis"] = {k: ans[-1][k] for k in ("speckle", "beam")}
        series = [t["meta"]["result"] for t in cmds if t["meta"]["tool"] in SELF_MEASURING and t["meta"]["result"].get("ok")]
        if series:
            ver = ver or {"tool": "camera"}
            ver["series"] = [{k: s[k] for k in ("settings", "n_frames", "duration_s", "summary")} for s in series]
        for tool, key, name in (("read_temperature", "temperature_c", "temperature_c"),
                                ("save_tiff", "tiff_file", "tiff_files")):
            vals = [t["meta"]["result"][key] for t in cmds if t["meta"]["tool"] == tool and t["meta"]["result"].get("ok")]
            if vals:
                ver = ver or {"tool": "camera"}
                ver[name] = vals
        return say("executor", f"Hardware turn report: {dec['summary']}\n\n[instrument record]\n" + "\n".join(lines),
                   hw="report", verified=ver)

    # -- routing -------------------------------------------------------------------------------
    def route_orch(state):
        m = state["dialogue"][-1].get("meta", {})
        if "stop" in m:
            return "end"
        nr = m["decision"]["next_role"]
        return {"finish": "end", "hardware": "hw_start"}.get(nr, nr)

    def route_human(state):
        m = state["dialogue"][-1]["meta"]
        if m.get("approved") and m["pending"]["role"] in ("executor", "hardware"):
            return "hw_start" if m["pending"]["role"] == "hardware" else "executor"
        return "orchestrator"

    def route_decide(state):
        return {"command": "hw_command", "measure": "hw_measure", "ask_person": "hw_person",
                "finish": "hw_report"}[state["dialogue"][-1]["meta"]["decision"]["action"]]

    def route_command(state):
        m = state["dialogue"][-1]["meta"]
        return "hw_measure" if m["result"].get("ok") and m["tool"] not in NO_CAPTURE else "hw_decide"

    g = StateGraph(DialogueState)
    for name, fn in (("orchestrator", orchestrator), ("scientist", scientist), ("general", general), ("executor", executor),
                     ("hw_start", hw_start), ("hw_decide", hw_decide), ("hw_command", hw_command),
                     ("hw_measure", hw_measure), ("hw_person", hw_person), ("hw_report", hw_report)):
        g.add_node(name, fn)
    dest = {"scientist": "scientist", "general": "general", "executor": "executor", "hw_start": "hw_start", "end": END}
    if allow_human:
        g.add_node("human", human)
        dest["human"] = "human"
        g.add_conditional_edges("human", route_human, {"executor": "executor", "hw_start": "hw_start",
                                                       "orchestrator": "orchestrator"})
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_orch, dest)
    for r in ("scientist", "general", "executor", "hw_report"):
        g.add_edge(r, "orchestrator")
    g.add_edge("hw_start", "hw_decide")
    g.add_conditional_edges("hw_decide", route_decide, {"hw_command": "hw_command", "hw_measure": "hw_measure",
                                                        "hw_person": "hw_person", "hw_report": "hw_report"})
    g.add_conditional_edges("hw_command", route_command, {"hw_measure": "hw_measure", "hw_decide": "hw_decide"})
    g.add_edge("hw_measure", "hw_decide")
    g.add_edge("hw_person", "hw_decide")
    return g.compile(checkpointer=checkpointer or InMemorySaver()), coding


def pending_interrupts(graph, config) -> list:
    snap = graph.get_state(config)
    out = list(getattr(snap, "interrupts", []) or [])
    if not out:
        for t in snap.tasks:
            out += list(t.interrupts)
    return out
