"""Homework 2 agent: the Homework 1 Task 6 graph, extended.

Kept from HW1: the four roles (orchestrator, scientist, general, executor), the shared dialogue as the graph
state, the orchestrator's choice of the next role as a model call (structured output), the 8-decision budget,
and the coding tool given only to the executor.

Task 1 (human in the loop): the orchestrator can choose a fifth destination, `human`, for
  * an ambiguous query  -> asks which reading to follow;
  * a hard decision     -> states the step (run the coding tool, overwrite a result on disk, or finish and
                           report a measurement) and asks whether to proceed.
The `human` node calls langgraph.types.interrupt(): the compiled graph (with a checkpointer) pauses on its
thread until it is resumed with Command(resume=<reply>).  The reply is appended to the dialogue with role
`human`.  Accepted hard decision -> the proposed step runs; refused -> back to the orchestrator, the step
does not run.  Ambiguous -> back to the orchestrator, which follows the reading the human gave.

Task 2: the executor also receives `focusing_tool` (bench.py); with allow_human=False the system works on its
own after the request.
"""
from __future__ import annotations

import json
import operator
import os
import re
import sys
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "task6_agentic_system"))
from agents import GENERAL_SYSTEM, SCIENTIST_SYSTEM, render_dialogue  # noqa: E402  (HW1 prompts reused)
from coding_tool import CodingTool  # noqa: E402

MAX_DECISIONS = 8


# ----------------------------------------------------------------------------
# state and decisions
# ----------------------------------------------------------------------------
class Turn(TypedDict, total=False):
    role: str          # user | orchestrator | scientist | general | executor | human
    content: str
    meta: dict


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]


class DecisionHITL(BaseModel):
    """Orchestrator decision when a person can be asked."""
    next_role: Literal["scientist", "general", "executor", "human", "finish"] = Field(
        description="who gets the turn; 'human' pauses the graph and asks the person")
    instruction: str = Field("", description="instruction for scientist/general/executor; for finish: the final report; empty for human")
    question: str = Field("", description="only for human: the question the person is asked (offer the readings / state the step)")
    reason: str = Field(description="one sentence: why this destination now")
    stop_kind: Literal["none", "ambiguous", "hard_decision"] = Field("none", description="only for human: why the graph stops")
    pending_role: Literal["none", "executor", "finish"] = Field(
        "none", description="only for a hard decision: the step that waits for approval")
    pending_instruction: str = Field(
        "", description="only for a hard decision: the executor instruction, or the final report, that runs if the person accepts")


class DecisionAuto(BaseModel):
    """Orchestrator decision when the system works on its own."""
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(description="instruction for that role; for finish: the final report")
    reason: str = Field(description="one sentence: why this role now")


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
ORCH_BASE = """You are the ORCHESTRATOR of a team. The user speaks only to you. You read the dialogue and decide who speaks
next. You never write theory, equations, plans or code, and you have no tools.
Roles:
- scientist: owns the theory and the plan (equations, model, experiment settings, numbered steps). Plan questions go here.
- general: answers a high-level side question: what a record/file is, what a term means, what a reported number measures.
- executor: the only role with tools ({tools}). Implements the scientist's latest plan and reports what the tools returned.
  Tracebacks and repairs go here.
- finish: end the run; the instruction is your final report to the user, quoting only numbers a tool actually returned
  (VERIFIED records). Finishing is allowed only after an executor turn holds a VERIFIED measurement.
You have at most {max_dec} decisions in total; budget them."""

ORCH_HUMAN = """
- human: pause the graph and ask the person. Use it in exactly two situations, judging from the dialogue:
  1. AMBIGUOUS QUERY (stop_kind="ambiguous"): the latest request has more than one reading, and the readings would send
     the work to different roles or would change the number that gets reported. Ask which reading to follow; list the
     readings in the question. After the reply, follow the reading the person chose.
  2. HARD DECISION (stop_kind="hard_decision"): the next step would run the coding tool (any executor turn), overwrite a
     result that is already on disk (see the files listed in the status), or end the dialogue and report a measurement
     (finish). Do not take such a step directly: put it in pending_role ("executor" or "finish") and pending_instruction
     (the exact executor instruction, or the exact final report), state it in the question, and ask whether to proceed.
     If the person accepts, the step runs exactly as written. If they refuse, the turn comes back to you and the step
     did not run: adapt (e.g. change the step, ask again, or take another route) and do not repeat the refused step
     unchanged without asking.
Do not ask the person anything else, and do not ask a question that the dialogue already answers."""

EXECUTOR_BASE = """You are the EXECUTOR, the only role with tools. Implement the latest instruction and the scientist's latest
plan, call the tools, and then reply with exactly what the tools returned (file names, exit codes, printed numbers,
enhancements, image paths) - or the traceback/error. Never state a number a tool did not return."""

CODING_TEXT = """
Tool coding_tool(action, filename, code): action="write" writes a NEW Python file into the working directory;
action="run" runs it (300 s limit) -> exit code, stdout, stderr. One write and one run per turn.
Environment: Python 3 with numpy, scipy, torch (CPU), matplotlib. Working-directory contents are listed in the prompt.
Scripts that measure the Silverbox model must print TRAIN_RMSE=<float> and TEST_RMSE=<float> (volts)."""

FOCUS_TEXT = """
Tool focusing_tool(...): runs ONE focusing experiment on the laboratory bench (laser -> DMD -> scattering medium -> camera)
and returns the enhancement and the camera images. Bench: {bench}
Algorithm inside the tool: the DMD is divided into macropixels of macropixel x macropixel mirrors, starting from a random
binary pattern; each iteration flips `flips` random macropixels (annealed linearly to `flips_final`), takes a camera shot,
reads the mean intensity in the focus square (focus_size x focus_size px) and in bg_regions sectors of the ring
bg_inner < r <= bg_outer px around it, enhancement = focus mean / mean of the sector means; the flip is kept if the
enhancement increased (with remeasure=True the current pattern is re-measured each iteration against noise).
It returns: enhancement (+ std over final_shots fresh shots), enhancement_far (focus against the camera outside the
ring, which the search does not use), enhancement_initial, the progress trace, segments, camera_shots, seconds, and
image paths. Limits: macropixel divides {dmd} (1..32), focus_size odd, focus_size//2+1 <= bg_inner < bg_outer <= {rmax},
2..16 regions, iterations*(2 if remeasure else 1) <= 80000. Up to {ncalls} experiments per executor turn; choose the
settings yourself from the plan and from earlier results."""


def orchestrator_system(allow_human, tools_text, max_dec):
    return ORCH_BASE.format(tools=tools_text, max_dec=max_dec) + (ORCH_HUMAN if allow_human else
                                                                  "\nThe person is not available during this run: decide on your own.")


# ----------------------------------------------------------------------------
# helpers on the dialogue
# ----------------------------------------------------------------------------
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
    """Accept only an explicit yes; anything else (no, not now, a counter-proposal) is a refusal."""
    return bool(APPROVE.match(reply or ""))


# ----------------------------------------------------------------------------
# model back-end
# ----------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    func: Callable[..., dict]          # python callable returning a JSON-able dict
    new_turn: Callable[[], None]
    log: list                          # list of call records of that tool


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def lc_tools(tools: list[ToolSpec]) -> dict:
    """The executor's tools as LangChain StructuredTools (name -> tool), shared by both back-ends."""
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
                              final_shots: int = 5, label: str = "", _f=spec.func) -> str:
                """Run one focusing experiment on the bench (DMD macropixel search) and return the enhancement and camera images."""
                return json.dumps(_f(macropixel=macropixel, focus_size=focus_size, bg_inner=bg_inner, bg_outer=bg_outer,
                                     bg_regions=bg_regions, iterations=iterations, flips=flips, flips_final=flips_final,
                                     remeasure=remeasure, final_shots=final_shots, label=label), default=str)
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

    def tool_loop(self, system, prompt, tools: list[ToolSpec], max_iters: int = 8):
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
    """The same three methods through the Claude Code CLI (task6_agentic_system/claude_cli.py), no API key."""

    def __init__(self, model: str):
        from claude_cli import ClaudeCLI
        self.cli = ClaudeCLI(model)

    def decide(self, system, prompt, schema):
        return schema.model_validate(self.cli.structured(system, prompt, schema.model_json_schema()))

    def chat(self, role, system, prompt):
        return self.cli.text(system, prompt)

    def tool_loop(self, system, prompt, tools: list[ToolSpec], max_iters: int = 8):
        return self.cli.tool_loop(system, prompt, list(lc_tools(tools).values()), max_iters,
                                  "Tool budget for this turn is used. Report now what the tools returned.")


def make_backend(model: str):
    """claude-cli:<model> -> Claude Code CLI; anything else -> a LangChain chat model id (provider:model)."""
    if model.startswith("claude-cli:"):
        return ClaudeCLIBackend(model.split(":", 1)[1])
    return LangChainBackend(model)


# ----------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------
def build_graph(backend, tools: list[ToolSpec], workdir: str, allow_human: bool = True, max_decisions: int = MAX_DECISIONS,
                checkpointer=None, on_turn=None, focus_bench_text: str = ""):
    emit = on_turn or (lambda t: None)
    names = [t.name for t in tools]
    schema = DecisionHITL if allow_human else DecisionAuto
    orch_sys = orchestrator_system(allow_human, ", ".join(names), max_decisions)
    exec_sys = EXECUTOR_BASE + (CODING_TEXT if "coding_tool" in names else "")
    if "focusing_tool" in names:
        ft = next(t for t in tools if t.name == "focusing_tool")
        b = ft.func.bench.cfg
        exec_sys += FOCUS_TEXT.format(bench=focus_bench_text or ft.func.describe_bench(), dmd=b.dmd_side,
                                      rmax=b.cam_side // 2 - 1, ncalls=ft.func.max_calls_per_turn)

    def n_tool_calls():
        return sum(len(t.log) for t in tools)

    def files_on_disk():
        out = []
        for root, _, files in os.walk(workdir):
            for f in files:
                if f.endswith((".json", ".png", ".csv", ".txt", ".pt", ".npz", ".md", ".log", ".py")) and not f.endswith(".mat"):
                    out.append(os.path.relpath(os.path.join(root, f), workdir))
        return sorted(out)[:60]

    # -- orchestrator ---------------------------------------------------------------
    def orchestrator(state):
        d = state["dialogue"]
        last = d[-1]
        lm = last.get("meta", {})
        if last["role"] == "human" and lm.get("approved") and lm.get("pending", {}).get("role") == "finish":
            turn = {"role": "orchestrator", "content": "Final report (accepted by the human):\n" + lm["pending"]["instruction"],
                    "meta": {"stop": "finished", "verified": verified(d)}}
            emit(turn)
            return {"dialogue": [turn]}
        k = n_decisions(d)
        if k >= max_decisions:
            rec = verified(d)
            turn = {"role": "orchestrator", "content": f"Decision budget ({max_decisions}) is used; stopping. Last verified result: {rec}.",
                    "meta": {"stop": "budget", "verified": rec}}
            emit(turn)
            return {"dialogue": [turn]}
        rec = verified(d)
        record = ""
        if last["role"] == "human" and lm.get("kind") == "hard_decision" and not lm.get("approved"):
            p = lm["pending"]
            record = (f"Record: the proposed step ({p['role']}: {p['instruction'][:300]}) was REFUSED by the human and did NOT run. "
                      f"Tool calls in this run: {lm['tool_calls_at_question']} when it was proposed, {n_tool_calls()} now.\n")
        status = (f"[status] decision {k + 1} of {max_decisions}. "
                  + (f"VERIFIED measurement exists: {json.dumps(rec)}. " if rec else "No VERIFIED measurement yet, so finish is not allowed. ")
                  + f"Files on disk in the working directory: {files_on_disk()}.")
        prompt = render_dialogue(d) + "\n\n" + record + status + "\nDecide the next turn."
        dec = backend.decide(orch_sys, prompt, schema)
        notes = []

        def wants_finish(x):
            return x.next_role == "finish" or (getattr(x, "next_role", "") == "human" and getattr(x, "pending_role", "none") == "finish")

        if wants_finish(dec) and rec is None:
            notes.append(f"finish proposed without a verified measurement ({dec.reason}); asked again")
            dec = backend.decide(orch_sys, prompt + "\n\nRejected: no executor turn holds a VERIFIED measurement, so the run "
                                 "cannot finish yet. Choose another destination.", schema)
            if wants_finish(dec):
                notes.append("finish again; routed to the scientist")
                dec = schema(next_role="scientist", instruction="Give a plan the executor can run to obtain the requested measurement.",
                             reason="A verified measurement is needed before finishing.")
        if dec.next_role == "human" and not dec.question.strip():
            dec.question = dec.reason
        if dec.next_role == "human" and dec.stop_kind == "none":
            dec.stop_kind = "hard_decision" if dec.pending_role != "none" else "ambiguous"
        if dec.next_role == "human":
            body = (f"Decision {k + 1}/{max_decisions} -> human ({dec.stop_kind.replace('_', ' ')})\n"
                    + (f"Proposed step: {dec.pending_role}: {dec.pending_instruction}\n" if dec.stop_kind == "hard_decision" else "")
                    + f"Question: {dec.question}\nReason: {dec.reason}")
        else:
            body = f"Decision {k + 1}/{max_decisions} -> {dec.next_role}\nInstruction: {dec.instruction}\nReason: {dec.reason}"
        if notes:
            body += "\nGuard: " + "; ".join(notes)
        turn = {"role": "orchestrator", "content": record + body,
                "meta": {"decision": dec.model_dump(), "index": k + 1, "guard_notes": notes, "verified": rec,
                         "refused_record": bool(record)}}
        emit(turn)
        return {"dialogue": [turn]}

    # -- human (interrupt) ----------------------------------------------------------------
    def human(state):
        d = state["dialogue"]
        dec = last_decision(d)
        hard = dec.get("stop_kind") == "hard_decision"
        payload = {"kind": dec.get("stop_kind"), "question": dec.get("question"),
                   "proposed_step": {"role": dec.get("pending_role"), "instruction": dec.get("pending_instruction")} if hard else None}
        reply = interrupt(payload)                       # the graph pauses here until resumed on the same thread
        reply = reply if isinstance(reply, str) else str(reply.get("reply", reply))
        approved = is_approval(reply) if hard else None
        turn = {"role": "human", "content": reply,
                "meta": {"kind": payload["kind"], "question": payload["question"], "approved": approved,
                         "pending": {"role": dec.get("pending_role"), "instruction": dec.get("pending_instruction")} if hard else None,
                         "tool_calls_at_question": n_tool_calls()}}
        emit(turn)
        return {"dialogue": [turn]}

    # -- specialists ----------------------------------------------------------------------
    def instruction_for(d):
        dec = last_decision(d)
        if dec.get("next_role") == "human":            # approved hard decision: the pending step runs as written
            return dec.get("pending_instruction", "")
        return dec.get("instruction", "")

    def specialist_prompt(d, role):
        human_note = ""
        if d[-1]["role"] == "human" and d[-1]["meta"].get("approved"):
            human_note = "\n(The human accepted this step.)"
        return (render_dialogue(d) + f"\n\n=== instruction from the orchestrator to the {role.upper()} ===\n"
                + instruction_for(d) + human_note)

    def scientist(state):
        text = backend.chat("scientist", SCIENTIST_SYSTEM, specialist_prompt(state["dialogue"], "scientist"))
        turn = {"role": "scientist", "content": text, "meta": {}}
        emit(turn)
        return {"dialogue": [turn]}

    def general(state):
        text = backend.chat("general", GENERAL_SYSTEM, specialist_prompt(state["dialogue"], "general"))
        turn = {"role": "general", "content": text, "meta": {}}
        emit(turn)
        return {"dialogue": [turn]}

    def executor(state):
        d = state["dialogue"]
        prompt = specialist_prompt(d, "executor") + f"\n\nFiles in the working directory: {files_on_disk()}"
        before = {t.name: len(t.log) for t in tools}
        for t in tools:
            t.new_turn()
        text = backend.tool_loop(exec_sys, prompt, tools)
        lines, ver = [], None
        for t in tools:
            for c in t.log[before[t.name]:]:
                if t.name == "coding_tool":
                    if c["action"] == "write":
                        lines.append(f"write {c['filename']}: " + ("ok" if c.get("ok") else c.get("error", "failed")))
                    elif "exit_code" in c:
                        lines.append(f"run {c['filename']}: exit code {c['exit_code']}, {c['seconds']} s")
                        if c["exit_code"] == 0 and c.get("parsed"):
                            ver = {"tool": "coding_tool", "script": c["filename"], **c["parsed"]}
                            lines.append(f"VERIFIED printed by the script: {c['parsed']}")
                        elif c["exit_code"] != 0:
                            lines.append("stderr tail:\n" + "\n".join((c.get("stderr") or "").strip().splitlines()[-10:]))
                    else:
                        lines.append(f"{c['action']} {c['filename']}: {c.get('error', '')}")
                elif t.name == "focusing_tool":
                    if c.get("ok"):
                        lines.append(f"experiment {c['experiment']}: {json.dumps(c['settings'])} -> enhancement "
                                     f"{c['enhancement']:.2f} +/- {c['enhancement_std']:.2f} (far field {c['enhancement_far']:.2f}), "
                                     f"{c['segments']} segments, image {os.path.basename(c['images']['figure'])}")
                        if ver is None or ver.get("tool") != "focusing_tool" or c["enhancement"] > ver["enhancement"]:
                            ver = {"tool": "focusing_tool", "experiment": c["experiment"], "enhancement": c["enhancement"],
                                   "enhancement_far": c["enhancement_far"], "settings": c["settings"]}
        if ver and ver.get("tool") == "focusing_tool":
            lines.append(f"VERIFIED best enhancement this turn: {ver['enhancement']:.2f} (experiment {ver['experiment']})")
        if not lines:
            lines.append("no tool was called in this turn")
        turn = {"role": "executor", "content": text.strip() + "\n\n[tool record]\n" + "\n".join(lines),
                "meta": {"verified": ver}}
        emit(turn)
        return {"dialogue": [turn]}

    # -- routing ---------------------------------------------------------------------------
    def route_orch(state):
        m = state["dialogue"][-1].get("meta", {})
        if "stop" in m:
            return "end"
        nr = m["decision"]["next_role"]
        return "end" if nr == "finish" else nr

    def route_human(state):
        m = state["dialogue"][-1]["meta"]
        if m.get("approved") and m["pending"]["role"] == "executor":
            return "executor"
        return "orchestrator"

    g = StateGraph(DialogueState)
    for name, fn in (("orchestrator", orchestrator), ("scientist", scientist), ("general", general), ("executor", executor)):
        g.add_node(name, fn)
    dest = {"scientist": "scientist", "general": "general", "executor": "executor", "end": END}
    if allow_human:
        g.add_node("human", human)
        dest["human"] = "human"
        g.add_conditional_edges("human", route_human, {"executor": "executor", "orchestrator": "orchestrator"})
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_orch, dest)
    for r in ("scientist", "general", "executor"):
        g.add_edge(r, "orchestrator")
    return g.compile(checkpointer=checkpointer)


def pending_interrupts(graph, config) -> list:
    snap = graph.get_state(config)
    out = list(getattr(snap, "interrupts", []) or [])
    if not out:
        for t in snap.tasks:
            out += list(t.interrupts)
    return out


def coding_spec(workdir) -> ToolSpec:
    tool = CodingTool(workdir=workdir)
    return ToolSpec("coding_tool", tool, tool.new_turn, tool.log)


def focusing_spec(out_dir, **kw) -> ToolSpec:
    from bench import FocusingTool
    tool = FocusingTool(out_dir, **kw)
    return ToolSpec("focusing_tool", tool, tool.new_turn, tool.log)
