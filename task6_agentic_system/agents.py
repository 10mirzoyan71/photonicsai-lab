"""Four-role agentic system as a LangGraph StateGraph.

    user -> orchestrator -> {scientist | general | executor} -> orchestrator -> ... -> END

* The state is the dialogue: a list of turns, each turn = {role, content, meta}.
  Nodes read the whole dialogue and append exactly one turn.
* The orchestrator is a model call that returns a structured decision
  (next_role, instruction, reason).  It writes no theory, no code, has no tool.
  It stops on `finish` or after 8 decisions.  `finish` is accepted only after an
  executor turn whose tool record holds a TEST_RMSE printed by a script that ran.
* The scientist owns the theory and the plan (no tool).
* The general agent answers high-level side questions (no tool, no plan).
* The executor is the only role with the coding tool (write a .py file, run it
  with a 5-minute limit).  It implements the scientist's latest plan.
"""
from __future__ import annotations

import json
import operator
import os
from typing import Annotated, Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from coding_tool import CodingTool

MAX_DECISIONS = 8
ROLES = ("scientist", "general", "executor")


# ----------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------
class Turn(TypedDict, total=False):
    role: str            # user | orchestrator | scientist | general | executor
    content: str         # the text that role said
    meta: dict           # decision fields / tool records (machine-readable)


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]     # each node appends its turn


class Decision(BaseModel):
    """The orchestrator's decision for the next turn."""
    next_role: Literal["scientist", "general", "executor", "finish"] = Field(
        description="who speaks next, or 'finish' to end the run")
    instruction: str = Field(description="the instruction that role should follow now (for finish: the final answer to the user)")
    reason: str = Field(description="one sentence: why this role, now")


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
ORCHESTRATOR_SYSTEM = f"""You are the ORCHESTRATOR of a team of four roles. The user speaks only to you.
You read the dialogue and decide who speaks next. You never write theory, equations, plans or code, and you have no tools.

Roles you can hand the turn to:
- scientist: owns the theory and the plan. Writes the differential equation, the proposed network, the state equation to
  integrate, how to train, the free-run test, and a numbered list of steps specific enough to code. Send every question
  about the fitting plan or the differential equation here, including revisions of the plan after results.
- general: answers a high-level question that sits beside the fit: what a record/channel in the data file is, what a term in
  the problem means, what a reported error is measuring. No plan, no code.
- executor: the only role with the coding tool. Implements the scientist's LATEST plan in a new Python file, runs it
  (5-minute limit) and returns the free-run RMSE the script printed, or the traceback. Send tracebacks and repairs here.
- finish: end the run. Allowed ONLY after an executor turn whose tool record is marked VERIFIED, i.e. it contains a
  TEST_RMSE printed by a script that actually ran. With finish, the instruction field is your final answer to the user
  (state the verified numbers exactly as printed).

You have at most {MAX_DECISIONS} decisions in total, so budget them: the plan must be written and executed, and a failed
script may need a repair turn. Return the next role, a concrete instruction for that role, and a one-sentence reason."""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own the theory and the plan. You have no coding tool and you write no code.
From the problem, write:
 1. the differential equation and the meaning (and expected size) of each coefficient;
 2. the network you propose for the acceleration: inputs, input/output scaling, depth, widths, activation;
 3. the state equation to integrate, the integrator and how the measured input enters each integrator stage;
 4. how to train it: initial states, loss, rollout length, optimizer, learning rate, number of steps; the whole script
    must finish within 5 minutes on a CPU, so give a budget and a wall-clock guard;
 5. the free-run test exactly as the problem defines it (train and test RMSE);
 6. a NUMBERED list of implementation steps, specific enough to code, ending with printing the lines
    TRAIN_RMSE=<float> and TEST_RMSE=<float> (volts) and saving the test-segment plot to test_segment.png.
Never claim a number was measured unless an executor turn (VERIFIED tool record) already printed it.
When asked to revise after results, keep the same differential equation and the same free-run test."""

GENERAL_SYSTEM = """You are the GENERAL agent. You answer one high-level question that sits beside the fit: what a record or
channel in a data file is, what a term in the problem means, what a reported error measures. Be accurate and brief.
You have no coding tool, you do not write the fitting plan, and you do not invent measured numbers."""

EXECUTOR_SYSTEM = """You are the EXECUTOR, the only role with the coding tool `coding_tool`:
  coding_tool(action="write", filename="name.py", code="...")   writes a NEW file into the working directory
  coding_tool(action="run",   filename="name.py")               runs it (time limit 300 s) -> exit code, stdout, stderr
Per turn you may write ONE new file and run it ONCE. Then reply.

Implement the scientist's LATEST plan exactly (same differential equation, same free-run test). Environment: Python 3 with
numpy, scipy, torch (CPU) and matplotlib. The data file ./SNLS80mV.mat is in the working directory (MATLAB v5; variables
V1 = input voltage u, V2 = output voltage y, each 1 x 131072). The whole script must finish within 5 minutes: put a
wall-clock guard on training. It must print the free-run results as the lines
  TRAIN_RMSE=<float>
  TEST_RMSE=<float>
(volts) and save the measured-vs-predicted test-segment plot as test_segment.png.
If an earlier script failed, repair it in a NEW file (e.g. suffix _v2) using the traceback; do not change the plan.
Your reply: file name, exit code, the RMSE lines exactly as printed - or the traceback. Never state a number the script
did not print."""


# ----------------------------------------------------------------------------
# rendering the dialogue for a model
# ----------------------------------------------------------------------------
def render_dialogue(dialogue: list[Turn]) -> str:
    out = []
    for i, t in enumerate(dialogue, 1):
        out.append(f"=== turn {i} | {t['role'].upper()} ===\n{t['content']}")
    return "\n\n".join(out)


def n_decisions(dialogue) -> int:
    return sum(1 for t in dialogue if t["role"] == "orchestrator" and "decision" in t.get("meta", {}))


def verified_rmse(dialogue) -> Optional[dict]:
    """Latest executor tool record with a TEST_RMSE printed by a script that ran (exit code 0)."""
    for t in reversed(dialogue):
        rec = t.get("meta", {}).get("verified")
        if t["role"] == "executor" and rec:
            return rec
    return None


def last_decision(dialogue) -> dict:
    for t in reversed(dialogue):
        if t["role"] == "orchestrator":
            return t["meta"]["decision"]
    raise RuntimeError("no orchestrator decision yet")


# ----------------------------------------------------------------------------
# model back-end (LangChain chat model).  A scripted stand-in with the same
# three methods is used by tests/offline_smoke_test.py.
# ----------------------------------------------------------------------------
class Backend(Protocol):
    def decide(self, system: str, prompt: str) -> Decision: ...
    def chat(self, role: str, system: str, prompt: str) -> str: ...
    def tool_loop(self, system: str, prompt: str, tool: CodingTool, max_iters: int = 6) -> str: ...


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


class LangChainBackend:
    def __init__(self, model: str, temperature: float = 0.2):
        from langchain.chat_models import init_chat_model
        self.model = model
        self.llm = init_chat_model(model, temperature=temperature, max_tokens=4000)
        self.llm_long = init_chat_model(model, temperature=temperature, max_tokens=16000)   # executor writes scripts

    def decide(self, system, prompt) -> Decision:
        from langchain_core.messages import HumanMessage, SystemMessage
        return self.llm.with_structured_output(Decision).invoke([SystemMessage(system), HumanMessage(prompt)])

    def chat(self, role, system, prompt) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        return _text(self.llm.invoke([SystemMessage(system), HumanMessage(prompt)]))

    def tool_loop(self, system, prompt, tool: CodingTool, max_iters: int = 6) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import StructuredTool

        def coding_tool(action: Literal["write", "run"], filename: str, code: str = "") -> str:
            """Write a new Python file into the run's working directory (action='write', needs code), or run an
            existing file with a 5-minute limit (action='run') and get exit code, stdout and stderr."""
            return json.dumps(tool(action, filename, code))

        lc_tool = StructuredTool.from_function(coding_tool)
        llm = self.llm_long.bind_tools([lc_tool])
        msgs = [SystemMessage(system), HumanMessage(prompt)]
        ai = None
        for _ in range(max_iters):
            ai = llm.invoke(msgs)
            msgs.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                msgs.append(ToolMessage(content=lc_tool.invoke(tc["args"]), tool_call_id=tc["id"]))
        if ai is not None and ai.tool_calls:          # ran out of iterations: ask for the report
            msgs.append(HumanMessage("Tool budget for this turn is used. Report now: file, exit code, RMSE lines or traceback."))
            ai = self.llm_long.invoke(msgs)
        return _text(ai) if ai is not None else ""


class ClaudeCLIBackend:
    """The same three methods through the Claude Code CLI (claude_cli.py): --model claude-cli:<model>, no API key."""

    def __init__(self, model: str):
        from claude_cli import ClaudeCLI
        self.model = "claude-cli:" + model
        self.cli = ClaudeCLI(model)

    def decide(self, system, prompt) -> Decision:
        return Decision.model_validate(self.cli.structured(system, prompt, Decision.model_json_schema()))

    def chat(self, role, system, prompt) -> str:
        return self.cli.text(system, prompt)

    def tool_loop(self, system, prompt, tool: CodingTool, max_iters: int = 6) -> str:
        from langchain_core.tools import StructuredTool

        def coding_tool(action: Literal["write", "run"], filename: str, code: str = "") -> str:
            """Write a new Python file into the run's working directory (action='write', needs code), or run an
            existing file with a 5-minute limit (action='run') and get exit code, stdout and stderr."""
            return json.dumps(tool(action, filename, code))

        return self.cli.tool_loop(system, prompt, [StructuredTool.from_function(coding_tool)], max_iters,
                                  "Tool budget for this turn is used. Report now: file, exit code, RMSE lines or traceback.")


def make_backend(model: str):
    """claude-cli:<model> -> Claude Code CLI; anything else -> a LangChain chat model id (provider:model)."""
    if model.startswith("claude-cli:"):
        return ClaudeCLIBackend(model.split(":", 1)[1])
    return LangChainBackend(model)


# ----------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------
def make_nodes(backend: Backend, tool: CodingTool, on_turn=None):
    emit = on_turn or (lambda turn: None)

    def orchestrator(state: DialogueState):
        d = state["dialogue"]
        k = n_decisions(d)
        if k >= MAX_DECISIONS:
            rec = verified_rmse(d)
            msg = (f"Decision budget ({MAX_DECISIONS}) is used; stopping. "
                   + (f"Verified free-run result: TEST_RMSE={rec['test']} (script {rec['script']})." if rec
                      else "No verified TEST_RMSE was produced."))
            turn = {"role": "orchestrator", "content": msg, "meta": {"stop": "budget", "verified": rec}}
            emit(turn)
            return {"dialogue": [turn]}

        rec = verified_rmse(d)
        status = (f"[status] this is decision {k + 1} of {MAX_DECISIONS}. "
                  + (f"VERIFIED executor result exists: TEST_RMSE={rec['test']} from {rec['script']}." if rec
                     else "No VERIFIED TEST_RMSE yet, so finish is not allowed."))
        prompt = render_dialogue(d) + "\n\n" + status + "\nDecide the next turn."
        dec = backend.decide(ORCHESTRATOR_SYSTEM, prompt)
        notes = []
        if dec.next_role == "finish" and rec is None:
            # the rule is checked in code; the model is asked once more with the reason
            notes.append(f"model chose finish without a verified RMSE ({dec.reason}); asked again")
            dec = backend.decide(ORCHESTRATOR_SYSTEM, prompt + "\n\nYour previous choice 'finish' was rejected: no executor "
                                 "turn holds a VERIFIED TEST_RMSE. Choose a role.")
            if dec.next_role == "finish":
                notes.append("second choice was finish again; routed to executor")
                dec = Decision(next_role="executor", reason="finish is not allowed before a verified RMSE",
                               instruction="Implement the scientist's latest plan in a new file, run it and report the printed TEST_RMSE.")
        content = (f"Decision {k + 1}/{MAX_DECISIONS} -> {dec.next_role}\n"
                   f"Instruction: {dec.instruction}\nReason: {dec.reason}")
        if notes:
            content += "\nGuard: " + "; ".join(notes)
        turn = {"role": "orchestrator", "content": content,
                "meta": {"decision": dec.model_dump(), "index": k + 1, "guard_notes": notes, "verified": rec}}
        emit(turn)
        return {"dialogue": [turn]}

    def _specialist_prompt(d, role):
        dec = last_decision(d)
        return (render_dialogue(d) + f"\n\n=== instruction from the orchestrator to the {role.upper()} ===\n"
                + dec["instruction"])

    def scientist(state: DialogueState):
        d = state["dialogue"]
        text = backend.chat("scientist", SCIENTIST_SYSTEM, _specialist_prompt(d, "scientist"))
        turn = {"role": "scientist", "content": text, "meta": {}}
        emit(turn)
        return {"dialogue": [turn]}

    def general(state: DialogueState):
        d = state["dialogue"]
        text = backend.chat("general", GENERAL_SYSTEM, _specialist_prompt(d, "general"))
        turn = {"role": "general", "content": text, "meta": {}}
        emit(turn)
        return {"dialogue": [turn]}

    def executor(state: DialogueState):
        d = state["dialogue"]
        prompt = _specialist_prompt(d, "executor")
        files = sorted(f for f in os.listdir(tool.workdir) if f.endswith(".py"))
        if files:                                   # give it the latest script so it can repair it
            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(tool.workdir, f)))
            with open(os.path.join(tool.workdir, latest)) as fh:
                prompt += (f"\n\nFiles already in the working directory: {files}.\n"
                           f"Latest script ({latest}):\n```python\n{fh.read()}\n```")
        n_before = len(tool.log)
        tool.new_turn()
        text = backend.tool_loop(EXECUTOR_SYSTEM, prompt, tool)
        calls = tool.log[n_before:]
        runs = [c for c in calls if c["action"] == "run" and "exit_code" in c]
        record_lines, verified = [], None
        for c in calls:
            if c["action"] == "write":
                record_lines.append(f"write {c['filename']}: " + ("ok" if c.get("ok") else c.get("error", "failed")))
        for r in runs:
            record_lines.append(f"run {r['filename']}: exit code {r['exit_code']}, {r['seconds']} s"
                                + (" (TIMEOUT)" if r.get("timed_out") else ""))
            if r["exit_code"] == 0 and "test" in r["parsed"]:
                verified = {"test": r["parsed"]["test"], "train": r["parsed"].get("train"), "script": r["filename"]}
                record_lines.append(f"VERIFIED printed by the script: TEST_RMSE={r['parsed']['test']}"
                                    + (f", TRAIN_RMSE={r['parsed']['train']}" if "train" in r["parsed"] else ""))
            elif r["exit_code"] != 0:
                tb = (r.get("stderr") or "").strip().splitlines()[-12:]
                record_lines.append("stderr tail:\n" + "\n".join(tb))
        if not runs:
            record_lines.append("no script was run in this turn")
        content = text.strip() + "\n\n[tool record]\n" + "\n".join(record_lines)
        turn = {"role": "executor", "content": content,
                "meta": {"tool_calls": [{k: v for k, v in c.items() if k not in ("stdout", "stderr")} for c in calls],
                         "verified": verified}}
        emit(turn)
        return {"dialogue": [turn]}

    return orchestrator, scientist, general, executor


def route(state: DialogueState) -> str:
    last = state["dialogue"][-1]
    meta = last.get("meta", {})
    if "stop" in meta:
        return "finish"
    return meta["decision"]["next_role"]


def build_graph(backend: Backend, tool: CodingTool, on_turn=None):
    orchestrator, scientist, general, executor = make_nodes(backend, tool, on_turn)
    g = StateGraph(DialogueState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("scientist", scientist)
    g.add_node("general", general)
    g.add_node("executor", executor)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route,
                            {"scientist": "scientist", "general": "general", "executor": "executor", "finish": END})
    for r in ROLES:                       # after a specialist speaks, the turn returns to the orchestrator
        g.add_edge(r, "orchestrator")
    return g.compile()
