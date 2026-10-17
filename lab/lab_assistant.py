"""Homework 4/5 - the lab assistant: a three-level hierarchy of agents on top of Homeworks 1-3.

Level 1  DIRECTOR (this file). The person speaks only to it. A model call splits the request into team tasks, passes
         verified numbers from one team to the next, stops for the person before hard steps, and writes the final
         report from the teams' VERIFIED records. LangGraph StateGraph on a checkpointed thread; its stops are interrupts.
Level 2  TEAMS: instrument (Homework 3: the lab camera behind its MCP server), optics (Homework 2: the simulated
         focusing bench), modeling (Homework 1: the Silverbox oscillator). Each team is the agent graph of Homeworks 1-3
         (orchestrator, scientist, general agent, executor, hardware turn) with its own tools, on its own thread.
Level 3  SPECIALISTS and TOOLS inside a team: scientist, general agent, executor; MCP camera tools, coding tool,
         focusing tool.
A question a team asks the person (e.g. "block the light" inside a hardware turn) is passed up as an interrupt of the
director, so the person answers everything in one place (the terminal of run_lab.py, or the web app).
"""
from __future__ import annotations

import json
import operator
import os
import shutil
import sqlite3
import sys
import threading
from typing import Annotated, Callable, Literal, TypedDict

from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "hw3", "agent"))
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

import agent_v3  # noqa: E402
from agent_v3 import is_approval, make_backend, pending_interrupts, verified  # noqa: E402

TEAMS = ("instrument", "optics", "modeling")
MAX_DIRECTOR_DECISIONS = 6
TEAM_MAX_DECISIONS = {"instrument": 8, "optics": 6, "modeling": 6}
CAMERA_LIMITS = {"max_exposure_us": 500000, "max_gain_db": 24, "max_frames": 16}


class Turn(TypedDict, total=False):
    role: str            # user | director | team | human
    content: str
    meta: dict


class LabState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]


class DirectorDecision(BaseModel):
    """The director's decision when the person can be asked."""
    next: Literal["instrument", "optics", "modeling", "human", "finish"] = Field(
        description="the team that gets the next task, 'human' to ask the person, or 'finish'")
    task: str = Field("", description="for a team: a self-contained task (the team does not see this dialogue, so copy "
                                      "every number it needs from earlier team reports); for finish: the final report")
    question: str = Field("", description="only for human: the question the person is asked")
    stop_kind: Literal["none", "ambiguous", "hard_decision"] = Field("none", description="only for human: why it stops")
    pending: Literal["none", "instrument", "optics", "modeling"] = Field(
        "none", description="only for a hard decision: the team whose task waits for approval")
    pending_task: str = Field("", description="only for a hard decision: the task that runs if accepted")
    reason: str = Field(description="one sentence: why this step now")


class DirectorDecisionAuto(BaseModel):
    """The director's decision when the person is not available."""
    next: Literal["instrument", "optics", "modeling", "finish"]
    task: str = Field(description="for a team: a self-contained task; for finish: the final report")
    reason: str = Field(description="one sentence: why this step now")


DIRECTOR_SYSTEM = """You are the DIRECTOR of a small photonics laboratory run by teams of AI agents. The person speaks only
to you. You never measure, compute or write code yourself: you give tasks to three teams, and each team has its own
orchestrator, scientist, general agent and executor with tools.
- instrument: the lab camera ({camera}) behind an MCP server: exposure, gain, region of interest, black level, ADC bit
  depth, temperature, captures, time series, frame statistics, TIFF export; plus a coding tool for analysis scripts on
  the saved frames. It can find a good exposure, measure
  the camera's noise model (conversion gain e-/DN, read noise e-, full well e-, dark level), check flicker and stability.
  Dark frames: on the real camera the person blocks the light (the team asks for that itself); the simulated camera
  switches its light off with set_illumination(0), so then nobody needs to be asked.
- optics: the SIMULATED wavefront-shaping bench of Homework 2 (laser -> DMD -> scattering medium -> camera) with its
  focusing tool. The bench's camera model can be set from measured numbers (read noise e-, full well e-, mean signal
  e-/px, intensity noise), so it can predict the focusing enhancement a real camera would allow.
- modeling: the Silverbox oscillator of Homework 1 (the measured record, the identified Duffing equation, the kept
  physics-informed network) with a coding tool; it can evaluate or re-run those models.
On each turn you decide: the next team and its task, or finish. Rules:
- Use the fewest team tasks that answer the request. A task must be self-contained: the team does not see this
  dialogue, so copy every number it needs from the earlier team reports (only VERIFIED numbers).
- Finish only when the teams' VERIFIED records answer the request. The final report quotes only numbers that a team
  reported as VERIFIED, says which team measured or computed each, and states what was simulated and what was real.
- You have at most {max_dec} decisions in total. Keep tasks and the final report short (the report: at most 12 lines)."""

DIRECTOR_HUMAN = """
- human: pause and ask the person, in exactly two situations: an AMBIGUOUS request (stop_kind="ambiguous": readings that
  would send the work to different teams or change the reported numbers; list the readings), or a HARD DECISION
  (stop_kind="hard_decision": the next task would make the instrument team change the settings of the REAL camera; put
  the task in pending/pending_task and ask once whether to proceed). Accepted -> the task runs as written; refused -> it
  does not run and the turn comes back to you. Do not ask anything else: never before finishing, never twice about the
  same task, never about the simulated camera or the simulated bench."""

BRIEF = (" Be brief: at most 12 numbered steps or 12 lines, no repetition; the people reading it are in a live demo.")
EXEC_BRIEF = "\nReport briefly: what ran and the numbers it printed, at most 10 lines."

TEAM_PROMPTS = {
    "instrument": {
        "team": "You are the ORCHESTRATOR of the INSTRUMENT team of a photonics lab; your team works with the lab camera.",
        "scientist": agent_v3.SCIENTIST_SYSTEM + BRIEF,
        "general": agent_v3.GENERAL_SYSTEM + BRIEF,
        "executor": EXEC_BRIEF,
    },
    "optics": {
        "team": ("You are the ORCHESTRATOR of the OPTICS team of a photonics lab; your team focuses light through a "
                 "scattering medium on the SIMULATED wavefront-shaping bench of Homework 2 (laser -> DMD -> medium -> "
                 "camera) with the focusing tool. Your team has no instrument."),
        "scientist": (
            "You are the SCIENTIST of the optics team. You own the theory and the plan for focusing light through a "
            "scattering medium with a binary-amplitude DMD on the simulated bench: enhancement = focus intensity over the "
            "mean of the background regions; binary amplitude shaping reaches about N/(2 pi) for N segments; a narrow "
            "background ring can be darkened by the search, so also look at the far-field enhancement; the focus is "
            "about enhancement x mean signal and must stay below the camera's full well. Give the tool settings "
            "(macropixel, focus size, ring radii, regions, iterations, flips) and, when the task gives a measured "
            "camera, its camera model (read_noise_e, full_well_e, mean_signal_e, intensity_noise), as numbered steps. "
            "No tools, no code; never claim a number that a tool did not return." + BRIEF),
        "executor": ("\nThis team has no camera folder: the focusing tool writes its figures and data to the bench "
                     "folder, and the coding tool can read them there." + EXEC_BRIEF),
    },
    "modeling": {
        "team": ("You are the ORCHESTRATOR of the MODELING team of a photonics lab; your team works on the Silverbox "
                 "oscillator of Homework 1 (the measured record, the identified Duffing equation, the kept "
                 "physics-informed network). Your team has no instrument."),
        "scientist": (
            "You are the SCIENTIST of the modeling team. You own the theory and the plan for the Silverbox oscillator: "
            "y'' + a y' + b y + c y^3 = g u(t), the excerpt [42650:46746] of SNLS80mV.mat centred by its means, 3072 "
            "training and 1024 test samples, the free-run RMSE metric (RK4 from the first sample, reading only u on the "
            "test record), the kept network A2_h5 and the identified equation R_duffing. Write the steps as a numbered "
            "list that one Python script can follow. No tools, no code; never claim a number a script did not print." + BRIEF),
        "executor": ("\nThis team has no camera: the working directory holds silverbox/silverbox_pinn.py (the Homework 1 "
                     "code), silverbox/results/*.json and *.pt (A2_h5 = the kept network, R_duffing = the identified "
                     "equation, kept.json, summary.md) and data/SNLS80mV.mat. The free-run metric of a saved model, run "
                     "from this directory, is exactly:\n"
                     "    import sys; sys.path.insert(0, 'silverbox')\n"
                     "    from silverbox_pinn import load_excerpt, load_model, half_sample_input, free_run\n"
                     "    u, y = load_excerpt()        # finds data/SNLS80mV.mat itself; centred, in volts\n"
                     "    f, cfg = load_model('silverbox/results/A2_h5.pt')     # or R_duffing.pt\n"
                     "    yhat, rmse_train, rmse_test = free_run(f, u, y, half_sample_input(u, cfg.interp))   # volts\n"
                     "Use it as it is: do not search for the data file and do not re-implement the metric." + EXEC_BRIEF),
    },
}


def prepare_modeling_workdir(work):
    """The modeling team's working directory: the Homework 1 Silverbox code, the kept results and the data record."""
    os.makedirs(os.path.join(work, "data"), exist_ok=True)
    os.makedirs(os.path.join(work, "silverbox", "results"), exist_ok=True)
    mat = os.path.join(ROOT, "data", "SNLS80mV.mat")
    if os.path.exists(mat):
        shutil.copy(mat, os.path.join(work, "data"))
    t5 = os.path.join(ROOT, "task5_pinn_silverbox")
    shutil.copy(os.path.join(t5, "silverbox_pinn.py"), os.path.join(work, "silverbox"))
    for f in ("A2_h5.pt", "A2_h5.json", "R_duffing.pt", "R_duffing.json", "kept.json", "summary.md"):
        shutil.copy(os.path.join(t5, "results", f), os.path.join(work, "silverbox", "results"))


def isolated(fn, *args, **kw):
    """Run fn in a fresh thread, so a team graph runs as an independent graph (own thread id and checkpoint), not as a
    subgraph of the director's node that is running it."""
    box = {}

    def target():
        try:
            box["r"] = fn(*args, **kw)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["e"] = e
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join()
    if "e" in box:
        raise box["e"]
    return box.get("r")


class Team:
    """One level-2 team: the Homework 1-3 agent graph with this team's tools, prompts and checkpointed threads."""

    def __init__(self, name: str, run_dir: str, backend, camera: str = "sim-room", seed: int = 1,
                 limits: dict | None = None, emit: Callable | None = None):
        self.name = name
        self.work = os.path.join(run_dir, "teams", name)
        os.makedirs(self.work, exist_ok=True)
        self.bridge, bench = None, None
        if name == "instrument":
            from mcp_bridge import MCPBridge
            self.bridge = MCPBridge(os.path.join(self.work, "camera"), kind=camera, seed=seed,
                                    limits=limits or CAMERA_LIMITS)
        elif name == "optics":
            bench = os.path.join(self.work, "bench")
        else:
            prepare_modeling_workdir(self.work)
        self.conn = sqlite3.connect(os.path.join(run_dir, f"team_{name}.sqlite"), check_same_thread=False)
        on_turn = (lambda t: emit(name, t)) if emit else None
        self.graph, self.coding = agent_v3.build_graph(
            backend, self.bridge, self.work, allow_human=False, max_decisions=TEAM_MAX_DECISIONS[name],
            checkpointer=SqliteSaver(self.conn), on_turn=on_turn, bench_dir=bench, prompts=TEAM_PROMPTS[name])

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        self.conn.close()


def n_decisions(d):
    return sum(1 for t in d if t["role"] == "director" and "decision" in t.get("meta", {}))


def last_decision(d):
    for t in reversed(d):
        if t["role"] == "director" and "decision" in t.get("meta", {}):
            return t["meta"]["decision"]
    raise RuntimeError("no director decision")


def team_records(d):
    """The VERIFIED records the teams returned so far: [(team, record), ...]."""
    return [(t["meta"]["team"], t["meta"]["verified"]) for t in d if t["role"] == "team" and t["meta"].get("verified")]


def render(d):
    return "\n\n".join(f"=== turn {i} | {t['role'].upper()}"
                       + (f" ({t['meta']['team']} team)" if t["role"] == "team" else "") + f" ===\n{t['content']}"
                       for i, t in enumerate(d, 1))


def build_lab(backend, get_team: Callable[[str], Team], allow_human: bool = True,
              max_decisions: int = MAX_DIRECTOR_DECISIONS, checkpointer=None, emit: Callable | None = None,
              camera_label: str = "simulated camera"):
    """The director graph. get_team(name) returns the Team that runs a task (built on first use)."""
    emit = emit or (lambda source, turn: None)
    schema = DirectorDecision if allow_human else DirectorDecisionAuto
    system = DIRECTOR_SYSTEM.format(camera=camera_label, max_dec=max_decisions) + (
        DIRECTOR_HUMAN if allow_human else "\nThe person is not available during this run: decide on your own.")

    def say(role, content, **meta):
        turn = {"role": role, "content": content, "meta": meta}
        emit("director", turn)
        return {"dialogue": [turn]}

    def director(state):
        d = state["dialogue"]
        last = d[-1]
        lm = last.get("meta", {})
        recs = team_records(d)
        if last["role"] == "human" and lm.get("approved") and (lm.get("pending") or {}).get("next") == "finish":
            return say("director", "Final report (accepted by the person):\n" + lm["pending"]["task"],
                       stop="finished", records=recs)
        k = n_decisions(d)
        if k >= max_decisions:
            return say("director", f"Decision budget ({max_decisions}) is used; stopping. Team records: {recs}",
                       stop="budget", records=recs)
        record = ""
        if last["role"] == "human" and lm.get("kind") == "hard_decision" and not lm.get("approved"):
            record = f"Record: the proposed step ({lm['pending']['next']}: {lm['pending']['task'][:300]}) was REFUSED by " \
                     f"the person and did NOT run.\n"
        status = (f"[status] decision {k + 1} of {max_decisions}. "
                  + (f"VERIFIED team records so far: {json.dumps(recs)[:3000]}." if recs else
                     "No VERIFIED team record yet, so finish is not allowed."))
        prompt = render(d) + "\n\n" + record + status + "\nDecide the next step."
        dec = backend.decide(system, prompt, schema)
        notes = []

        def wants_finish(x):
            return x.next == "finish" or (x.next == "human" and getattr(x, "pending", "none") == "finish")
        if wants_finish(dec) and not recs:
            notes.append("finish proposed without a verified team record; asked again")
            dec = backend.decide(system, prompt + "\n\nRejected: no team has returned a VERIFIED record yet, so the run "
                                 "cannot finish. Give a task to a team.", schema)
            if wants_finish(dec):
                return say("director", "Stopping: no team returned a verified result. " + dec.reason,
                           stop="no_result", records=recs, guard_notes=notes)
        if dec.next == "human" and dec.stop_kind == "hard_decision" and dec.pending in TEAMS:
            approved_before = any(t["role"] == "human" and t["meta"].get("kind") == "hard_decision"
                                  and t["meta"].get("approved") for t in d)
            if dec.pending != "instrument" or approved_before:
                notes.append("approval asked for " + ("a team without hardware" if dec.pending != "instrument" else
                             "the camera again after the person already approved it") + "; the task runs directly")
                dec.next, dec.task, dec.stop_kind = dec.pending, dec.pending_task, "none"
        if dec.next == "human":
            if not dec.question.strip():
                dec.question = dec.reason
            if dec.stop_kind == "none":
                dec.stop_kind = "hard_decision" if dec.pending != "none" else "ambiguous"
            body = (f"Decision {k + 1}/{max_decisions} -> human ({dec.stop_kind.replace('_', ' ')})\n"
                    + (f"Proposed step: {dec.pending}: {dec.pending_task}\n" if dec.stop_kind == "hard_decision" else "")
                    + f"Question: {dec.question}\nReason: {dec.reason}")
        elif dec.next == "finish":
            return say("director", dec.task, decision=dec.model_dump(), index=k + 1, stop="finished", records=recs)
        else:
            body = f"Decision {k + 1}/{max_decisions} -> {dec.next} team\nTask: {dec.task}\nReason: {dec.reason}"
        if notes:
            body += "\nGuard: " + "; ".join(notes)
        return say("director", record + body, decision=dec.model_dump(), index=k + 1, guard_notes=notes)

    def human(state):
        dec = last_decision(state["dialogue"])
        hard = dec.get("stop_kind") == "hard_decision"
        payload = {"kind": dec.get("stop_kind"), "source": "director", "question": dec.get("question"),
                   "proposed_step": {"next": dec.get("pending"), "task": dec.get("pending_task")} if hard else None}
        reply = interrupt(payload)
        reply = reply if isinstance(reply, str) else str(reply)
        return say("human", reply, kind=payload["kind"], question=payload["question"],
                   approved=is_approval(reply) if hard else None,
                   pending={"next": dec.get("pending"), "task": dec.get("pending_task")} if hard else None)

    def make_team_node(name):
        def node(state, config):
            d = state["dialogue"]
            dec = last_decision(d)
            task = dec["pending_task"] if dec["next"] == "human" else dec["task"]
            k = sum(1 for t in d if t["role"] == "team") + 1
            team = get_team(name)
            tcfg = {"configurable": {"thread_id": f"{config['configurable']['thread_id']}-{k}-{name}"},
                    "recursion_limit": 300}
            if not isolated(team.graph.get_state, tcfg).values:
                isolated(team.graph.invoke, {"dialogue": [{"role": "user", "content": task, "meta": {}}]}, tcfg)
            # LangGraph runs this node again from the top after every resume, and interrupt() then returns the stored
            # replies in order: skip those the team has already received, then pass on the next question.
            applied = sum(1 for t in isolated(team.graph.get_state, tcfg).values["dialogue"] if t["role"] == "human")
            for _ in range(applied):
                interrupt({"kind": "replay"})
            while True:
                intr = isolated(pending_interrupts, team.graph, tcfg)
                if not intr:
                    break
                q = intr[0].value
                reply = interrupt({"kind": q.get("kind", "question"), "source": f"{name} team",
                                   "question": q.get("question"), "proposed_step": q.get("proposed_step")})
                isolated(team.graph.invoke, Command(resume=reply if isinstance(reply, str) else str(reply)), tcfg)
            td = isolated(team.graph.get_state, tcfg).values["dialogue"]
            final = td[-1]["content"] if td else ""
            rec = verified(td)
            return say("team", f"[{name} team, task {k}] {final}", team=name, task=task, verified=rec,
                       thread=tcfg["configurable"]["thread_id"], turns=len(td))
        return node

    def route_director(state):
        m = state["dialogue"][-1]["meta"]
        if m.get("stop"):
            return END
        return m["decision"]["next"]

    def route_human(state):
        m = state["dialogue"][-1]["meta"]
        if m.get("approved") and (m.get("pending") or {}).get("next") in TEAMS:
            return m["pending"]["next"]
        return "director"

    g = StateGraph(LabState)
    g.add_node("director", director)
    g.add_node("human", human)
    for name in TEAMS:
        g.add_node(name, make_team_node(name))
        g.add_edge(name, "director")
    g.add_edge(START, "director")
    g.add_conditional_edges("director", route_director, {**{n: n for n in TEAMS}, "human": "human", END: END})
    g.add_conditional_edges("human", route_human, {**{n: n for n in TEAMS}, "director": "director"})
    return g.compile(checkpointer=checkpointer)


class LabRun:
    """One run of the lab assistant on a thread: the director graph, its teams (built on first use), and the record.
    Used by run_lab.py (terminal) and app.py (web)."""

    def __init__(self, thread: str, run_root: str, camera: str = "sim-room", director_model: str = "claude-cli:claude-sonnet-5-5@low",
                 team_model: str = "claude-cli:claude-haiku-5-5@low", allow_human: bool = True, seed: int = 1,
                 limits: dict | None = None, emit: Callable | None = None, director_backend=None, team_backend=None):
        self.thread = thread
        self.dir = os.path.join(run_root, thread)
        os.makedirs(self.dir, exist_ok=True)
        self.camera, self.seed, self.limits = camera, seed, limits
        from claude_cli import usage_snapshot
        self.events, self.usage_before = [], {}
        old = os.path.join(self.dir, "transcript.json")
        if os.path.exists(old):                       # a run resumed in a new process keeps its record and token count
            rec = json.load(open(old, encoding="utf-8"))
            self.events, self.usage_before = rec.get("events", []), rec.get("usage", {})
        self.usage_base = usage_snapshot()            # the process meter counts every run; this run adds the difference
        self.lock = threading.Lock()
        self.user_emit = emit
        self.teams: dict[str, Team] = {}
        self.team_backend = team_backend or make_backend(team_model)
        self.conn = sqlite3.connect(os.path.join(self.dir, "director.sqlite"), check_same_thread=False)
        self.graph = build_lab(director_backend or make_backend(director_model), self.get_team, allow_human=allow_human,
                               checkpointer=SqliteSaver(self.conn), emit=self.emit,
                               camera_label="Hikrobot MV-CS028-10UM, real hardware" if camera == "hikrobot"
                               else f"simulated camera ({camera})")
        self.config = {"configurable": {"thread_id": thread}, "recursion_limit": 300}
        meta = {"thread": thread, "camera": camera, "director_model": director_model, "team_model": team_model,
                "allow_human": allow_human, "seed": seed}
        json.dump(meta, open(os.path.join(self.dir, "run.json"), "w", encoding="utf-8"), indent=1)

    def emit(self, source, turn):
        with self.lock:
            self.events.append({"source": source, **turn})
        if self.user_emit:
            self.user_emit(source, turn)

    def get_team(self, name) -> Team:
        if name not in self.teams:
            self.teams[name] = Team(name, self.dir, self.team_backend, camera=self.camera, seed=self.seed,
                                    limits=self.limits, emit=self.emit)
        return self.teams[name]

    def start(self, request: str):
        self.emit("user", {"role": "user", "content": request, "meta": {}})
        self.graph.invoke({"dialogue": [{"role": "user", "content": request, "meta": {}}]}, self.config)
        self.save()

    def resume(self, reply: str):
        self.emit("user", {"role": "human", "content": reply, "meta": {"reply": True}})
        self.graph.invoke(Command(resume=reply), self.config)
        self.save()

    def retry(self):
        """Run on from the last checkpoint, after an error stopped the run (e.g. a model call that timed out)."""
        self.graph.invoke(None, self.config)
        self.save()

    def pending(self):
        """The question the run is waiting on (a dict), or None."""
        intr = pending_interrupts(self.graph, self.config)
        for i in intr:
            if i.value.get("kind") != "replay":
                return i.value
        return None

    def dialogue(self):
        return self.graph.get_state(self.config).values.get("dialogue", [])

    def usage(self) -> dict:
        """Model calls and tokens of this run (all its processes): calls, input, output, cache_read, cache_write,
        cost_usd (the CLI's estimate at API prices) and by_model."""
        from claude_cli import usage_snapshot
        now, base, before = usage_snapshot(), self.usage_base, self.usage_before
        out = {k: before.get(k, 0) + now[k] - base[k] for k in ("calls", "input", "output", "cache_read", "cache_write",
                                                                   "cost_usd")}
        out["by_model"] = {}
        for m in set(now["by_model"]) | set(before.get("by_model", {})):
            n, b, o = now["by_model"].get(m, {}), base["by_model"].get(m, {}), before.get("by_model", {}).get(m, {})
            c = {k: o.get(k, 0) + n.get(k, 0) - b.get(k, 0) for k in ("calls", "output")}
            if c["calls"]:
                out["by_model"][m] = c
        return out

    def save(self):
        with self.lock:
            ev = list(self.events)
        json.dump({"thread": self.thread, "dialogue": self.dialogue(), "events": ev, "usage": self.usage()},
                  open(os.path.join(self.dir, "transcript.json"), "w", encoding="utf-8"), indent=1, default=str)

    def close(self):
        for t in self.teams.values():
            t.close()
        self.conn.close()
