"""PhotonicsAI Lab Assistant - the web app of Homework 4/5.

    python -m streamlit run lab/app.py

The person talks to the director in the chat. Questions from the director (approve a step on the real camera) and from
the teams ("block the light") appear as cards with buttons. The camera panel shows the newest frame the instrument
team captured, the results panel the teams' VERIFIED records and figures, and the token panel what the run used.
The side bar picks the camera (simulated or the real Hikrobot), opens earlier runs, and replays a finished run step by
step (the fallback for a demo without the camera or without model calls).
"""
import os
import sys
import time

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app_core as core  # noqa: E402

st.set_page_config(page_title="PhotonicsAI Lab Assistant", page_icon="🔬", layout="wide")
# Keep the scroll bars' room reserved: otherwise, when the page is about one screen high, the bar appearing narrows the
# columns, the camera image gets shorter, the bar disappears, the columns widen again... and the page shakes sideways.
st.markdown("<style>[data-testid='stMain'] {overflow-y: scroll !important; scrollbar-gutter: stable;} "
            "[data-testid='stVerticalBlock'] {scrollbar-gutter: stable;}</style>", unsafe_allow_html=True)

CAMERAS = {"sim-room": "Simulated camera, room light", "sim": "Simulated camera, laser speckle",
           "hikrobot": "REAL camera: Hikrobot MV-CS028-10UM"}
TEAM_ICON = {"instrument": "📷", "optics": "🔦", "modeling": "📈"}
ROLE_LABEL = {"orchestrator": "🎯 orchestrator", "scientist": "🧪 scientist", "general": "🧠 general agent",
              "executor": "⚙️ executor", "instrument": "📷 camera (MCP)", "human": "🙋 person"}
EXAMPLES = [
    "Find an exposure at which the camera image uses about half of its range.",
    "Measure the camera's noise model (conversion gain, read noise, full well) and predict the focusing enhancement "
    "the optics bench would reach with this camera.",
    "How well does the kept Silverbox network predict the test record? Report its free-run RMSE.",
]
HIERARCHY = """digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f8", fontname="Helvetica", fontsize=11];
  person [label="Person\\n(chat, buttons)", fillcolor="#fde9c9"];
  director [label="Director\\nClaude Sonnet", fillcolor="#d9e8fb"];
  instrument [label="Instrument team\\n(HW3 camera)"]; optics [label="Optics team\\n(HW2 bench)"];
  modeling [label="Modeling team\\n(HW1 Silverbox)"];
  spec [label="orchestrator · scientist\\ngeneral agent · executor\\n(Claude Haiku)", fillcolor="#e8f5e9"];
  tools [label="MCP camera server · focusing tool\\ncoding tool", fillcolor="#eeeeee"];
  person -> director [dir=both]; director -> instrument; director -> optics; director -> modeling;
  instrument -> spec; optics -> spec; modeling -> spec; spec -> tools;
}"""

# Shown in the top right corner while the lab works (instead of Streamlit's running figure, which the toolbar setting in
# .streamlit/config.toml hides): a laser wave packet travelling to the right.
PHOTON_BADGE = """<div class="photon-badge"><svg width="120" height="28" viewBox="0 0 120 28"><defs>
<linearGradient id="pfade" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/>
<stop offset="0.5" stop-color="#fff" stop-opacity="1"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<mask id="ppacket"><rect width="120" height="28" fill="url(#pfade)"/></mask></defs>
<g mask="url(#ppacket)"><path class="pwave" d="M-40 14 Q-30 2 -20 14 T0 14 T20 14 T40 14 T60 14 T80 14 T100 14 T120 14 T140 14"
fill="none" stroke="#ff4b4b" stroke-width="2.5"/></g></svg><span>the lab is working</span></div>
<style>[data-testid="stHeader"]{background:transparent!important}
.photon-badge{position:fixed;top:12px;right:28px;z-index:999999;display:flex;align-items:center;gap:6px;
padding:3px 12px 3px 6px;border-radius:16px;background:rgba(18,18,28,.88);color:#fff;font-size:13px;
box-shadow:0 0 14px rgba(255,75,75,.45)}
.photon-badge .pwave{animation:ptravel .45s linear infinite;filter:drop-shadow(0 0 3px #ff4b4b)}
@keyframes ptravel{from{transform:translateX(0)}to{transform:translateX(40px)}}</style>"""


@st.cache_resource
def workers() -> dict:
    """The live runs of this server, by thread (they outlive a page reload)."""
    return {}


def md(text: str) -> str:
    """Agent text as Markdown: keep line breaks, and do not read '$' as the start of a formula."""
    return (text or "").replace("$", "\\$").replace("\n", "  \n")


def free_camera(ws: dict) -> bool:
    """Close the idle runs (they hold the camera); False if a run is still working."""
    if any(w.busy for w in ws.values()):
        st.session_state.flash = "Another run is still working; wait until it pauses or finishes."
        return False
    for t in list(ws):
        ws.pop(t).close()
    return True


# -- callbacks (they run before the page is drawn again) ---------------------------------------------------------------
def start_run(request: str):
    request = (request or "").strip()
    ws = workers()
    if not request or not free_camera(ws):
        return
    thread = "run-" + time.strftime("%Y%m%d-%H%M%S")
    w = core.Worker(thread, camera=st.session_state.camera, allow_human=st.session_state.allow_human,
                    director_model=st.session_state.dm, team_model=st.session_state.tm)
    w.send("start", request)
    ws[thread] = w
    st.session_state.current, st.session_state.mode = thread, "Live"


def continue_run(thread: str):
    ws = workers()
    if free_camera(ws):
        ws[thread] = core.Worker(thread, **core.run_config(os.path.join(core.RUNS, thread)))


def reply(thread: str, text: str):
    w = workers().get(thread)
    if w and not w.busy and text.strip():
        w.send("resume", text.strip())


def retry(thread: str):
    w = workers().get(thread)
    if w and not w.busy:
        w.send("retry")


def move(key: str, to: int):
    st.session_state[key] = to


# -- drawing -------------------------------------------------------------------------------------------------------------
def show_events(events: list, open_last: bool):
    i = 0
    while i < len(events):
        e = events[i]
        src, role, meta = e.get("source"), e.get("role"), e.get("meta", {})
        if src in core.TEAMS:                                    # a team at work: its internal turns in one box
            j = i
            while j < len(events) and events[j].get("source") == src:
                j += 1
            with st.expander(f"{TEAM_ICON[src]} **{src} team** at work: {j - i} steps",
                             expanded=open_last and j == len(events)):
                for t in events[i:j]:
                    text = t.get("content", "")
                    st.markdown(f"**{ROLE_LABEL.get(t.get('role'), t.get('role'))}**  \n"
                                + md(text[:1200] + (" …" if len(text) > 1200 else "")))
            i = j
            continue
        if src == "user":
            with st.chat_message("user"):
                st.markdown(("**Reply:** " if role == "human" else "") + md(e.get("content") or "(empty reply)"))
        elif src == "director" and role == "director":
            with st.chat_message("assistant", avatar="🧭"):
                if meta.get("stop") == "finished":
                    st.success("**Final report of the director**  \n" + md(e["content"]))
                else:
                    st.markdown("**Director**  \n" + md(e["content"]))
        elif src == "director" and role == "team":
            with st.chat_message("assistant", avatar=TEAM_ICON.get(meta.get("team"), "🔧")):
                text = e.get("content", "")
                st.markdown(f"**{meta.get('team')} team reports back**  \n" + md(text[:900] + (" …" if len(text) > 900 else "")))
                if meta.get("verified"):
                    with st.expander("VERIFIED record"):
                        st.json(meta["verified"])
                else:
                    st.warning("No VERIFIED record from this team.")
        i += 1


def pending_card(thread: str, q: dict):
    kind, src, question = q.get("kind"), q.get("source", "director"), q.get("question") or ""
    with st.container(border=True):
        if kind == "hard_decision":
            st.markdown(f"#### ✋ Approval needed ({src})\n" + md(question))
            step = q.get("proposed_step") or {}
            if step.get("task"):
                st.info(f"**Proposed step for the {step.get('next')} team:**  \n" + md(step["task"]))
            reason = st.text_input("Reason, sent with Refuse (optional)", key=f"reason_{thread}")
            c1, c2 = st.columns(2)
            c1.button("✅ Approve", type="primary", width="stretch", on_click=reply, args=(thread, "yes"))
            c2.button("❌ Refuse", width="stretch", on_click=reply,
                      args=(thread, "no" + (f", {reason}" if reason.strip() else "")))
        elif kind == "person_action":
            st.markdown(f"#### 🙋 The {src} asks you to do something\n" + md(question))
            st.button("✅ Done", type="primary", width="stretch", on_click=reply, args=(thread, "done"))
            other = st.text_input("Or answer in words (e.g. what you could not do)", key=f"other_{thread}")
            st.button("Send this answer", on_click=reply, args=(thread, other))
        else:
            st.markdown(f"#### ❓ Question ({src})\n" + md(question))
            ans = st.text_input("Your answer", key=f"answer_{thread}")
            st.button("Send", type="primary", on_click=reply, args=(thread, ans))


def last_capture(events: list) -> dict | None:
    for e in reversed(events):
        cap = e.get("meta", {}).get("capture") or {}
        if cap.get("ok"):
            return cap
    return None


def side_panel(run_dir: str | None, events: list, usage: dict, replay: bool):
    st.subheader("📷 Camera")
    path = (core.frame_at(run_dir, events) if replay else core.newest_frame(run_dir)) if run_dir else None
    cap = last_capture(events)
    if path:
        img, info = core.frame_image(path)
        cap_text = os.path.basename(path)
        if cap:
            s = cap.get("settings", {})
            cap_text += f" · {s.get('exposure_us')} µs, {s.get('gain_db')} dB"
        if info:
            cap_text += f" · mean {info['mean_dn']:.0f} DN, max {info['max_dn']} DN"
        st.image(img, caption=cap_text, width="stretch")
    else:
        st.caption("No frame yet. This panel shows the newest frame the instrument team captured.")
    call = core.last_call(run_dir) if run_dir and not replay else None
    if call:
        res = call.get("result", {})
        st.caption(f"Last MCP call: `{call.get('tool')}` {str(call.get('args', {}))[:80]} → "
                   + ("ok" if res.get("ok") else "refused"))

    st.subheader("✅ Verified results")
    recs = core.records(events)
    if not recs:
        st.caption("None yet. A team's result counts only when its executor's numbers are VERIFIED.")
    for team, rec in recs:
        with st.expander(f"{TEAM_ICON.get(team, '')} {team}: {rec.get('tool')}"):
            st.json(rec)
    figs = core.figures(run_dir)[:3] if run_dir else []
    for f in figs:
        st.image(f, caption=os.path.relpath(f, run_dir), width="stretch")

    st.subheader("🔢 Tokens of this run")
    if usage.get("calls"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Model calls", usage["calls"])
        c2.metric("Output tokens", f"{usage['output']:,}")
        c3.metric("≈ cost at API prices", f"${usage['cost_usd']:.2f}")
        st.caption(" · ".join(f"{m.split(':')[-1]}: {v['calls']} calls, {v['output']:,} out"
                              for m, v in usage.get("by_model", {}).items())
                   + ". With a Claude subscription nothing is charged; the plan's usage limit is used.")
    else:
        st.caption("No model calls yet.")


# -- page ----------------------------------------------------------------------------------------------------------------
ws = workers()
runs = core.list_runs()
known = {r["thread"]: r for r in runs}
options = [""] + [t for t in ws if t not in known] + [r["thread"] for r in runs]
# the selected run lives in session state, not in the widget: the list changes as runs start and finish, and a
# changed list makes Streamlit draw the select box as a new widget, which then starts from this index
current = st.session_state.get("current", "")
if current not in options:
    current = ""

with st.sidebar:
    st.title("🔬 PhotonicsAI Lab")
    st.caption("Director → teams (instrument · optics · modeling) → specialists and tools")
    st.radio("Mode", ["Live", "Replay"], horizontal=True, key="mode")
    st.selectbox("Camera for a new run", list(CAMERAS), format_func=CAMERAS.get, key="camera")
    st.toggle("The director may ask me", value=True, key="allow_human",
              help="On: the director asks before a task changes the real camera. Off: it decides alone "
                   "(a team still asks when it needs your hands, e.g. to block the light).")
    with st.expander("Models"):
        st.text_input("Director", "claude-cli:claude-sonnet-5-5@low", key="dm")
        st.text_input("Teams", "claude-cli:claude-haiku-5-5@low", key="tm")
    st.divider()

    def label(t):
        if not t:
            return "➕ New run"
        if t in ws and ws[t].busy:
            icon = "⏳"
        else:
            icon = {"finished": "✅", "paused": "⏸️"}.get(known.get(t, {}).get("status"), "•")
        req = known.get(t, {}).get("request", "")
        return f"{icon} {t}" + (f" · {req[:38]}…" if req else "")
    picked = st.selectbox("Runs", options, index=options.index(current), format_func=label)
    st.session_state.current = current = picked

thread = current
run_dir = os.path.join(core.RUNS, thread) if thread else None
w = ws.get(thread) if thread else None
replay = st.session_state.mode == "Replay"
if st.session_state.get("flash"):
    st.warning(st.session_state.pop("flash"))

left, right = st.columns([3, 2], gap="large")
refresh = False
with left:
    if not thread:
        st.header("PhotonicsAI Lab Assistant")
        st.markdown("Ask the lab in plain words. The **director** splits the request into tasks for three **teams**, "
                    "each with its own specialists and tools, and reports only numbers a team **verified**. "
                    "You approve steps on the real camera and do what needs your hands.")
        st.graphviz_chart(HIERARCHY)
        st.markdown("**Try one:**")
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=start_run, args=(ex,), disabled=replay)
        events, usage = [], {}
    elif replay:
        tr = core.load_transcript(run_dir)
        all_ev = tr.get("events", [])
        usage = tr.get("usage", {})
        st.header(f"Replay · {thread}")
        if not all_ev:
            st.info("This run has no saved events.")
            events = []
        else:
            key = f"pos_{thread}"
            st.session_state.setdefault(key, 1)
            pos = min(st.session_state[key], len(all_ev))
            c = st.columns([1, 1, 1, 1, 3])
            c[0].button("⏮", on_click=move, args=(key, 1), help="start")
            c[1].button("◀", on_click=move, args=(key, max(1, pos - 1)), help="back")
            c[2].button("▶", on_click=move, args=(key, min(len(all_ev), pos + 1)), help="next")
            c[3].button("⏭", on_click=move, args=(key, len(all_ev)), help="end")
            play = c[4].toggle("Auto-play", key=f"play_{thread}")
            st.progress(pos / len(all_ev), text=f"step {pos} of {len(all_ev)}")
            events = all_ev[:pos]
            with st.container(height=560):
                show_events(events, open_last=True)
            if play and pos < len(all_ev):
                st.session_state[key] = pos + 1
                refresh = True
    else:
        st.header(f"Run · {thread}")
        if w is None:
            tr = core.load_transcript(run_dir)
            events, usage = tr.get("events", []), tr.get("usage", {})
            with st.container(height=560):
                show_events(events, open_last=False)
            if known.get(thread, {}).get("status") == "paused":
                st.info("This run is paused (started earlier or in the terminal).")
                st.button("▶ Continue this run here", type="primary", on_click=continue_run, args=(thread,))
        else:
            events, usage = w.events(), w.usage()
            with st.container(height=480 if (w.pending and not w.busy) else 560):
                show_events(events, open_last=w.busy)
            if w.error:
                st.error(f"The run stopped with an error: {w.error}")
                st.button("🔁 Retry from the last checkpoint", on_click=retry, args=(thread,))
            elif w.busy:
                last = events[-1] if events else {}
                who = last.get("source", "director")
                st.info(f"⏳ Working… ({who}{' team' if who in core.TEAMS else ''}; model calls take a few seconds "
                        "each, a team task a few minutes)")
                refresh = True
            elif w.pending:
                pending_card(thread, w.pending)
            elif w.finished or any(e.get("meta", {}).get("stop") for e in events):
                st.success("Run finished. Ask a new question below to start a new run.")
            elif events:                                     # stopped half-way (e.g. the app was closed during a run)
                st.warning("This run stopped before it finished.")
                st.button("▶ Run on from the last checkpoint", type="primary", on_click=retry, args=(thread,))
    if not replay:
        busy_or_waiting = bool(w and (w.busy or w.pending))
        st.chat_input("Ask the lab…" if not busy_or_waiting else "Answer the question above first",
                      key="request", disabled=busy_or_waiting,
                      on_submit=lambda: start_run(st.session_state.get("request", "")))

with right:
    side_panel(run_dir, events, usage, replay)

if refresh:                                            # while the lab works: a light wave packet in the corner
    st.markdown(PHOTON_BADGE, unsafe_allow_html=True)
    time.sleep(1.0 if not replay else 1.2)
    st.rerun()
