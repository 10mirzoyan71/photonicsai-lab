# PhotonicsAI Lab Assistant — Homeworks 4 and 5

Erik Mirzoyan. Course: PhotonicsAI — AI-Driven Science and Autonomous Laboratories.

Homeworks 1–3 built three agent systems: physics-informed modelling of the Silverbox oscillator (HW1), human-in-the-loop
focusing of light through a scattering medium on a simulated bench (HW2), and an agent that drives a real lab camera
through MCP (HW3). This project puts them under one roof as a **laboratory with a director and teams**, and gives it a
web interface: a person asks the lab a question in plain words, approves steps on the real camera with a button, does
what needs hands (blocking the light), and watches the camera and the results while the agents work.

## The hierarchy
```
Level 1  DIRECTOR (Claude Sonnet)        the person talks only to it; splits the request into team tasks, passes
                                          VERIFIED numbers from one team to the next, asks the person before a task
                                          changes the real camera, writes the final report from VERIFIED records only
Level 2  TEAMS (each the HW1-3 graph)    instrument  - the lab camera (HW3) behind its MCP server
                                          optics      - the simulated wavefront-shaping bench and focusing tool (HW2)
                                          modeling    - the Silverbox record, identified equation and kept network (HW1)
Level 3  SPECIALISTS and TOOLS            orchestrator, scientist, general agent, executor (Claude Haiku); MCP camera
                                          tools, focusing tool, coding tool
```
Each team runs on its own checkpointed thread, as an independent LangGraph graph. A question a team asks the person
(e.g. "block the light" inside a hardware turn) is passed up as an interrupt of the director, so the person answers
everything in one place. The director's decisions are structured outputs with a budget of 6; finishing needs a VERIFIED
team record, and the person is asked for approval at most once per run, only before an instrument-team task (the
director's prompt keeps it to tasks that change the real camera).

## Layout
```
lab/                    the lab assistant (Homeworks 4/5)
  lab_assistant.py      the director graph, the teams, LabRun (start / resume / retry / usage / save)
  app.py                the web app (Streamlit): chat, approval cards, camera panel, results, tokens, replay
  app_core.py           the app's engine without Streamlit: a worker thread per run, saved runs, frames, figures
  run_lab.py            the same in the terminal
  tests/                test_lab_offline.py, test_app_offline.py (scripted models, real MCP server, no tokens)
  runs/<thread>/        every run: run.json, transcript.json (events, dialogue, token usage), the teams' folders
                        (camera frames and calls.jsonl, bench figures, analysis scripts) and the checkpoints
hw3/                    Homework 3: camera connectors, MCP camera server (13 tools), agent with the hardware turn, hw3.py
hw2_agentic_hardware/   Homework 2 (the bench and the focusing tool are used by the optics team)
task5_pinn_silverbox/   Homework 1 (the Silverbox code and the kept results used by the modeling team)
task6_agentic_system/   Homework 1, task 6 (the first agent system)
data/                   SNLS80mV.mat (not in the repository: download it from the Silverbox benchmark)
```

## Setup
Python 3.11–3.13. On Windows, in the PyCharm terminal or PowerShell:
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
* **Models:** the Claude Code CLI logged in with a Claude subscription (no API key). The director uses
  `claude-cli:claude-sonnet-5-5@low`, the teams `claude-cli:claude-haiku-5-5@low` (`@low` = low thinking effort).
  Any LangChain chat model also works, e.g. `anthropic:claude-sonnet-5-5` with `ANTHROPIC_API_KEY`.
* **Real camera:** Hikrobot MVS with its SDK; plug the camera into a USB 3 port and close the MVS client. Check it with
  `cd hw3` and `python hw3.py check-camera`.

## Use it
```
python -m streamlit run lab/app.py
```
Open the address it prints (http://localhost:8501; Ctrl+click it in the PyCharm terminal). In the side bar choose the camera (simulated room light, simulated laser speckle, or the
real Hikrobot), whether the director may ask you, and earlier runs. Ask a question in the chat or click an example.
* **Approval card** (✅ Approve / ❌ Refuse with an optional reason) when the director wants to change the real camera.
* **Action card** (✅ Done, or an answer in words) when a team needs your hands, e.g. to block the light.
* **Camera panel**: the newest frame the instrument team captured, with its exposure, gain and levels, and the last
  MCP call. **Verified results**: each team's VERIFIED record and the figures it wrote. **Tokens**: model calls,
  tokens and the CLI's cost estimate for this run.
* **Replay** mode steps through a saved run (buttons or auto-play) with the camera frames in sync: the fallback for a
  demo without the camera or without model calls. A paused run (also one started in the terminal) can be continued in
  the app; after an error, *Retry* runs on from the last checkpoint.

The same in the terminal:
```
python lab/run_lab.py --thread t1 --camera hikrobot --request "Measure the camera's noise model and predict the focusing enhancement with it."
python lab/run_lab.py --thread t1 --reply "yes"        # resume a paused run, also from a new terminal
```

## Tests (no model calls, no camera)
```
python lab/tests/test_lab_offline.py    # hierarchy: approval, a team's question passed up, restart between replies
python lab/tests/test_app_offline.py    # web app: worker thread, start page, finished run, replay, approval card
cd hw3 && python hw3.py test            # Homework 3: simulated camera, MCP server, agent graph
```

## Token budget
A Claude subscription has a 5-hour and a weekly usage limit, shared with every other use. The lab keeps a run small:
Haiku for the teams, low thinking effort, short prompts and reports, a budget of decisions per level, at most one
approval stop. The first run (find an exposure at half range, simulated camera) took 24 model calls (19 Haiku, 5 Sonnet)
and about 33 000 output tokens; the CLI estimated $2.14 at API prices (with the subscription nothing is charged, the usage
limit is used).
