# Task 6 — Agentic system (LangGraph)

Four nodes in a `StateGraph` whose state is the dialogue (`{role, content, meta}` turns, appended with `operator.add`):

* **orchestrator** — structured model call `Decision(next_role, instruction, reason)`, next_role in
  {scientist, general, executor, finish}; no tools, no theory, no code; at most 8 decisions; `finish` only after an
  executor turn holds a `TEST_RMSE` printed by a script that ran (checked in code from the real stdout).
* **scientist** — differential equation, network, state equation, training, free-run test, numbered steps. No tool.
* **general** — high-level side questions (what a record/term/error is). No tool, no plan.
* **executor** — only role with `coding_tool(action, filename, code)`: `write` a new .py file into the run's working
  directory, `run` it with a 300 s limit -> exit code, stdout, stderr. One write + one run per turn; repairs go to new files.

Files: `agents.py`, `coding_tool.py`, `run_task5.py`, `render_transcript.py`, `task5_statement.md`,
`tests/offline_smoke_test.py` (scripted stand-in for the model; tests the wiring without an API key).
