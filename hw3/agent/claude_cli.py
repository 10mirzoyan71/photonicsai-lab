"""Model calls through the Claude Code CLI in print mode (`claude -p`), for running the agents without an API key.

Select it with  --model claude-cli:<model>  (e.g. claude-cli:claude-sonnet-5-5).  The CLI uses its own login (a Claude
subscription).  Every built-in tool of the CLI is switched off (--tools ""), so the CLI is only the language model; it
runs in an empty folder with no settings, MCP servers or skills loaded.  Each call is independent: the system prompt
and the full prompt are sent every time.

Structured outputs (the orchestrator's decision) use --json-schema.  The executor's tools are offered the same way:
each step returns {"tool_calls": [...], "reply": "..."}; the tools named there are run here, by the project's own
tool objects, and their results are sent back in the next call, until a step returns no tool call.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading

# Token meter shared by every model call in this process (all teams of the lab assistant): the Claude subscription has
# usage limits, so each run reports what it used. Keys: calls, input, output, cache_read, cache_write, cost_usd,
# and per-model counts in by_model.
USAGE = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost_usd": 0.0, "by_model": {}}
_USAGE_LOCK = threading.Lock()


def _count(model: str, out: dict):
    u = out.get("usage") or {}
    with _USAGE_LOCK:
        USAGE["calls"] += 1
        USAGE["input"] += int(u.get("input_tokens") or 0)
        USAGE["output"] += int(u.get("output_tokens") or 0)
        USAGE["cache_read"] += int(u.get("cache_read_input_tokens") or 0)
        USAGE["cache_write"] += int(u.get("cache_creation_input_tokens") or 0)
        USAGE["cost_usd"] += float(out.get("total_cost_usd") or 0.0)
        m = USAGE["by_model"].setdefault(model, {"calls": 0, "output": 0})
        m["calls"] += 1
        m["output"] += int(u.get("output_tokens") or 0)


def usage_snapshot() -> dict:
    with _USAGE_LOCK:
        return {**USAGE, "by_model": {k: dict(v) for k, v in USAGE["by_model"].items()}}


def find_cli() -> str:
    exe = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if exe:
        return exe
    hits = glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                                  "Anthropic.ClaudeCode*", "claude.exe"))
    if hits:
        return hits[0]
    raise RuntimeError("Claude Code CLI not found: install it, or set CLAUDE_CLI to the path of claude(.exe)")


def tool_schema(t) -> dict:
    """Argument schema of a LangChain tool: a pydantic model, or a JSON schema dict (tools that come from MCP)."""
    s = t.args_schema
    return s if isinstance(s, dict) else t.tool_call_schema.model_json_schema()


TOOL_PROTOCOL = """

TOOLS. You call tools through your structured output:
  {{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}], "reply": ""}}  -> the tools run in that order and the next
      message shows what they returned;
  {{"tool_calls": [], "reply": "<your reply>"}}  -> ends your turn with that reply (no tool runs).
These tools are NOT tools of this session: calling one directly fails ("No such tool available"). The only way to run
one is to list it in tool_calls of your structured output.
The tools and their arguments (JSON schema):
{tools}"""


class ClaudeCLI:
    def __init__(self, model: str, timeout: int = 1200, retries: int = 3):
        self.exe = find_cli()
        # "claude-haiku-5-5@low": the part after @ is the CLI's --effort (low thinking = fewer output tokens)
        self.model, _, self.effort = model.partition("@")
        self.timeout = timeout
        self.retries = retries
        self.cwd = os.path.join(tempfile.gettempdir(), "claude_cli_agent")     # empty folder: no project files
        os.makedirs(self.cwd, exist_ok=True)
        self.n_calls = 0

    def call(self, system: str, prompt: str, schema: dict | None = None) -> dict:
        """One model call; returns the CLI's JSON result (fields `result`, and `structured_output` with a schema)."""
        cmd = [self.exe, "-p", "--output-format", "json", "--model", self.model, "--tools", "",
               "--system-prompt", system, "--no-session-persistence", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands"]
        if self.effort:
            cmd += ["--effort", self.effort]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]
        err = ""
        for _ in range(self.retries):
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", cwd=self.cwd, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                err = f"timeout after {self.timeout} s"
                continue
            try:
                out = json.loads(p.stdout)
            except json.JSONDecodeError:
                err = f"exit {p.returncode}: {(p.stdout + p.stderr).strip()[-1500:]}"
                continue
            if out.get("is_error"):
                err = f"{out.get('subtype')}: {str(out.get('result'))[-1500:]}"
                continue
            if schema is not None and not isinstance(out.get("structured_output"), dict):
                err = "no structured output in the reply"
                continue
            self.n_calls += 1
            _count(self.model, out)
            return out
        raise RuntimeError(f"claude -p failed {self.retries} times: {err}")

    def text(self, system: str, prompt: str) -> str:
        return self.call(system, prompt)["result"]

    def structured(self, system: str, prompt: str, schema: dict) -> dict:
        return self.call(system, prompt, schema)["structured_output"]

    def tool_loop(self, system: str, prompt: str, lc_tools: list, max_iters: int, budget_note: str) -> str:
        """Executor turn with tools (LangChain StructuredTools: name, description, tool_call_schema, invoke)."""
        by_name = {t.name: t for t in lc_tools}
        listing = "\n".join(f"- {t.name}: {t.description}\n  {json.dumps(tool_schema(t))}" for t in lc_tools)
        sys_t = system + TOOL_PROTOCOL.format(tools=listing)
        step_schema = {
            "type": "object",
            "properties": {
                "tool_calls": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": list(by_name)}, "args": {"type": "object"}},
                    "required": ["name", "args"]}},
                "reply": {"type": "string", "description": "your reply; used only when tool_calls is empty"}},
            "required": ["tool_calls", "reply"]}
        history, slips = [], 0
        for _ in range(max_iters):
            step = self.structured(sys_t, self._with_history(prompt, history), step_schema)
            calls = step.get("tool_calls") or []
            if not calls and "No such tool" in step.get("reply", "") and slips < 2:
                slips += 1                                   # the model tried a direct call: remind it, try again
                history.append(("protocol_note", {}, "A direct tool call cannot work in this session. To run a "
                                "tool, put it in tool_calls of your structured output, e.g. "
                                '{"tool_calls": [{"name": "' + next(iter(by_name)) + '", "args": {...}}], "reply": ""}.'))
                continue
            if not calls:
                return step.get("reply", "")
            for c in calls:
                tool = by_name.get(c.get("name"))
                try:
                    res = tool.invoke(c.get("args") or {}) if tool else json.dumps({"ok": False, "error": "unknown tool"})
                except Exception as e:                       # invalid arguments: report them to the model
                    res = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
                history.append((c.get("name"), c.get("args"), res))
        return self.text(system, self._with_history(prompt, history) + "\n\n" + budget_note)

    @staticmethod
    def _with_history(prompt: str, history: list) -> str:
        if not history:
            return prompt
        parts = [prompt, "\n\n=== YOUR TOOL CALLS IN THIS TURN SO FAR, AND WHAT THEY RETURNED ==="]
        for i, (name, args, res) in enumerate(history, 1):
            parts.append(f"--- call {i}: {name} {json.dumps(args)}\n--- returned:\n{res}")
        parts.append("\nContinue: call the next tool, or end the turn with tool_calls = [] and your reply.")
        return "\n".join(parts)
