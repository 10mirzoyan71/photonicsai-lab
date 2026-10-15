"""MCP client for the agent: starts the camera server (stdio), keeps ONE session open for the whole run (so the camera
keeps its settings between calls), and exposes the server's tools as LangChain tools with synchronous calls.

The MCP session lives in an asyncio loop on a background thread; the graph (synchronous) calls tools through it.
Tool names, descriptions and argument schemas come from the server (list_tools), not from this file.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "..", "mcp_server", "camera_server.py")


class MCPBridge:
    def __init__(self, out_dir: str, kind: str = "sim", seed: int = 1, limits: dict | None = None, timeout: float = 900):
        env = {**os.environ, "CAMERA_OUT": os.path.abspath(out_dir), "CAMERA_KIND": kind, "CAMERA_SEED": str(seed),
               "CAMERA_LIMITS": json.dumps(limits or {}), "PYTHONIOENCODING": "utf-8"}
        self.params = StdioServerParameters(command=sys.executable, args=[SERVER], env=env)
        self.timeout = timeout
        self.log: list[dict] = []                       # every call made through the bridge (tool, args, result)
        # the server is a subprocess: on Windows that needs the proactor loop, whatever loop policy the host app set
        self.loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        self._ready, self._stop = threading.Event(), None
        self._error = None
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(), self.loop)
        if not self._ready.wait(timeout) or self._error:
            raise RuntimeError(f"camera MCP server did not start: {self._error}")

    async def _main(self):
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self.params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.tool_defs = (await session.list_tools()).tools
                    self._ready.set()
                    await self._stop.wait()             # the session stays open until close()
        except Exception as e:                          # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            self._ready.set()

    # -- calls ---------------------------------------------------------------------------
    def call(self, name: str, args: dict | None = None) -> dict:
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, args or {}), self.loop)
        res = fut.result(self.timeout)
        if res.structuredContent is not None:
            out = res.structuredContent
            out = out.get("result", out) if isinstance(out, dict) and set(out) == {"result"} else out
        else:
            text = "".join(getattr(c, "text", "") for c in res.content)
            try:
                out = json.loads(text)
            except json.JSONDecodeError:
                out = {"ok": not res.isError, "text": text}
        if res.isError and isinstance(out, dict):
            out.setdefault("ok", False)
        self.log.append({"tool": name, "args": args or {}, "result": out})
        return out

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_defs]

    def langchain_tools(self, names: list[str] | None = None) -> list[StructuredTool]:
        """The server's tools as LangChain tools (schema from the server; calls go through this session)."""
        out = []
        for t in self.tool_defs:
            if names and t.name not in names:
                continue
            out.append(StructuredTool(name=t.name, description=t.description or "", args_schema=t.inputSchema,
                                      func=lambda _n=t.name, **kw: json.dumps(self.call(_n, kw), default=str)))
        return out

    def close(self):
        if self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
            try:
                self._main_future.result(30)
            except Exception:                           # noqa: BLE001
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
