"""One entry point for Homework 3 (run from this folder, with the project's Python):

    python hw3.py test                 # all offline tests: simulated camera, MCP server, agent graph (no model, no camera)
    python hw3.py check-camera         # the MVS SDK, then the plugged-in Hikrobot camera (close the MVS client first)
    python hw3.py tools                # the camera MCP server's tools, as an MCP client sees them
    python hw3.py run --thread t1 --camera hikrobot --request "..."   # a run of the agent (options of agent/run_hw3.py)
    python hw3.py live [--thread t1]   # live window with the newest frame the agent captured
    python hw3.py demo [--thread t1]   # live window + a run on the real camera, for a screen recording
    python hw3.py mcp-config           # the entry that adds the camera server to an MCP client (e.g. Claude Desktop)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TESTS = ["test_camera.py", "test_mcp_server.py", "test_agent_offline.py"]


def sh(args, cwd=HERE):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.call([PY, *args], cwd=cwd, env=env)


def test():
    results = []
    for t in TESTS:
        print(f"\n===== {t} =====", flush=True)
        t0 = time.time()
        results.append((t, sh([os.path.join("tests", t)]), time.time() - t0))
    print("\n===== summary =====")
    for t, code, dt in results:
        print(f"{'PASSED' if code == 0 else 'FAILED'}  {t}  ({dt:.0f} s)")
    return 0 if all(code == 0 for _, code, _ in results) else 1


def tools():
    sys.path.insert(0, os.path.join(HERE, "agent"))
    import tempfile
    from mcp_bridge import MCPBridge
    br = MCPBridge(tempfile.mkdtemp(prefix="hw3_tools_"))
    try:
        for t in br.tool_defs:
            args = ", ".join(t.inputSchema.get("properties", {}))
            print(f"\n{t.name}({args})\n    " + " ".join((t.description or "").split()))
        print(f"\n{len(br.tool_defs)} tools (server: mcp_server/camera_server.py, transport: stdio)")
    finally:
        br.close()
    return 0


def mcp_config():
    entry = {"mcpServers": {"lab-camera": {
        "command": PY, "args": [os.path.join(HERE, "mcp_server", "camera_server.py")],
        "env": {"CAMERA_KIND": "hikrobot", "CAMERA_OUT": os.path.join(HERE, "camera_out"),
                "CAMERA_LIMITS": json.dumps({"max_exposure_us": 500000, "max_gain_db": 24, "max_frames": 16})}}}}
    print("Add this to the MCP client's configuration (CAMERA_KIND sim or sim-room for the simulated camera):\n")
    print(json.dumps(entry, indent=2))
    return 0


def main():
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("help", [])
    agent = os.path.join(HERE, "agent")
    if cmd == "test":
        return test()
    if cmd == "check-camera":
        return sh([os.path.join("tests", "check_mvs_sdk.py")]) or sh([os.path.join("tests", "check_real_camera.py"), *rest])
    if cmd == "tools":
        return tools()
    if cmd == "run":
        if "--model" not in rest:
            rest = ["--model", "claude-cli:claude-sonnet-5-5", *rest]
        return sh(["run_hw3.py", *rest], cwd=agent)
    if cmd == "live":
        return sh(["live_view.py", *rest], cwd=agent)
    if cmd == "demo":
        return sh(["demo_real.py", *rest], cwd=agent)
    if cmd == "mcp-config":
        return mcp_config()
    print(__doc__)
    return 0 if cmd in ("help", "-h", "--help") else 2


if __name__ == "__main__":
    sys.exit(main())
