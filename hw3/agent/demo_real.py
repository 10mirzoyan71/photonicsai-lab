"""Demo run for the video: the agent drives the real camera (Hikrobot MV-CS028-10UM) through MCP.
    python demo_real.py [--thread real_uc1_autoexposure] [--request "..."] [--human]
Opens the live view, waits for Enter (start the screen recording first), clears the screen and runs run_hw3.py."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", ".venv", "Scripts", "python.exe")
PYW = os.path.join(HERE, "..", ".venv", "Scripts", "pythonw.exe")

p = argparse.ArgumentParser()
p.add_argument("--thread", default="real_uc1_autoexposure")
p.add_argument("--request", default="Set the camera so that the white surface uses most of the camera's range without "
                                    "saturating, and report the exposure and gain you chose.")
p.add_argument("--human", action="store_true")
p.add_argument("--max-decisions", default="10")
a = p.parse_args()

env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
subprocess.Popen([PYW, "live_view.py", "--thread", a.thread], cwd=HERE, env=env)
input("Start the screen recording (Win+Alt+R), then press Enter ")
os.system("cls" if os.name == "nt" else "clear")
cmd = [PY, "run_hw3.py", "--thread", a.thread, "--camera", "hikrobot", "--model", "claude-cli:claude-sonnet-5-5",
       "--max-decisions", a.max_decisions, "--request", a.request] + (["--human"] if a.human else [])
sys.exit(subprocess.call(cmd, cwd=HERE, env=env))
