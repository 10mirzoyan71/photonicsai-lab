"""Opens the camera the MCP server talks to: the simulated camera, or the lab camera through its SDK."""
from __future__ import annotations

from .base import Camera, CameraError


def open_camera(kind: str = "sim", **kwargs) -> Camera:
    if kind in ("sim", "sim-room"):          # simulated camera: laser speckle scene, or a white surface under room light
        from .sim_camera import SimCamera
        return SimCamera(**kwargs, scene="room" if kind == "sim-room" else "speckle")
    if kind in ("hikrobot", "vendor"):       # Hikrobot MV-CS028-10UM through the MVS SDK
        from .hikrobot_camera import HikrobotCamera
        return HikrobotCamera(**kwargs)
    raise CameraError(f"unknown camera kind {kind!r} (use 'sim', 'sim-room' or 'hikrobot')")
