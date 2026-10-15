"""The camera MCP server through a real MCP client session (stdio): python tests/test_mcp_server.py"""
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
from mcp_bridge import MCPBridge  # noqa: E402


def main():
    out = tempfile.mkdtemp(prefix="cam_mcp_")
    br = MCPBridge(out, seed=2, limits={"max_exposure_us": 100000, "max_frames": 8})
    try:
        names = br.tool_names()
        print("tools:", names)
        assert {"camera_info", "set_exposure", "set_gain", "set_roi", "reset_roi", "set_illumination", "capture",
                "analyze_frame"} <= set(names)
        info = br.call("camera_info")
        assert info["ok"] and info["max_dn"] == 4095 and info["lab_limits"]["max_exposure_us"] == 100000
        assert br.call("set_exposure", {"exposure_us": 250})["exposure_us"] == 250
        refused = br.call("set_exposure", {"exposure_us": 200000})
        assert refused["ok"] is False and "lab limit" in refused["error"], refused
        cap = br.call("capture", {"n_frames": 2, "label": "test", "regions": [{"name": "c", "x": 300, "y": 236, "width": 40, "height": 40}]})
        assert cap["ok"] and cap["settings"]["exposure_us"] == 250 and "temporal_noise" in cap and os.path.exists(cap["frame_file"])
        print("capture:", json.dumps({k: cap[k] for k in ("settings", "stats")})[:400])
        an = br.call("analyze_frame", {"frame_file": cap["frame_file"], "black_level_dn": 40,
                                       "region": {"name": "c", "x": 270, "y": 206, "width": 100, "height": 100}})
        assert an["ok"] and 0.8 < an["speckle"]["contrast"] < 1.2
        print("analyze:", an["speckle"])
        se = br.call("capture_series", {"n_frames": 4, "interval_s": 0.2, "label": "stab"})
        assert se["ok"] and len(se["series"]["mean_dn"]) == 4 and se["duration_s"] >= 0.55 and os.path.exists(se["series_file"])
        print("series:", se["summary"], f"{se['duration_s']} s")
        assert br.call("capture_series", {"n_frames": 100, "interval_s": 10})["ok"] is False      # above max_series_s
        lc = {t.name: t for t in br.langchain_tools()}
        r = json.loads(lc["set_gain"].invoke({"gain_db": 6}))
        assert r["ok"] and r["gain_db"] == 6.0
        assert json.loads(lc["camera_info"].invoke({}))["gain_db"] == 6.0          # state kept in the one session
        # added tools: temperature, ADC bit depth, black level, TIFF export
        t = br.call("read_temperature")
        assert t["ok"] and 30 < t["temperature_c"] < 50, t
        assert br.call("set_adc_bit_depth", {"bits": 8})["adc_bit_depth"] == 8
        br.call("set_illumination", {"level": 0})
        fr = np.load(br.call("capture", {"n_frames": 1, "label": "dark8"})["frame_file"])[0]
        step = 4095 / 255
        assert np.all(np.abs(np.unique(fr) / step - np.round(np.unique(fr) / step)) < 0.05), np.unique(fr)[:10]
        assert br.call("set_adc_bit_depth", {"bits": 10})["ok"] is False
        assert br.call("set_adc_bit_depth", {"bits": 12})["adc_bit_depth"] == 12
        assert br.call("set_black_level", {"value": 100})["black_level"] == 100
        dark = br.call("capture", {"n_frames": 2, "label": "dark_bl100"})
        assert dark["settings"]["black_level_setting"] == 100 and abs(dark["stats"]["mean_dn"] - 100) < 2, dark["stats"]
        tif = br.call("save_tiff", {"frame_file": os.path.basename(dark["frame_file"])})
        img = Image.open(tif["tiff_file"])
        assert tif["pages"] == 2 and img.n_frames == 2 and img.mode.startswith("I;16")
        assert np.array_equal(np.array(img), np.load(dark["frame_file"])[0])
        print(f"temperature {t['temperature_c']} C; 8-bit ADC levels {np.unique(fr)[:4]}...; dark level at black level "
              f"100: {dark['stats']['mean_dn']:.1f} DN; TIFF {os.path.basename(tif['tiff_file'])} ({tif['pages']} pages)")
        calls = [json.loads(line) for line in open(os.path.join(out, "calls.jsonl"), encoding="utf-8")]
        print(f"calls.jsonl: {len(calls)} records, tools {[c['tool'] for c in calls]}")
        assert len(calls) == 18
        br.call("set_adc_bit_depth", {"bits": 8})
    finally:
        br.close()
    br = MCPBridge(out, seed=2)                      # a restarted server restores the settings from calls.jsonl
    try:
        info = br.call("camera_info")
        assert info["adc_bit_depth"] == 8 and info["black_level_setting"] == 100 and info["gain_db"] == 6.0, info
        print("restart: ADC 8 bit, black level 100 and gain 6 dB restored")
    finally:
        br.close()
    print("ALL MCP SERVER CHECKS PASSED")


if __name__ == "__main__":
    main()
