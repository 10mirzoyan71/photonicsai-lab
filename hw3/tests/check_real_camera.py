"""First contact with the lab camera (Hikrobot MV-CS028-10UM through MVS), run on the PC the camera is plugged into:
    python tests/check_real_camera.py            # connector checks; saves frames in tests/real_camera_check/
    python tests/check_real_camera.py --dark     # also asks you to cover the lens and measures the dark level
Close the MVS client first (the connector opens the camera exclusively). Every step prints OK or the error.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from camera import open_camera  # noqa: E402
from camera.stats import frame_stats, temporal_noise  # noqa: E402


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        shown = f"array {out.shape} {out.dtype}" if isinstance(out, np.ndarray) else out
        print(f"[OK]   {name} ({time.time() - t0:.2f} s)" + (f": {shown}" if out is not None else ""), flush=True)
        return out
    except Exception as e:                    # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dark", action="store_true")
    p.add_argument("--exposure", type=float, default=2000.0, help="first exposure in us")
    a = p.parse_args()
    out = os.path.join(HERE, "real_camera_check")
    os.makedirs(out, exist_ok=True)

    cam = step("load MVS SDK and open the camera", lambda: open_camera("hikrobot"))
    info = step("read camera info", cam.info)
    json.dump(info, open(os.path.join(out, "camera_info.json"), "w"), indent=1)
    mx = info["max_dn"]
    try:
        step("set gain 0 dB", lambda: cam.set_gain(max(0.0, info["gain_range_db"][0])))
        e1 = step(f"set exposure {a.exposure:g} us", lambda: cam.set_exposure(a.exposure))
        f1 = step("grab one frame", cam.grab)
        s1 = frame_stats(f1, mx)
        print("       frame:", f1.shape, f1.dtype, {k: s1[k] for k in ("mean_dn", "max_dn", "saturated_fraction")})
        np.save(os.path.join(out, "normal_frame.npy"), f1)
        e2 = step(f"set exposure {2 * a.exposure:g} us", lambda: cam.set_exposure(2 * a.exposure))
        f2 = step("grab one frame", cam.grab)
        print(f"       mean {f1.mean():.1f} DN at {e1:g} us -> {f2.mean():.1f} DN at {e2:g} us "
              f"(ratio {f2.mean() / max(f1.mean(), 1e-9):.2f}; about 2 when well above the black level and not saturated)")
        stack = step("grab 4 frames", lambda: np.stack([cam.grab() for _ in range(4)]))
        print("       temporal noise:", temporal_noise(stack, mx))
        t0 = time.time()
        for _ in range(10):
            cam.grab()
        print(f"       {10 / (time.time() - t0):.1f} frames/s with software trigger at {e2:g} us")
        step("read the device temperature (C)", cam.temperature_c)
        bl0 = info.get("black_level_setting") or 180
        step("set black level 200 (a larger value lowers the dark level)", lambda: cam.set_black_level(200))
        fb = step("grab one frame", cam.grab)
        print(f"       mean {fb.mean():.1f} DN at black level 200, {f2.mean():.1f} DN at {bl0}")
        step(f"black level back to {bl0}", lambda: cam.set_black_level(bl0))
        step("ADC 8 bit", lambda: cam.set_adc_bit_depth(8))
        f8 = step("grab one frame", cam.grab)
        step("ADC 12 bit", lambda: cam.set_adc_bit_depth(12))
        f12 = step("grab one frame", cam.grab)
        print(f"       distinct levels in a frame: {len(np.unique(f8))} with the 8-bit ADC, {len(np.unique(f12))} with 12 bits")
        w, h = info["sensor_width"], info["sensor_height"]
        roi = step("set a 256x256 ROI in the centre", lambda: cam.set_roi(w // 2 - 128, h // 2 - 128, 256, 256))
        step("grab with the ROI", lambda: cam.grab().shape)
        step("back to the full sensor", lambda: cam.set_roi(0, 0, w, h))
        if a.dark:
            input("\nCover the lens (or block all light), then press Enter... ")
            cam.set_exposure(e1)
            d = step("grab 4 dark frames", lambda: np.stack([cam.grab() for _ in range(4)]))
            np.save(os.path.join(out, "dark_frames.npy"), d)
            tn = temporal_noise(d, mx)
            print(f"       black level {d.mean():.2f} DN, temporal noise {tn.get('temporal_std_dn')} DN rms")
        print(f"\nALL CONNECTOR CHECKS PASSED - {info['model']}, {info['sensor_width']}x{info['sensor_height']}, "
              f"{info['pixel_format']}, ROI used {roi}. Files in {out}")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
