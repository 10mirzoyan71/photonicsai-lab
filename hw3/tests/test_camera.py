"""Checks of the simulated camera and the statistics (no MCP, no model): python tests/test_camera.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from camera import CameraError, open_camera  # noqa: E402
from camera.stats import beam_stats, frame_stats, speckle_stats, temporal_noise  # noqa: E402


def stack(cam, n):
    return np.stack([cam.grab() for _ in range(n)])


def main():
    cam = open_camera("sim", seed=3)
    cam._clock = 0.0                                          # freeze the slow drift for the noise checks
    cam.laser_noise = 0.0
    truth, mx = cam.truth(), cam.max_dn

    # ranges are enforced
    for bad in (lambda: cam.set_exposure(1.0), lambda: cam.set_gain(40), lambda: cam.set_roi(600, 0, 100, 100)):
        try:
            bad()
            raise AssertionError("out-of-range command accepted")
        except CameraError:
            pass

    # dark frames: level = black level, temporal noise = read noise (short exposure, so dark current is negligible)
    cam.set_illumination(0.0)
    cam.set_exposure(100)
    dark = stack(cam, 4)
    tn = temporal_noise(dark, mx)
    read_dn = tn["temporal_std_dn"]
    print(f"dark: mean {dark.mean():.2f} DN (black {truth['black_level_dn']}), read noise {read_dn:.3f} DN "
          f"(truth {truth['read_noise_dn_at_0dB']:.3f})")
    assert abs(dark.mean() - truth["black_level_dn"]) < 0.5
    assert abs(read_dn / truth["read_noise_dn_at_0dB"] - 1) < 0.05

    # photon transfer: temporal variance vs mean signal above dark; slope = 1/K
    cam.set_illumination(1.0)
    cam.set_roi(220, 156, 200, 200)
    dark_roi = None
    means, vars_ = [], []
    for t in (50, 100, 200, 400, 800, 1600):
        cam.set_exposure(t)
        s = stack(cam, 2)
        st = temporal_noise(s, mx)
        cam.set_illumination(0.0)
        dark_roi = stack(cam, 2).mean()
        cam.set_illumination(1.0)
        means.append(st["mean_dn"] - dark_roi)
        vars_.append(st["temporal_var_dn2"] - read_dn ** 2)
    slope = np.polyfit(means, vars_, 1)[0]
    k_est = 1.0 / slope
    print(f"photon transfer: K = {k_est:.3f} e-/DN (truth {truth['conversion_gain_e_per_dn_at_0dB']}), "
          f"read noise = {read_dn * k_est:.2f} e- (truth {truth['read_noise_e']})")
    assert abs(k_est / truth["conversion_gain_e_per_dn_at_0dB"] - 1) < 0.05

    # saturation appears at long exposure and is reported
    cam.set_roi(0, 0, 640, 512)
    cam.set_exposure(20000)
    st = frame_stats(cam.grab(), mx, regions=[{"name": "centre", "x": 300, "y": 236, "width": 40, "height": 40}])
    print(f"20 ms: saturated fraction {st['saturated_fraction']}, max {st['max_dn']}, centre {st['regions']['centre']}")
    assert st["saturated_fraction"] > 0.01

    # speckle: contrast near 1 and a grain of a few pixels in the flat centre of the beam
    cam.set_exposure(300)
    sp = speckle_stats(cam.grab(), truth["black_level_dn"], {"x": 270, "y": 206, "width": 100, "height": 100})
    bm = beam_stats(cam.grab(), truth["black_level_dn"])
    print(f"speckle: {sp}\nbeam: {bm}")
    assert 0.8 < sp["contrast"] < 1.2 and 1.5 < sp["grain_fwhm_x_px"] < 6
    assert abs(bm["centroid_x_px"] - 319.5) < 15 and abs(bm["centroid_y_px"] - 255.5) < 15

    # ROI crop
    cam.set_roi(10, 20, 64, 32)
    assert cam.grab().shape == (32, 64)

    # room-light scene: no speckle, 100 Hz flicker; the frame mean fluctuates unless the exposure spans whole periods
    room = open_camera("sim-room", seed=4)
    room._clock = 0.0
    tr = room.truth()
    assert tr["scene"] == "room" and tr["flicker_hz"] == 100.0
    rel = {}
    for t in (3000, 7000, 10000, 20000):
        room.set_exposure(t)
        m = np.array([room.grab().mean() for _ in range(40)]) - tr["black_level_dn"]
        rel[t] = 100 * m.std() / m.mean()
    print("room light, frame-to-frame std of the mean [%] vs exposure [us]:", {k: round(v, 2) for k, v in rel.items()})
    assert rel[3000] > 3 * rel[10000] and rel[7000] > 3 * rel[20000] and rel[10000] < 1.0
    room.set_exposure(10000)
    flat = room.grab().astype(float) - tr["black_level_dn"]
    assert flat.std() / flat.mean() < 0.15                    # nearly uniform (vignetting + noise), no speckle
    print("ALL CAMERA CHECKS PASSED")


if __name__ == "__main__":
    main()
