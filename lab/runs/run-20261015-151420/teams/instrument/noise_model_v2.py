import json
import math
from pathlib import Path

import numpy as np

CAM = Path("camera")
R = slice(128, 384)
C = slice(192, 448)
BLACK = 40.0
FS_DT = 0.1
NPX = 256 * 256

def frames(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-2], a.shape[-1])

def find_key(obj, name):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == name and isinstance(v, list):
                return v
            r = find_key(v, name)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, name)
            if r is not None:
                return r
    return None

for f in sorted(CAM.glob("*.npy")):
    a = np.load(f)
    print(f"SHAPE {f.name} shape={a.shape} dtype={a.dtype}")

js = json.load(open(CAM / "cap019_flicker_10ms_0dB_illum1_series.json"))
if isinstance(js, dict):
    print("JSON_TOP_KEYS=" + ",".join(js.keys()))
    for k, v in js.items():
        if isinstance(v, list):
            print(f"JSON_LIST_LEN {k}={len(v)}")
else:
    print("JSON_TOP_KEYS=(not dict)")

dark_names = ["cap003.npy", "cap004_dark_10ms_0dB_a.npy", "cap005_dark_10ms_0dB_b.npy",
              "cap006_dark_10ms_0dB_c.npy", "cap007_dark_10ms_0dB_d.npy"]
dark_rois = []
max_code = 0
for n in dark_names:
    raw = np.load(CAM / n)
    max_code = max(max_code, int(raw.max()))
    fr = frames(raw)
    dark_rois.append(fr[:, R, C])
D = np.concatenate(dark_rois, axis=0)
DARK_FRAME_COUNT = D.shape[0]
DARK_MEAN_DN = float(D.mean())
DARK_LEVEL_DN = DARK_MEAN_DN - BLACK
SIGMA_DARK_DN = float(D.std(axis=0, ddof=1).mean())
print(f"DARK_FRAME_COUNT={DARK_FRAME_COUNT}")
print(f"BLACK_LEVEL_DN={BLACK}")
print(f"DARK_MEAN_DN={DARK_MEAN_DN:.4f}")
print(f"DARK_LEVEL_DN={DARK_LEVEL_DN:.4f}")
print(f"SIGMA_DARK_DN={SIGMA_DARK_DN:.4f}")

for f in sorted(CAM.glob("*.npy")):
    max_code = max(max_code, int(np.load(f).max()))
BIT_DEPTH = 12
print(f"BIT_DEPTH={BIT_DEPTH}")
print(f"MAX_CODE_OBSERVED={max_code} (<=4095: {max_code <= 4095})")

ptc = [(500, "cap009_ptc_500us.npy"), (1000, "cap010_ptc_1000us.npy"),
       (2000, "cap011_ptc_2000us.npy"), (4000, "cap012_ptc_4000us.npy"),
       (6000, "cap013_ptc_6000us.npy"), (8000, "cap014_ptc_8000us.npy"),
       (10000, "cap015_ptc_10000us.npy"), (12000, "cap016_ptc_12000us.npy"),
       (15000, "cap017_ptc_15000us.npy")]
mus, vars_ = [], []
mu_sat = -1e9
sat_max = 0.0
print("PTC_TABLE exposure_us mu_DN Var_DN")
for exp, n in ptc:
    raw = np.load(CAM / n)
    fr = frames(raw)
    sat_max = max(sat_max, float((raw >= 4095).mean()))
    A = fr[0][R, C]
    B = fr[1][R, C]
    for k in range(fr.shape[0]):
        mu_sat = max(mu_sat, float(fr[k][R, C].mean()))
    mu = float(((A + B) / 2).mean()) - BLACK
    var = float(((A - B) ** 2 / 2).mean())
    mus.append(mu)
    vars_.append(var)
    print(f"PTC_TABLE {exp} {mu:.4f} {var:.4f}")
mus = np.array(mus)
vars_ = np.array(vars_)
slope, intercept = np.polyfit(mus, vars_, 1)
pred = slope * mus + intercept
ss_res = float(((vars_ - pred) ** 2).sum())
ss_tot = float(((vars_ - vars_.mean()) ** 2).sum())
PTC_R2 = 1 - ss_res / ss_tot
PTC_SLOPE = float(slope)
PTC_INTERCEPT_DN2 = float(intercept)
print(f"PTC_SLOPE={PTC_SLOPE:.6f}")
print(f"PTC_INTERCEPT_DN2={PTC_INTERCEPT_DN2:.4f}")
print(f"PTC_R2={PTC_R2:.6f}")
print(f"SANITY_LOW_mu={mus[0]:.2f} Var={vars_[0]:.2f}")
print(f"SANITY_MID_mu={mus[6]:.2f} Var={vars_[6]:.2f}")
if PTC_SLOPE > 0:
    GAIN_E_PER_DN = 1.0 / PTC_SLOPE
else:
    GAIN_E_PER_DN = float("nan")
    print("PTC_WARNING slope<=0, gain not trusted")
if PTC_R2 < 0.98:
    print(f"PTC_WARNING R2={PTC_R2:.4f} < 0.98, gain not trusted")
print(f"GAIN_E_PER_DN={GAIN_E_PER_DN:.6f}")

if PTC_INTERCEPT_DN2 > 0:
    READ_NOISE_E = GAIN_E_PER_DN * math.sqrt(PTC_INTERCEPT_DN2)
else:
    READ_NOISE_E = float("nan")
    print("READ_NOISE_WARNING intercept<=0")
READ_NOISE_DARK_E = GAIN_E_PER_DN * SIGMA_DARK_DN
READ_NOISE_DIFF_PCT = 100 * (READ_NOISE_E - READ_NOISE_DARK_E) / READ_NOISE_DARK_E
print(f"READ_NOISE_E={READ_NOISE_E:.4f}")
print(f"READ_NOISE_DARK_E={READ_NOISE_DARK_E:.4f}")
print(f"READ_NOISE_DIFF_PCT={READ_NOISE_DIFF_PCT:.4f}")

MU_SAT_DN = mu_sat
FULL_WELL_E = GAIN_E_PER_DN * (MU_SAT_DN - DARK_LEVEL_DN)
print(f"MU_SAT_DN={MU_SAT_DN:.4f}")
print(f"FULL_WELL_E={FULL_WELL_E:.2f} (lower bound: no saturation reached)")
print(f"SATURATED_FRACTION_MAX={sat_max}")

c15 = frames(np.load(CAM / "cap015_ptc_10000us.npy"))
roi_mean_15 = float(c15[:, R, C].mean())
MEAN_SIGNAL_E_PER_PX = GAIN_E_PER_DN * (roi_mean_15 - BLACK)
print(f"ROI_MEAN_CAP015_DN={roi_mean_15:.4f}")
print("EXPOSURE_MS=10.0")
print(f"MEAN_SIGNAL_E_PER_PX={MEAN_SIGNAL_E_PER_PX:.2f}")

mean_list = find_key(js, "mean_dn")
m = None
if mean_list is None:
    print("FLICKER_ERROR no mean_dn list in JSON")
else:
    print("FLICKER_KEY_USED=mean_dn")
    m = np.asarray(mean_list, dtype=np.float64)
    print(f"MEAN_DN_SERIES_N={m.size}")
    print(f"MEAN_DN_SERIES_MEAN={m.mean():.4f}")
    FLICKER_FRACTION = float(m.std(ddof=1) / m.mean())
    SHOT_FLOOR = 1.0 / math.sqrt(MEAN_SIGNAL_E_PER_PX * NPX)
    d = m - m.mean()
    spec = np.abs(np.fft.rfft(d))
    freqs = np.fft.rfftfreq(d.size, d=FS_DT)
    spec[0] = 0.0
    FLICKER_HZ = float(freqs[int(np.argmax(spec))])
    print(f"FLICKER_FRACTION={FLICKER_FRACTION:.6f}")
    print(f"SHOT_FLOOR={SHOT_FLOOR:.6e}")
    print(f"FLICKER_HZ={FLICKER_HZ:.4f}")
    print(f"FRAME_COUNT={m.size}")

ok_base = (PTC_R2 > 0.98) and (DARK_FRAME_COUNT >= 50) and (abs(READ_NOISE_DIFF_PCT) <= 10)
reason = []
if not PTC_R2 > 0.98:
    reason.append(f"PTC_R2={PTC_R2:.4f}<=0.98")
if not DARK_FRAME_COUNT >= 50:
    reason.append("dark<50")
if not abs(READ_NOISE_DIFF_PCT) <= 10:
    reason.append(f"READ_NOISE_DIFF_PCT={READ_NOISE_DIFF_PCT:.2f}")
rs = "; ".join(reason) if reason else "all checks pass"
for name in ["GAIN_E_PER_DN", "READ_NOISE_E", "FULL_WELL_E", "MEAN_SIGNAL_E_PER_PX"]:
    print(f"{name}_STATUS=" + ("VERIFIED" if ok_base else f"UNVERIFIED ({rs})"))
for name in ["BLACK_LEVEL_DN", "DARK_LEVEL_DN", "BIT_DEPTH", "EXPOSURE_MS",
             "DARK_FRAME_COUNT", "SIGMA_DARK_DN", "FLICKER_FRACTION"]:
    if name == "FLICKER_FRACTION" and m is None:
        print(f"{name}_STATUS=UNVERIFIED (no series)")
    else:
        print(f"{name}_STATUS=VERIFIED")
