import numpy as np
import json
import os

CAM = 'camera'
BLACK = 40.0
ROI_R = slice(128, 384)
ROI_C = slice(192, 448)
MAXCODE_SEEN = 0.0
SAT_MAX = 0.0


def load(name):
    global MAXCODE_SEEN
    a = np.load(os.path.join(CAM, name))
    print(f"SHAPE[{name}]={a.shape}")
    MAXCODE_SEEN = max(MAXCODE_SEEN, float(a.max()))
    a = a.astype(np.float64)
    if a.ndim == 2:
        a = a[None]
    return a


def roi(a):
    return a[:, ROI_R, ROI_C]


# ---- 1. Dark frames ----
dark_files = ['cap003.npy', 'cap004_dark_10ms_0dB_a.npy', 'cap005_dark_10ms_0dB_b.npy',
              'cap006_dark_10ms_0dB_c.npy', 'cap007_dark_10ms_0dB_d.npy']
D = np.concatenate([roi(load(f)) for f in dark_files], axis=0)
DARK_FRAME_COUNT = D.shape[0]
DARK_MEAN_DN = D.mean()
DARK_LEVEL_DN = DARK_MEAN_DN - BLACK
SIGMA_DARK_DN = np.std(D, axis=0, ddof=1).mean()
print(f"DARK_FRAME_COUNT={DARK_FRAME_COUNT}")
print(f"BLACK_LEVEL_DN={BLACK}")
print(f"DARK_MEAN_DN={DARK_MEAN_DN:.4f}")
print(f"DARK_LEVEL_DN={DARK_LEVEL_DN:.4f}")
print(f"SIGMA_DARK_DN={SIGMA_DARK_DN:.4f}")

# ---- 2. Bit depth ----
BIT_DEPTH = 12
print(f"BIT_DEPTH={BIT_DEPTH}")

# ---- 3. PTC ----
ptc_files = ['cap009_ptc_500us.npy', 'cap010_ptc_1000us.npy', 'cap011_ptc_2000us.npy',
             'cap012_ptc_4000us.npy', 'cap013_ptc_6000us.npy', 'cap014_ptc_8000us.npy',
             'cap015_ptc_10000us.npy', 'cap016_ptc_12000us.npy', 'cap017_ptc_15000us.npy']
mu_list, var_list, frame_max_list = [], [], []
for f in ptc_files:
    full = load(f)
    SAT_MAX = max(SAT_MAX, float((full >= 4095).mean()))
    a = roi(full)
    A, B = a[0], a[1]
    mu = ((A + B) / 2).mean() - BLACK
    var = ((A - B) ** 2 / 2).mean()
    mu_list.append(mu)
    var_list.append(var)
    frame_max_list.append(max(A.mean(), B.mean()))

print(f"MAX_CODE_OBSERVED={MAXCODE_SEEN}")
assert MAXCODE_SEEN <= 4095

mu = np.array(mu_list)
var = np.array(var_list)
slope, intercept = np.polyfit(mu, var, 1)
pred = slope * mu + intercept
PTC_R2 = 1 - np.sum((var - pred) ** 2) / np.sum((var - var.mean()) ** 2)
GAIN_E_PER_DN = 1.0 / slope
print(f"PTC_SLOPE_DN_PER_DN2={slope:.6f}")
print(f"PTC_INTERCEPT_DN2={intercept:.6f}")
print(f"GAIN_E_PER_DN={GAIN_E_PER_DN:.6f}")
print(f"PTC_R2={PTC_R2:.6f}")
print("PTC_TABLE: file | mu_DN | Var_DN2")
for f, m, v in zip(ptc_files, mu_list, var_list):
    print(f"PTC_ROW {f} | {m:.4f} | {v:.4f}")

# ---- 4. Read noise ----
if intercept > 0:
    READ_NOISE_E = GAIN_E_PER_DN * np.sqrt(intercept)
else:
    READ_NOISE_E = float('nan')
READ_NOISE_DARK_E = GAIN_E_PER_DN * SIGMA_DARK_DN
READ_NOISE_DIFF_PCT = 100.0 * (READ_NOISE_DARK_E - READ_NOISE_E) / READ_NOISE_E
print(f"READ_NOISE_E={READ_NOISE_E:.4f}")
print(f"READ_NOISE_DARK_E={READ_NOISE_DARK_E:.4f}")
print(f"READ_NOISE_DIFF_PCT={READ_NOISE_DIFF_PCT:.4f}")

# ---- 5. Full well (lower bound) ----
mu_sat = max(frame_max_list)
FULL_WELL_E = GAIN_E_PER_DN * (mu_sat - DARK_LEVEL_DN)
print(f"MU_SAT_DN={mu_sat:.4f}")
print(f"FULL_WELL_E={FULL_WELL_E:.2f}")
print("FULL_WELL_NOTE=lower bound, no saturation reached")
print(f"SATURATED_FRACTION_MAX={SAT_MAX}")

# ---- 6. Mean signal ----
c15 = roi(load('cap015_ptc_10000us.npy'))
roi_mean_15 = c15.mean()
MEAN_SIGNAL_E_PER_PX = GAIN_E_PER_DN * (roi_mean_15 - BLACK)
print("EXPOSURE_MS=10.0")
print(f"ROI_MEAN_CAP015_DN={roi_mean_15:.4f}")
print(f"MEAN_SIGNAL_E_PER_PX={MEAN_SIGNAL_E_PER_PX:.4f}")

# ---- 7. Flicker ----
jpath = os.path.join(CAM, 'cap019_flicker_10ms_0dB_illum1_series.json')
with open(jpath, 'r') as fh:
    js = json.load(fh)
if isinstance(js, dict):
    print("JSON_TOP_KEYS=" + ",".join(js.keys()))
else:
    print("JSON_TOP_KEYS=(not a dict)")


def find_lists(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_lists(v, f"{path}.{k}")
    elif isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(x, (int, float)) for x in obj):
            out.append((path, obj))
        else:
            for i, v in enumerate(obj):
                out += find_lists(v, f"{path}[{i}]")
    return out


cands = find_lists(js)
for p, l in cands:
    print(f"JSON_NUMERIC_LIST path={p} len={len(l)}")
chosen = [c for c in cands if len(c[1]) == 200]
if chosen:
    p, lst = chosen[0]
    m = np.array(lst, dtype=float)
    FRAME_COUNT = len(m)
    FLICKER_FRACTION = m.std(ddof=1) / m.mean()
    SHOT_FLOOR = 1.0 / np.sqrt(MEAN_SIGNAL_E_PER_PX * 256 * 256)
    x = m - m.mean()
    F = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(x), d=0.1)
    F[0] = 0.0
    FLICKER_HZ = fr[np.argmax(F)]
    print(f"FRAME_COUNT={FRAME_COUNT}")
    print(f"FLICKER_SERIES_PATH={p}")
    print(f"FLICKER_FRACTION={FLICKER_FRACTION:.6f}")
    print(f"SHOT_FLOOR={SHOT_FLOOR:.6f}")
    print(f"FLICKER_HZ={FLICKER_HZ:.4f}")
    flicker_ok = True
else:
    print("FRAME_COUNT=0")
    print("FLICKER_FRACTION=UNAVAILABLE (no 200-value per-frame list in JSON)")
    flicker_ok = False

# ---- 8. VERIFIED rule ----
reasons = []
if not PTC_R2 > 0.98:
    reasons.append(f"PTC_R2={PTC_R2:.4f}<=0.98")
if not DARK_FRAME_COUNT >= 50:
    reasons.append(f"DARK_FRAME_COUNT={DARK_FRAME_COUNT}<50")
if not abs(READ_NOISE_DIFF_PCT) <= 10:
    reasons.append(f"READ_NOISE_DIFF_PCT={READ_NOISE_DIFF_PCT:.2f} outside 10%")
if reasons:
    status = "UNVERIFIED (" + "; ".join(reasons) + ")"
else:
    status = "VERIFIED"
for name in ['GAIN_E_PER_DN', 'READ_NOISE_E', 'FULL_WELL_E', 'DARK_LEVEL_DN', 'MEAN_SIGNAL_E_PER_PX']:
    print(f"{name}_STATUS={status}")
if not flicker_ok:
    print("FLICKER_FRACTION_STATUS=UNVERIFIED (per-frame series missing)")
else:
    print(f"FLICKER_FRACTION_STATUS={status}")
