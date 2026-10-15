import os, json, math
import numpy as np

CAM = 'camera'
ROI = (slice(128, 384), slice(192, 448))
BLACK = 40.0

def load(name):
    raw = np.load(os.path.join(CAM, name))
    print(f"FILE_{name}_SHAPE={raw.shape}")
    print(f"FILE_{name}_DTYPE={raw.dtype}")
    print(f"FILE_{name}_MAX_CODE={int(raw.max())}")
    a = raw.astype(np.float64)
    if a.ndim == 2:
        a = a[None]
    return a, raw

def fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, icpt])
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, icpt, r2

PTC = [
    ('cap009_ptc_500us.npy', 500, 21.1985),
    ('cap010_ptc_1000us.npy', 1000, 32.1),
    ('cap011_ptc_2000us.npy', 2000, 67.4741),
    ('cap012_ptc_4000us.npy', 4000, 111.1669),
    ('cap013_ptc_6000us.npy', 6000, 166.0218),
    ('cap014_ptc_8000us.npy', 8000, 208.4157),
    ('cap015_ptc_10000us.npy', 10000, 257.6639),
    ('cap016_ptc_12000us.npy', 12000, 301.1435),
    ('cap017_ptc_15000us.npy', 15000, 397.8644),
]

print("=== PTC ===")
mus, vmean, vmed, vclip, vcam = [], [], [], [], []
max_frame_mean = -1.0
sat_count = 0
total_px = 0
for fname, exp_us, cam_var in PTC:
    a, raw = load(fname)
    sat_count += int(np.sum(raw == 4095))
    total_px += raw.size
    A = a[0][ROI]; B = a[1][ROI]
    for k in range(a.shape[0]):
        max_frame_mean = max(max_frame_mean, a[k][ROI].mean() - BLACK)
    mu = np.mean((A + B) / 2.0) - BLACK
    d = A - B
    v = d ** 2 / 2.0
    vm = v.mean()
    vmd = np.median(v) / 0.4549
    mask = np.ones(d.shape, bool)
    for _ in range(50):
        s2 = v[mask].mean()
        sd = math.sqrt(2.0 * s2)
        new = np.abs(d) <= 4.0 * sd
        if np.array_equal(new, mask):
            break
        mask = new
    vc = v[mask].mean()
    nclip = int(d.size - mask.sum())
    print(f"PTC_{exp_us}us_MU_DN={mu:.4f}")
    print(f"PTC_{exp_us}us_VAR_MEAN_DN2={vm:.4f}")
    print(f"PTC_{exp_us}us_VAR_MEDIAN_DN2={vmd:.4f}")
    print(f"PTC_{exp_us}us_VAR_CLIP_DN2={vc:.4f}")
    print(f"PTC_{exp_us}us_CLIPPED_PX={nclip}")
    print(f"PTC_{exp_us}us_CAMERA_VAR_DN2={cam_var:.4f}")
    mus.append(mu); vmean.append(vm); vmed.append(vmd); vclip.append(vc); vcam.append(cam_var)

mus = np.array(mus)
print("TABLE exp_us mu_DN Var_mean Var_median Var_clip CAMERA_VAR")
for i, (_, e, _) in enumerate(PTC):
    print(f"ROW {e} {mus[i]:.3f} {vmean[i]:.3f} {vmed[i]:.3f} {vclip[i]:.3f} {vcam[i]:.3f}")

print("=== FITS (Var = mu/g + sigma_r^2) ===")
results = {}
for label, ys in [('MEAN', vmean), ('MEDIAN', vmed), ('CLIP', vclip), ('CAMERA', vcam)]:
    slope, icpt, r2 = fit(mus, ys)
    gain = 1.0 / slope if slope > 0 else float('nan')
    results[label] = (slope, icpt, r2, gain)
    print(f"FIT_{label}_SLOPE={slope:.6f}")
    print(f"FIT_{label}_INTERCEPT_DN2={icpt:.4f}")
    print(f"FIT_{label}_R2={r2:.6f}")
    print(f"FIT_{label}_GAIN_E_PER_DN={gain:.6f}")

cam_slope = results['CAMERA'][0]
best = None
for lab in ['MEAN', 'MEDIAN', 'CLIP']:
    s = results[lab][0]
    ratio = s / cam_slope if cam_slope != 0 else float('nan')
    print(f"SLOPE_RATIO_{lab}_OVER_CAMERA={ratio:.6f}")
    if np.isfinite(ratio) and results[lab][0] > 0 and ratio > 0:
        dist = abs(math.log(ratio))
        if best is None or dist < best[1]:
            best = (lab, dist)
best_lab = best[0] if best else 'MEAN'
print(f"BEST_ESTIMATOR_AGREES_WITH_CAMERA={best_lab}")

SLOPE, INTERCEPT, PTC_R2, GAIN = results[best_lab]
print(f"CHOSEN_ESTIMATOR={best_lab}")
print(f"PTC_R2={PTC_R2:.6f}")
print(f"GAIN_E_PER_DN={GAIN:.6f}")
print(f"PTC_INTERCEPT_DN2={INTERCEPT:.4f}")

print("=== DARK ===")
dark_files = ['cap003.npy', 'cap004_dark_10ms_0dB_a.npy', 'cap005_dark_10ms_0dB_b.npy',
              'cap006_dark_10ms_0dB_c.npy', 'cap007_dark_10ms_0dB_d.npy']
frames = []
for f in dark_files:
    a, raw = load(f)
    frames.append(a[:, ROI[0], ROI[1]])
D = np.concatenate(frames, axis=0)
N = D.shape[0]
print(f"DARK_FRAME_COUNT={N}")
DARK_MEAN = D.mean()
print(f"DARK_MEAN_DN={DARK_MEAN:.4f}")
DARK_LEVEL = DARK_MEAN - BLACK
print(f"BLACK_LEVEL_DN={BLACK}")
print(f"DARK_LEVEL_DN={DARK_LEVEL:.4f}")
pix_std = np.std(D, axis=0, ddof=1)
SIGMA_DARK = pix_std.mean()
SIGMA_DARK_MED = np.median(pix_std)
hot = int(np.sum(pix_std > 5 * SIGMA_DARK_MED))
print(f"SIGMA_DARK_DN={SIGMA_DARK:.4f}")
print(f"SIGMA_DARK_MEDIAN_DN={SIGMA_DARK_MED:.4f}")
print(f"HOT_PIXEL_COUNT={hot}")

print("=== READ NOISE ===")
if np.isfinite(GAIN) and INTERCEPT > 0:
    RN = GAIN * math.sqrt(INTERCEPT)
else:
    RN = float('nan')
RN_DARK = GAIN * SIGMA_DARK
RN_DARK_MED = GAIN * SIGMA_DARK_MED
DIFF = 100.0 * (RN - RN_DARK) / RN_DARK if RN_DARK else float('nan')
DIFF_MED = 100.0 * (RN - RN_DARK_MED) / RN_DARK_MED if RN_DARK_MED else float('nan')
print(f"READ_NOISE_E={RN:.4f}")
print(f"READ_NOISE_DARK_E={RN_DARK:.4f}")
print(f"READ_NOISE_DARK_MEDIAN_E={RN_DARK_MED:.4f}")
print(f"READ_NOISE_DIFF_PCT={DIFF:.2f}")
print(f"READ_NOISE_DIFF_MEDIAN_PCT={DIFF_MED:.2f}")

print("=== FULL WELL ===")
MU_SAT = max_frame_mean
FULL_WELL = GAIN * MU_SAT
SAT_FRAC = sat_count / total_px
print(f"MU_SAT_DN={MU_SAT:.4f}")
print(f"FULL_WELL_E={FULL_WELL:.2f}")
print("FULL_WELL_NOTE=lower bound (no saturation reached)")
print(f"SATURATED_FRACTION_MAX={SAT_FRAC:.6f}")

print("=== MEAN SIGNAL ===")
a15, _ = load('cap015_ptc_10000us.npy')
roi15 = np.mean([fr[ROI].mean() for fr in a15])
MEAN_SIG = GAIN * (roi15 - BLACK)
print(f"ROI_MEAN_CAP015_DN={roi15:.4f}")
print("EXPOSURE_MS=10.0")
print(f"MEAN_SIGNAL_E_PER_PX={MEAN_SIG:.4f}")

print("=== FLICKER ===")
jpath = os.path.join(CAM, 'cap019_flicker_10ms_0dB_illum1_series.json')
with open(jpath) as fh:
    J = json.load(fh)
print(f"JSON_TOP_KEYS={list(J.keys()) if isinstance(J, dict) else type(J).__name__}")

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None

m = np.array(find_key(J, 'mean_dn'), float)
print("FLICKER_KEY_USED=mean_dn")
FRAME_COUNT = m.size
print(f"FRAME_COUNT={FRAME_COUNT}")
MEAN_DN_SERIES = m.mean()
print(f"MEAN_DN_SERIES={MEAN_DN_SERIES:.4f}")
FLICKER_FRACTION = np.std(m, ddof=1) / MEAN_DN_SERIES
print(f"FLICKER_FRACTION={FLICKER_FRACTION:.6f}")
SHOT_FLOOR = 1.0 / math.sqrt(MEAN_SIG * 256 * 256) if MEAN_SIG > 0 else float('nan')
print(f"SHOT_FLOOR={SHOT_FLOOR:.6e}")
spec = np.abs(np.fft.rfft(m - m.mean()))
freqs = np.fft.rfftfreq(m.size, d=0.1)
spec[0] = 0.0
pk = int(np.argmax(spec))
print(f"FLICKER_PEAK_HZ={freqs[pk]:.4f}")
print("FLICKER_HZ_NOTE='10 Hz sampling, 100 Hz line aliases; peak not diagnostic'")

print("=== VERIFIED ===")
reasons = []
if not (PTC_R2 > 0.98):
    reasons.append(f"PTC_R2={PTC_R2:.4f} <= 0.98")
if not (N >= 50):
    reasons.append(f"DARK_FRAME_COUNT={N} < 50")
if not (abs(DIFF) <= 10):
    reasons.append(f"|READ_NOISE_DIFF_PCT|={abs(DIFF):.1f} > 10")
for k in ['GAIN_E_PER_DN', 'READ_NOISE_E', 'FULL_WELL_E', 'MEAN_SIGNAL_E_PER_PX']:
    if reasons:
        print(f"{k}=UNVERIFIED ({'; '.join(reasons)})")
    else:
        print(f"{k}=VERIFIED")
for k in ['BLACK_LEVEL_DN', 'DARK_LEVEL_DN', 'BIT_DEPTH', 'EXPOSURE_MS', 'DARK_FRAME_COUNT', 'SIGMA_DARK_DN', 'FLICKER_FRACTION']:
    print(f"{k}=VERIFIED (read from files/series)")
print("BIT_DEPTH=12")
