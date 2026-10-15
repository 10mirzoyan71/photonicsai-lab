import numpy as np

FILES = {
    1: 'camera/cap023_ptc_v2_1ms.npy',
    5: 'camera/cap024_ptc_v2_5ms.npy',
    10: 'camera/cap026_ptc_v2_10ms.npy',
    15: 'camera/cap027_ptc_v2_15ms.npy',
    20: 'camera/cap028_ptc_v2_20ms.npy',
    25: 'camera/cap029_ptc_v2_25ms.npy',
    30: 'camera/cap030_ptc_v2_30ms.npy',
}
BL = 40.0
DARK = 2.5151
DVAR = DARK ** 2
RS, RE, CS, CE = 128, 384, 192, 448
LIM_LO, LIM_HI = 20.0, 0.8 * (4095 - 40)

ts, means, vars_, pvars = [], [], [], []
for t, f in FILES.items():
    a = np.load(f).astype(np.float64)
    print(f'FRAMES_{t}ms={a.shape[0]}')
    roi = a[:, RS:RE, CS:CE]
    m = roi.mean(axis=0)
    keep = (m - BL >= LIM_LO) & (m - BL <= LIM_HI)
    r = roi[:, keep]
    mean_dn = m[keep].mean()
    rr = r - r.mean(axis=1, keepdims=True)
    var_t = rr.var(axis=0, ddof=1).mean() - DVAR
    d = rr[0::2] - rr[1::2]
    pvar_t = (d ** 2 / 2.0).mean(axis=0).mean() - DVAR
    diff_pct = 100.0 * (pvar_t - var_t) / var_t
    print(f'MEAN_DN_{t}ms={mean_dn:.4f}')
    print(f'KEPT_PX_{t}ms={keep.sum()}')
    print(f'VAR_DN2_{t}ms={var_t:.4f}')
    print(f'VAR_PAIR_DN2_{t}ms={pvar_t:.4f}')
    print(f'PAIR_VS_FRAME_PCT_{t}ms={diff_pct:.2f}')
    ts.append(t); means.append(mean_dn); vars_.append(var_t)

x = np.array(means) - BL
y = np.array(vars_)

def fit(x, y, label):
    s, b = np.polyfit(x, y, 1)
    yhat = s * x + b
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f'SLOPE_{label}={s:.6f}')
    print(f'INTERCEPT_{label}={b:.6f}')
    print(f'R2_{label}={r2:.5f}')
    return s, b

slope, intercept = fit(x, y, 'ALL7')
mask = np.array(ts) != 1
fit(x[mask], y[mask], 'EXCL1MS')

GAIN = 1.0 / slope
print(f'GAIN_e_per_DN={GAIN:.6f}')
if intercept > 0:
    READ_fit = GAIN * np.sqrt(intercept)
    print(f'READ_e_fit={READ_fit:.4f}')
else:
    READ_fit = float('nan')
    print('READ_e_fit=NA (intercept<=0)')
READ_dark = GAIN * DARK
print(f'READ_e_dark={READ_dark:.4f}')
print(f'READ_DIFF_e={READ_fit - READ_dark:.4f}')
FW = GAIN * (4095 - 40)
print(f'FULL_WELL_e_lower_bound={FW:.1f}')

a40 = np.load('camera/cap031_ptc_v2_40ms_check.npy').astype(np.float64)
print(f'MAX_CODE_40ms={a40.max():.0f}')
print(f'P99_9_40ms={np.percentile(a40, 99.9):.1f}')

mean10 = means[ts.index(10)]
MS = GAIN * (mean10 - 40.1229)
print(f'MEAN_SIGNAL_e_per_px_10ms={MS:.2f}')
print('CAMERA_GAIN=2.50')
print(f'RATIO={GAIN / 2.50:.4f}')
