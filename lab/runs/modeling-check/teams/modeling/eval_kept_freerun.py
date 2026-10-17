import sys, json, re
import numpy as np
sys.path.insert(0, 'silverbox')
from silverbox_pinn import load_excerpt, load_model, half_sample_input, free_run

print('KEPT_JSON=' + open('silverbox/results/kept.json').read().replace('\n', ' '))
src = open('silverbox/silverbox_pinn.py').read()
for line in src.splitlines():
    if re.search(r'half|test|train|split|init|n_a|n_b|lag', line, re.I):
        print('SRC:', line.strip())

u, y = load_excerpt()
print('RECORD_LEN=', len(y), 'UNITS=volts (centred, per load_excerpt)')
f, cfg = load_model('silverbox/results/A2_h5.pt')
yhat, rmse_train, rmse_test = free_run(f, u, y, half_sample_input(u, cfg.interp))
yhat = np.asarray(yhat).ravel(); y = np.asarray(y).ravel(); u = np.asarray(u).ravel()
n = len(y)
h = n // 2
seg = slice(h, n)
N = len(y[seg])
rmse = np.sqrt(np.mean((yhat[seg] - y[seg])**2))
std = np.std(y[seg])
print('TEST_RECORD=SNLS80mV (data/SNLS80mV.mat, load_excerpt)')
print('TEST_SEGMENT=second half, indices %d..%d (ASSUMED split n//2; verify against silverbox_pinn.py SRC lines)' % (h, n-1))
print('N_SAMPLES=%d' % N)
print('FREERUN_RMSE=%.6g' % rmse)
print('TEST_OUTPUT_STD=%.6g' % std)
print('FREERUN_RMSE_OVER_STD=%.6g' % (rmse/std))
print('METRIC_RMSE_TEST_RETURNED=%s' % rmse_test)
print('MODEL_LOADED_FROM=silverbox/results/A2_h5.pt')
print('RETRAINED=no')
print('INIT=as in free_run/half_sample_input of silverbox_pinn.py (not re-implemented)')
