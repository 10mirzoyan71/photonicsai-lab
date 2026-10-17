import sys
import numpy as np
sys.path.insert(0, 'silverbox')
import silverbox_pinn as sp
from silverbox_pinn import load_excerpt, load_model, half_sample_input, free_run

print('N_TRAIN=', getattr(sp, 'N_TRAIN', None))
src = open('silverbox/silverbox_pinn.py', encoding='utf-8').read().splitlines()
for i, l in enumerate(src):
    if 'N_TRAIN' in l or 'test' in l.lower():
        print('SRC_LINE', i + 1, l.strip())

u, y = load_excerpt()
f, cfg = load_model('silverbox/results/A2_h5.pt')
yhat, rmse_train, rmse_test = free_run(f, u, y, half_sample_input(u, cfg.interp))
print('TEST_RECORD=SNLS80mV')
print('MODEL_LOADED_FROM=silverbox/results/A2_h5.pt')
print('RETRAINED=no')
print('UNITS=V (volts, per free_run metric)')
print('INIT_CONDITIONS=as implemented in free_run/half_sample_input in silverbox_pinn.py (not re-implemented)')
print('METRIC_RMSE_TEST_RETURNED=', float(rmse_test))

yv = np.asarray(y, dtype=float).ravel()
yh = np.asarray(yhat, dtype=float).ravel()
s = slice(3072, 4096)
ym = yv[s]
ys = yh[s]
e = ys - ym
N = int(e.size)
rmse = float(np.sqrt(np.mean(e ** 2)))
std = float(np.std(ym))
print('TEST_SEGMENT=3072..4095')
print('N_SAMPLES=', N)
print('FREERUN_RMSE=', rmse)
print('TEST_OUTPUT_STD=', std)
print('FREERUN_RMSE_OVER_STD=', rmse / std)
