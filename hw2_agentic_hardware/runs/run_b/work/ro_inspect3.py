import sys, json, inspect, math, traceback, time
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, 'silverbox')
R = 'silverbox/results/'
import silverbox_pinn as sp
def hist_free(d):
    return {k: (v if not isinstance(v, (list, dict)) or k in ('config','ls_coeffs','duffing_coeffs','pinn_coeffs') else '<%d>' % len(v)) for k, v in d.items()}
kept = json.load(open(R + 'kept.json'))
A = json.load(open(R + 'A2_h5.json')); D = json.load(open(R + 'R_duffing.json'))
for f in ('load_excerpt', 'free_run', 'half_sample_input'):
    src = inspect.getsource(getattr(sp, f)).splitlines()
    print('\n'.join(src[:28])); print('---')
for l in open(R + 'summary.md', encoding='utf-8'):
    if 'A2_h5' in l or 'R_duffing' in l: print('SUM:', l.strip()[:200])
print('A2_h5.json:', json.dumps(hist_free(A))[:1200])
print('R_duffing.json:', json.dumps(hist_free(D))[:900])
print('kept.json:', json.dumps(kept)[:1500])
print('load_excerpt sig', inspect.signature(sp.load_excerpt), 'free_run sig', inspect.signature(sp.free_run))
new = {}
try:
    ex = sp.load_excerpt()
    print('excerpt', type(ex), [getattr(x, 'shape', type(x).__name__) for x in (ex if isinstance(ex, (tuple, list)) else [ex])])
    u, y = ex[0], ex[1]
    uh = sp.half_sample_input(u)
    for nm, pt in (('A2_h5', R + 'A2_h5.pt'), ('R_duffing', R + 'R_duffing.pt')):
        t0 = time.time()
        f, cfg = sp.load_model(pt)
        yhat, tr, te = sp.free_run(f, u, y, uh)
        assert math.isfinite(tr) and math.isfinite(te)
        new[nm] = (tr, te)
        print(nm, 'TRAIN_RMSE=%.6g TEST_RMSE=%.6g' % (tr, te), 'sec', round(time.time() - t0, 1))
except Exception:
    traceback.print_exc(file=sys.stdout)
for nm, d in (('A2_h5', A), ('R_duffing', D)):
    if nm in new:
        print(nm, 'json rmse_train/test', d.get('rmse_train'), d.get('rmse_test'), '| new', new[nm])
        for k, o, n in (('train', d.get('rmse_train'), new[nm][0]), ('test', d.get('rmse_test'), new[nm][1])):
            if o: print('  ', k, 'abs diff', n - o, 'rel %.4f%%' % (100 * (n - o) / o), 'FLAG' if abs(n - o) / o > 0.01 else 'ok')
if 'A2_h5' in new: print('TRAIN_RMSE=%.6g' % new['A2_h5'][0]); print('TEST_RMSE=%.6g' % new['A2_h5'][1])
print('kept.json unchanged (never opened for writing)')
