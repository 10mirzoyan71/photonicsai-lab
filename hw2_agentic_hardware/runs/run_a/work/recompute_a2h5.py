import sys, os, math
sys.path.insert(0, 'silverbox')
import numpy as np
import silverbox_pinn as sp

u, y = sp.load_excerpt('data/SNLS80mV.mat')
f, cfg = sp.load_model('silverbox/results/A2_h5.pt')
kind = getattr(cfg, 'interp', None)
if kind is None:
    for name in ('interpolation', 'interp_kind', 'half_interp', 'kind'):
        if hasattr(cfg, name):
            kind = getattr(cfg, name); break
print('cfg interp kind found:', kind)
if kind is None:
    kind = 'sinc'
uh = sp.half_sample_input(u, kind)
yhat, rtr, rte = sp.free_run(f, u, y, uh)
st_te = 0.00042225465297085084
st_tr = 0.00040721431617798747
print('len(u)=', len(u))
print('yhat finite:', bool(np.all(np.isfinite(np.asarray(yhat)))))
print('TRAIN_RMSE=%r' % float(rtr))
print('TEST_RMSE=%r' % float(rte))
print('stored train', st_tr, 'abs diff', abs(float(rtr) - st_tr))
print('stored test', st_te, 'abs diff', abs(float(rte) - st_te))
