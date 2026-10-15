import sys, inspect
sys.path.insert(0, 'silverbox')
import silverbox_pinn as m
for n in ['load_excerpt', 'load_model', 'free_run', 'half_sample_input']:
    print('=' * 20, n)
    try:
        print(inspect.getsource(getattr(m, n)))
    except Exception as e:
        print('ERR', repr(e))
