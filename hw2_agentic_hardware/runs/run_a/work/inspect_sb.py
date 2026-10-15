import re
src = open('silverbox/silverbox_pinn.py').read()
for name in ['load_excerpt', 'load_model', 'free_run', 'half_sample_input']:
    m = re.search(r'^def %s\(.*?(?=^def |\Z)' % name, src, re.S | re.M)
    print('=====', name)
    print(m.group(0) if m else 'NOT FOUND')
print('===== A2_h5.json')
print(open('silverbox/results/A2_h5.json').read())
print('===== kept.json')
print(open('silverbox/results/kept.json').read())
