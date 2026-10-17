import json
print(open('silverbox/results/kept.json').read())
src = open('silverbox/silverbox_pinn.py').read().splitlines()
for i, line in enumerate(src):
    if any(k in line for k in ['def ', 'class ', 'n_a', 'n_b', 'lag', 'test', 'split', 'segment', 'excerpt', 'interp', 'save', 'load']):
        print(f'{i+1}: {line}')
