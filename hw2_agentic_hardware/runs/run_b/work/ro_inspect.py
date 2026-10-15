import json, sys
from pathlib import Path
r = Path('silverbox/results')
for n in ['kept.json', 'A2_h5.json', 'R_duffing.json']:
    print('=====', n)
    print((r/n).read_text())
print('===== summary.md')
print((r/'summary.md').read_text())
print('===== silverbox_pinn.py')
print(Path('silverbox/silverbox_pinn.py').read_text())
