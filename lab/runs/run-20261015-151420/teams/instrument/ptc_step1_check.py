import os
import numpy as np

names = [
    "cap009_ptc_500us",
    "cap010_ptc_1000us",
    "cap011_ptc_2000us",
    "cap012_ptc_4000us",
    "cap013_ptc_6000us",
    "cap014_ptc_8000us",
    "cap015_ptc_10000us",
    "cap016_ptc_12000us",
    "cap017_ptc_15000us",
]

all_ge_20 = True
any_4095_global = False
for name in names:
    path = os.path.join("camera", name + ".npy")
    arr = np.load(path)
    n = arr.shape[0] if arr.ndim == 3 else 1
    max_code = int(arr.max())
    has_4095 = bool(np.any(arr == 4095))
    if n < 20:
        all_ge_20 = False
    any_4095_global = any_4095_global or has_4095
    print(f"FILE_{name}_SHAPE={arr.shape}")
    print(f"FILE_{name}_DTYPE={arr.dtype}")
    print(f"FILE_{name}_MAX_CODE={max_code}")
    print(f"FILE_{name}_ANY_4095={has_4095}")
    print(f"FRAMES_{name}={n}")

print(f"ALL_GE_20={'true' if all_ge_20 else 'false'}")
print(f"ANY_PIXEL_4095_IN_ANY_FILE={'true' if any_4095_global else 'false'}")
