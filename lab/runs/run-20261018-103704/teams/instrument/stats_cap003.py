import numpy as np

FULL_SCALE = 4095
arr = np.load("camera/cap003.npy")
frame = arr[0] if arr.ndim == 3 else arr
frame = frame.astype(np.float64)

mean_dn = frame.mean()
median_dn = np.median(frame)
max_dn = frame.max()
saturated_pixels = int(np.count_nonzero(frame == FULL_SCALE))
saturated_fraction = saturated_pixels / frame.size
mean_fraction_of_full_scale = mean_dn / FULL_SCALE

print(f"MEAN_DN={mean_dn:.2f}")
print(f"MEDIAN_DN={median_dn:.2f}")
print(f"MAX_DN={max_dn:.0f}")
print(f"SATURATED_PIXELS={saturated_pixels}")
print(f"SATURATED_FRACTION={saturated_fraction:.6f}")
print(f"MEAN_FRACTION_OF_FULL_SCALE={mean_fraction_of_full_scale:.4f}")
