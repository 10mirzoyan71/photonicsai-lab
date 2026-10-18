import numpy as np

FILES = ["camera/cap001_dark_batch1.npy", "camera/cap002.npy"]

H, W = None, None
S = None
S2 = None
frame_means = []
batch_sum = {}
batch_sumsq = {}
batch_n = {}
n_total = 0

for bi, path in enumerate(FILES):
    arr = np.load(path, mmap_mode="r")
    print(f"SHAPE_BATCH{bi+1}={arr.shape}")
    if S is None:
        H, W = arr.shape[1], arr.shape[2]
        S = np.zeros((H, W), dtype=np.float64)
        S2 = np.zeros((H, W), dtype=np.float64)
    batch_sum[bi] = 0.0
    batch_sumsq[bi] = 0.0
    batch_n[bi] = 0
    for k in range(arr.shape[0]):
        x = np.asarray(arr[k], dtype=np.float64)
        S += x
        S2 += x * x
        frame_means.append(x.mean())
        batch_sum[bi] += x.sum()
        batch_sumsq[bi] += (x * x).sum()
        batch_n[bi] += x.size
        n_total += 1

N = n_total
print(f"N_FRAMES={N}")
print(f"FRAME_SHAPE={H}x{W}")

total_sum = batch_sum[0] + batch_sum[1]
total_sumsq = batch_sumsq[0] + batch_sumsq[1]
total_n = batch_n[0] + batch_n[1]
mean_all = total_sum / total_n
std_all = np.sqrt(total_sumsq / total_n - mean_all ** 2)
print(f"mean_DN={mean_all:.4f}")
print(f"std_DN={std_all:.4f}")

m1 = batch_sum[0] / batch_n[0]
s1 = np.sqrt(batch_sumsq[0] / batch_n[0] - m1 ** 2)
m2 = batch_sum[1] / batch_n[1]
s2 = np.sqrt(batch_sumsq[1] / batch_n[1] - m2 ** 2)
print(f"batch1_n_frames=16 batch1_mean_DN={m1:.4f} batch1_std_DN={s1:.4f}")
print(f"batch2_n_frames=14 batch2_mean_DN={m2:.4f} batch2_std_DN={s2:.4f}")

pix_mean = S / N
pix_var = np.maximum(S2 / N - pix_mean ** 2, 0.0)
pix_tstd = np.sqrt(pix_var)
print(f"pixel_mean_std_DN={pix_mean.std():.4f}")
print(f"pixel_mean_min_DN={pix_mean.min():.4f}")
print(f"pixel_mean_max_DN={pix_mean.max():.4f}")
print(f"per_pixel_temporal_std_mean_DN={pix_tstd.mean():.4f}")
p1, p99 = np.percentile(pix_mean, [1, 99])
print(f"pixel_mean_p1_DN={p1:.4f}")
print(f"pixel_mean_p99_DN={p99:.4f}")

diff = m2 - m1
se = std_all * np.sqrt(1.0 / 16 + 1.0 / 14)
print(f"batch2_minus_batch1_DN={diff:.4f}")
print(f"se_diff_DN={se:.4f}")
print(f"diff_over_se={diff / se:.3f}")

print("FRAME_MEANS=" + ",".join(f"{v:.2f}" for v in frame_means))
