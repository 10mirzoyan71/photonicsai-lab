import os
import numpy as np

CAM = os.path.join(os.getcwd(), "camera")
FULL_MAX = 4095


def frames_of(arr):
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError("unexpected ndim %d" % arr.ndim)


def frame_stats(img, tag):
    img = np.asarray(img)
    total = img.size
    mean = float(img.astype(np.float64).mean())
    p999 = float(np.percentile(img.astype(np.float64), 99.9))
    mx = int(img.max())
    n4095 = int(np.count_nonzero(img == FULL_MAX))
    sat = float(np.count_nonzero(img >= FULL_MAX)) / total
    print("%s_MEAN_DN=%.4f" % (tag, mean))
    print("%s_P999_DN=%.4f" % (tag, p999))
    print("%s_MAX_DN=%d" % (tag, mx))
    print("%s_COUNT_4095=%d" % (tag, n4095))
    print("%s_SAT_FRAC=%.6f" % (tag, sat))


def load(name):
    path = os.path.join(CAM, name)
    arr = np.load(path)
    print("%s_SHAPE=%s" % (name.replace(".", "_"), str(arr.shape)))
    print("%s_DTYPE=%s" % (name.replace(".", "_"), str(arr.dtype)))
    return arr


# ---- cap006_verify_34200us.npy (expected 5 frames at 34200 us) ----
a6 = load("cap006_verify_34200us.npy")
f6 = frames_of(a6)
per_means = []
for i, fr in enumerate(f6):
    print("CAP006_FRAME_INDEX=%d" % i)
    frame_stats(fr, "CAP006_F%d" % i)
    per_means.append(float(np.asarray(fr).astype(np.float64).mean()))

allpix = np.concatenate([np.asarray(fr).ravel() for fr in f6])
agg_mean = float(allpix.astype(np.float64).mean())
agg_p999 = float(np.percentile(allpix.astype(np.float64), 99.9))
agg_max = int(allpix.max())
print("CAP006_NFRAMES=%d" % len(f6))
print("CAP006_AGG_MEAN_DN=%.4f" % agg_mean)
print("CAP006_AGG_P999_DN=%.4f" % agg_p999)
print("CAP006_AGG_MAX_DN=%d" % agg_max)
print("CAP006_MEAN_OF_FRAME_MEANS_DN=%.4f" % float(np.mean(per_means)))

# ---- cap005.npy (1 frame at 34200 us) ----
a5 = load("cap005.npy")
f5 = frames_of(a5)
for i, fr in enumerate(f5):
    print("CAP005_FRAME_INDEX=%d" % i)
    frame_stats(fr, "CAP005_F%d" % i)

# ---- restore frames at 1000 us ----
for name, tag in [("cap007.npy", "CAP007"), ("cap008.npy", "CAP008")]:
    a = load(name)
    fs = frames_of(a)
    for i, fr in enumerate(fs):
        print("%s_FRAME_INDEX=%d" % (tag, i))
        frame_stats(fr, "%s_F%d" % (tag, i))
