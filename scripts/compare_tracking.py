"""Compare two tracking pickles (e.g. YOLO+MediaPipe vs DWPose single pass).

Usage:
    python compare_tracking.py <tracking_A.pickle> <tracking_B.pickle> [offset_B]

Prints the per-limb position disagreement on the common time range, so a
DWPose run can be validated against the existing MediaPipe tracking before
switching pipelines. Both files use the format written by extract_tracking.
offset_B (seconds, optional) shifts B's time base — useful when B comes
from a clip cut from the original video (e.g. ffmpeg -ss 8 -> offset 8).
"""
import pickle
import sys

import numpy as np

LIMBS = ['LH', 'RH', 'LF', 'RF', 'H']


def main(path_a, path_b, offset_b=0.0):
    with open(path_a, 'rb') as f:
        a = pickle.load(f)
    with open(path_b, 'rb') as f:
        b = pickle.load(f)
    ta = np.asarray(a['Time (s)'])
    tb = np.asarray(b['Time (s)']) + offset_b
    print(f"A: {path_a}  {len(ta)} frames  t {ta[0]:.2f}..{ta[-1]:.2f} s")
    print(f"B: {path_b}  {len(tb)} frames  t {tb[0]:.2f}..{tb[-1]:.2f} s")

    dx = float(np.asarray(a['dx']).ravel()[0])
    lo = max(ta[0], tb[0])
    hi = min(ta[-1], tb[-1])
    print(f"common time range: {lo:.2f}..{hi:.2f} s\n")

    print(f"{'limb':4} {'median err':>10} {'p90':>8} {'cm (median)':>12}")
    for limb in LIMBS:
        xa = np.asarray(a[limb + 'x (pix)'])
        ya = np.asarray(a[limb + 'y (pix)'])
        xb = np.asarray(b[limb + 'x (pix)'])
        yb = np.asarray(b[limb + 'y (pix)'])
        errs = []
        for i, t in enumerate(ta):
            if not (lo <= t <= hi):
                continue
            j = int(np.argmin(np.abs(tb - t)))
            if abs(tb[j] - t) < 0.03:
                errs.append(np.hypot(xa[i] - xb[j], ya[i] - yb[j]))
        errs = np.array(errs)
        if len(errs) == 0:
            print(f"{limb:4} no common frames")
            continue
        print(f"{limb:4} {np.median(errs):10.1f} "
              f"{np.percentile(errs, 90):8.1f} "
              f"{np.median(errs) * dx * 100:12.1f}")
    print("\nrule of thumb: median within ~15 px (7 cm) during movement "
          "means the pipelines are interchangeable for the hold detection.")


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    offset = float(sys.argv[3]) if len(sys.argv) == 4 else 0.0
    main(sys.argv[1], sys.argv[2], offset)
