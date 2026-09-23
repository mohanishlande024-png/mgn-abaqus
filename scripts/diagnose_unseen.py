r"""
Where does the unseen-geometry error come from?

    python scripts/diagnose_unseen.py --ckpt results_aug20_L8/multi_geometry_mgn.pt
    python scripts/diagnose_unseen.py --ckpt results/multi_geometry_mgn.pt

Two tables, one per question:

1. IS IT JUST A FEW PEAK NODES?
   R2 on all nodes vs R2 with the top 1% of stress values removed. If removing
   them fixes R2, the "failure" is a handful of (possibly mesh-dependent) peak
   nodes. If R2 stays low, the whole field is wrong.

2. NEAR THE HOLE OR FAR AWAY?
   Share of the total squared error in distance bands from the hole boundary.
       mostly < 1 in   -> LOCAL problem: the model cannot read the geometry of
                          this boundary. More depth / more shortcut edges will
                          not help; node geometry features and training data will.
       mostly > 3 in   -> FAR-FIELD problem: the whole-plate stress pattern is
                          wrong. This is where propagation (depth, augmentation)
                          matters.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case, build_topology, _read_name   # noqa: E402
from mgn.augmented_trainer import AugMGN                    # noqa: E402
from mgn.graph import NODE_TYPE_TO_ID                       # noqa: E402

HOLE = NODE_TYPE_TO_ID["hole"]
BINS = [0, 1, 3, 6, 15, 1e9]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a multi_geometry_mgn.pt file")
    ap.add_argument("--npz-dir", default="unseen_npz")
    a = ap.parse_args()

    m = AugMGN.load(a.ckpt)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.npz_dir, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        co = d["coords"].astype(np.float32)
        tr = d["tris"].astype(np.int64)
        y = d["mises"].astype(np.float32)
        g = _read_name(d["geometry"])
        ei, nt, fl = build_topology(co, tr, g, aug_perc=m.aug_perc, aug_seed=m.aug_seed)
        p = m.predict(Case(g, co, ei, nt, y, {"load": float(d["load"])}, fl))
        rows.append((g.replace("plate_", ""), co, nt, y, p))

    print("\n1. IS IT JUST A FEW PEAK NODES?")
    print("%-16s %8s %11s %10s %10s" % ("geometry", "R2 all", "R2 -top1%", "peak true", "peak pred"))
    for g, co, nt, y, p in rows:
        top = y >= np.quantile(y, 0.99)
        print("%-16s %8.3f %11.3f %10.0f %10.0f"
              % (g, r2(y, p), r2(y[~top], p[~top]), y.max(), p.max()))

    print("\n2. NEAR THE HOLE OR FAR AWAY?  (share of total squared error)")
    labels = ["<1 in", "1-3", "3-6", "6-15", ">15"]
    print("%-16s" % "geometry" + "".join("%8s" % s for s in labels) + "   verdict")
    for g, co, nt, y, p in rows:
        dist = cKDTree(co[nt == HOLE]).query(co)[0]
        res = (y - p) ** 2
        sh = [100 * res[(dist >= lo) & (dist < hi)].sum() / res.sum()
              for lo, hi in zip(BINS[:-1], BINS[1:])]
        verdict = ("LOCAL" if sh[0] >= 60 else
                   "FAR-FIELD" if sum(sh[2:]) >= 50 else "mixed")
        print("%-16s" % g + "".join("%7.0f%%" % s for s in sh) + "   " + verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
