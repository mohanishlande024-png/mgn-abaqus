r"""
Check edge augmentation on real meshes BEFORE training.

    python scripts\check_augment.py                       every .npz in data\raw
    python scripts\check_augment.py --raw unseen_npz      the 7 unseen meshes
    python scripts\check_augment.py --aug-perc 0.2 --aug-seed 0

For every geometry it builds the baseline graph and the augmented graph with
mgn.dataset.build_topology() - the same function training and inference use -
and checks:

    PASS/FAIL  node types identical to the baseline (augmentation must not touch them)
    PASS/FAIL  real edges unchanged and first in edge_index
    PASS/FAIL  augmented count == round(perc * real undirected edges)
    PASS/FAIL  no self-loops, no duplicates, no pair that is already a mesh edge
    PASS/FAIL  every augmented edge exists in both directions
    PASS/FAIL  deterministic: a second build gives byte-identical arrays

and prints what augmentation does to the graph:

    degree     mean neighbours before/after, % nodes that got no shortcut
    length     mean edge length of real vs augmented edges (why the two are
               normalised separately in mgn/augmented_trainer.py)

Exit code is 1 if any check fails.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import _read_name, build_topology    # noqa: E402


def check_one(path, perc, seed):
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    name = (_read_name(d["geometry"]) if "geometry" in d
            else os.path.basename(path).split("__")[0])
    n = len(coords)

    ei0, nt0, fl0 = build_topology(coords, tris, name, aug_perc=0.0)
    ei, nt, fl = build_topology(coords, tris, name, aug_perc=perc, aug_seed=seed)
    ei2, _, fl2 = build_topology(coords, tris, name, aug_perc=perc, aug_seed=seed)
    ei0, ei, ei2 = ei0.numpy(), ei.numpy(), ei2.numpy()
    fl0, fl, fl2 = fl0.numpy(), fl.numpy(), fl2.numpy()

    e_real = ei0.shape[1]
    real_und = e_real // 2
    aug_mask = fl > 0.5
    aug = ei[:, aug_mask]
    n_aug_und = aug.shape[1] // 2

    und_real = set(map(tuple, np.sort(ei0.T, axis=1).tolist()))
    und_aug = [tuple(p) for p in np.sort(aug.T, axis=1).tolist()]
    directed = set(map(tuple, aug.T.tolist()))

    checks = [
        ("node types unchanged", np.array_equal(nt0, nt)),
        ("real edges unchanged, first",
         np.array_equal(ei[:, :e_real], ei0) and not aug_mask[:e_real].any()),
        ("augmented count", n_aug_und == int(round(perc * real_und))),
        ("no self-loops", bool((aug[0] != aug[1]).all())),
        ("no overlap with mesh", not any(p in und_real for p in und_aug)),
        ("no duplicates", len(set(und_aug)) * 2 == aug.shape[1]),
        ("both directions", all((b, a) in directed for (a, b) in directed)),
        ("deterministic", np.array_equal(ei, ei2) and np.array_equal(fl, fl2)),
    ]

    deg0 = np.bincount(ei0[1], minlength=n)
    deg_extra = np.bincount(aug[1], minlength=n)
    L = np.linalg.norm(coords[ei[1]] - coords[ei[0]], axis=1)

    ok = all(c for _, c in checks)
    print("\n%s  %s   N=%d" % ("PASS" if ok else "FAIL", name, n))
    for label, c in checks:
        print("    %-30s %s" % (label, "ok" if c else "** FAILED **"))
    print("    edges      %d real + %d augmented = %d  (+%.1f%%)"
          % (e_real, aug.shape[1], ei.shape[1], 100.0 * aug.shape[1] / e_real))
    print("    degree     %.2f -> %.2f mean neighbours;  %.0f%% of nodes got no "
          "shortcut, %.0f%% got 3+"
          % (deg0.mean(), (deg0 + deg_extra).mean(),
             100 * (deg_extra == 0).mean(), 100 * (deg_extra >= 3).mean()))
    print("    length     real %.3f +/- %.3f in   augmented %.2f +/- %.2f in"
          % (L[~aug_mask].mean(), L[~aug_mask].std(),
             L[aug_mask].mean(), L[aug_mask].std()))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--aug-perc", type=float, default=0.2)
    ap.add_argument("--aug-seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "*.npz")))
    if not files:
        print("No .npz files in %s" % a.raw)
        return 1

    # one mesh per geometry is enough - load cases share it
    seen, results = set(), []
    for f in files:
        key = os.path.basename(f).split("__")[0]
        if key in seen:
            continue
        seen.add(key)
        results.append(check_one(f, a.aug_perc, a.aug_seed))

    print("\n%d/%d geometries passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
