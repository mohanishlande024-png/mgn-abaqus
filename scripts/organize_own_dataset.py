r"""
Sort the solved own-dataset files into training and test folders.

    python scripts\organize_own_dataset.py

INPUT
    data\raw_own_solved\plate_own_<name>__<load>.npz   every geometry at every load
    data\odb_own\own_manifest.csv                       train / test labels (from Abaqus)

OUTPUT - files are COPIED exactly as extract_odb.py wrote them, nothing is scaled
    data\raw_own_train\      27 training geometries x 20 training loads   (540 files)
    data\unseen_own_loads\   27 training geometries x  3 test loads       ( 81 files)
    data\unseen_own_interp\  13 interpolation-test geometries x all 23 loads
    data\unseen_own_extrap\   5 extrapolation-test geometries x all 23 loads

The three tests:
    unseen loads         training geometries at 800 / 5000 / 12000 psi  -> data\unseen_own_loads
    unseen geometries    new shapes inside each family's range          -> data\unseen_own_interp
    extrapolation        each family's extreme end                      -> data\unseen_own_extrap
Every test folder works with fig3_unseen.py and diagnose_unseen.py (--npz-dir).
The paper evaluates unseen geometries at 5000 psi: use fig3_unseen.py --load 5000.

CHECKS, before anything is written
    1. every geometry x load in the plan has a solved file (missing ones are listed)
    2. QUALITY GATE: the a = b circle (E_H1, d/W = 0.30) against the Peterson
       finite-width formula - more than 5% off means the hole mesh is too coarse
    3. LINEARITY: every geometry's solves must be exact multiples of its 1000 psi
       solve. They should be (linear elastic, no prescribed displacement); if not,
       something in the model is nonlinear and the load feature means something else.
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import numpy as np

TRAIN_LOADS = [1000, 1388, 1777, 2166, 2555, 2944, 3333, 3722, 4111, 4500,
               5500, 5999, 6499, 6999, 7499, 7999, 8499, 8999, 9499, 10000]   # the paper's
TEST_LOADS = [800, 5000, 12000]                                                # the paper's
ALL_LOADS = sorted(set(TRAIN_LOADS + TEST_LOADS))
PLATE_D = 10.0          # plate height = the "D" of the Kt formula, in


def peterson_peak(d, D=PLATE_D, gross=1000.0):
    """Peak stress at a centred circular hole in a finite-width plate, tension."""
    r = d / D
    kt = 3.000 - 3.140 * r + 3.667 * r ** 2 - 1.527 * r ** 3
    return kt * gross * D / (D - d)                 # Kt applies to the net-section stress


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solved", default=os.path.join("data", "raw_own_solved"))
    ap.add_argument("--manifest", default=os.path.join("data", "odb_own", "own_manifest.csv"))
    ap.add_argument("--out", default="data", help="parent folder of the four output folders")
    ap.add_argument("--skip-quality-gate", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.manifest):
        print("No manifest at %s - it is written by abaqus/generate_own_dataset.py" % a.manifest)
        return 1
    with open(a.manifest) as fh:
        rows = list(csv.DictReader(fh))

    solved = {}
    for f in glob.glob(os.path.join(a.solved, "*.npz")):
        geom, load = os.path.splitext(os.path.basename(f))[0].rsplit("__", 1)
        solved[(geom, int(float(load)))] = f

    # ---- 1. completeness -----------------------------------------------------
    missing = [(r["geometry"], L) for r in rows for L in ALL_LOADS if (r["geometry"], L) not in solved]
    if missing:
        print("%d of %d solves are missing from %s:" % (len(missing), len(rows) * len(ALL_LOADS), a.solved))
        for g, L in missing[:25]:
            print("   %s__%d" % (g, L))
        if len(missing) > 25:
            print("   ... and %d more" % (len(missing) - 25))
        print("Re-run generate_own_dataset.py (it resumes), then extract_odb.py --out %s." % a.solved)
        return 1
    print("completeness  %d geometries x %d loads = %d solves, all present"
          % (len(rows), len(ALL_LOADS), len(rows) * len(ALL_LOADS)))

    # ---- 2. quality gate -----------------------------------------------------
    circ = [r for r in rows if r["name"] == "E_H1"]
    if circ:
        d = np.load(solved[(circ[0]["geometry"], 1000)], allow_pickle=False)
        got, want = float(d["mises"].max()), peterson_peak(2 * float(circ[0]["a_in"]))
        print("quality gate  circle d = 3 in at 1000 psi: peak %.0f psi vs Peterson %.0f psi -> %.1f%%"
              % (got, want, 100 * got / want))
        if abs(got / want - 1) > 0.05 and not a.skip_quality_gate:
            print("   More than 5% off: the hole mesh is not resolving the peak. Refine HOLE_SEED")
            print("   in generate_own_dataset.py and re-run, or pass --skip-quality-gate.")
            return 1

    # ---- 3. linearity ----------------------------------------------------------
    worst, worst_at = 0.0, ""
    for r in rows:
        ref = np.load(solved[(r["geometry"], 1000)], allow_pickle=False)["mises"].astype(np.float64)
        for L in ALL_LOADS:
            m = np.load(solved[(r["geometry"], L)], allow_pickle=False)["mises"].astype(np.float64)
            if m.shape != ref.shape:
                print("   %s: mesh at %d psi differs from 1000 psi - loads must share one mesh"
                      % (r["geometry"], L))
                return 1
            dev = float(np.abs(m - ref * L / 1000.0).max() / max(np.abs(m).max(), 1e-12))
            if dev > worst:
                worst, worst_at = dev, "%s at %d psi" % (r["geometry"], L)
    print("linearity     largest deviation from exact scaling: %.2e (%s)" % (worst, worst_at or "-"))
    if worst > 1e-3:
        print("   WARNING: solves are not proportional to the load. Check the model is linear.")

    # ---- write -----------------------------------------------------------------
    dirs = {"train": "raw_own_train", "loads": "unseen_own_loads",
            "test_interp": "unseen_own_interp", "test_extrap": "unseen_own_extrap"}
    dirs = dict((k, os.path.join(a.out, v)) for k, v in dirs.items())
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    n = dict((k, 0) for k in dirs)
    for r in rows:
        g = r["geometry"]
        if r["split"] == "train":
            plan = [(L, "train") for L in TRAIN_LOADS] + [(L, "loads") for L in TEST_LOADS]
        else:
            plan = [(L, r["split"]) for L in ALL_LOADS]
        for L, where in plan:
            shutil.copy2(solved[(g, L)], dirs[where])
            n[where] += 1

    print("\n%-22s %4d files  (27 geometries x 20 training loads)" % (dirs["train"], n["train"]))
    print("%-22s %4d files  (training geometries at 800 / 5000 / 12000 psi)" % (dirs["loads"], n["loads"]))
    print("%-22s %4d files  (interpolation geometries x 23 loads)" % (dirs["test_interp"], n["test_interp"]))
    print("%-22s %4d files  (extrapolation geometries x 23 loads)" % (dirs["test_extrap"], n["test_extrap"]))
    print("\nNext:")
    print("   python -m mgn.dataset --raw %s --aug-perc 0.2 --out data/dataset_own_aug20.pt" % dirs["train"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
