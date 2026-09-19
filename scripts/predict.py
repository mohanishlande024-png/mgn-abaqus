r"""
Predict von Mises stress for a load the model was never trained on.

    # a new load on a geometry that is already in the dataset
    python scripts/predict.py --geometry circle8 --load 7250

    # several loads at once
    python scripts/predict.py --geometry circle8 --load 7250 12000 15000

    # a mesh that is not in dataset.pt at all (from extract_odb.py)
    python scripts/predict.py --npz data/raw/newshape__5000.npz --load 7250

    # compare against Abaqus truth stored in the npz
    python scripts/predict.py --npz data/raw/circle8__7499.npz --load 7499 --compare

WHAT "UNSEEN LOAD" MEANS HERE
    The mesh gives the graph: node positions, edges, node types. The load
    enters only as a GLOBAL FEATURE, re-injected at every message-passing
    layer. So predicting a new load on a known mesh needs no Abaqus run at
    all - only a number.

    A new GEOMETRY is different. There is no mesh until Abaqus makes one,
    so extract_odb.py must run first and produce a .npz. The stress array
    in that .npz is ignored here (a zero array is fine).

OUTPUT (into results/predictions/)
    <geometry>__<load>.csv      x, y, node_type, predicted_von_mises
    <geometry>__<load>.png      field plot, unless --no-plots

CAUTION ON EXTRAPOLATION
    The model saw loads inside one range only. A load far outside it is an
    extrapolation, and this script warns when you ask for one. Linear
    elasticity says stress scales linearly with load, but the network does
    not know that - it only knows what it was shown.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case, load_dataset          # noqa: E402
from mgn.graph import NODE_TYPE_TO_ID, build_edges, classify_nodes  # noqa: E402
from mgn.trainer import MGN                         # noqa: E402

OUT = os.path.join("results", "predictions")
ID_TO_NAME = dict((v, k) for k, v in NODE_TYPE_TO_ID.items())


def case_from_dataset(path, geometry):
    """Reuse a mesh already compiled into dataset.pt."""
    cases = load_dataset(path)
    names = sorted(set(c.geometry for c in cases))
    match = [c for c in cases if c.geometry == geometry]
    if not match:
        print("No geometry '%s' in %s." % (geometry, path))
        print("Available: %s" % ", ".join(names))
        return None, None
    loads = sorted(set(c.metadata["load"] for c in match))
    return match[0], loads


def case_from_npz(path):
    """Build a mesh straight from an extract_odb.py export."""
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    truth = d["mises"].astype(np.float32) if "mises" in d else None
    geom = os.path.splitext(os.path.basename(path))[0].split("__")[0]

    case = Case(
        geometry=geom,
        coordinates=coords,
        edge_index=torch.from_numpy(build_edges(tris)),
        node_types=classify_nodes(coords, tris),
        von_mises=np.zeros(len(coords), dtype=np.float32),
        metadata={"load": 0.0},
    )
    return case, truth


def plot(case, pred, load, path):
    import matplotlib
    matplotlib.use("Agg")          # no display over SSH
    import matplotlib.pyplot as plt

    x, y = case.coordinates[:, 0], case.coordinates[:, 1]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    s = ax.scatter(x, y, c=pred, s=9, cmap="jet")
    peak = int(np.argmax(pred))
    ax.plot(x[peak], y[peak], "o", mfc="none", mec="red", ms=16, mew=2)
    ax.set_aspect("equal")
    ax.set_title("%s  at %.0f psi   (peak %.0f psi, red circle)"
                 % (case.geometry, load, pred[peak]))
    fig.colorbar(s, ax=ax, label="predicted von Mises (psi)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--geometry", help="name of a mesh already in dataset.pt")
    src.add_argument("--npz", help="path to an extract_odb.py export")
    ap.add_argument("--load", type=float, nargs="+", required=True,
                    help="one or more load values in psi")
    ap.add_argument("--ckpt", default=os.path.join("results", "multi_geometry_mgn.pt"),
                    help="a model saved by MGN.save()")
    ap.add_argument("--data", default=os.path.join("data", "dataset.pt"))
    ap.add_argument("--compare", action="store_true",
                    help="score against the truth stored in the npz")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.ckpt):
        print("No model at %s" % a.ckpt)
        print("train.py writes that file only when the full run finishes.")
        print("An interrupted run leaves results/mgn_latest.pt instead, which")
        print("is a different format and cannot be loaded here.")
        return 1

    # ---- mesh ----------------------------------------------------------
    truth = None
    if a.geometry:
        case, trained_loads = case_from_dataset(a.data, a.geometry)
        if case is None:
            return 1
    else:
        if not os.path.isfile(a.npz):
            print("No such file: %s" % a.npz)
            return 1
        case, truth = case_from_npz(a.npz)
        trained_loads = None

    counts = {}
    for t in case.node_types:
        n = ID_TO_NAME[int(t)]
        counts[n] = counts.get(n, 0) + 1

    print("=" * 62)
    print("geometry      %s" % case.geometry)
    print("nodes         %d" % len(case.coordinates))
    print("edges         %d" % case.edge_index.shape[1])
    print("node types    %s" % ", ".join("%s=%d" % (k, counts[k])
                                         for k in sorted(counts)))
    if trained_loads:
        print("trained loads %.0f to %.0f psi  (%d of them)"
              % (min(trained_loads), max(trained_loads), len(trained_loads)))
    print("=" * 62)

    mgn = MGN.load(a.ckpt)

    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    # ---- predict -------------------------------------------------------
    for load in a.load:
        if trained_loads and not (min(trained_loads) <= load <= max(trained_loads)):
            print("\nWARNING: %.0f psi is outside the trained range "
                  "%.0f to %.0f. This is extrapolation."
                  % (load, min(trained_loads), max(trained_loads)))

        case.metadata = {"load": float(load)}
        pred = mgn.predict(case).squeeze()

        stem = "%s__%.0f" % (case.geometry, load)
        csv = os.path.join(OUT, stem + ".csv")
        with open(csv, "w") as f:
            f.write("x,y,node_type,predicted_von_mises_psi\n")
            for i in range(len(pred)):
                f.write("%.6f,%.6f,%s,%.4f\n"
                        % (case.coordinates[i, 0], case.coordinates[i, 1],
                           ID_TO_NAME[int(case.node_types[i])], pred[i]))

        peak = int(np.argmax(pred))
        print("\n%.0f psi" % load)
        print("  mean      %10.1f psi" % pred.mean())
        print("  peak      %10.1f psi  at (%.3f, %.3f)  node type %s"
              % (pred[peak], case.coordinates[peak, 0],
                 case.coordinates[peak, 1], ID_TO_NAME[int(case.node_types[peak])]))
        print("  min       %10.1f psi" % pred.min())
        print("  csv       %s" % csv)

        if a.compare and truth is not None:
            ss_res = ((truth - pred) ** 2).sum()
            ss_tot = ((truth - truth.mean()) ** 2).sum()
            print("  R2 vs Abaqus       %.6f" % (1 - ss_res / ss_tot))
            print("  RMSE               %.1f psi"
                  % np.sqrt(((truth - pred) ** 2).mean()))
            print("  peak truth         %.1f psi (predicted %.1f)"
                  % (truth.max(), pred[peak]))

        if not a.no_plots:
            png = os.path.join(OUT, stem + ".png")
            plot(case, pred, load, png)
            print("  plot      %s" % png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
