r"""
Draw the graph MGN will actually see.

Two panels per geometry:
  left   the graph itself - every edge, nodes coloured by type
  right  the von Mises field, to confirm the physics looks sane

Needs numpy + matplotlib only. No torch. Run from the project root:

    py scripts\plot_graph.py                          # all geometries
    py scripts\plot_graph.py data\raw\plate_circle8__1000.npz
    py scripts\plot_graph.py --no-edges               # faster, nodes only

PNGs land in data\plots\.

This is the strongest check available before training: if 'hole' nodes are
not sitting on the hole boundary, or 'fixed' is not the left edge, you will
see it instantly here even though the histogram looked fine.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402
from matplotlib.tri import Triangulation     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.graph import (                      # noqa: E402
    NODE_TYPE_TO_ID,
    build_edges,
    classify_nodes,
    edge_features,
)

# distinct, colour-blind-safe-ish, and 'hole' deliberately loud
COLOURS = {
    "applied_load": "#d62728",
    "fixed":        "#1f77b4",
    "free":         "#7f7f7f",
    "hole":         "#ff7f0e",
    "interior":     "#c7c7c7",
}
ID_TO_NAME = {v: k for k, v in NODE_TYPE_TO_ID.items()}


def plot_one(path, out_dir, draw_edges=True):
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    mises = d["mises"].astype(np.float64)
    load = float(d["load"])
    name = os.path.splitext(os.path.basename(path))[0]

    edge_index = build_edges(tris)
    ids = classify_nodes(coords, tris)
    ef = edge_features(coords, edge_index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # ---- panel 1: the graph -------------------------------------------
    if draw_edges:
        # undirected half is enough to draw
        half = edge_index[:, :edge_index.shape[1] // 2]
        segs = np.stack([coords[half[0]], coords[half[1]]], axis=1)
        ax1.add_collection(LineCollection(segs, colors="#dddddd",
                                          linewidths=0.4, zorder=1))

    for tid in sorted(ID_TO_NAME):
        m = ids == tid
        if not m.any():
            continue
        nm = ID_TO_NAME[tid]
        ax1.scatter(coords[m, 0], coords[m, 1], s=14 if nm != "interior" else 4,
                    c=COLOURS[nm], label="%s (%d)" % (nm, int(m.sum())),
                    zorder=3 if nm != "interior" else 2, edgecolors="none")

    ax1.set_title("%s  -  graph MGN sees: %d nodes, %d directed edges"
                  % (name, len(coords), edge_index.shape[1]))
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (in)")
    ax1.set_ylabel("y (in)")

    # ---- panel 2: the target ------------------------------------------
    triang = Triangulation(coords[:, 0], coords[:, 1], tris)
    tpc = ax2.tripcolor(triang, mises, shading="gouraud", cmap="inferno")
    fig.colorbar(tpc, ax=ax2, label="von Mises (psi)", fraction=0.02)
    peak = int(np.argmax(mises))
    ax2.plot(coords[peak, 0], coords[peak, 1], "wo", ms=7, mfc="none", mew=1.5)
    ax2.set_title("von Mises at %.0f psi   -   peak %.1f psi (SCF %.2f) at %s node"
                  % (load, mises.max(), mises.max() / load,
                     ID_TO_NAME[int(ids[peak])]))
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (in)")
    ax2.set_ylabel("y (in)")

    fig.tight_layout()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out = os.path.join(out_dir, name + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("%-30s nodes=%4d edges=%5d  edge_len min/med/max = %.3f/%.3f/%.3f in"
          % (name, len(coords), edge_index.shape[1],
             ef[:, 2].min(), np.median(ef[:, 2]), ef[:, 2].max()))
    print("%-30s peak %.1f psi on a '%s' node  -> %s"
          % ("", mises.max(), ID_TO_NAME[int(ids[peak])], out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", default=os.path.join("data", "plots"))
    ap.add_argument("--no-edges", action="store_true")
    a = ap.parse_args()

    paths = a.paths or sorted(glob.glob(os.path.join("data", "raw", "*.npz")))
    if not paths:
        print("No .npz found. Run extract_odb.py first.")
        return 1
    for p in paths:
        plot_one(p, a.out, draw_edges=not a.no_edges)
    print("\nWrote %d plot(s) to %s" % (len(paths), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
