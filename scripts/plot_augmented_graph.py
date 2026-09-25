r"""
See the edge-augmented graphs - exactly the graphs training uses.

    py scripts\plot_augmented_graph.py                         all 11 training geometries
    py scripts\plot_augmented_graph.py --raw unseen_npz        the 7 unseen geometries
    py scripts\plot_augmented_graph.py --geometry plate_circle8
    py scripts\plot_augmented_graph.py --aug-perc 0.2 --aug-seed 0

Every graph is built with mgn.dataset.build_topology(), the same function the
dataset, predict.py and fig3_unseen.py use, and with the same fraction and seed
you used for data\dataset_aug20.pt (defaults: 0.2 and 0) - so the picture is
the graph the model actually trains on.

Four panels per geometry, one PNG each (default folder data\plots_aug20\):

  1  mesh edges only     the baseline graph, nodes coloured by type
  2  + shortcut edges    the random long-range edges added on top (red)
  3  shortcuts per node  how many shortcut edges touch each node
                         (about a third of nodes get none)
  4  one node's view     a hole-boundary node: its mesh neighbours (1 hop away
                         anyway) vs its shortcut partners (1 hop instead of many)
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import _read_name, build_topology    # noqa: E402
from mgn.graph import NODE_TYPE_TO_ID                 # noqa: E402

COLOURS = {            # same colours as scripts\plot_graph.py
    "applied_load": "#d62728",
    "fixed":        "#1f77b4",
    "free":         "#7f7f7f",
    "hole":         "#ff7f0e",
    "interior":     "#c7c7c7",
}
ID_TO_NAME = {v: k for k, v in NODE_TYPE_TO_ID.items()}
SHORTCUT = "#c0392b"


def undirected(edge_index):
    """Each edge once (the graph stores both directions)."""
    return edge_index[:, edge_index[0] < edge_index[1]]


def segments(coords, pairs):
    return np.stack([coords[pairs[0]], coords[pairs[1]]], axis=1)


def draw_nodes(ax, coords, types, small=False):
    for tid in sorted(ID_TO_NAME):
        m = types == tid
        if not m.any():
            continue
        nm = ID_TO_NAME[tid]
        size = (6 if nm != "interior" else 2) if small else (14 if nm != "interior" else 4)
        ax.scatter(coords[m, 0], coords[m, 1], s=size, c=COLOURS[nm],
                   label="%s (%d)" % (nm, int(m.sum())),
                   zorder=3 if nm != "interior" else 2, edgecolors="none")


def frame(ax, coords, title):
    pad = 1.0
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8)


def plot_one(path, out_dir, perc, seed):
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    name = (_read_name(d["geometry"]) if "geometry" in d
            else os.path.basename(path).split("__")[0])

    ei, types, flag = build_topology(coords, tris, name, aug_perc=perc, aug_seed=seed)
    ei, flag = ei.numpy(), flag.numpy()
    real = undirected(ei[:, flag < 0.5])
    aug = undirected(ei[:, flag > 0.5])
    n = len(coords)

    n_short = np.bincount(aug.ravel(), minlength=n)          # shortcuts per node
    L_real = np.linalg.norm(coords[real[1]] - coords[real[0]], axis=1)
    L_aug = np.linalg.norm(coords[aug[1]] - coords[aug[0]], axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12.5))
    mesh_lines = segments(coords, real)

    # 1 - the baseline graph
    ax = axes[0]
    ax.add_collection(LineCollection(mesh_lines, colors="#cfcfcf", linewidths=0.4, zorder=1))
    draw_nodes(ax, coords, types)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=5)
    frame(ax, coords, "1   Mesh edges only (baseline graph):  %d nodes, %d edges"
          % (n, real.shape[1]))

    # 2 - with shortcuts
    ax = axes[1]
    ax.add_collection(LineCollection(mesh_lines, colors="#e3e3e3", linewidths=0.3, zorder=1))
    ax.add_collection(LineCollection(segments(coords, aug), colors=SHORTCUT,
                                     linewidths=0.5, alpha=0.35, zorder=2))
    draw_nodes(ax, coords, types, small=True)
    frame(ax, coords, "2   + %d random shortcut edges (red, +%.0f%%):  mesh edges %.2f in "
          "long on average, shortcuts %.1f in"
          % (aug.shape[1], 100.0 * aug.shape[1] / real.shape[1],
             L_real.mean(), L_aug.mean()))

    # 3 - how many shortcuts each node got
    ax = axes[2]
    ax.add_collection(LineCollection(mesh_lines, colors="#ececec", linewidths=0.3, zorder=1))
    levels = np.minimum(n_short, 3)
    shades = ["#d9d9d9", "#9ecae1", "#3182bd", "#08306b"]
    labels = ["0 shortcuts", "1", "2", "3 or more"]
    for lv in range(4):
        m = levels == lv
        ax.scatter(coords[m, 0], coords[m, 1], c=shades[lv], s=9, edgecolors="none",
                   zorder=3, label="%s (%d nodes)" % (labels[lv], int(m.sum())))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=4)
    frame(ax, coords, "3   Shortcuts per node:  %.0f%% of nodes got none, %.0f%% got one, "
          "%.0f%% got 3 or more"
          % (100 * (n_short == 0).mean(), 100 * (n_short == 1).mean(),
             100 * (n_short >= 3).mean()))

    # 4 - one hole node's neighbourhood (or any boundary node if there is no hole)
    ax = axes[3]
    ax.add_collection(LineCollection(mesh_lines, colors="#ececec", linewidths=0.3, zorder=1))
    cand = np.where(types == NODE_TYPE_TO_ID["hole"])[0]
    kind = "hole-boundary"
    if len(cand) == 0:
        cand = np.where(types != NODE_TYPE_TO_ID["interior"])[0]
        kind = "boundary"
    node = cand[np.argmax(n_short[cand])]            # the one with most shortcuts
    mesh_nb = np.unique(np.r_[real[1][real[0] == node], real[0][real[1] == node]])
    short_nb = np.unique(np.r_[aug[1][aug[0] == node], aug[0][aug[1] == node]])
    ax.add_collection(LineCollection([[coords[node], coords[j]] for j in mesh_nb],
                                     colors="#2ca02c", linewidths=1.8, zorder=3))
    if len(short_nb):
        ax.add_collection(LineCollection([[coords[node], coords[j]] for j in short_nb],
                                         colors=SHORTCUT, linewidths=1.4, zorder=3))
    ax.scatter(coords[mesh_nb, 0], coords[mesh_nb, 1], s=30, c="#2ca02c", zorder=4,
               label="mesh neighbours (%d)" % len(mesh_nb))
    ax.scatter(coords[short_nb, 0], coords[short_nb, 1], s=30, c=SHORTCUT, zorder=4,
               label="shortcut partners (%d)" % len(short_nb))
    ax.scatter(*coords[node], s=90, c="black", marker="*", zorder=5,
               label="node %d (%s)" % (node, ID_TO_NAME[int(types[node])]))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=3)
    frame(ax, coords, "4   One %s node's view: its mesh neighbours are close by; each "
          "shortcut reaches a distant node in a single hop" % kind)

    for ax in axes:
        ax.set_ylabel("y", fontsize=8)
    axes[-1].set_xlabel("x", fontsize=8)
    fig.suptitle("%s  -  edge augmentation %.0f%%, seed %d  (the graph training uses)"
                 % (name, 100 * perc, seed), fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("%-26s N=%4d  mesh edges %5d  shortcuts %4d  no-shortcut nodes %2.0f%%  -> %s"
          % (name, n, real.shape[1], aug.shape[1], 100 * (n_short == 0).mean(), out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join("data", "raw"),
                    help="folder of .npz meshes (data\\raw or unseen_npz)")
    ap.add_argument("--geometry", default=None, help="only this geometry, e.g. plate_circle8")
    ap.add_argument("--aug-perc", type=float, default=0.2)
    ap.add_argument("--aug-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: data\\plots_augNN")
    a = ap.parse_args()

    if a.aug_perc <= 0 or a.aug_perc >= 1:
        ap.error("--aug-perc must be a fraction between 0 and 1, e.g. 0.2")
    out = a.out or os.path.join("data", "plots_aug%d" % round(100 * a.aug_perc))

    files = sorted(glob.glob(os.path.join(a.raw, "*.npz")))
    seen, done = set(), 0
    for f in files:                      # one mesh per geometry is enough
        geom = os.path.basename(f).split("__")[0]
        if geom in seen or (a.geometry and geom != a.geometry):
            continue
        seen.add(geom)
        plot_one(f, out, a.aug_perc, a.aug_seed)
        done += 1
    if not done:
        print("No matching .npz files in %s" % a.raw)
        return 1
    print("\nWrote %d plot(s) to %s" % (done, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
