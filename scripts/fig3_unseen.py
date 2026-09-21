r"""
Reproduce Fig. 3 of Kunz & Choudhary (2026): MGN predictions on the 7 UNSEEN geometries.

    python scripts/fig3_unseen.py
    python scripts/fig3_unseen.py --npz-dir unseen_npz --ckpt results/multi_geometry_mgn.pt

For every .npz in --npz-dir (written by odb_to_npz.py) it
    1. builds the graph exactly like scripts/predict.py (same node types, same edges),
    2. predicts von Mises at the load stored in the file (5000 psi for Fig. 3),
    3. compares with the Abaqus truth stored in the same file.

LAYOUT (same as the paper)
    top    : predicted von Mises field on the plate (holes stay white)
    bottom : predicted vs actual, one dot per node, red dashed line = perfect prediction
    title  : R^2 = 1 - sum((truth - pred)^2) / sum((truth - mean(truth))^2)
    grey   : the paper's R^2 for the same panel, for comparison

WRITES (into --out, default results/fig3/)
    fig3_unseen_<load>psi.png    the full figure (7 panels, (a)-(g))
    fig3_<geometry>.png          one panel per geometry
    fig3_r2.csv                  R2, RMSE, MAE, peaks, and the paper's R2
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                 # no display over SSH
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case                                  # noqa: E402
from mgn.graph import build_edges, classify_nodes             # noqa: E402
from mgn.trainer import MGN                                   # noqa: E402

# paper order, labels and R2 (Fig. 3)
PANELS = [
    ("plate_hex8",           "(a)", '8" Hex',            0.9706),
    ("plate_tri8",           "(b)", '8" Tri',            0.7139),
    ("plate_j8",             "(c)", '8" J',              0.1853),
    ("plate_j4",             "(d)", '4" J',             -2.8400),
    ("plate_figure8",        "(e)", '8" Figure8',        0.3249),
    ("plate_track8",         "(f)", '8" Track',          0.7620),
    ("plate_tracksideways8", "(g)", '8" Track Sideways', 0.8099),
]


def read_name(arr):
    """geometry names written by Abaqus' Python 2.7 come back as bytes"""
    v = arr.item() if hasattr(arr, "item") else arr
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def load_case(path):
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    truth = d["mises"].astype(np.float64)
    load = float(d["load"])
    name = read_name(d["geometry"]) if "geometry" in d else os.path.basename(path).split("__")[0]
    case = Case(
        geometry=name,
        coordinates=coords,
        edge_index=torch.from_numpy(build_edges(tris)),
        node_types=classify_nodes(coords, tris),
        von_mises=np.zeros(len(coords), dtype=np.float32),
        metadata={"load": load},
    )
    return case, tris, truth, load


def r2_score(truth, pred):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def draw_field(ax, coords, tris, values):
    x, y = coords[:, 0].astype(float), coords[:, 1].astype(float)
    tri = mtri.Triangulation(x, y, tris)
    ax.tripcolor(tri, values, shading="gouraud", cmap="viridis")
    pad = 0.04 * (x.max() - x.min())
    ax.add_patch(Rectangle((x.min() - pad, y.min() - pad), x.max() - x.min() + 2 * pad,
                           y.max() - y.min() + 2 * pad, fill=False, lw=0.8, ec="black"))
    ax.set_xlim(x.min() - 1.5 * pad, x.max() + 1.5 * pad)
    ax.set_ylim(y.min() - 1.5 * pad, y.max() + 1.5 * pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_scatter(ax, truth, pred, r2, paper_r2):
    ax.scatter(truth, pred, s=4, alpha=0.4, color="#4a86c5", edgecolors="none")
    lo = min(truth.min(), pred.min())
    hi = max(truth.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.3)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("R\u00b2 = %.4f" % r2, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect(1)
    if paper_r2 is not None:
        ax.text(0.97, 0.04, "paper %.4f" % paper_r2, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.45")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="unseen_npz")
    ap.add_argument("--ckpt", default=os.path.join("results", "multi_geometry_mgn.pt"))
    ap.add_argument("--out", default=os.path.join("results", "fig3"))
    ap.add_argument("--load", type=float, default=None,
                    help="use only files at this load (default: all loads found)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.npz_dir, "*.npz")))
    if not files:
        print("No .npz files in %s" % a.npz_dir)
        return 1
    if not os.path.isfile(a.ckpt):
        print("No model at %s" % a.ckpt)
        return 1

    mgn = MGN.load(a.ckpt)
    os.makedirs(a.out, exist_ok=True)

    # ---- predict every file ------------------------------------------------
    results = {}                          # (name, load) -> dict
    for f in files:
        case, tris, truth, load = load_case(f)
        if a.load is not None and abs(load - a.load) > 1e-6:
            continue
        pred = np.asarray(mgn.predict(case), dtype=np.float64).ravel()
        err = pred - truth
        results[(case.geometry, load)] = dict(
            case=case, tris=tris, truth=truth, pred=pred, r2=r2_score(truth, pred),
            rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()))
        print("%-24s %6.0f psi  N=%4d  R2=%8.4f  RMSE=%7.1f psi"
              % (case.geometry, load, len(truth), results[(case.geometry, load)]["r2"],
                 results[(case.geometry, load)]["rmse"]))
    if not results:
        print("No files at %g psi in %s" % (a.load, a.npz_dir))
        return 1

    known = dict((p[0], p) for p in PANELS)
    loads = sorted(set(k[1] for k in results))

    # ---- csv ---------------------------------------------------------------
    csv = os.path.join(a.out, "fig3_r2.csv")
    with open(csv, "w") as fh:
        fh.write("geometry,panel,load_psi,n_nodes,r2,paper_r2,rmse_psi,mae_psi,peak_truth_psi,peak_pred_psi\n")
        for (name, load), r in sorted(results.items(), key=lambda kv: (kv[0][1], [p[0] for p in PANELS].index(kv[0][0])
                                                                        if kv[0][0] in known else 99, kv[0][0])):
            p = known.get(name)
            fh.write("%s,%s,%g,%d,%.6f,%s,%.2f,%.2f,%.2f,%.2f\n" % (
                name, p[1] if p else "", load, len(r["truth"]), r["r2"], ("%.4f" % p[3]) if p else "",
                r["rmse"], r["mae"], r["truth"].max(), r["pred"].max()))

    # ---- one figure per load, laid out like the paper ----------------------
    for load in loads:
        order = [p for p in PANELS if (p[0], load) in results]
        order += [(n, "", n, None) for (n, L) in sorted(results) if L == load and n not in known]
        n = len(order)
        # rows of 3, 2, 2 like the paper (falls back to rows of 3 for other counts)
        rows = [order[0:3], order[3:5], order[5:7]] if n == 7 else [order[i:i + 3] for i in range(0, n, 3)]
        fig = plt.figure(figsize=(11, 4.4 * len(rows) + 0.6))
        outer = fig.add_gridspec(len(rows), 6, hspace=0.35, wspace=0.9,
                                 top=1 - 1.05 / fig.get_figheight(), bottom=0.04, left=0.06, right=0.98)
        for ri, row in enumerate(rows):
            start = (6 - 2 * len(row)) // 2
            for ci, (name, letter, label, paper_r2) in enumerate(row):
                r = results[(name, load)]
                cell = outer[ri, start + 2 * ci:start + 2 * ci + 2].subgridspec(2, 1, height_ratios=[1, 3.6], hspace=0.28)
                axf = fig.add_subplot(cell[0])
                axs = fig.add_subplot(cell[1])
                draw_field(axf, r["case"].coordinates, r["tris"], r["pred"])
                axf.set_title(label, fontsize=12, fontweight="bold")
                if letter:
                    axf.text(-0.08, 1.15, letter, transform=axf.transAxes, fontsize=11, fontweight="bold",
                             ha="right", va="top")
                draw_scatter(axs, r["truth"], r["pred"], r["r2"], paper_r2)
            if ri < len(rows) - 1:                              # divider lines like the paper
                y = outer[ri + 1, 0].get_position(fig).y1 + 0.012
                fig.add_artist(Line2D([0.06, 0.98], [y, y], color="0.6", lw=0.8))
        fig.suptitle("MGN Predictions on Unseen Geometries @ %g psi" % load, fontsize=13, fontweight="bold",
                     y=1 - 0.25 / fig.get_figheight())
        png = os.path.join(a.out, "fig3_unseen_%gpsi.png" % load)
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print("\nfigure   %s" % png)

        # single panels, handy for slides
        for name, letter, label, paper_r2 in order:
            r = results[(name, load)]
            f1 = plt.figure(figsize=(4.2, 5.2))
            g = f1.add_gridspec(2, 1, height_ratios=[1, 3.6], hspace=0.3)
            axf, axs = f1.add_subplot(g[0]), f1.add_subplot(g[1])
            draw_field(axf, r["case"].coordinates, r["tris"], r["pred"])
            axf.set_title(("%s %s" % (letter, label)).strip(), fontsize=12, fontweight="bold")
            draw_scatter(axs, r["truth"], r["pred"], r["r2"], paper_r2)
            f1.savefig(os.path.join(a.out, "fig3_%s__%g.png" % (name, load)), dpi=150, bbox_inches="tight")
            plt.close(f1)

    print("panels   %s" % os.path.join(a.out, "fig3_<geometry>__<load>.png"))
    print("table    %s" % csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
