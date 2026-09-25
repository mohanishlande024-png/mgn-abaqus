r"""
Predictions at the UNSEEN loads (default 800, 5000, 12000 psi).

    python scripts/unseen_loads.py --ckpt results_aug20_L8/mgn_best.pt
    python scripts/unseen_loads.py --loads 800 5000 12000 --geometry plate_circle8
    python scripts/unseen_loads.py --ckpt results/multi_geometry_mgn.pt --layers 20

WHY THESE THREE LOADS
    Training uses 1000-4500 and 5500-10000 psi, with a deliberate gap. So:
        800    extrapolation below the trained range
        5000   interpolation, inside the gap
        12000  extrapolation above the trained range

WHERE THE TRUTH COMES FROM
    No Abaqus solve exists at these loads - none is needed. The problem is
    linear elastic with no prescribed displacement, so sigma(k*P) = k*sigma(P)
    EXACTLY. The truth at 5000 psi is the solved 1000 psi field times 5. (Your
    own spot check confirmed this: circle8 max 5354.8 psi at 1000 -> 26774.0 at
    5000.) That also means a perfect model would be exactly linear in load, so
    the peak-vs-load plot below is a direct test of whether the model learned
    the physics or just memorised the trained load range.

WRITES (into <ckpt folder>/unseen_loads/)
    r2_by_load.csv      R2, RMSE, peak error for every geometry x load
    summary.png         R2 vs load, and predicted peak vs load against the
                        exact linear truth, with the trained range shaded
    field_<geom>.png    truth / prediction / error maps at each unseen load
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import matplotlib.tri as mtri                     # noqa: E402
import torch                                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case, load_dataset        # noqa: E402
from mgn.augmented_trainer import AugMGN          # noqa: E402

TRAIN_BANDS = [(1000, 4500), (5500, 10000)]


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - p) ** 2).sum() / ss if ss > 0 else float("nan")


def reference(cases):
    """One solved case per geometry, plus its load - the basis for scaling."""
    ref = {}
    for c in cases:
        L = c.metadata["load"]
        if c.geometry not in ref or L < ref[c.geometry].metadata["load"]:
            ref[c.geometry] = c
    return ref


def tris_for(geom, raw_dir):
    import glob
    hits = sorted(glob.glob(os.path.join(raw_dir, geom + "__*.npz")))
    if not hits:
        return None
    return np.load(hits[0], allow_pickle=False)["tris"].astype(np.int64)


def field_figure(geom, coords, tris, results, path):
    """rows = unseen loads, columns = truth / prediction / error."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 1.55 * n + 0.9), squeeze=False)
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], tris) if tris is not None else None
    for r, (L, y, p) in enumerate(results):
        err = p - y
        for c, (vals, title, cmap, lim) in enumerate([
                (y, "Abaqus (scaled) %.0f psi" % L, "viridis", (y.min(), y.max())),
                (p, "MGN prediction", "viridis", (y.min(), y.max())),
                (err, "error (pred - truth)", "coolwarm",
                 (-np.abs(err).max(), np.abs(err).max()))]):
            ax = axes[r][c]
            if tri is not None:
                im = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap,
                                  vmin=lim[0], vmax=lim[1])
            else:
                im = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=4, cmap=cmap,
                                vmin=lim[0], vmax=lim[1])
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
        axes[r][0].set_ylabel("%.0f psi" % L, fontsize=10)
    fig.suptitle("%s - unseen loads  (R2 %s)"
                 % (geom, ", ".join("%.4f" % r2(y, p) for _, y, p in results)), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120); plt.close(fig)


def summary_figure(rows, loads, path):
    geoms = sorted({r["geometry"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for a in (ax1, ax2):
        for lo, hi in TRAIN_BANDS:
            a.axvspan(lo, hi, color="#1C7293", alpha=0.10)
        a.set_xlabel("load (psi)")
    for g in geoms:
        rr = sorted([r for r in rows if r["geometry"] == g], key=lambda r: r["load"])
        ax1.plot([r["load"] for r in rr], [r["r2"] for r in rr], "o-", ms=4, lw=1, label=g)
        ax2.plot([r["load"] for r in rr], [r["pred_peak"] for r in rr], "o-", ms=4, lw=1)
    ax1.set_ylabel("R2"); ax1.set_title("Accuracy at unseen loads (shaded = trained range)")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.legend(fontsize=7, ncol=2)
    # exact-linear reference through the lowest unseen load, per geometry
    for g in geoms:
        rr = sorted([r for r in rows if r["geometry"] == g], key=lambda r: r["load"])
        k = rr[0]["true_peak"] / rr[0]["load"]
        xs = np.array([min(loads) * 0.8, max(loads) * 1.05])
        ax2.plot(xs, k * xs, color="0.6", lw=0.8, ls="--", zorder=0)
    ax2.set_ylabel("peak von Mises (psi)")
    ax2.set_title("Predicted peak vs load  (dashed = exact linear truth)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join("results", "multi_geometry_mgn.pt"))
    ap.add_argument("--data", default=None,
                    help="dataset for the meshes; default matches the checkpoint")
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--loads", type=float, nargs="+", default=[800, 5000, 12000])
    ap.add_argument("--geometry", default=None, help="field maps for this geometry "
                    "(default: the one with the largest error)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mgn = AugMGN.load(a.ckpt)
    out = a.out or os.path.join(os.path.dirname(a.ckpt) or ".", "unseen_loads")
    os.makedirs(out, exist_ok=True)
    data = a.data or ("data/dataset_aug%d.pt" % round(100 * mgn.aug_perc)
                      if mgn.augmented else "data/dataset.pt")
    cases, meta = load_dataset(data, return_meta=True)
    if (meta["aug_perc"], meta["aug_seed"]) != (mgn.aug_perc, mgn.aug_seed):
        print("MISMATCH: %s has augmentation %s, checkpoint has %g/%d"
              % (data, meta, mgn.aug_perc, mgn.aug_seed))
        return 1

    rows, fields = [], {}
    for geom, ref in sorted(reference(cases).items()):
        res = []
        for L in a.loads:
            y = ref.von_mises * np.float32(L / ref.metadata["load"])   # exact, by linearity
            case = Case(geometry=geom, coordinates=ref.coordinates, edge_index=ref.edge_index,
                        node_types=ref.node_types, von_mises=y, metadata={"load": float(L)},
                        edge_flag=ref.edge_flag)
            p = mgn.predict(case)
            rows.append(dict(geometry=geom, load=L, r2=r2(y, p),
                             rmse=float(np.sqrt(((y - p) ** 2).mean())),
                             true_peak=float(y.max()), pred_peak=float(p.max()),
                             peak_err_pct=float(100 * (p.max() - y.max()) / y.max())))
            res.append((L, y, p))
            print("%-20s %7.0f psi   R2 %8.4f   RMSE %8.1f   peak %9.1f vs %9.1f (%+.1f%%)"
                  % (geom, L, rows[-1]["r2"], rows[-1]["rmse"], rows[-1]["pred_peak"],
                     rows[-1]["true_peak"], rows[-1]["peak_err_pct"]))
        fields[geom] = (ref.coordinates, res)

    with open(os.path.join(out, "r2_by_load.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary_figure(rows, a.loads, os.path.join(out, "summary.png"))

    pick = a.geometry or min({r["geometry"] for r in rows},
                             key=lambda g: min(r["r2"] for r in rows if r["geometry"] == g))
    coords, res = fields[pick]
    field_figure(pick, coords, tris_for(pick, a.raw), res,
                 os.path.join(out, "field_%s.png" % pick))

    print("\nmean R2 per load:")
    for L in a.loads:
        v = [r["r2"] for r in rows if r["load"] == L]
        tag = "inside gap" if 4500 < L < 5500 else ("extrapolation" if L < 1000 or L > 10000 else "")
        print("   %7.0f psi : %.4f   %s" % (L, np.mean(v), tag))
    print("\nWrote %s  (r2_by_load.csv, summary.png, field_%s.png)" % (out, pick))
    return 0


if __name__ == "__main__":
    sys.exit(main())
