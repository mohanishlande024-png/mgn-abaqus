r"""
Evaluate a trained MGN: R2 overall, per geometry, and per node type.

    python scripts\evaluate.py                          uses results\mgn_best.pt
    python scripts\evaluate.py --ckpt results\mgn_latest.pt
    python scripts\evaluate.py --no-plots               numbers only, faster

WHAT R2 MEANS HERE
    R2 = 1 - (sum of squared errors) / (total variance of the truth)

    1.0    perfect
    0.9    explains 90% of the variation in stress
    0.0    no better than always guessing the mean
    < 0    worse than guessing the mean

    The paper reports R2 per geometry in Tables II and III. Their classical
    baselines (Random Forest / Gradient Boosting / KNN) score 0.01 to 0.86;
    their converged MGN reaches about 0.9999.

IMPORTANT
    Run against the training set, this is TRAINING performance, not a
    result. It tells you the model fits what it was shown. Held-out numbers
    require the unseen load cases and the 7 unseen geometries.

WRITES (into results\eval\)
    scatter.png          predicted vs actual, all nodes
    field_<geom>.png     FE truth vs prediction vs error, per geometry
    per_geometry.csv     R2, RMSE, MAE per geometry
    per_nodetype.csv     R2, RMSE, MAE per node type
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import load_dataset                    # noqa: E402
from mgn.graph import NODE_TYPE_TO_ID                   # noqa: E402
from mgn.trainer import MGN                             # noqa: E402

ID_TO_NAME = dict((v, k) for k, v in NODE_TYPE_TO_ID.items())
OUT = os.path.join("results", "eval")


def r2(truth, pred):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def metrics(truth, pred):
    err = pred - truth
    return (r2(truth, pred),
            float(np.sqrt((err ** 2).mean())),
            float(np.abs(err).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join("data", "dataset.pt"))
    ap.add_argument("--ckpt", default=os.path.join("results", "mgn_best.pt"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cases = load_dataset(a.data)
    y_all = np.concatenate([c.von_mises for c in cases])

    # Rebuild the model exactly as training did, then load the weights.
    mgn = MGN(num_layers=20, hidden_channels=64, embedding_dim=16,
              learning_rate=1e-5, epochs=1, global_features=["load"])
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y_all, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    mgn._build_model()

    ck = torch.load(a.ckpt, map_location=mgn.device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    mgn._model.load_state_dict(state)
    mgn._model.eval()

    trained_epochs = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    print("=" * 70)
    print("EVALUATION   checkpoint: %s  (epoch %s)" % (a.ckpt, trained_epochs))
    print("=" * 70)

    # ---- predict -------------------------------------------------------
    preds, truths, types, geoms, loads = [], [], [], [], []
    with torch.no_grad():
        for c in cases:
            p = np.asarray(mgn.predict(c)).ravel()
            preds.append(p)
            truths.append(c.von_mises)
            types.append(c.node_types)
            geoms.append(np.array([c.geometry] * len(p)))
            loads.append(np.full(len(p), c.metadata["load"]))

    pred = np.concatenate(preds)
    truth = np.concatenate(truths)
    ntype = np.concatenate(types)
    geom = np.concatenate(geoms)
    load = np.concatenate(loads)

    R2, rmse, mae = metrics(truth, pred)
    print("\nOVERALL")
    print("  nodes        %d" % len(truth))
    print("  R2           %.6f" % R2)
    print("  RMSE         %.1f psi" % rmse)
    print("  MAE          %.1f psi" % mae)
    print("  mean stress  %.1f psi   -> RMSE is %.1f%% of mean"
          % (truth.mean(), 100 * rmse / truth.mean()))

    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    # ---- per geometry --------------------------------------------------
    print("\nPER GEOMETRY")
    print("  %-16s %10s %10s %10s" % ("geometry", "R2", "RMSE", "MAE"))
    print("  " + "-" * 48)
    rows = ["geometry,r2,rmse_psi,mae_psi,n_nodes"]
    for g in sorted(set(geom.tolist())):
        m = geom == g
        gr2, grm, gma = metrics(truth[m], pred[m])
        print("  %-16s %10.6f %10.1f %10.1f" % (g, gr2, grm, gma))
        rows.append("%s,%.6f,%.4f,%.4f,%d" % (g, gr2, grm, gma, int(m.sum())))
    open(os.path.join(OUT, "per_geometry.csv"), "w").write("\n".join(rows))

    # ---- per node type -------------------------------------------------
    print("\nPER NODE TYPE")
    print("  %-16s %10s %10s %10s" % ("node type", "R2", "RMSE", "MAE"))
    print("  " + "-" * 48)
    rows = ["node_type,r2,rmse_psi,mae_psi,n_nodes"]
    for tid in sorted(ID_TO_NAME):
        m = ntype == tid
        if not m.any():
            continue
        name = ID_TO_NAME[tid]
        tr2, trm, tma = metrics(truth[m], pred[m])
        print("  %-16s %10.6f %10.1f %10.1f" % (name, tr2, trm, tma))
        rows.append("%s,%.6f,%.4f,%.4f,%d" % (name, tr2, trm, tma,
                                              int(m.sum())))
    open(os.path.join(OUT, "per_nodetype.csv"), "w").write("\n".join(rows))

    # ---- per load ------------------------------------------------------
    ul = sorted(set(load.tolist()))
    print("\nPER LOAD (first, middle, last)")
    for L in (ul[0], ul[len(ul) // 2], ul[-1]):
        m = load == L
        lr2, lrm, _ = metrics(truth[m], pred[m])
        print("  %8.0f psi   R2 %.6f   RMSE %.1f psi" % (L, lr2, lrm))

    if a.no_plots:
        print("\nWrote CSVs to %s" % OUT)
        return 0

    # ---- plots ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    step = max(1, len(truth) // 40000)          # keep the PNG light
    ax.scatter(truth[::step], pred[::step], s=2, alpha=0.25,
               edgecolors="none")
    lo, hi = float(truth.min()), float(truth.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="perfect")
    ax.set_xlabel("FE von Mises (psi)")
    ax.set_ylabel("MGN predicted (psi)")
    ax.set_title("Predicted vs actual   R2 = %.5f" % R2)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scatter.png"), dpi=130)
    plt.close(fig)

    # ---- per-geometry scatter grid, coloured by node type ---------------
    # Colouring by node type is what makes this worth looking at: if the
    # model is struggling, the off-diagonal points are almost always 'hole'
    # nodes, because that is where the stress gradient is sharpest.
    colours = {"applied_load": "#d62728", "fixed": "#1f77b4",
               "free": "#7f7f7f", "hole": "#ff7f0e", "interior": "#c7c7c7"}
    gs = sorted(set(geom.tolist()))
    ncol = 4
    nrow = int(np.ceil(len(gs) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, g in zip(axes, gs):
        m = geom == g
        t, p, nt = truth[m], pred[m], ntype[m]
        gr2, grm, _ = metrics(t, p)
        # interior first so the boundary types draw on top of it
        for tid in (4, 2, 1, 0, 3):
            if tid not in ID_TO_NAME:
                continue
            k = nt == tid
            if not k.any():
                continue
            ax.scatter(t[k], p[k], s=3, alpha=0.35, edgecolors="none",
                       c=colours[ID_TO_NAME[tid]], label=ID_TO_NAME[tid])
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title("%s\nR2 = %.5f   RMSE %.0f psi" % (g, gr2, grm),
                     fontsize=10)
        ax.set_xlabel("FE truth (psi)", fontsize=8)
        ax.set_ylabel("MGN predicted (psi)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    for ax in axes[len(gs):]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=6,
                          color=colours[n], label=n)
               for n in ("applied_load", "fixed", "free", "hole", "interior")]
    axes[len(gs) - 1].legend(handles=handles, loc="upper left", fontsize=7,
                             framealpha=0.9)
    fig.suptitle("Predicted vs actual by geometry   (overall R2 = %.5f)" % R2,
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "scatter_by_geometry.png"), dpi=120)
    plt.close(fig)

    # ---- one scatter file per geometry, for slides ----------------------
    for g in gs:
        m = geom == g
        t, p, nt = truth[m], pred[m], ntype[m]
        gr2, grm, gma = metrics(t, p)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for tid in (4, 2, 1, 0, 3):
            k = nt == tid
            if k.any():
                ax.scatter(t[k], p[k], s=6, alpha=0.5, edgecolors="none",
                           c=colours[ID_TO_NAME[tid]], label=ID_TO_NAME[tid])
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.set_xlabel("FE von Mises (psi)")
        ax.set_ylabel("MGN predicted (psi)")
        ax.set_title("%s\nR2 = %.5f   RMSE %.0f psi   MAE %.0f psi"
                     % (g, gr2, grm, gma))
        ax.legend(fontsize=8, markerscale=2)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "scatter_%s.png" % g), dpi=130)
        plt.close(fig)

    # one field comparison per geometry, at the lowest load
    seen = set()
    for c in cases:
        if c.geometry in seen:
            continue
        seen.add(c.geometry)
        with torch.no_grad():
            p = np.asarray(mgn.predict(c)).ravel()
        t = c.von_mises
        tri = Triangulation(c.coordinates[:, 0], c.coordinates[:, 1],
                            _tris_from(c))
        if tri is None:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(13, 8))
        for axx, vals, ttl, cmap in (
                (axes[0], t, "FE truth", "inferno"),
                (axes[1], p, "MGN prediction", "inferno"),
                (axes[2], p - t, "error (pred - truth)", "coolwarm")):
            tc = axx.tripcolor(tri, vals, shading="gouraud", cmap=cmap)
            fig.colorbar(tc, ax=axx, fraction=0.02)
            axx.set_title("%s   %s   %.0f psi"
                          % (c.geometry, ttl, c.metadata["load"]))
            axx.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "field_%s.png" % c.geometry), dpi=110)
        plt.close(fig)

    print("\nWrote %d plot(s) and 2 CSV(s) to %s" % (len(seen) + 1, OUT))
    return 0


def _tris_from(case):
    """Recover triangles from the edge list is not possible, so re-read them
    from the matching .npz. Returns None if it is not on this machine."""
    import glob
    for p in glob.glob(os.path.join("data", "raw", "*.npz")):
        d = np.load(p, allow_pickle=False)
        if len(d["coords"]) == len(case.coordinates):
            if np.allclose(d["coords"], case.coordinates, atol=1e-5):
                return d["tris"].astype(np.int64)
    return None


if __name__ == "__main__":
    sys.exit(main())
