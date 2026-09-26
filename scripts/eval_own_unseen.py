r"""
Results on the UNSEEN LOADS and the UNSEEN GEOMETRIES of your own dataset,
with contour plots (Abaqus truth | MGN prediction | error).

    python scripts/eval_own_unseen.py
    python scripts/eval_own_unseen.py --ckpt results_aug20/mgn_best.pt
    python scripts/eval_own_unseen.py --contour-loads all        contours at every load, not just the test loads
    python scripts/eval_own_unseen.py --no-contours              numbers and summary plots only (fast)

    SI dataset later:
    python scripts/eval_own_unseen.py --train-data data/dataset_si_aug20.pt --ckpt results_si_aug20/mgn_best.pt \
        --loads-dir data/unseen_si_loads --interp-dir data/unseen_si_interp --extrap-dir data/unseen_si_extrap \
        --manifest data/odb_si/si_manifest.csv --units MPa

SAFE WHILE TRAINING IS STILL RUNNING
    train.py keeps overwriting results_aug20/mgn_best.pt. This script copies
    that file first and evaluates the copy, so it never reads a half-written
    file and never touches the running job. It prints which epoch it is
    evaluating. Re-run it when training finishes for the final numbers.

WHICH MODEL
    mgn_best.pt / mgn_latest.pt hold weights only. The normalisation (mean and
    std of stress, load and edge features) is recomputed from the training
    dataset (--train-data) exactly as train.py computed it - the same data
    gives the same numbers. multi_geometry_mgn.pt (written when a run
    finishes) carries its own normalisation and is loaded directly.

THE THREE TESTS  (folders written by scripts/organize_own_dataset.py)
    unseen loads          data/unseen_own_loads    the 27 training geometries at 800 / 5000 / 12000 psi
                                                   800 below the trained range, 5000 in the gap, 12000 above
    unseen geometries     data/unseen_own_interp   13 new shapes inside each family's range, all 23 loads
    (extrapolation)       data/unseen_own_extrap    5 shapes past each family's end, all 23 loads
    The truth is a real Abaqus solve at that exact load - nothing is scaled.

WRITES  (into <checkpoint folder>/unseen_eval/)
    metrics_all.csv                        every file: R2, RMSE, MAE, peak truth / prediction / peak error
    summary_unseen_loads.png               R2 and peak error of each training geometry at the 3 unseen loads
    summary_r2_vs_load.png                 R2 against load for all three tests, trained load bands shaded
    summary_unseen_geometries_heatmap.png  R2 of every unseen geometry at every load
    scatter_unseen_geometries_<L>.png      predicted vs actual per unseen geometry at the gap load (paper Fig. 3 style)
    contours/unseen_loads/<geometry>__<load>.png
    contours/unseen_interp/<geometry>__<load>.png
    contours/unseen_extrap/<geometry>__<load>.png
        each: FE truth and MGN prediction on the SAME colour levels, then the
        error (MGN - FE); left the whole plate, right a zoom on the hole.
        The circles mark where the peak is in the truth and in the prediction.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                     # no display over SSH
import matplotlib.pyplot as plt           # noqa: E402
import matplotlib.tri as mtri             # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import Case, _read_name, build_topology, load_dataset   # noqa: E402
from mgn.augmented_trainer import AugMGN                                  # noqa: E402

HOLE_ID = 3                                # NODE_TYPE_TO_ID["hole"]
R2_FLOOR = -1.0                            # R2 axes stop here; worse values are marked, not plotted to scale

# colours: validated categorical slots (blue, orange, aqua), text stays in ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de", "#2a78d6"
SET_COLOUR = {"unseen_loads": BLUE, "unseen_interp": ORANGE, "unseen_extrap": AQUA}
SET_TITLE = {"unseen_loads": "unseen LOAD (training geometry)",
             "unseen_interp": "unseen GEOMETRY - interpolation",
             "unseen_extrap": "unseen GEOMETRY - extrapolation"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 11, "axes.titlecolor": INK})


# ----------------------------------------------------------------- metrics

def r2_score(t, p):
    ss = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / ss if ss > 0 else float("nan")


def metrics(t, p):
    e = p - t
    ip, it = int(np.argmax(p)), int(np.argmax(t))
    return dict(r2=r2_score(t, p), rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()),
                max_abs_err=float(np.abs(e).max()), peak_true=float(t[it]), peak_pred=float(p[ip]),
                peak_err_pct=100.0 * (float(p[ip]) - float(t[it])) / float(t[it]),
                i_peak_true=it, i_peak_pred=ip)


# ----------------------------------------------------------------- model

def snapshot(path):
    """Copy the checkpoint (train.py may be rewriting it right now) and load the copy."""
    tmp = os.path.join(tempfile.gettempdir(), "mgn_eval_snapshot_%d.pt" % os.getpid())
    last = None
    for _ in range(6):
        try:
            shutil.copy2(path, tmp)
            return tmp, torch.load(tmp, map_location="cpu", weights_only=False)
        except Exception as exc:          # caught mid-write: wait and copy again
            last = exc
            time.sleep(3)
    raise RuntimeError("could not read %s: %s" % (path, last))


def load_model(a):
    snap, ck = snapshot(a.ckpt)
    train_loads = []
    if os.path.isfile(a.train_data):
        cases, aug = load_dataset(a.train_data, return_meta=True)
        train_loads = sorted(set(float(c.metadata["load"]) for c in cases))
    else:
        cases, aug = None, None

    if isinstance(ck, dict) and "model_state_dict" in ck:          # multi_geometry_mgn.pt
        mgn = AugMGN.load(snap)
        desc = "final model (%d epochs)" % len(ck.get("train_losses", []))
        return mgn, desc, train_loads

    if cases is None:
        sys.exit("%s holds weights only, so the training dataset is needed to rebuild the "
                 "normalisation - pass --train-data." % a.ckpt)
    if isinstance(ck, dict) and "augmentation" in ck and ck["augmentation"] != aug:
        sys.exit("MISMATCH: %s was trained with augmentation %s but %s has %s."
                 % (a.ckpt, ck["augmentation"], a.train_data, aug))
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    epoch = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    loss = ck.get("loss", float("nan")) if isinstance(ck, dict) else float("nan")

    print("rebuilding normalisation from %s (%d training cases) ..." % (a.train_data, len(cases)))
    y = np.concatenate([c.von_mises for c in cases])
    mgn = AugMGN(num_layers=a.layers, hidden_channels=a.hidden, embedding_dim=a.embedding,
                 learning_rate=1e-5, epochs=1, global_features=["load"],
                 aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"])
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    del fem_data
    mgn._build_model()
    try:
        mgn._model.load_state_dict(state)
    except RuntimeError as exc:
        sys.exit("The weights do not fit a %d-layer / %d-hidden model. If you trained with "
                 "--layers or --hidden, pass the same values here.\n%s"
                 % (a.layers, a.hidden, str(exc).splitlines()[0]))
    mgn._model.eval()
    mgn._train_fem = None                     # free the training cases
    desc = "best weights so far: epoch %s, training loss %.3e" % (epoch, loss)
    return mgn, desc, train_loads


# ----------------------------------------------------------------- data

def read_manifest(path):
    info, order = {}, {}
    if path and os.path.isfile(path):
        with open(path) as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                info[r["geometry"]] = r
                order[r["geometry"]] = i
    return info, order


def short(name):
    for p in ("plate_own_", "plate_si_", "plate_"):
        if name.startswith(p):
            return name[len(p):]
    return name


def label(name, info):
    r = info.get(name)
    return "%s  (%s)" % (short(name), r["param"]) if r and r.get("param") else short(name)


def read_set(folder):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        name = (_read_name(d["geometry"]) if "geometry" in d.files
                else os.path.splitext(os.path.basename(f))[0].rsplit("__", 1)[0])
        out.append(dict(path=f, name=name, load=float(d["load"]),
                        coords=d["coords"].astype(np.float32), tris=d["tris"].astype(np.int64),
                        truth=d["mises"].astype(np.float64)))
    return out


# ----------------------------------------------------------------- contour plot

def fmt(v):
    return "{:,.0f}".format(v) if abs(v) >= 100 else "{:,.2f}".format(v)


def contour_figure(item, res, info, set_key, units, path):
    c, tris = item["coords"], item["tris"]
    t, p = item["truth"], res["pred"]
    m = res["m"]
    tri = mtri.Triangulation(c[:, 0].astype(float), c[:, 1].astype(float), tris)
    x0, x1 = float(c[:, 0].min()), float(c[:, 0].max())
    y0, y1 = float(c[:, 1].min()), float(c[:, 1].max())
    H = y1 - y0

    hole = c[item["node_types"] == HOLE_ID]
    cx = float(hole[:, 0].mean()) if len(hole) else 0.5 * (x0 + x1)
    hx = float(np.abs(hole[:, 0] - cx).max()) if len(hole) else 0.1 * H
    w = max(0.6 * H, 1.5 * hx)

    lo = min(float(t.min()), float(p.min()), 0.0)
    hi = max(float(t.max()), float(p.max()))
    lv = MaxNLocator(nbins=20).tick_values(lo, hi)           # round-number levels, shared by truth and MGN
    err = p - t
    el = max(float(np.abs(err).max()), 1e-9)
    elv = MaxNLocator(nbins=20, symmetric=True).tick_values(-el, el)

    fig = plt.figure(figsize=(15.5, 7.4), facecolor="white")
    gs = fig.add_gridspec(3, 3, width_ratios=[6.0, 1.45, 0.07], wspace=0.06, hspace=0.42,
                          left=0.03, right=0.94, top=0.855, bottom=0.03)
    rows = [
        (t, lv, "viridis", "Abaqus FE (truth)    peak %s %s" % (fmt(m["peak_true"]), units), m["i_peak_true"]),
        (p, lv, "viridis", "MGN prediction    peak %s %s  (%+.1f %%)" % (fmt(m["peak_pred"]), units,
                                                                        m["peak_err_pct"]), m["i_peak_pred"]),
        (err, elv, "RdBu_r", "Error = MGN - FE    RMSE %s %s,  max |error| %s %s"
         % (fmt(m["rmse"]), units, fmt(m["max_abs_err"]), units), None),
    ]
    for r, (vals, levels, cmap, title, ipk) in enumerate(rows):
        for col, (xa, xb) in enumerate(((x0, x1), (cx - w, cx + w))):
            ax = fig.add_subplot(gs[r, col])
            cs = ax.tricontourf(tri, vals, levels=levels, cmap=cmap, extend="neither")
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=INK2, lw=0.7)
            if ipk is not None:
                ax.plot(c[ipk, 0], c[ipk, 1], "o", mfc="none", mec="#ff2d55", ms=11 if col else 8, mew=1.8)
            ax.set_xlim(xa, xb)
            ax.set_ylim(y0 - 0.02 * H, y1 + 0.02 * H)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col == 0:
                ax.set_title(title, loc="left", fontsize=11.5)
            else:
                ax.set_title("zoom on the hole", fontsize=9.5, color=MUTED)
        cb = fig.colorbar(cs, cax=fig.add_subplot(gs[r, 2]))
        cb.ax.tick_params(labelsize=8)
        cb.set_label(units, fontsize=9)
    fig.suptitle("%s   -   %s   -   load %g %s\nR\u00b2 = %.4f     RMSE %s %s     MAE %s %s     %d nodes"
                 % (label(item["name"], info), SET_TITLE[set_key], item["load"], units, m["r2"],
                    fmt(m["rmse"]), units, fmt(m["mae"]), units, len(t)),
                 fontsize=13, y=0.975)
    fig.savefig(path, dpi=115, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- summaries

def bands_from(loads):
    """Split the sorted training loads into contiguous bands (the paper has two)."""
    if not loads:
        return []
    steps = np.diff(loads)
    if len(steps) == 0:
        return [(loads[0], loads[0])]
    cut = 1.5 * float(np.median(steps))
    bands, start = [], loads[0]
    for i, s in enumerate(steps):
        if s > cut:
            bands.append((start, loads[i]))
            start = loads[i + 1]
    bands.append((start, loads[-1]))
    return bands


def role(L, bands):
    if not bands:
        return ""
    if L < bands[0][0]:
        return "below trained range"
    if L > bands[-1][1]:
        return "above trained range"
    if any(lo <= L <= hi for lo, hi in bands):
        return "inside a trained band"
    return "in the gap between bands"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def summary_unseen_loads(rows, order, info, test_loads, bands, units, path):
    rs = [r for r in rows if r["set"] == "unseen_loads"]
    if not rs:
        return
    geoms = sorted(set(r["geometry"] for r in rs), key=lambda g: (order.get(g, 999), g))
    ypos = dict((g, i) for i, g in enumerate(geoms))
    cols = [BLUE, ORANGE, AQUA, "#e87ba4", "#008300"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, max(4.8, 0.33 * len(geoms) + 2.6)), sharey=True,
                                 gridspec_kw=dict(wspace=0.06))
    for k, L in enumerate(test_loads):
        sel = [r for r in rs if r["load"] == L]
        off = (k - (len(test_loads) - 1) / 2.0) * 0.22
        yy = [ypos[r["geometry"]] + off for r in sel]
        lab = "%g %s  (%s)" % (L, units, role(L, bands))
        r2v = np.array([r["r2"] for r in sel])
        a1.scatter(np.maximum(r2v, R2_FLOOR), yy, s=34, color=cols[k % 5], edgecolor="white", lw=0.8, label=lab,
                   zorder=3, marker="o")
        low = r2v < R2_FLOOR                      # off the left edge: drawn as '<' at the floor
        if low.any():
            a1.scatter(np.full(low.sum(), R2_FLOOR), np.array(yy)[low], s=60, color=cols[k % 5], marker="<", zorder=4)
        a2.scatter([r["peak_err_pct"] for r in sel], yy, s=34, color=cols[k % 5], edgecolor="white", lw=0.8, zorder=3)
    a1.set_yticks(range(len(geoms)))
    a1.set_yticklabels([label(g, info) for g in geoms], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("R\u00b2  (1 = perfect;  '<' = below %g)" % R2_FLOOR)
    lo_r2 = min([r["r2"] for r in rs if np.isfinite(r["r2"])] + [0.9])
    a1.set_xlim(max(lo_r2, R2_FLOOR) - 0.02, 1.005)
    a2.axvline(0, color=INK2, lw=1)
    a2.set_xlabel("peak stress error  (MGN peak - FE peak) / FE peak   [%]")
    for ax in (a1, a2):
        style(ax)
    a1.set_title("R\u00b2 per geometry", loc="left")
    a2.set_title("peak stress error", loc="left")
    h, l = a1.get_legend_handles_labels()
    H = fig.get_figheight()                  # keep the title and legend a fixed height in inches
    fig.suptitle("Unseen LOADS - the %d training geometries at loads the model never saw" % len(geoms),
                 fontsize=13, y=1 - 0.12 / H, va="top")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / H), ncol=len(test_loads), frameon=False,
               fontsize=10)
    W = fig.get_figwidth()
    fig.subplots_adjust(left=2.3 / W, right=1 - 0.3 / W, bottom=0.8 / H, top=1 - 1.15 / H, wspace=0.06)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def summary_r2_vs_load(rows, bands, units, path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for lo, hi in bands:
        ax.axvspan(lo, hi, color=BAND, alpha=0.08, lw=0)
    if bands:
        ax.text(bands[0][0], 1.0, " trained load bands", color=INK2, fontsize=9, va="top",
                transform=ax.get_xaxis_transform())
    for key in ("unseen_interp", "unseen_extrap"):
        rs = [r for r in rows if r["set"] == key]
        if not rs:
            continue
        for g in sorted(set(r["geometry"] for r in rs)):
            gr = sorted([r for r in rs if r["geometry"] == g], key=lambda r: r["load"])
            ax.plot([r["load"] for r in gr], [r["r2"] for r in gr], color=SET_COLOUR[key], lw=0.8, alpha=0.25)
        loads = sorted(set(r["load"] for r in rs))
        mean = [np.mean([r["r2"] for r in rs if r["load"] == L]) for L in loads]
        ax.plot(loads, mean, color=SET_COLOUR[key], lw=2.2, marker="o", ms=5,
                label="%s - mean of %d geometries" % (SET_TITLE[key], len(set(r["geometry"] for r in rs))))
    rs = [r for r in rows if r["set"] == "unseen_loads"]
    if rs:
        loads = sorted(set(r["load"] for r in rs))
        mean = [np.mean([r["r2"] for r in rs if r["load"] == L]) for L in loads]
        ax.plot(loads, mean, ls="", marker="D", ms=10, color=BLUE, mec="white", mew=1.5,
                label="%s - mean of %d geometries" % (SET_TITLE["unseen_loads"], len(set(r["geometry"] for r in rs))))
    style(ax)
    ax.set_xlabel("applied traction (%s)" % units)
    ax.set_ylabel("R\u00b2")
    allr2 = [r["r2"] for r in rows if np.isfinite(r["r2"])]
    if allr2:
        ax.set_ylim(max(min(allr2) - 0.05, -1.0), 1.01)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_title("R\u00b2 against load - thin lines are single geometries, thick lines the mean", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def summary_heatmap(rows, order, info, test_loads, units, path):
    rs = [r for r in rows if r["set"] in ("unseen_interp", "unseen_extrap")]
    if not rs:
        return
    geoms = sorted(set(r["geometry"] for r in rs),
                   key=lambda g: (0 if any(r["set"] == "unseen_interp" for r in rs if r["geometry"] == g) else 1,
                                  order.get(g, 999), g))
    loads = sorted(set(r["load"] for r in rs))
    M = np.full((len(geoms), len(loads)), np.nan)
    setof = {}
    for r in rs:
        M[geoms.index(r["geometry"]), loads.index(r["load"])] = r["r2"]
        setof[r["geometry"]] = r["set"]
    vmin = min(0.5, np.floor(np.nanmin(M) * 10) / 10.0) if np.isfinite(np.nanmin(M)) else 0.5
    vmin = max(vmin, 0.0)
    fig, ax = plt.subplots(figsize=(max(12, 0.62 * len(loads) + 4), 0.42 * len(geoms) + 2.6))
    im = ax.imshow(np.clip(M, vmin, 1.0), cmap="Blues", vmin=vmin, vmax=1.0, aspect="auto")
    for i in range(len(geoms)):
        for j in range(len(loads)):
            v = M[i, j]
            if np.isfinite(v):
                frac = (min(max(v, vmin), 1.0) - vmin) / (1.0 - vmin + 1e-12)
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.8,
                        color="white" if frac > 0.55 else INK)
    ax.set_xticks(range(len(loads)))
    ax.set_xticklabels(["%g%s" % (L, "\n(test)" if L in test_loads else "") for L in loads], fontsize=7.5)
    ax.set_yticks(range(len(geoms)))
    ax.set_yticklabels(["%s   [%s]" % (label(g, info), "interp" if setof[g] == "unseen_interp" else "EXTRAP")
                        for g in geoms], fontsize=9)
    n_int = sum(1 for g in geoms if setof[g] == "unseen_interp")
    if 0 < n_int < len(geoms):
        ax.axhline(n_int - 0.5, color=INK, lw=1.5)
    ax.set_xlabel("load (%s)" % units)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("R\u00b2  (below %.1f: lightest colour, number exact)" % vmin, fontsize=8.5)
    ax.set_title("Unseen GEOMETRIES - R\u00b2 at every load (above the line: interpolation, below: extrapolation)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def scatter_grid(keep, order, info, units, load, path):
    items = sorted(keep, key=lambda k: (0 if k["set"] == "unseen_interp" else 1, order.get(k["name"], 999)))
    if not items:
        return
    ncol = 6
    nrow = int(np.ceil(len(items) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.25 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, items):
        t, p = k["truth"], k["pred"]
        ax.scatter(t, p, s=3, alpha=0.35, color=SET_COLOUR[k["set"]], edgecolors="none")
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=1)
        ax.set_title("%s\nR\u00b2 = %.4f%s" % (label(k["name"], info), r2_score(t, p),
                                              "   EXTRAP" if k["set"] == "unseen_extrap" else ""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(1)
        style(ax)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.supxlabel("Abaqus FE von Mises (%s)" % units, fontsize=11)
    fig.supylabel("MGN predicted (%s)" % units, fontsize=11)
    fig.suptitle("Unseen GEOMETRIES at %g %s - predicted vs actual, one dot per node (dashed = perfect)"
                 % (load, units), fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join("results_aug20", "mgn_best.pt"))
    ap.add_argument("--train-data", default=os.path.join("data", "dataset_own_aug20.pt"))
    ap.add_argument("--loads-dir", default=os.path.join("data", "unseen_own_loads"))
    ap.add_argument("--interp-dir", default=os.path.join("data", "unseen_own_interp"))
    ap.add_argument("--extrap-dir", default=os.path.join("data", "unseen_own_extrap"))
    ap.add_argument("--manifest", default=os.path.join("data", "own_manifest.csv"))
    ap.add_argument("--out", default=None, help="default: <checkpoint folder>/unseen_eval")
    ap.add_argument("--contour-loads", nargs="+", default=["test"],
                    help="'test' (default: the unseen test loads), 'all', or numbers, e.g. 5000")
    ap.add_argument("--no-contours", action="store_true")
    ap.add_argument("--units", default="psi")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    a = ap.parse_args()

    if not os.path.isfile(a.ckpt):
        sys.exit("No checkpoint at %s" % a.ckpt)
    out = a.out or os.path.join(os.path.dirname(a.ckpt) or ".", "unseen_eval")
    os.makedirs(out, exist_ok=True)

    sets = [("unseen_loads", a.loads_dir), ("unseen_interp", a.interp_dir), ("unseen_extrap", a.extrap_dir)]
    data = {}
    for key, folder in sets:
        data[key] = read_set(folder) if os.path.isdir(folder) else []
        print("%-14s %-28s %4d files" % (key, folder, len(data[key])))
    if not any(data.values()):
        sys.exit("No test files found. Copy data/unseen_own_* over from the machine that ran "
                 "organize_own_dataset.py.")

    t0 = time.time()
    mgn, desc, train_loads = load_model(a)
    bands = bands_from(train_loads)
    info, order = read_manifest(a.manifest)
    test_loads = sorted(set(d["load"] for d in data["unseen_loads"])) or \
        sorted(set(d["load"] for k in data for d in data[k] if d["load"] not in train_loads))
    gap_load = [L for L in test_loads if role(L, bands) == "in the gap between bands"]
    fig3_load = gap_load[0] if gap_load else (test_loads[len(test_loads) // 2] if test_loads else None)
    if a.contour_loads == ["all"]:
        contour_loads = None
    elif a.contour_loads == ["test"]:
        contour_loads = set(test_loads)
    else:
        contour_loads = set(float(x) for x in a.contour_loads)

    print("=" * 78)
    print("checkpoint   %s  ->  %s" % (a.ckpt, desc))
    print("device       %s    edge augmentation %.0f%% (seed %d)"
          % (mgn.device, 100 * mgn.aug_perc, mgn.aug_seed))
    print("trained on   %s" % ", ".join("%g-%g" % b for b in bands) + " %s" % a.units)
    print("test loads   %s %s   (Fig. 3 style plot at %s)"
          % (", ".join("%g" % L for L in test_loads), a.units, "%g" % fig3_load if fig3_load else "-"))
    print("output       %s" % out)
    print("=" * 78)

    rows, keep, topo = [], [], {}
    n_total = sum(len(v) for v in data.values())
    done = 0
    for key, _ in sets:
        cdir = os.path.join(out, "contours", key)
        for item in data[key]:
            tk = (item["name"], len(item["coords"]))
            if tk not in topo:
                ei, nt, fl = build_topology(item["coords"], item["tris"], item["name"],
                                            aug_perc=mgn.aug_perc, aug_seed=mgn.aug_seed)[:3]
                topo[tk] = (ei, nt, fl)
            ei, nt, fl = topo[tk]
            item["node_types"] = nt
            case = Case(geometry=item["name"], coordinates=item["coords"], edge_index=ei, node_types=nt,
                        von_mises=np.zeros(len(item["coords"]), dtype=np.float32),
                        metadata={"load": item["load"]}, edge_flag=fl)
            with torch.no_grad():
                pred = np.asarray(mgn.predict(case), dtype=np.float64).ravel()
            m = metrics(item["truth"], pred)
            rows.append(dict(set=key, geometry=item["name"], load=item["load"], n_nodes=len(pred),
                             **dict((k, v) for k, v in m.items() if not k.startswith("i_"))))
            if key != "unseen_loads" and fig3_load is not None and abs(item["load"] - fig3_load) < 1e-6:
                keep.append(dict(set=key, name=item["name"], truth=item["truth"], pred=pred))
            if not a.no_contours and (contour_loads is None or item["load"] in contour_loads):
                os.makedirs(cdir, exist_ok=True)
                contour_figure(item, dict(pred=pred, m=m), info, key, a.units,
                               os.path.join(cdir, "%s__%g.png" % (short(item["name"]), item["load"])))
            done += 1
            if done % 25 == 0 or done == n_total:
                print("  %4d/%d files   %.0f s" % (done, n_total, time.time() - t0))

    # ---- tables -----------------------------------------------------------
    csv_path = os.path.join(out, "metrics_all.csv")
    cols = ["set", "geometry", "load", "n_nodes", "r2", "rmse", "mae", "max_abs_err",
            "peak_true", "peak_pred", "peak_err_pct"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols[:4] + ["r2", "rmse_" + a.units, "mae_" + a.units, "max_abs_err_" + a.units,
                               "peak_true_" + a.units, "peak_pred_" + a.units, "peak_err_pct"])
        for r in sorted(rows, key=lambda r: (r["set"], order.get(r["geometry"], 999), r["geometry"], r["load"])):
            w.writerow([r["set"], r["geometry"], "%g" % r["load"], r["n_nodes"]] +
                       ["%.6f" % r[c] for c in cols[4:]])

    print("\n%-16s %6s %9s %9s %9s  %-34s %s" % ("test", "files", "mean R2", "median", "worst", "worst case",
                                                  "mean |peak err|"))
    print("-" * 104)
    for key, _ in sets:
        rs = [r for r in rows if r["set"] == key]
        if not rs:
            continue
        r2s = np.array([r["r2"] for r in rs])
        wr = rs[int(np.nanargmin(r2s))]
        print("%-16s %6d %9.4f %9.4f %9.4f  %-34s %.1f %%"
              % (key, len(rs), np.nanmean(r2s), np.nanmedian(r2s), wr["r2"],
                 "%s @ %g" % (short(wr["geometry"]), wr["load"]),
                 np.mean([abs(r["peak_err_pct"]) for r in rs])))
    for key, title in (("unseen_loads", "UNSEEN LOADS - R2 per geometry"),
                       ("unseen_interp", "UNSEEN GEOMETRIES (interpolation) - R2 at the test loads"),
                       ("unseen_extrap", "UNSEEN GEOMETRIES (extrapolation) - R2 at the test loads")):
        rs = [r for r in rows if r["set"] == key and r["load"] in test_loads]
        if not rs:
            continue
        print("\n" + title)
        print("  %-30s" % "geometry" + "".join("%12s" % ("%g %s" % (L, a.units)) for L in test_loads))
        for g in sorted(set(r["geometry"] for r in rs), key=lambda g: (order.get(g, 999), g)):
            vals = dict((r["load"], r["r2"]) for r in rs if r["geometry"] == g)
            print("  %-30s" % label(g, info)[:30] + "".join("%12.4f" % vals.get(L, np.nan) for L in test_loads))

    # ---- summary figures --------------------------------------------------
    summary_unseen_loads(rows, order, info, test_loads, bands, a.units, os.path.join(out, "summary_unseen_loads.png"))
    summary_r2_vs_load(rows, bands, a.units, os.path.join(out, "summary_r2_vs_load.png"))
    summary_heatmap(rows, order, info, test_loads, a.units,
                    os.path.join(out, "summary_unseen_geometries_heatmap.png"))
    if fig3_load is not None:
        scatter_grid(keep, order, info, a.units, fig3_load,
                     os.path.join(out, "scatter_unseen_geometries_%g.png" % fig3_load))

    n_png = len(glob.glob(os.path.join(out, "contours", "*", "*.png")))
    print("\nwrote        %s" % csv_path)
    print("             %s/summary_*.png, scatter_unseen_geometries_*.png" % out)
    print("             %d contour plots in %s" % (n_png, os.path.join(out, "contours")))
    print("time         %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
