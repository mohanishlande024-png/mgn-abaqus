r"""
Train the MGN on the Abaqus dataset, with checkpointing and resume.

    python scripts\train.py
    python scripts\train.py --epochs 50        quick sanity run
    python scripts\train.py --resume           continue from last checkpoint

    python scripts\train.py --data data\dataset_aug20.pt
        EDGE AUGMENTATION run (EA-GNN, 20% random long-range edges). The
        augmentation settings are read from the dataset file itself, and
        results go to results_aug20\ so the baseline results\ is untouched.

WHY NOT JUST CALL MGN.fit()
    fit() calls _build_model() every time, which re-initialises the weights.
    Calling it in chunks would restart training from scratch each chunk, so
    there would be no way to checkpoint a multi-hour run.

    This script runs the epoch loop itself but calls THEIR _train_batch()
    for every step. That is where the forward pass, loss, backward pass,
    gradient clipping and optimiser step all happen - so the maths is
    entirely theirs and unchanged. All this file adds is the loop, the
    checkpoints and the log.

WHAT IT WRITES (into results\, or results_augNN\ for an augmented dataset)
    mgn_latest.pt        full state: weights + optimiser + epoch counter
    mgn_best.pt          weights only, at the lowest loss seen
    loss_log.csv         epoch, loss, seconds - open in Excel
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.dataset import load_dataset                          # noqa: E402
from mgn.augmented_trainer import AugMGN, results_dir_for     # noqa: E402


def fmt(seconds):
    s = int(seconds)
    return "%dh%02dm%02ds" % (s // 3600, (s % 3600) // 60, s % 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join("data", "dataset.pt"))
    ap.add_argument("--epochs", type=int, default=10000,
                    help="paper uses 10000")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="the repo's script uses 1; changing this changes "
                         "the optimisation and is a deviation")
    ap.add_argument("--lr", type=float, default=1e-5, help="paper uses 1e-5")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="the paper sets no seed; we do, for reproducibility")
    ap.add_argument("--results", default=None,
                    help="output folder (default: results, or results_augNN "
                         "for an augmented dataset)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- data ----------------------------------------------------------
    if not os.path.isfile(a.data):
        print("No %s. Build it first:" % a.data)
        print("    python -m mgn.dataset --raw data/raw --out data/dataset.pt")
        return 1

    cases, aug = load_dataset(a.data, return_meta=True)

    RESULTS = a.results or results_dir_for(aug["aug_perc"])
    LATEST = os.path.join(RESULTS, "mgn_latest.pt")
    BEST = os.path.join(RESULTS, "mgn_best.pt")
    LOG = os.path.join(RESULTS, "loss_log.csv")
    if not os.path.isdir(RESULTS):
        os.makedirs(RESULTS)
    y = np.concatenate([c.von_mises for c in cases])
    geoms = sorted(set(c.geometry for c in cases))

    print("=" * 66)
    print("MGN TRAINING")
    print("=" * 66)
    print("cases         %d  (%d geometries x %d loads)"
          % (len(cases), len(geoms), len(cases) // max(len(geoms), 1)))
    print("nodes total   %d" % len(y))
    print("stress range  %.1f to %.1f psi" % (y.min(), y.max()))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device        %s%s" % (dev, "  (" + torch.cuda.get_device_name(0)
                                  + ")" if dev == "cuda" else ""))
    print("epochs        %d   batch %d   lr %g"
          % (a.epochs, a.batch_size, a.lr))
    if aug["aug_perc"] > 0:
        n_aug = sum(int(c.edge_flag.sum()) for c in cases) // len(cases)
        n_all = sum(c.edge_index.shape[1] for c in cases) // len(cases)
        print("augmentation  %.0f%%  (seed %d)  ~%d of %d edges per graph "
              "are augmented" % (100 * aug["aug_perc"], aug["aug_seed"],
                                 n_aug, n_all))
    else:
        print("augmentation  none (baseline graph)")
    print("results       %s" % RESULTS)
    print("=" * 66)

    # ---- model ---------------------------------------------------------
    mgn = AugMGN(num_layers=a.layers, hidden_channels=a.hidden,
                 embedding_dim=a.embedding, learning_rate=a.lr,
                 epochs=a.epochs, global_features=["load"],
                 aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"])

    # Reproduce fit()'s setup, but stop short of the training loop so we can
    # drive the epochs ourselves.
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    mgn._build_model()

    print("\nnode types    %s" % mgn.node_type_to_id)
    n_par = sum(p.numel() for p in mgn._model.parameters())
    print("parameters    %d" % n_par)
    print("y mean/std    %.1f / %.1f psi"
          % (mgn._y_mean.item(), mgn._y_std.item()))

    optimizer = torch.optim.Adam(mgn._model.parameters(), lr=a.lr)
    loss_fn = torch.nn.MSELoss()

    start_epoch = 0
    best = float("inf")

    # ---- resume --------------------------------------------------------
    if a.resume and os.path.isfile(LATEST):
        ck = torch.load(LATEST, map_location=mgn.device, weights_only=False)
        ck_aug = ck.get("augmentation", {"aug_perc": 0.0, "aug_seed": 0})
        if ck_aug != aug:
            print("\nREFUSING TO RESUME: %s was trained with augmentation %s "
                  "but %s has %s." % (LATEST, ck_aug, a.data, aug))
            return 1
        mgn._model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best = ck.get("best", float("inf"))
        mgn.train_losses = ck.get("train_losses", [])
        print("\nRESUMED from epoch %d  (best loss %.6f)" % (start_epoch, best))
    elif a.resume:
        print("\n--resume given but no %s found; starting fresh." % LATEST)

    if not os.path.isfile(LOG) or start_epoch == 0:
        open(LOG, "w").write("epoch,loss,seconds\n")

    # ---- train ---------------------------------------------------------
    print("\nepoch      loss        elapsed     eta")
    print("-" * 52)
    t0 = time.time()
    interrupted = False

    try:
        for epoch in range(start_epoch, a.epochs):
            # THEIR code does the actual work: shuffle, batch, forward,
            # loss, backward, clip, step.
            loss = mgn._train_epoch(cases, fem_data, a.batch_size,
                                    optimizer, loss_fn)
            mgn.train_losses.append(loss)

            elapsed = time.time() - t0
            done = epoch - start_epoch + 1
            eta = elapsed / done * (a.epochs - epoch - 1)

            open(LOG, "a").write("%d,%.8f,%.1f\n" % (epoch + 1, loss, elapsed))

            if loss < best:
                best = loss
                torch.save({"model": mgn._model.state_dict(),
                            "epoch": epoch + 1, "loss": loss,
                            "augmentation": aug}, BEST)

            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print("%6d   %.6f   %s   %s"
                      % (epoch + 1, loss, fmt(elapsed), fmt(eta)))

            if (epoch + 1) % a.ckpt_every == 0:
                torch.save({"model": mgn._model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch + 1, "best": best,
                            "train_losses": mgn.train_losses,
                            "augmentation": aug}, LATEST)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Saving checkpoint...")

    # always leave a resumable checkpoint behind
    torch.save({"model": mgn._model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": len(mgn.train_losses),
                "best": best, "train_losses": mgn.train_losses,
                "augmentation": aug}, LATEST)

    print("\n" + "=" * 66)
    print("%s at epoch %d after %s"
          % ("Stopped" if interrupted else "Finished",
             len(mgn.train_losses), fmt(time.time() - t0)))
    if mgn.train_losses:
        print("final loss    %.6f" % mgn.train_losses[-1])
        print("best loss     %.6f" % best)
    print("checkpoint    %s" % LATEST)
    print("log           %s" % LOG)
    if interrupted or len(mgn.train_losses) < a.epochs:
        print("\nContinue with:  python scripts\\train.py --resume --data %s"
              % a.data)
    else:
        # save in the paper's own format so it can be compared to theirs
        mgn.save(os.path.join(RESULTS, "multi_geometry_mgn.pt"))
        print("saved          %s"
              % os.path.join(RESULTS, "multi_geometry_mgn.pt"))
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
