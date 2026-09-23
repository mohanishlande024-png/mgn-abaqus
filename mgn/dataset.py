"""
Compile Abaqus .npz exports into one serialized dataset.

    python -m mgn.dataset --raw data/raw --out data/dataset.pt --scale-loads
    python -m mgn.dataset --raw data/raw --scale-loads --aug-perc 0.2
        -> writes data/dataset_aug20.pt  (baseline dataset.pt is left alone)

A Case duck-types the FEMObject attributes MGN actually reads:
    .coordinates  .edge_index  .node_types  .metadata  .von_mises
so the vendored trainer consumes it with `_classify_nodes` returning
`case.node_types` and nothing else changed.

Topology is stored ONCE PER GEOMETRY, not once per load case. The 20 load
cases of a geometry share a mesh, so this cuts the file ~10x and makes the
linear-scaling shortcut natural.

EDGE AUGMENTATION
    --aug-perc 0.2 adds 20% random long-range edges per geometry (EA-GNN,
    Gladstone et al. 2024). Every graph in the project - training, predict.py,
    fig3_unseen.py - is built by build_topology() below, so all of them get
    the same augmentation from the same seed. --aug-perc 0 (the default) is
    the exact baseline graph.
"""

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .graph import (
    NODE_TYPE_TO_ID,
    augment_edges,
    build_edges,
    classify_nodes,
    geometry_seed,
    type_counts,
)


@dataclass
class Case:
    """One (geometry, load) sample. Attribute names match FEMObject."""
    geometry: str
    coordinates: np.ndarray        # (N, 2) float32, inches
    edge_index: torch.Tensor       # (2, E) int64   real edges, then augmented
    node_types: np.ndarray         # (N,) int64 ids
    von_mises: np.ndarray          # (N,) float32, psi
    metadata: Dict = field(default_factory=dict)   # {"load": psi}
    edge_flag: Optional[torch.Tensor] = None       # (E,) float32, 1 = augmented

    @property
    def n_nodes(self):
        return len(self.coordinates)


def build_topology(coords, tris, geometry, aug_perc=0.0, aug_seed=0):
    """
    The ONE function that turns a mesh into a graph. Used for training data
    and for every inference script, so train and test graphs cannot drift.

    Order matters: node types come from the REAL mesh first, and only then
    are augmented edges appended.

    Returns (edge_index torch int64 (2,E), node_types np int64 (N,),
             edge_flag torch float32 (E,))
    """
    coords = np.asarray(coords, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int64)
    edge_index = build_edges(tris)                    # real mesh edges only
    node_types = classify_nodes(coords, tris)         # real boundary only
    edge_index, edge_flag = augment_edges(
        edge_index, len(coords), perc=aug_perc,
        seed=geometry_seed(geometry, aug_seed))
    return (torch.from_numpy(edge_index),
            node_types,
            torch.from_numpy(edge_flag))


class Topology:
    """Mesh-derived quantities shared by every load case of one geometry."""

    def __init__(self, coords, tris, geometry="", aug_perc=0.0, aug_seed=0):
        self.coords = coords.astype(np.float32)
        self.tris = tris
        self.edge_index, self.node_types, self.edge_flag = build_topology(
            self.coords, tris, geometry, aug_perc, aug_seed)

    @property
    def n_augmented(self):
        return int(self.edge_flag.sum().item())


def _read_name(arr):
    """np.savez from Abaqus' Python 2.7 stores the name as bytes. numpy 2.x
    returns it as np.bytes_, whose str() is the repr - "np.bytes_(b'x')" -
    not the text. Decode it properly, or every geometry filter downstream
    (per-geometry R2, train/test splits) silently fails to match."""
    try:
        val = arr.item()
    except Exception:
        val = arr
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def compile_dataset(raw_dir="data/raw", scale_loads=None, verbose=True,
                    aug_perc=0.0, aug_seed=0):
    """
    Read every .npz in raw_dir and return (cases, topologies).

    scale_loads: optional list of psi values. For each geometry, the single
    solved case is scaled to produce these loads. Linear elasticity with no
    prescribed displacement makes sigma(k*P) == k*sigma(P) exactly -- but
    VERIFY IT on two real Abaqus runs before you trust it.

    aug_perc / aug_seed: edge augmentation, see build_topology().
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError("No .npz files in %s" % raw_dir)

    topologies: Dict[str, Topology] = {}
    cases: List[Case] = []

    for p in paths:
        d = np.load(p, allow_pickle=False)
        geom = _read_name(d["geometry"])
        coords = d["coords"].astype(np.float32)
        tris = d["tris"].astype(np.int64)
        mises = d["mises"].astype(np.float32)
        load = float(d["load"])

        if len(coords) != len(mises):
            raise ValueError(
                "%s: %d nodes but %d stress values -- label remap is wrong."
                % (p, len(coords), len(mises))
            )

        if geom not in topologies:
            topologies[geom] = Topology(coords, tris, geom, aug_perc, aug_seed)
            if verbose:
                t = topologies[geom]
                aug = ("  +%d aug" % t.n_augmented) if aug_perc > 0 else ""
                print("%-28s N=%4d E=%5d%s  %s"
                      % (geom, len(coords), t.edge_index.shape[1], aug,
                         type_counts(t.node_types)))
        topo = topologies[geom]

        if len(coords) != len(topo.coords):
            raise ValueError(
                "%s: mesh differs from the first case of '%s'. Every load "
                "case of a geometry must reuse the same mesh." % (p, geom)
            )

        targets = ([(load, mises)] if scale_loads is None
                   else [(L, mises * np.float32(L / load)) for L in scale_loads])

        for L, y in targets:
            cases.append(Case(
                geometry=geom,
                coordinates=topo.coords,
                edge_index=topo.edge_index,
                node_types=topo.node_types,
                von_mises=y,
                metadata={"load": float(L)},
                edge_flag=topo.edge_flag,
            ))

    if verbose:
        print("\n%d geometries -> %d cases, %d nodes total"
              % (len(topologies), len(cases), sum(c.n_nodes for c in cases)))
        if aug_perc > 0:
            print("edge augmentation: %.0f%% of mesh edges, base seed %d"
                  % (100 * aug_perc, aug_seed))
    return cases, topologies


def save(cases, path="data/dataset.pt", aug_perc=0.0, aug_seed=0):
    """Serialize. Topology is deduplicated by geometry name on write."""
    has_aug = any(c.edge_flag is not None and float(c.edge_flag.sum()) > 0
                  for c in cases)
    if has_aug != (aug_perc > 0):
        raise ValueError("save(): cases %s augmented edges but aug_perc=%g"
                         % ("contain" if has_aug else "have no", aug_perc))
    topo = {}
    for c in cases:
        if c.geometry not in topo:
            flag = (c.edge_flag if c.edge_flag is not None
                    else torch.zeros(c.edge_index.shape[1]))
            topo[c.geometry] = {
                "coords": c.coordinates,
                "edge_index": c.edge_index,
                "node_types": c.node_types,
                "edge_flag": flag,
            }
    payload = {
        "node_type_to_id": NODE_TYPE_TO_ID,
        "augmentation": {"aug_perc": float(aug_perc), "aug_seed": int(aug_seed)},
        "topologies": topo,
        "samples": [{"geometry": c.geometry,
                     "load": c.metadata["load"],
                     "von_mises": c.von_mises} for c in cases],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    mb = os.path.getsize(path) / 1e6
    print("Wrote %s  (%d samples, %.1f MB)" % (path, len(cases), mb))


def load_dataset(path="data/dataset.pt", return_meta=False):
    """
    Rehydrate into Case objects. torch>=2.6 needs weights_only=False.

    return_meta=True -> (cases, {"aug_perc": .., "aug_seed": ..}).
    Datasets written before augmentation existed load as aug_perc = 0 with an
    all-zero edge_flag, i.e. exactly the baseline.
    """
    p = torch.load(path, weights_only=False)
    meta = p.get("augmentation", {"aug_perc": 0.0, "aug_seed": 0})
    cases = []
    for s in p["samples"]:
        t = p["topologies"][s["geometry"]]
        flag = t.get("edge_flag")
        if flag is None:
            flag = torch.zeros(t["edge_index"].shape[1])
        cases.append(Case(geometry=s["geometry"],
                          coordinates=t["coords"],
                          edge_index=t["edge_index"],
                          node_types=t["node_types"],
                          von_mises=s["von_mises"],
                          metadata={"load": s["load"]},
                          edge_flag=flag))
    return (cases, meta) if return_meta else cases


def default_dataset_path(aug_perc):
    """data/dataset.pt for the baseline, data/dataset_aug20.pt for 20% etc.
    Keeps an augmented build from ever overwriting the baseline dataset."""
    if not aug_perc or aug_perc <= 0:
        return os.path.join("data", "dataset.pt")
    return os.path.join("data", "dataset_aug%d.pt" % round(100 * aug_perc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=None,
                    help="default: data/dataset.pt, or data/dataset_augNN.pt "
                         "when --aug-perc > 0")
    ap.add_argument("--scale-loads", action="store_true",
                    help="synthesize the paper's 20 loads from one solve each")
    ap.add_argument("--aug-perc", type=float, default=0.0,
                    help="edge augmentation as a FRACTION of mesh edges, "
                         "e.g. 0.2 = Gladstone's 20%%. 0 = baseline (default)")
    ap.add_argument("--aug-seed", type=int, default=0,
                    help="base seed; each geometry adds a crc32 of its name")
    a = ap.parse_args()

    if a.aug_perc >= 1.0:
        ap.error("--aug-perc is a fraction: use 0.2 for 20%%, not 20")

    loads = None
    if a.scale_loads:
        loads = (np.linspace(1000, 4500, 10, dtype=int).tolist()
                 + np.linspace(5500, 10000, 10, dtype=int).tolist())

    out = a.out or default_dataset_path(a.aug_perc)
    cases, _ = compile_dataset(a.raw, scale_loads=loads,
                               aug_perc=a.aug_perc, aug_seed=a.aug_seed)
    save(cases, out, aug_perc=a.aug_perc, aug_seed=a.aug_seed)
