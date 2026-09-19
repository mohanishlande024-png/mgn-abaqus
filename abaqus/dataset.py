"""
Compile Abaqus .npz exports into one serialized dataset.

    python -m mgn.dataset --raw data/raw --out data/dataset.pt

A Case duck-types the FEMObject attributes MGN actually reads:
    .coordinates  .edge_index  .node_types  .metadata  .von_mises
so the vendored trainer consumes it with `_classify_nodes` returning
`case.node_types` and nothing else changed.

Topology is stored ONCE PER GEOMETRY, not once per load case. The 20 load
cases of a geometry share a mesh, so this cuts the file ~10x and makes the
linear-scaling shortcut natural.
"""

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .graph import (
    NODE_TYPE_TO_ID,
    build_edges,
    classify_nodes,
    type_counts,
)


@dataclass
class Case:
    """One (geometry, load) sample. Attribute names match FEMObject."""
    geometry: str
    coordinates: np.ndarray        # (N, 2) float32, inches
    edge_index: torch.Tensor       # (2, 2E) int64
    node_types: np.ndarray         # (N,) int64 ids
    von_mises: np.ndarray          # (N,) float32, psi
    metadata: Dict = field(default_factory=dict)   # {"load": psi}

    @property
    def n_nodes(self):
        return len(self.coordinates)


class Topology:
    """Mesh-derived quantities shared by every load case of one geometry."""

    def __init__(self, coords, tris):
        self.coords = coords.astype(np.float32)
        self.tris = tris
        self.edge_index = torch.from_numpy(build_edges(tris))
        self.node_types = classify_nodes(self.coords, tris)


def compile_dataset(raw_dir="data/raw", scale_loads=None, verbose=True):
    """
    Read every .npz in raw_dir and return (cases, topologies).

    scale_loads: optional list of psi values. For each geometry, the single
    solved case is scaled to produce these loads. Linear elasticity with no
    prescribed displacement makes sigma(k*P) == k*sigma(P) exactly -- but
    VERIFY IT on two real Abaqus runs before you trust it.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError("No .npz files in %s" % raw_dir)

    topologies: Dict[str, Topology] = {}
    cases: List[Case] = []

    for p in paths:
        d = np.load(p, allow_pickle=False)
        geom = str(d["geometry"])
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
            topologies[geom] = Topology(coords, tris)
            if verbose:
                t = topologies[geom]
                print("%-28s N=%4d E=%5d  %s"
                      % (geom, len(coords), t.edge_index.shape[1],
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
            ))

    if verbose:
        print("\n%d geometries -> %d cases, %d nodes total"
              % (len(topologies), len(cases), sum(c.n_nodes for c in cases)))
    return cases, topologies


def save(cases, path="data/dataset.pt"):
    """Serialize. Topology is deduplicated by geometry name on write."""
    topo = {}
    for c in cases:
        if c.geometry not in topo:
            topo[c.geometry] = {
                "coords": c.coordinates,
                "edge_index": c.edge_index,
                "node_types": c.node_types,
            }
    payload = {
        "node_type_to_id": NODE_TYPE_TO_ID,
        "topologies": topo,
        "samples": [{"geometry": c.geometry,
                     "load": c.metadata["load"],
                     "von_mises": c.von_mises} for c in cases],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    mb = os.path.getsize(path) / 1e6
    print("Wrote %s  (%d samples, %.1f MB)" % (path, len(cases), mb))


def load_dataset(path="data/dataset.pt"):
    """Rehydrate into Case objects. torch>=2.6 needs weights_only=False."""
    p = torch.load(path, weights_only=False)
    return [Case(geometry=s["geometry"],
                 coordinates=p["topologies"][s["geometry"]]["coords"],
                 edge_index=p["topologies"][s["geometry"]]["edge_index"],
                 node_types=p["topologies"][s["geometry"]]["node_types"],
                 von_mises=s["von_mises"],
                 metadata={"load": s["load"]})
            for s in p["samples"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/dataset.pt")
    ap.add_argument("--scale-loads", action="store_true",
                    help="synthesize the paper's 20 loads from one solve each")
    a = ap.parse_args()

    loads = None
    if a.scale_loads:
        loads = (np.linspace(1000, 4500, 10, dtype=int).tolist()
                 + np.linspace(5500, 10000, 10, dtype=int).tolist())

    cases, _ = compile_dataset(a.raw, scale_loads=loads)
    save(cases, a.out)
