"""
Extract mesh + nodal von Mises from Abaqus ODB files into .npz.

RUN THIS WITH ABAQUS' OWN PYTHON (2.7), NOT YOUR VENV:

    abaqus python extract_odb.py data/odb/8in__1000.odb
    abaqus python extract_odb.py data/odb/*.odb

Filename convention:  <geometry>__<load>.odb   e.g.  8in__1000.odb
The load value is parsed from the filename. Override with --load <psi>.

Writes one file per ODB to data/raw/<geometry>__<load>.npz containing:
    coords    (N, 2) float32   inches, node order = sorted node label
    tris      (T, 3) int32     0-based, remapped to coords order
    mises     (N,)   float32   psi, nodal-averaged from ELEMENT_NODAL
    load      scalar float32   psi
    geometry  str

Deliberately imports nothing from `abaqus` or `abaqusConstants.__main__` so it
runs under `abaqus python` (no CAE licence token needed).
"""

from __future__ import print_function

import os
import sys
import glob

import numpy as np
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL


OUT_DIR = os.path.join("data", "raw")


def parse_name(odb_path):
    """'.../8in__1000.odb' -> ('8in', 1000.0)"""
    stem = os.path.splitext(os.path.basename(odb_path))[0]
    if "__" in stem:
        geom, load = stem.rsplit("__", 1)
        try:
            return geom, float(load)
        except ValueError:
            pass
    return stem, None


def extract(odb_path, load_override=None, out_dir=OUT_DIR):
    geometry, load = parse_name(odb_path)
    if load_override is not None:
        load = load_override
    if load is None:
        raise ValueError(
            "Could not parse load from '%s'. Rename to <geom>__<load>.odb "
            "or pass --load <psi>." % odb_path
        )

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        inst_names = odb.rootAssembly.instances.keys()
        if len(inst_names) != 1:
            raise ValueError("Expected 1 instance, found %d: %s"
                             % (len(inst_names), list(inst_names)))
        inst = odb.rootAssembly.instances[inst_names[0]]

        # ---- nodes -------------------------------------------------------
        # Abaqus labels are 1-based and may have gaps. Sort by label and
        # remap to 0-based contiguous indices. EVERYTHING downstream
        # (connectivity, stress) must use this same map.
        labels = np.array([n.label for n in inst.nodes], dtype=np.int64)
        coords3 = np.array([n.coordinates for n in inst.nodes], dtype=np.float64)

        order = np.argsort(labels)
        labels = labels[order]
        coords = coords3[order][:, :2].astype(np.float32)   # drop z

        lab2idx = {}
        for i in range(len(labels)):
            lab2idx[int(labels[i])] = i
        n_nodes = len(labels)

        # ---- elements ----------------------------------------------------
        tris = []
        bad = set()
        for e in inst.elements:
            conn = e.connectivity
            if len(conn) != 3:
                bad.add(e.type)
                continue
            tris.append([lab2idx[int(c)] for c in conn])

        if bad:
            raise ValueError(
                "Non-triangular elements found (%s). The MGN graph assumes "
                "linear triangles - remesh the part with CPS3." % sorted(bad)
            )
        tris = np.array(tris, dtype=np.int32)

        # ---- stress ------------------------------------------------------
        # S lives at integration points by default, where nodeLabel is None.
        # getSubset(position=ELEMENT_NODAL) extrapolates to nodes and gives
        # ONE VALUE PER ELEMENT-NODE PAIR, so a shared node appears several
        # times. Accumulate and divide -- that is what Abaqus' own nodal
        # averaging does at a 100% threshold.
        step_name = odb.steps.keys()[-1]
        frame = odb.steps[step_name].frames[-1]
        if "S" not in frame.fieldOutputs:
            raise ValueError("No 'S' field output in %s. Request S, Mises."
                             % odb_path)

        s = frame.fieldOutputs["S"].getSubset(position=ELEMENT_NODAL)

        acc = np.zeros(n_nodes, dtype=np.float64)
        cnt = np.zeros(n_nodes, dtype=np.int64)
        for v in s.values:
            i = lab2idx[int(v.nodeLabel)]
            acc[i] += v.mises
            cnt[i] += 1

        if (cnt == 0).any():
            raise ValueError("%d orphan nodes with no stress contribution."
                             % int((cnt == 0).sum()))
        mises = (acc / cnt).astype(np.float32)

        # ---- write -------------------------------------------------------
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out = os.path.join(out_dir, "%s__%d.npz" % (geometry, int(load)))
        np.savez_compressed(
            out,
            coords=coords,
            tris=tris,
            mises=mises,
            load=np.float32(load),
            geometry=np.array(geometry),
        )
        print("%-34s N=%4d  T=%4d  max_mises=%10.1f  -> %s"
              % (os.path.basename(odb_path), n_nodes, len(tris),
                 mises.max(), out))
        return out
    finally:
        odb.close()


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    load_override = None
    if "--load" in argv:
        load_override = float(argv[argv.index("--load") + 1])
        args = [a for a in args if a != str(load_override)]

    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    if not paths:
        print(__doc__)
        return 1

    for p in paths:
        extract(p, load_override)
    print("\nDone: %d ODB(s)." % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
