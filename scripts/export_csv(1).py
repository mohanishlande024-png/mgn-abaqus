r"""
Export the extracted .npz data to CSV.

Numpy only -- no torch needed. Run from the project root:

    py scripts\export_csv.py                    # one long nodes.csv
    py scripts\export_csv.py --scale-loads      # all 20 loads per geometry
    py scripts\export_csv.py --per-case         # one CSV per case
    py scripts\export_csv.py --edges            # also write edge lists

Outputs land in data\csv\.

WHAT CSV IS AND ISN'T GOOD FOR
    Fine for: eyeballing the data, and training the paper's Random Forest /
    Gradient Boosting / KNN baselines, which take (x, y, load) -> von_mises
    per node and ignore topology entirely.

    NOT enough for MGN: the graph needs the edge list, which does not fit a
    flat node table. Use mgn/dataset.py -> dataset.pt for training. The
    --edges flag writes the connectivity separately if you want it.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.graph import NODE_TYPE_TO_ID, build_edges, classify_nodes  # noqa: E402


ID_TO_NAME = {v: k for k, v in NODE_TYPE_TO_ID.items()}

PAPER_LOADS = ([1000, 1388, 1777, 2166, 2555, 2944, 3333, 3722, 4111, 4500] +
               [5500, 5999, 6499, 6999, 7499, 7999, 8499, 8999, 9499, 10000])


def _read_name(arr):
    """Abaqus' Python 2.7 writes the name as bytes; numpy 2.x hands it back
    as np.bytes_, whose str() is the repr rather than the text."""
    try:
        val = arr.item()
    except Exception:
        val = arr
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return (_read_name(d["geometry"]),
            d["coords"].astype(np.float64),
            d["tris"].astype(np.int64),
            d["mises"].astype(np.float64),
            float(d["load"]))


def node_rows(geometry, coords, types, mises, load):
    out = []
    for i in range(len(coords)):
        out.append("%s,%.1f,%d,%.6f,%.6f,%s,%.4f" % (
            geometry, load, i, coords[i, 0], coords[i, 1],
            ID_TO_NAME[int(types[i])], mises[i]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--out", default=os.path.join("data", "csv"))
    ap.add_argument("--scale-loads", action="store_true",
                    help="expand each solve into the paper's 20 load cases")
    ap.add_argument("--per-case", action="store_true",
                    help="one CSV per case instead of one long file")
    ap.add_argument("--edges", action="store_true",
                    help="also write an edge list per geometry")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.raw, "*.npz")))
    if not paths:
        print("No .npz in %s -- run extract_odb.py first." % a.raw)
        return 1
    if not os.path.isdir(a.out):
        os.makedirs(a.out)

    header = "geometry,load_psi,node_id,x_in,y_in,node_type,von_mises_psi"
    all_rows = []
    summary = ["geometry,n_nodes,n_triangles,n_edges,load_psi,max_mises_psi,scf"]

    for p in paths:
        geometry, coords, tris, mises, ref_load = load_npz(p)
        types = classify_nodes(coords.astype(np.float32), tris)
        edge_index = build_edges(tris)

        loads = PAPER_LOADS if a.scale_loads else [ref_load]
        for L in loads:
            y = mises * (L / ref_load)
            rows = node_rows(geometry, coords, types, y, L)
            if a.per_case:
                f = os.path.join(a.out, "%s__%d.csv" % (geometry, int(L)))
                open(f, "w").write(header + "\n" + "\n".join(rows) + "\n")
            else:
                all_rows.extend(rows)
            summary.append("%s,%d,%d,%d,%.1f,%.4f,%.4f" % (
                geometry, len(coords), len(tris), edge_index.shape[1],
                L, y.max(), y.max() / L))

        if a.edges:
            f = os.path.join(a.out, "%s_edges.csv" % geometry)
            lines = ["source,target"]
            for k in range(edge_index.shape[1]):
                lines.append("%d,%d" % (edge_index[0, k], edge_index[1, k]))
            open(f, "w").write("\n".join(lines) + "\n")

        print("%-14s %5d nodes x %2d load(s)" % (geometry, len(coords),
                                                 len(loads)))

    if not a.per_case:
        f = os.path.join(a.out, "nodes.csv")
        open(f, "w").write(header + "\n" + "\n".join(all_rows) + "\n")
        print("\nnodes.csv      %d rows  (%.1f MB)"
              % (len(all_rows), os.path.getsize(f) / 1e6))

    f = os.path.join(a.out, "summary.csv")
    open(f, "w").write("\n".join(summary) + "\n")
    print("summary.csv    %d rows" % (len(summary) - 1))
    print("\nWrote to %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
