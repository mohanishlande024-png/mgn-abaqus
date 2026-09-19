r"""
Check one extracted .npz against the paper's mesh.

Needs only numpy -- no torch, no venv. Run from the project root:

    py scripts\check_npz.py
    py scripts\check_npz.py data\raw\circle8__1000.npz

Reference values below come from the paper's own cached mesh for the
equivalent geometry, so matching them means your Abaqus data and theirs
are interchangeable.
"""

import os
import sys

import numpy as np

# make `import mgn.graph` work no matter where this is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgn.graph import build_edges, classify_nodes, type_counts   # noqa: E402


REFERENCE = {
    "N": 788,
    "E": 4388,
    "types": {"applied_load": 12, "fixed": 12, "free": 120,
              "hole": 26, "interior": 618},
}


def check(path):
    d = np.load(path, allow_pickle=False)
    coords = d["coords"].astype(np.float32)
    tris = d["tris"].astype(np.int64)
    mises = d["mises"]
    load = float(d["load"])

    edge_index = build_edges(tris)
    ids = classify_nodes(coords, tris)
    counts = type_counts(ids)

    n, e = len(coords), edge_index.shape[1]
    x, y = coords[:, 0], coords[:, 1]

    print("\n%s" % os.path.basename(path))
    print("-" * 58)
    print("  nodes          %5d      (paper: %d)" % (n, REFERENCE["N"]))
    print("  triangles      %5d      T/N = %.2f  (3.0 means quads slipped in)"
          % (len(tris), len(tris) / float(n)))
    print("  edges          %5d      (paper: %d)" % (e, REFERENCE["E"]))
    print("  extent         x [%.2f, %.2f]   y [%.2f, %.2f]"
          % (x.min(), x.max(), y.min(), y.max()))
    print("  load           %.0f psi" % load)
    print("  max mises      %.1f psi   (SCF = %.2f)" % (mises.max(),
                                                        mises.max() / load))
    print("\n  node types")
    for k in sorted(counts):
        ref = REFERENCE["types"].get(k, 0)
        flag = "" if abs(counts[k] - ref) <= max(3, 0.15 * ref) else "   <-- off"
        print("    %-14s %5d      (paper: %4d)%s" % (k, counts[k], ref, flag))

    print("\n  verdict")
    ok = True
    if counts["hole"] == 0:
        print("    FAIL  no hole nodes -- boundary loop detection failed")
        ok = False
    if counts["fixed"] == 0 or counts["applied_load"] == 0:
        print("    FAIL  left or right edge not detected -- check plate extent")
        ok = False
    if abs(len(tris) / float(n) - 1.75) > 0.4:
        print("    WARN  T/N looks wrong -- mesh may not be pure triangles")
    if not (0.85 * REFERENCE["N"] <= n <= 1.15 * REFERENCE["N"]):
        print("    WARN  node count %d is >15%% from %d -- adjust MESH_SIZE"
              % (n, REFERENCE["N"]))
    elif ok:
        print("    OK    mesh and node types match the paper closely enough")
    print()
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        import glob
        args = sorted(glob.glob(os.path.join("data", "raw", "*.npz")))
    if not args:
        print("No .npz found. Run extract_odb.py first.")
        sys.exit(1)
    for p in args:
        check(p)
