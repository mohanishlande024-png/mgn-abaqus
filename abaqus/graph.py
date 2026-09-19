"""
Pure-numpy replacement for every dolfin-dependent step in MGN-Public.

Mirrors MGN._classify_boundary_nodes / _classify_nodes without FEniCS.
No torch, no dolfin, no meshio -- so it can be unit-tested standalone.
"""

import numpy as np


# Must match the released checkpoint exactly. MGN builds this map with
# sorted(set(node_types)), so alphabetical order is the contract.
# Note there is deliberately NO 'corner' type: the paper's checkpoint has 5.
NODE_TYPE_TO_ID = {
    "applied_load": 0,
    "fixed": 1,
    "free": 2,
    "hole": 3,
    "interior": 4,
}


def build_edges(tris):
    """
    Triangle connectivity -> bidirectional edge_index (2, 2E), int64.

    Matches FEMObject.__post_init__: dolfin's unique edge list, then
    concatenated with its own flip.
    """
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)                      # undirected canonical form
    e = np.unique(e, axis=0)                    # dedupe shared edges
    return np.concatenate([e.T, e.T[::-1]], axis=1).astype(np.int64)


def boundary_edges(tris):
    """Edges belonging to exactly one triangle."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_loops(coords, tris):
    """
    Split the boundary into loops and label them outer vs hole.

    Same rule as MGN._classify_boundary_nodes: connected components of the
    boundary edge graph, and the component with the LARGEST BOUNDING-BOX
    AREA is the outer contour. Everything else is a hole.

    Returns (on_outer, on_hole), both bool arrays of length len(coords).
    """
    bedges = boundary_edges(tris)
    n = len(coords)

    adj = {}
    for a, b in bedges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    seen = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(x for x in adj[node] if x not in seen)
        loops.append(comp)

    on_outer = np.zeros(n, dtype=bool)
    on_hole = np.zeros(n, dtype=bool)
    if not loops:
        return on_outer, on_hole

    def bbox_area(loop):
        c = coords[loop]
        lo, hi = c.min(axis=0), c.max(axis=0)
        return float((hi[0] - lo[0]) * (hi[1] - lo[1]))

    outer = max(range(len(loops)), key=lambda i: bbox_area(loops[i]))
    for i, loop in enumerate(loops):
        (on_outer if i == outer else on_hole)[loop] = True

    return on_outer, on_hole


def classify_nodes(coords, tris, tol=1e-6):
    """
    Node type ids, in the same priority order MGN uses:

        fixed (left edge) > applied_load (right edge) > hole > free > interior

    Only boundary nodes can be anything but interior -- that is MGN's fast
    path and it matters, because an interior node near x_min must NOT be
    called 'fixed'.

    Returns int64 array of ids indexed by NODE_TYPE_TO_ID.
    """
    on_outer, on_hole = boundary_loops(coords, tris)
    is_boundary = on_outer | on_hole

    x = coords[:, 0]
    x_min, x_max = x.min(), x.max()

    ids = np.full(len(coords), NODE_TYPE_TO_ID["interior"], dtype=np.int64)
    ids[is_boundary & on_hole] = NODE_TYPE_TO_ID["hole"]
    ids[is_boundary & on_outer] = NODE_TYPE_TO_ID["free"]
    ids[is_boundary & (x >= x_max - tol)] = NODE_TYPE_TO_ID["applied_load"]
    ids[is_boundary & (x <= x_min + tol)] = NODE_TYPE_TO_ID["fixed"]
    return ids


def edge_features(coords, edge_index):
    """[dx, dy, length] per directed edge -> (2E, 3) float32. Relative only."""
    src, dst = edge_index[0], edge_index[1]
    d = coords[dst] - coords[src]
    length = np.sqrt((d ** 2).sum(axis=1, keepdims=True))
    return np.concatenate([d, length], axis=1).astype(np.float32)


def type_counts(ids):
    """Readable histogram, for sanity-checking a mesh."""
    inv = {v: k for k, v in NODE_TYPE_TO_ID.items()}
    return {inv[i]: int((ids == i).sum()) for i in sorted(inv)}
