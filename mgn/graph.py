"""
Pure-numpy replacement for every dolfin-dependent step in MGN-Public.

Mirrors MGN._classify_boundary_nodes / _classify_nodes without FEniCS.
No torch, no dolfin, no meshio -- so it can be unit-tested standalone.

EDGE AUGMENTATION (optional, off by default)
    augment_edges() adds random long-range "shortcut" edges, as in the EA-GNN
    of Gladstone et al. (2024). It runs AFTER classify_nodes(), so node types
    are always decided from the real mesh boundary only - a random edge
    belongs to no triangle and would otherwise corrupt boundary detection.
"""

import zlib

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


def geometry_seed(name, base_seed=0):
    """
    Stable per-geometry random seed for edge augmentation.

    Python's built-in hash() of a string is salted per process, so using it
    would give a DIFFERENT augmented graph every run (and a different one on
    Windows vs Ubuntu). crc32 is identical everywhere.
    """
    return (zlib.crc32(str(name).encode("utf-8")) + int(base_seed)) % (2 ** 32)


def augment_edges(edge_index, n_nodes, perc=0.20, seed=0):
    """
    EA-GNN edge augmentation (Gladstone et al. 2024, Sci. Rep. 14:3394).

    Adds round(perc * E_undirected) random long-range node pairs, each stored
    in BOTH directions, appended AFTER the real mesh edges:

        [ real fwd | real rev | aug fwd | aug rev ]

    Pairs are drawn uniformly over all nodes, and a pair is rejected if it is
    a self-loop, an existing mesh edge, or already drawn. perc is a fraction
    of the EXISTING undirected edge count (paper: A_perc = 20% -> perc=0.20),
    so the graph grows to about 1.2x its edges.

    perc <= 0 returns edge_index unchanged -> byte-identical baseline.

    Returns
        edge_index_aug  (2, E')  int64
        edge_flag       (E',)    float32   0 = real mesh edge, 1 = augmented
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    flag = np.zeros(edge_index.shape[1], dtype=np.float32)
    if perc is None or perc <= 0:
        return edge_index, flag

    und = np.unique(np.sort(edge_index.T, axis=1), axis=0)
    n_und = len(und)
    n_new = int(round(perc * n_und))
    free_pairs = n_nodes * (n_nodes - 1) // 2 - n_und
    if n_new > free_pairs:
        raise ValueError("asked for %d augmented pairs but only %d node pairs "
                         "are not already mesh edges" % (n_new, free_pairs))
    if n_new == 0:
        return edge_index, flag

    # undirected pair (a < b) -> one integer key, for fast membership tests
    existing = set((und[:, 0] * n_nodes + und[:, 1]).tolist())
    rng = np.random.default_rng(seed)
    chosen, chosen_keys = [], set()
    while len(chosen) < n_new:
        need = n_new - len(chosen)
        for i, j in rng.integers(0, n_nodes, size=(2 * need + 16, 2)):
            i, j = int(i), int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            key = a * n_nodes + b
            if key in existing or key in chosen_keys:
                continue
            chosen_keys.add(key)
            chosen.append((a, b))
            if len(chosen) == n_new:
                break

    new = np.array(chosen, dtype=np.int64)                       # (n_new, 2)
    aug = np.concatenate([new.T, new.T[::-1]], axis=1)           # both directions
    edge_index_aug = np.concatenate([edge_index, aug], axis=1)
    flag = np.concatenate([flag, np.ones(aug.shape[1], dtype=np.float32)])
    return edge_index_aug, flag


def edge_features(coords, edge_index, edge_flag=None):
    """
    [dx, dy, length] per directed edge -> (E, 3) float32. Relative only.

    With edge_flag given (augmented graph), the flag is appended as a 4th
    column -> (E, 4): [dx, dy, length, is_augmented]. Without it the output
    is exactly the baseline (E, 3).

    NOTE: training does not call this - mgn/trainer.py computes its own edge
    features. It is used by scripts/plot_graph.py and scripts/check_augment.py.
    """
    src, dst = edge_index[0], edge_index[1]
    d = coords[dst] - coords[src]
    length = np.sqrt((d ** 2).sum(axis=1, keepdims=True))
    cols = [d, length]
    if edge_flag is not None:
        cols.append(np.asarray(edge_flag, dtype=np.float32).reshape(-1, 1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def type_counts(ids):
    """Readable histogram, for sanity-checking a mesh."""
    inv = {v: k for k, v in NODE_TYPE_TO_ID.items()}
    return {inv[i]: int((ids == i).sum()) for i in sorted(inv)}
