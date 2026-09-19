r"""
Vendor the MGN trainer out of MGN-Public, patched to run without FEniCS.

WHY THIS EXISTS
    `pip install meshgraphnet` will NOT import on a machine without dolfin.
    The package's __init__.py imports mesh_object and fem_object, both of
    which `import dolfin`, and mgn.py has `import dolfin as df` on line 40.
    There is no dolfin on Windows, so nothing loads.

    The saving grace: dolfin appears in the ML half of the package in only
    three places, all of which we replace with the Abaqus topology we
    already compute in mgn/graph.py.

USAGE
    1. Clone their repo next to your project:

           cd /d E:\mgn-abaqus
           git clone --depth 1 https://github.com/Josiah-Kunz/MGN-Public.git

       No git? Download the ZIP from that URL (Code -> Download ZIP) and
       extract it so you have E:\mgn-abaqus\MGN-Public\

    2. Run this:

           py scripts\vendor_mgn.py

    3. It writes:
           mgn\model.py      their mgn_model.py, VERBATIM (pure torch)
           mgn\trainer.py    their mgn.py, patched

WHAT GETS PATCHED (5 changes, all reported when you run it)
    1. `import dolfin as df`                      -> removed
    2. `_classify_boundary_nodes()`               -> removed (dolfin BoundaryMesh)
    3. `_classify_nodes()`                        -> returns case.node_types
    4. `torch.load(...)`                          -> weights_only=False
    5. `torch.cuda.amp.autocast/GradScaler`       -> torch.amp equivalents
    plus `visualize_mesh()` and `plot_ml_vs_fem()` removed, which reach into
    fem.mesh (a dolfin object we do not have).

    Everything else - the message-passing loop, normalisation, training loop,
    predict, save/load - is THEIR code, unchanged. That matters: we are
    replicating their training, not a rewrite of it.
"""

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CANDIDATES = [
    os.path.join(HERE, "MGN-Public", "meshgraphnet", "ml_object"),
    os.path.join(HERE, "..", "MGN-Public", "meshgraphnet", "ml_object"),
]
DST = os.path.join(HERE, "mgn")


def find_source():
    for c in SRC_CANDIDATES:
        if os.path.isfile(os.path.join(c, "mgn.py")):
            return os.path.abspath(c)
    print("Could not find MGN-Public. Clone it first:\n")
    print("    cd /d %s" % HERE)
    print("    git clone --depth 1 "
          "https://github.com/Josiah-Kunz/MGN-Public.git\n")
    print("Looked in:")
    for c in SRC_CANDIDATES:
        print("   ", os.path.abspath(c))
    sys.exit(1)


def drop_method(src, name):
    """Remove a whole `    def name(...)` block from a class body."""
    pat = re.compile(r"\n    def %s\(.*?(?=\n    def |\n\nclass |\Z)" % name,
                     re.S)
    new, n = pat.subn("\n", src)
    return new, n


def main():
    src_dir = find_source()
    print("source: %s" % src_dir)
    if not os.path.isdir(DST):
        os.makedirs(DST)

    # ---- 1. model.py, verbatim except the autocast deprecation ----------
    model_src = open(os.path.join(src_dir, "mgn_model.py")).read()
    model_out = model_src.replace(
        "with torch.cuda.amp.autocast(enabled=False):",
        "with torch.amp.autocast('cuda', enabled=False):")
    open(os.path.join(DST, "model.py"), "w").write(model_out)
    print("  model.py    %d lines  (1 autocast fix, otherwise verbatim)"
          % len(model_out.splitlines()))

    # ---- 2. trainer.py, patched ----------------------------------------
    s = open(os.path.join(src_dir, "mgn.py")).read()
    changes = []

    # (1) dolfin import
    s2 = s.replace("import dolfin as df\n", "")
    if s2 != s:
        changes.append("removed `import dolfin as df`")
    s = s2

    # model import now points at our vendored copy
    s2 = re.sub(r"from .*mgn_model import", "from .model import", s)
    if s2 != s:
        changes.append("repointed mgn_model import -> .model")
    s = s2

    # (2) dolfin-dependent methods
    for meth in ("_classify_boundary_nodes", "visualize_mesh",
                 "plot_ml_vs_fem"):
        s, n = drop_method(s, meth)
        if n:
            changes.append("removed %s()" % meth)

    # (3) _classify_nodes -> read precomputed types off the Case
    s, n = drop_method(s, "_classify_nodes")
    if n:
        changes.append("replaced _classify_nodes() with Case passthrough")
    replacement = '''
    # PATCHED: node types are precomputed in mgn/graph.py from Abaqus
    # connectivity. Returned as the paper's five strings so that
    # _encode_node_types() produces the same alphabetical map as their
    # released checkpoint: applied_load=0 fixed=1 free=2 hole=3 interior=4
    def _classify_nodes(self, fem):
        from .graph import NODE_TYPE_TO_ID
        id_to_name = dict((v, k) for k, v in NODE_TYPE_TO_ID.items())
        return [id_to_name[int(i)] for i in fem.node_types]
'''
    anchor = "\n    def _encode_node_types("
    s = s.replace(anchor, replacement + anchor, 1)

    # (4) torch.load default flipped to weights_only=True in torch 2.6.
    # The checkpoint holds a dict, a list and floats alongside tensors, so
    # it raises UnpicklingError without this.
    s2 = s.replace("torch.load(filepath, map_location=device)",
                   "torch.load(filepath, map_location=device, "
                   "weights_only=False)")
    if s2 != s:
        changes.append("torch.load -> weights_only=False")
    s = s2

    # (5) AMP deprecations
    s2 = s.replace("torch.cuda.amp.GradScaler()",
                   "torch.amp.GradScaler('cuda')")
    s2 = s2.replace("with torch.cuda.amp.autocast():",
                    "with torch.amp.autocast('cuda'):")
    if s2 != s:
        changes.append("torch.cuda.amp -> torch.amp")
    s = s2

    header = ('"""VENDORED from MGN-Public/meshgraphnet/ml_object/mgn.py.\n'
              'Patched by scripts/vendor_mgn.py to run without FEniCS.\n'
              'Do not edit by hand - re-run the vendor script instead.\n'
              '"""\n')
    open(os.path.join(DST, "trainer.py"), "w").write(header + s)

    print("  trainer.py  %d lines" % len(s.splitlines()))
    for c in changes:
        print("      - %s" % c)

    # ---- 3. sanity ------------------------------------------------------
    leftovers = [ln for ln in s.splitlines() if "dolfin" in ln or "df." in ln]
    print("\n  dolfin references remaining: %d" % len(leftovers))
    for ln in leftovers[:5]:
        print("      %s" % ln.strip())

    print("\nDone. Now:")
    print("    python -c \"from mgn.trainer import MGN; print('import OK')\"")
    return 0 if not leftovers else 1


if __name__ == "__main__":
    sys.exit(main())
