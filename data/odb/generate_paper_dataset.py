# -*- coding: utf-8 -*-
"""
Paper-spec plate-with-hole generator for Abaqus.

Retargeted from generate_plate_sample(5).py to match Kunz & Choudhary:
    * units   in-lbf-psi (was N-mm-MPa)
    * steel   E=29.0e6 psi, nu=0.3, t=1.0 (was aluminium 70000, 0.33, 5.0)
    * plate   60 x 10 in, centred on origin (was 140 x 100 mm)
    * BCs     left edge FULLY clamped, u1=u2=0 (was roller + corner pin)
    * mesh    uniform 0.984 in seed, PURE CPS3 triangles, no hole refinement
    * shapes  the paper's 11 named geometries (was random circle radii)

Run with:
    abaqus cae noGUI=generate_paper_dataset.py

Writes plate_<geom>__<load>.odb, ready for extract_odb.py.

NOTE ON LOAD CASES
    Linear elasticity with no prescribed displacement gives
    sigma(k*P) == k*sigma(P) exactly, so ONE solve per geometry is enough --
    build_dataset.py --scale-loads synthesises the other 19. That is 11 jobs
    instead of 220. Set ALL_LOADS = True to solve all of them anyway.
"""

import mesh
from abaqus import mdb, session
from abaqusConstants import *
from regionToolset import Region


# ---------------------------------------------------------------- paper spec

WIDTH = 60.0          # in
HEIGHT = 10.0         # in
E = 29.0e6            # psi  (200 GPa)
NU = 0.3
THICKNESS = 1.0       # in
MESH_SIZE = 0.90     # in   (their 25 mm)

REF_LOAD = 1000.0     # psi, the single solved load per geometry
ALL_LOADS = False     # True -> solve all 20 instead of scaling

# Validate the chain on one geometry before committing to 11 jobs.
# Set back to None (or []) for the full run.
ONLY = ['circle8']

LOADS = ([1000, 1388, 1777, 2166, 2555, 2944, 3333, 3722, 4111, 4500] +
         [5500, 5999, 6499, 6999, 7499, 7999, 8499, 8999, 9499, 10000])


# ------------------------------------------------------------- the 11 shapes
# Each entry draws its hole(s) into an already-rectangular sketch.
# Paper labels from Fig. 6 are given in the comments.

def _circle(sketch, cx, r):
    sketch.CircleByCenterPerimeter(center=(cx, 0.0), point1=(cx + r, 0.0))


GEOMETRIES = {
    # (a) 8 in circle, centred
    'circle8':   lambda s: _circle(s, 0.0, 4.0),
    # (b) 8 in square hole
    'square8':   lambda s: s.rectangle(point1=(-4.0, 4.0), point2=(4.0, -4.0)),
    # (c) 8 x 4 in ellipse
    'ellipse84': lambda s: s.EllipseByCenterPerimeter(
        center=(0.0, 0.0), axisPoint1=(4.0, 0.0), axisPoint2=(0.0, 2.0)),
    # (d) 4 in circle
    'circle4':   lambda s: _circle(s, 0.0, 2.0),
    # (e) two 8 in circles at x = +/-15
    'double8':   lambda s: (_circle(s, -15.0, 4.0), _circle(s, 15.0, 4.0)),
    # (f) 8 in circle offset right 15 in
    'offset15':  lambda s: _circle(s, 15.0, 4.0),
    # (g) 8 in circle offset right 1 in
    'offset1':   lambda s: _circle(s, 1.0, 4.0),
    # (h) two 4 in circles at x = +/-15
    'double4':   lambda s: (_circle(s, -15.0, 2.0), _circle(s, 15.0, 2.0)),
    # (i) 1 in circle
    'circle1':   lambda s: _circle(s, 0.0, 0.5),
    # (j) no hole
    'nohole':    lambda s: None,
    # (k) three holes  -- SIZES/SPACING ARE A GUESS, see notes
    'three4':    lambda s: (_circle(s, -15.0, 2.0), _circle(s, 0.0, 2.0),
                            _circle(s, 15.0, 2.0)),
}


def build_and_solve(geom_name, load, job_name):
    half_w, half_h = WIDTH / 2.0, HEIGHT / 2.0

    model_name = 'PlateModel'
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    # --- geometry: model-level '__profile__' sketch (BaseShell fails otherwise)
    model.ConstrainedSketch(name='__profile__', sheetSize=WIDTH * 2)
    sketch = model.sketches['__profile__']
    sketch.rectangle(point1=(-half_w, half_h), point2=(half_w, -half_h))
    GEOMETRIES[geom_name](sketch)

    part = model.Part(name='plate', dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches['__profile__']

    # --- material and section
    model.Material(name='Material-1')
    model.materials['Material-1'].Elastic(table=((E, NU),))
    model.HomogeneousSolidSection(name='Section-1', material='Material-1',
                                  thickness=THICKNESS)
    part.SectionAssignment(region=Region(faces=part.faces),
                           sectionName='Section-1')

    assembly = model.rootAssembly
    instance = assembly.Instance(name='plate-1', part=part, dependent=ON)
    model.StaticStep(name='Step-1', previous='Initial')
    # No explicit FieldOutputRequest. StaticStep's default request already
    # contains S and U, which is everything extract_odb.py reads, and the
    # constructor/repository names for it differ between Abaqus builds.
    # extract_odb.py raises loudly if 'S' is missing, so this fails visibly
    # rather than silently.

    # --- BCs: left edge FULLY clamped.
    # The paper uses FixedBoundary(value=(0,0,0)), i.e. u1=u2=0 on the whole
    # edge. This is over-constrained versus a roller and DOES create corner
    # stress spikes -- that is the behaviour we are replicating. No corner
    # pin is needed; a clamped edge already removes rigid-body motion.
    left_edges = instance.edges.getByBoundingBox(
        xMin=-half_w - 1e-6, xMax=-half_w + 1e-6,
        yMin=-half_h - 1e-6, yMax=half_h + 1e-6)
    left_set = assembly.Set(edges=left_edges, name='LeftEdgeSet')
    model.DisplacementBC(name='BC-LeftEdge', createStepName='Initial',
                         region=left_set, u1=SET, u2=SET, ur3=UNSET,
                         amplitude=UNSET, distributionType=UNIFORM,
                         fieldName='', localCsys=None)

    # --- load: tension on the right edge (negative pressure pulls outward)
    right_edges = instance.edges.getByBoundingBox(
        xMin=half_w - 1e-6, xMax=half_w + 1e-6,
        yMin=-half_h - 1e-6, yMax=half_h + 1e-6)
    load_surface = assembly.Surface(side1Edges=right_edges, name='LoadSurface')
    model.Pressure(name='Load-1', createStepName='Step-1',
                   region=load_surface, magnitude=-float(load),
                   distributionType=UNIFORM, field='', amplitude=UNSET)

    # --- mesh: uniform, PURE TRIANGLES.
    # setElementType alone is NOT enough -- without setMeshControls(TRI) the
    # free mesher stays quad-dominated and you get a CPS3 + CPS4R mix.
    # No hole refinement: the paper seeds uniformly, and refining the hole
    # would change node density exactly where peak stress lives.
    part.setMeshControls(regions=part.faces, elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces,),
        elemTypes=(mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD),))
    part.seedPart(size=MESH_SIZE, deviationFactor=0.1)
    part.generateMesh()

    n_nodes = len(part.nodes)

    job = mdb.Job(name=job_name, model=model_name)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    print('%-28s nodes=%4d  load=%6.0f psi' % (job_name, n_nodes, load))
    return n_nodes


def main():
    loads = LOADS if ALL_LOADS else [REF_LOAD]
    names = sorted(ONLY) if ONLY else sorted(GEOMETRIES)
    total = 0
    for geom_name in names:
        for load in loads:
            job_name = 'plate_%s__%d' % (geom_name, int(load))
            build_and_solve(geom_name, load, job_name)
            total += 1
    print('\nDone: %d job(s).' % total)


if __name__ == '__main__':
    main()
