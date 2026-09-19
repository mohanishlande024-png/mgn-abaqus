# -*- coding: utf-8 -*-
"""
Paper-spec plate-with-hole generator - FULL 220 CASE RUN.

11 geometries x 20 loads, matching Kunz & Choudhary exactly. No linear
scaling shortcut: every load case is solved.

Run with:
    abaqus cae noGUI=generate_paper_dataset_full.py

Writes plate_<geom>__<load>.odb, ready for extract_odb.py.


WHAT CHANGED FROM THE 11-JOB VERSION
------------------------------------
1. MODEL REUSE. The 20 loads of a geometry share one mesh, so the model is
   built and meshed ONCE per geometry and only the load magnitude changes
   between submissions. Saves roughly 20-40 minutes over the full run, and
   it GUARANTEES all 20 cases of a geometry have byte-identical meshes,
   which dataset.py requires.

2. RESUME. Jobs whose .odb exists and whose .sta says COMPLETED are skipped.
   Kill the run and restart it and you pick up where you left off. Partially
   written jobs are cleaned before resubmission, which also avoids Abaqus'
   refusal to overwrite an existing ODB.

3. PROGRESS + ETA, so you know whether to wait or go home.


BEFORE YOU START
----------------
  TIME   roughly 1.5-3 hours. About 25-45 s per job, most of which is job
         spawn and ODB writing rather than the solve (these models are
         about 1600 DOF).
  DISK   220 ODBs at ~400 KB each is ~90 MB, plus .dat/.msg/.sta/.prt.
         Budget 500 MB in data\\odb.
  RUN    from data\\odb - Abaqus writes output to the CURRENT directory.

     cd /d E:\\Mohanish_Mtech_svr_3\\mgn-abaqus\\data\\odb
     abaqus cae noGUI=..\\..\\abaqus\\generate_paper_dataset_full.py

  AFTERWARDS
     findstr /i "COMPLETED" *.sta          <- expect 220 lines
     cd ..\\..
     abaqus python abaqus\\extract_odb.py data\\odb\\*.odb
"""

from __future__ import print_function

import os
import time

import mesh
from abaqus import mdb
from abaqusConstants import *
from regionToolset import Region


# ---------------------------------------------------------------- paper spec

WIDTH = 60.0          # in
HEIGHT = 10.0         # in
E = 29.0e6            # psi  (200 GPa)
NU = 0.3
THICKNESS = 1.0       # in
MESH_SIZE = 0.90      # in   tuned to hit the paper's ~788 node count

# The paper's 20 loads. Note the deliberate gap between 4500 and 5500: the
# test load of 5000 psi sits INSIDE the training range but is never trained
# on, which is what makes it an "unseen" load (interpolation, not
# extrapolation).
LOADS = ([1000, 1388, 1777, 2166, 2555, 2944, 3333, 3722, 4111, 4500] +
         [5500, 5999, 6499, 6999, 7499, 7999, 8499, 8999, 9499, 10000])

# Restrict the run. None = all 11 geometries.
#   ONLY = ['circle8']              one geometry, all 20 loads
#   ONLY = None                     the full 220
ONLY = None

# Restrict the loads. None = all 20.
#   LOAD_SUBSET = [1000]            one load per geometry (the 11-job run)
LOAD_SUBSET = None


# ------------------------------------------------------------- the 11 shapes

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
    # (k) three holes  -- SIZES/SPACING ARE A GUESS, the paper does not say
    'three4':    lambda s: (_circle(s, -15.0, 2.0), _circle(s, 0.0, 2.0),
                            _circle(s, 15.0, 2.0)),
}


# ------------------------------------------------------------------- helpers

JOB_EXTS = ('.odb', '.dat', '.msg', '.sta', '.prt', '.com', '.log', '.sim',
            '.lck', '.res', '.mdl', '.stt', '.ipm', '.abq', '.pac', '.sel',
            '.selle')


def already_done(job_name):
    """True if this job finished cleanly on an earlier run."""
    odb, sta = job_name + '.odb', job_name + '.sta'
    if not (os.path.exists(odb) and os.path.exists(sta)):
        return False
    try:
        return 'COMPLETED' in open(sta).read()
    except Exception:
        return False


def clean_partial(job_name):
    """Remove leftovers from a killed job. Abaqus refuses to overwrite an
    existing ODB and simply aborts, so this must happen before resubmit."""
    for ext in JOB_EXTS:
        p = job_name + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def fmt_hms(seconds):
    seconds = int(seconds)
    return '%dh%02dm%02ds' % (seconds // 3600, (seconds % 3600) // 60,
                              seconds % 60)


# ------------------------------------------------------------ model building

def build_model(geom_name):
    """Build geometry, material, assembly, step, BCs, load and mesh ONCE.

    Returns (model, n_nodes). The load magnitude is set later per job.
    """
    half_w, half_h = WIDTH / 2.0, HEIGHT / 2.0

    model_name = 'PlateModel'
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    # --- geometry: model-level '__profile__' sketch (BaseShell fails
    #     if the sketch is created any other way)
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
    # No explicit FieldOutputRequest: StaticStep's default already contains
    # S and U, and the constructor name for it differs between Abaqus builds.

    # --- BCs: left edge FULLY clamped, u1 = u2 = 0.
    # The paper uses FixedBoundary(value=(0,0,0)). This is over-constrained
    # versus a roller and DOES produce corner stress spikes -- that is the
    # behaviour being replicated, not a mistake. A clamped edge already
    # removes rigid-body motion, so no corner pin is needed.
    left_edges = instance.edges.getByBoundingBox(
        xMin=-half_w - 1e-6, xMax=-half_w + 1e-6,
        yMin=-half_h - 1e-6, yMax=half_h + 1e-6)
    left_set = assembly.Set(edges=left_edges, name='LeftEdgeSet')
    model.DisplacementBC(name='BC-LeftEdge', createStepName='Initial',
                         region=left_set, u1=SET, u2=SET, ur3=UNSET,
                         amplitude=UNSET, distributionType=UNIFORM,
                         fieldName='', localCsys=None)

    # --- load: tension on the right edge. Magnitude is a placeholder here
    #     and is overwritten per job by set_load().
    right_edges = instance.edges.getByBoundingBox(
        xMin=half_w - 1e-6, xMax=half_w + 1e-6,
        yMin=-half_h - 1e-6, yMax=half_h + 1e-6)
    load_surface = assembly.Surface(side1Edges=right_edges, name='LoadSurface')
    model.Pressure(name='Load-1', createStepName='Step-1',
                   region=load_surface, magnitude=-1000.0,
                   distributionType=UNIFORM, field='', amplitude=UNSET)

    # --- mesh: uniform, PURE TRIANGLES.
    # setElementType alone is NOT enough -- without setMeshControls(TRI) the
    # free mesher stays quad-dominated and produces a CPS3 + CPS4R mix.
    # No hole refinement: the paper seeds uniformly, and refining the hole
    # would change node density exactly where peak stress lives.
    part.setMeshControls(regions=part.faces, elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces,),
        elemTypes=(mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD),))
    part.seedPart(size=MESH_SIZE, deviationFactor=0.1)
    part.generateMesh()

    return model, len(part.nodes)


def set_load(model, load):
    """Negative pressure = outward traction = tension."""
    model.loads['Load-1'].setValues(magnitude=-float(load))


def submit(model_name, job_name):
    job = mdb.Job(name=job_name, model=model_name)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()


# ----------------------------------------------------------------- main loop

def main():
    names = sorted(ONLY) if ONLY else sorted(GEOMETRIES)
    loads = LOAD_SUBSET if LOAD_SUBSET else LOADS
    total = len(names) * len(loads)

    print('=' * 70)
    print('FULL RUN: %d geometries x %d loads = %d jobs'
          % (len(names), len(loads), total))
    print('mesh size %.3f in, output -> %s' % (MESH_SIZE, os.getcwd()))
    print('=' * 70)

    t0 = time.time()
    done = 0
    skipped = 0
    failed = []

    for geom_name in names:
        # Build the mesh once for this geometry, but only if at least one of
        # its jobs still needs running.
        pending = [L for L in loads
                   if not already_done('plate_%s__%d' % (geom_name, int(L)))]
        if not pending:
            print('\n[%s] all %d jobs already complete - skipping'
                  % (geom_name, len(loads)))
            skipped += len(loads)
            done += len(loads)
            continue

        tg = time.time()
        model, n_nodes = build_model(geom_name)
        print('\n[%s] meshed: %d nodes, %d job(s) to run'
              % (geom_name, n_nodes, len(pending)))

        for load in loads:
            job_name = 'plate_%s__%d' % (geom_name, int(load))
            done += 1

            if already_done(job_name):
                skipped += 1
                continue

            clean_partial(job_name)
            set_load(model, load)
            try:
                submit('PlateModel', job_name)
            except Exception as exc:
                failed.append((job_name, str(exc)))
                print('    FAILED %-28s %s' % (job_name, exc))
                continue

            if not already_done(job_name):
                failed.append((job_name, 'did not reach COMPLETED'))
                print('    FAILED %-28s did not reach COMPLETED' % job_name)
                continue

            elapsed = time.time() - t0
            rate = elapsed / max(done - skipped, 1)
            eta = rate * (total - done)
            print('    %4d/%-4d %-28s %6.0f psi   elapsed %s   eta %s'
                  % (done, total, job_name, load,
                     fmt_hms(elapsed), fmt_hms(eta)))

        print('[%s] geometry done in %s' % (geom_name, fmt_hms(time.time() - tg)))

    print('\n' + '=' * 70)
    print('Finished %d job(s) in %s  (%d skipped as already complete)'
          % (total, fmt_hms(time.time() - t0), skipped))
    if failed:
        print('\n%d FAILURE(S):' % len(failed))
        for name, why in failed:
            print('   %-30s %s' % (name, why))
        print('\nRe-run this script to retry only the failed jobs.')
    else:
        print('All jobs completed cleanly.')
    print('=' * 70)
    print('\nNext:')
    print('   findstr /i "COMPLETED" *.sta          (expect %d lines)' % total)
    print('   cd ..\\..')
    print('   abaqus python abaqus\\extract_odb.py data\\odb\\*.odb')


if __name__ == '__main__':
    main()
