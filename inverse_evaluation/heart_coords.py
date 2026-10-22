"""Patient-independent epicardial coordinates, so per-vertex errors can be
averaged ACROSS patients whose hearts are different meshes (1650-2284 vertices,
different geometry).

Each patient's epicardium is parameterised by two anatomically-consistent
coordinates:

  z in [0,1] : apico-basal position along the heart's own long axis
               (0 = apex, 1 = base).  Normalised per patient, so heart size
               differences are absorbed.
  theta in (-pi, pi] : circumferential angle about the long axis, measured from
               the body's ANTERIOR direction.  Using an anatomical reference
               (not an arbitrary one) is what makes cross-patient averaging
               meaningful.

Anatomical frame is taken from the TORSO mesh, which shares the patient
coordinate frame with the heart (the MFS solver uses them jointly):
  * body height axis  = torso axis of LARGEST extent
  * antero-posterior  = torso axis of SMALLEST extent (chest is shallower
                        front-to-back than side-to-side)
  * anterior sign     = the side the heart sits on (the heart is anterior of
                        the torso centre) -> verified across all test patients
Apex/base axis uses the vendor's VALVE surfaces (.valve file: the caps of the
atrio-ventricular and semilunar openings of the ventricular mesh): the long
axis runs from the valve-surface centroid (centre of the base) to the apex
(mean of the 2% of vertices farthest from it), oriented apex -> base. The
former label-free heuristic (base = broader half) is WRONG for these meshes --
their basal end is a narrow great-vessel neck while the ventricles are widest
mid-way -- and put the base at the apex end for ~80% of geometries (2026-09-05
audit over 509 geometries). Without a valve surface the heuristic is used and
a warning is printed.

`bin_map` reduces a per-vertex quantity to a fixed (n_theta x n_z) grid, giving
every patient the SAME shaped map regardless of their vertex count.
"""
import numpy as np

N_THETA, N_Z = 36, 20          # common grid (10 deg x 5% of long axis)

# Where the patient's LEFT lies in the theta frame (theta = 0 anterior): the
# left-right body axis projected into the short-axis plane sits at about
# +137 deg (circular mean over 150 test geometries, quartiles 113-142 deg;
# 2026-09-06) because the apex->base axis is tilted only ~36 deg from the
# left-right axis. Hence the RV free wall occupies roughly theta in
# (-120, +30) deg (anterior-right) and the LV free wall theta in (+60, 180)
# deg (left-posterior); |theta| ~ 180 is the inferior wall.
THETA_LEFT_DEG = 137.0
RV_SECTOR = (-120.0, 30.0)
LV_SECTOR = (60.0, 180.0)


def add_sector_axis(ax, fontsize=8, y0=3.0, h=7.0, lw=1.2, color='0.2'):
    """Downward-opening brackets above a (theta, z) map marking the RV/LV
    free-wall sectors, with the sector name centred above each bracket.

    `y0` = gap between the top spine and the bracket tips, `h` = bracket height
    (both in points), so the geometry is layout-independent.  The caller should
    leave room for it (e.g. ``ax.set_title(..., pad=36)``)."""
    from matplotlib.transforms import blended_transform_factory, offset_copy
    base = blended_transform_factory(ax.transData, ax.transAxes)   # x data, y axes
    fig = ax.figure
    t_lo = offset_copy(base, fig=fig, x=0, y=y0, units='points')
    t_hi = offset_copy(base, fig=fig, x=0, y=y0 + h, units='points')
    t_lab = offset_copy(base, fig=fig, x=0, y=y0 + h + 2.0, units='points')
    arrow = dict(arrowstyle='-', lw=lw, color=color, shrinkA=0, shrinkB=0)
    for (lo, hi), lab in ((RV_SECTOR, 'RV free wall\n(anterior-right)'),
                          (LV_SECTOR, 'LV free wall\n(left-posterior)')):
        ax.annotate('', xy=(lo, 1), xytext=(hi, 1), xycoords=t_hi, textcoords=t_hi,
                    arrowprops=arrow, annotation_clip=False)            # top bar
        for x in (lo, hi):                                              # end ticks, pointing down
            ax.annotate('', xy=(x, 1), xytext=(x, 1), xycoords=t_lo, textcoords=t_hi,
                        arrowprops=arrow, annotation_clip=False)
        ax.text(0.5 * (lo + hi), 1, lab, transform=t_lab, ha='center', va='bottom',
                fontsize=fontsize, clip_on=False)


# Cohort convention (validated on ALL 219 test geometries by _selfcheck):
#   antero-posterior axis = column 1 for 217/219 patients; the heart sits on the
#   NEGATIVE side of the torso centre along it for 218/219.  All recordings share
#   one acquisition frame, so we FIX the convention rather than re-detecting it
#   per patient -- otherwise the 1-2 atypical torsos would contribute a mirrored
#   map and smear the cross-patient average.
AP_AXIS, ANT_SIGN = 1, -1.0


def body_axes(torso_V):
    """(height_axis, ap_axis, lr_axis) as indices into the coordinate columns."""
    ext = torso_V.max(0) - torso_V.min(0)
    h = int(np.argmax(ext))            # body height (superior-inferior)
    ap = int(np.argmin(ext))           # antero-posterior (shallowest)
    lr = int(3 - h - ap)               # remaining = left-right
    return h, ap, lr


def heart_frame(heart_V, torso_V, valve_V=None):
    """Return (origin, e_long, e_ref) for one patient.

    e_long : unit long axis, oriented APEX -> BASE (base = the end on which the
             valve-surface centroid lies; broader-half heuristic if valve_V is None)
    e_ref  : unit vector in the plane perpendicular to e_long pointing ANTERIOR
             (theta = 0 direction)

    `torso_V` is kept in the signature (callers pass the patient's torso) but is
    no longer read: the anterior reference now comes from the FIXED cohort
    convention above rather than per-patient detection, which is what keeps
    theta comparable across patients.
    """
    hV = np.asarray(heart_V, float)
    origin = hV.mean(0)
    X = hV - origin
    # long axis = principal direction of the epicardium
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e_long = Vt[0] / np.linalg.norm(Vt[0])

    # orient apex->base
    t = X @ e_long
    if valve_V is not None and len(valve_V):
        # anatomical long axis: from the centre of the base (centroid of the
        # valve surfaces that cap the AV and semilunar openings) to the apex
        # (mean of the 2% of vertices farthest from that base centre). The SVD
        # axis is kept only as a fallback: these ventricular meshes are squat
        # (extents ~14x12x11 cm), so their principal direction is often
        # oblique or near-horizontal and would mis-orient z for ~20% of hearts.
        base_c = np.asarray(valve_V, float).mean(0)
        dist = np.linalg.norm(hV - base_c, axis=1)
        k = max(10, int(0.02 * len(hV)))
        apex_c = hV[np.argsort(dist)[-k:]].mean(0)
        e_long = (base_c - apex_c) / np.linalg.norm(base_c - apex_c)   # apex -> base
        t = X @ e_long
    else:
        import warnings
        warnings.warn('heart_frame: no valve surface given -- falling back to '
                      'the broader-half heuristic, which mis-orients ~80% of '
                      'these meshes', RuntimeWarning)
        perp = X - np.outer(t, e_long)
        rad = np.linalg.norm(perp, axis=1)
        if rad[t > 0].mean() < rad[t < 0].mean():
            e_long = -e_long                  # flip so +e_long points to base
            t = -t

    # anterior reference: FIXED cohort convention (see AP_AXIS/ANT_SIGN above),
    # so every patient's theta=0 points the same anatomical way.
    ap_ax, lr_ax = AP_AXIS, (0 if AP_AXIS != 0 else 2)
    ant = np.zeros(3)
    ant[ap_ax] = ANT_SIGN
    # project out the long-axis component -> lies in the circumferential plane
    e_ref = ant - (ant @ e_long) * e_long
    n = np.linalg.norm(e_ref)
    if n < 1e-8:                              # degenerate: fall back to left-right
        alt = np.zeros(3); alt[lr_ax] = 1.0
        e_ref = alt - (alt @ e_long) * e_long
        n = np.linalg.norm(e_ref)
    e_ref /= n
    return origin, e_long, e_ref


def vertex_coords(heart_V, torso_V, valve_V=None):
    """Per-vertex (theta, z) in the patient-independent frame.
    theta in (-pi,pi] from anterior; z in [0,1] apex->base (base = valve side)."""
    hV = np.asarray(heart_V, float)
    origin, e_long, e_ref = heart_frame(hV, torso_V, valve_V)
    X = hV - origin
    t = X @ e_long
    z = (t - t.min()) / max(t.max() - t.min(), 1e-9)      # 0 = apex, 1 = base
    e_perp2 = np.cross(e_long, e_ref)                     # right-handed pair
    perp = X - np.outer(t, e_long)
    theta = np.arctan2(perp @ e_perp2, perp @ e_ref)      # 0 = anterior
    return theta, z


def coords_for_geometry(gdir):
    """(theta, z, n_h) for a patient geometry directory, loading heart, torso and
    valve surfaces via mesh_io (the valve surface fixes the apex->base sign)."""
    from mesh_io import load_surface, find_patient_files
    f = find_patient_files(gdir)
    hV = load_surface(f['heart'])[0]
    tV = load_surface(f['torso'])[0]
    vV = load_surface(f['valve'])[0] if f.get('valve') else None
    theta, z = vertex_coords(hV, tV, vV)
    return theta, z, hV.shape[0]


def bin_map(theta, z, values, mask=None, n_theta=N_THETA, n_z=N_Z):
    """Average `values` over vertices into a fixed (n_theta, n_z) grid.

    Returns (map, count) where map is nan where no vertex contributed.
    `mask` optionally restricts to valid vertices (e.g. the reference-gated set).
    Every patient yields the SAME shaped map -> averageable across patients.
    """
    theta = np.asarray(theta, float); z = np.asarray(z, float)
    v = np.asarray(values, float)
    keep = np.isfinite(v)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    if not keep.any():
        return np.full((n_theta, n_z), np.nan), np.zeros((n_theta, n_z), int)
    ti = np.clip(((theta[keep] + np.pi) / (2 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    zi = np.clip((z[keep] * n_z).astype(int), 0, n_z - 1)
    flat = ti * n_z + zi
    cnt = np.bincount(flat, minlength=n_theta * n_z).astype(int)
    tot = np.bincount(flat, weights=v[keep], minlength=n_theta * n_z)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return m.reshape(n_theta, n_z), cnt.reshape(n_theta, n_z)


# ---------------------------------------------------------------------------
# self-check across ALL test geometries (internal + external)
# ---------------------------------------------------------------------------
def _selfcheck():
    import os, sys, collections
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction'))
    from mesh_io import load_surface, find_patient_files
    from dataset import collect_split
    b, _ = collect_split(seed=42)
    stats = collections.defaultdict(list)
    for split in ['test_internal', 'test_external']:
        gdirs = sorted({os.path.dirname(t) for _, _, t in b[split] if t})
        for g in gdirs:
            f = find_patient_files(g)
            if not (f['heart'] and f['torso']):
                continue
            hV, _, _ = load_surface(f['heart']); tV, _, _ = load_surface(f['torso'])
            h_ax, ap_ax, lr_ax = body_axes(tV)
            ant_sign = np.sign(hV.mean(0)[ap_ax] - tV.mean(0)[ap_ax])
            th, z = vertex_coords(hV, tV)
            stats[split].append((hV.shape[0], h_ax, ap_ax, ant_sign,
                                 z.min(), z.max(), th.min(), th.max()))
    for split, S in stats.items():
        n = len(S)
        print(f"[{split}] {n} patient geometries")
        print(f"   vertices: {min(s[0] for s in S)}-{max(s[0] for s in S)}")
        print(f"   height-axis choice: {collections.Counter(s[1] for s in S).most_common()}")
        print(f"   AP-axis choice    : {collections.Counter(s[2] for s in S).most_common()}")
        print(f"   anterior sign     : {collections.Counter(int(s[3]) for s in S).most_common()}"
              f"   <- must be CONSISTENT for cross-patient averaging")
        print(f"   z range ok: {all(abs(s[4]) < 1e-9 and abs(s[5]-1) < 1e-9 for s in S)}")


if __name__ == '__main__':
    _selfcheck()
