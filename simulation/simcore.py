"""
Route A surface-source simulator for FixdElectrodsN_Recon (2026-09-14).

Physics (see results/sim_plan.md, "Route A"):
  * source   : transmembrane potential Vm(v,t) on the patient's CLOSED
               epicardial surface (vendor .heart mesh + fan caps over the two
               valve openings), built from an activation-time map AT(v)
               (eikonal / Dijkstra on the mesh with a conduction-velocity field)
               and a repolarisation-time map RT(v) = AT + APD(v), through a
               template action potential (tanh upstroke, sloped plateau, tanh
               repolarisation).
  * forward  : equivalent-double-layer (solid-angle) potential of Vm in an
               infinite homogeneous medium, phi_inf(x,t) = -kappa/(4 pi)
               sum_f Omega_f(x) Vm_f(t) (Omega positive when the outward normal
               faces x; extracellular potential negative over depolarised
               tissue), bounded by the closed vest surface as
               an insulated homogeneous conductor via the Barr-Geselowitz
               boundary-element equations (TorsoBEM).  Body-surface potentials
               are read at the electrodes; the ground-truth epicardial
               potential is the exterior limit of phi on the source surface
               (evaluated at v + eps n_v).  The whole map Vm_f -> (BSP, phi_H)
               is LINEAR, assembled once per (geometry, pose jitter) as two
               matrices, so a beat costs one matmul.
  * inverse-crime mitigation: the forward operator (solid-angle double layer
               + surface BEM) is not the dual-layer MFS of the inverse; the true
               electrode/heart pose is jittered while the vendor registration
               is what the dataset/inverse see; measurement noise, missing and
               bad channels are added downstream (simulation/noise.py).  The torso is
               still homogeneous -- a known limitation of Route A.
Units: geometry in cm (vendor), time in ms at fs = 1 kHz, Vm in mV; the final
per-recording gain is calibrated to the real recordings by simulation/make_dataset.py.
"""
import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(_HERE, '..', 'inverse_evaluation'), os.path.join(_HERE, '..', 'bsp_reconstruction')):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from mesh_io import load_surface, find_patient_files, vertex_normals   # noqa: E402
from heart_coords import vertex_coords                                  # noqa: E402

FS = 1000.0
KAPPA = 0.4          # sigma_i / (sigma_i + sigma_e): EDL strength (dimensionless)
EPS_EPI = 0.05       # cm: exterior-limit offset for the epicardial potential
# repolarisation / source tunables (matched to the clinical T/QRS ratio, 2026-09-14)
APD_HET_SD = (3.0, 10.0)   # ms: SD of the smooth random APD field
APD_AB = (0.0, 20.0)       # ms: apex->base APD gradient amplitude
APD_AP = 12.0              # ms: anterior/posterior APD modulation amplitude (+-)
TAU_R = (15.0, 35.0)       # ms: phase-3 repolarisation time constant
PLATEAU_DROP = 0.10        # fraction of the plateau lost linearly over the APD
SINUS_CV = (0.10, 0.22)    # cm/ms: apparent epicardial CV in sinus rhythm (1.0-2.2 m/s)
K_AT = (0.5, 0.9)          # AT-APD coupling: later activation -> shorter APD (RT dispersion << AT dispersion)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _boundary_loops(F, n_v):
    """Ordered vertex loops of the open boundary of a triangle mesh."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Es = np.sort(E, axis=1)
    u, inv, cnt = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    bnd = E[cnt[inv] == 1]                        # directed boundary edges (face order)
    nxt = {int(a): int(b) for a, b in bnd}
    loops, seen = [], set()
    for a in list(nxt):
        if a in seen:
            continue
        loop, cur = [], a
        while cur not in seen:
            seen.add(cur); loop.append(cur)
            cur = nxt.get(cur)
            if cur is None:
                break
        loops.append(np.array(loop, int))
    return loops


def _orient_outward(V, F):
    N = vertex_normals(V, F)
    if np.mean(np.einsum('ij,ij->i', N, V - V.mean(0))) < 0:
        F = F[:, [0, 2, 1]].copy()
    return F


def check_closed(V, F, ext_points=None):
    """(interior solid-angle sum / 4pi, max |exterior sum|): 1 and ~0 for a
    closed, consistently outward-oriented surface."""
    c = V.mean(0)[None]
    s_in = solid_angles(c, V, F).sum(1)[0] / (4 * np.pi)
    if ext_points is None:
        ext_points = c + 3.0 * (V.max(0) - V.min(0))[None]
    s_out = np.abs(solid_angles(ext_points, V, F).sum(1)).max()
    return float(s_in), float(s_out)


class SimGeometry:
    """Closed epicardial source surface + closed vest surface of one patient."""

    def __init__(self, patient_dir, torso_path=None):
        f = find_patient_files(patient_dir)
        if torso_path is None:
            torso_path = f['torso']
        if f['heart'] is None or torso_path is None:
            raise FileNotFoundError(patient_dir)
        hV, hF, _ = load_surface(f['heart'])
        hV = hV.astype(np.float64); hF = _orient_outward(hV, hF.astype(np.int64))
        self.n_epi = hV.shape[0]
        # fan caps over the valve openings (cap centroid = new vertex, Vm interpolated)
        loops = _boundary_loops(hF, self.n_epi)
        capV, capF, self.cap_rings = [], [], []
        for lp in loops:
            if len(lp) < 3:
                continue
            c_idx = self.n_epi + len(capV)
            capV.append(hV[lp].mean(0)); self.cap_rings.append(lp)
            for i in range(len(lp)):
                # boundary edges run (a -> b) in face order; the cap triangle
                # (b, a, c) keeps the surface consistently oriented
                a, b = lp[i], lp[(i + 1) % len(lp)]
                capF.append([b, a, c_idx])
        self.V = np.vstack([hV, np.array(capV)]) if capV else hV
        self.F = np.vstack([hF, np.array(capF, np.int64)]) if capF else hF
        self.F = _orient_outward(self.V, self.F)
        self.N = vertex_normals(self.V, self.F).astype(np.float64)
        self.heart_V, self.heart_F = hV, hF
        # patient-independent frame (theta from anterior, z apex->base) on the
        # epicardial vertices, valve surface fixes the apex/base sign
        tV, tF, t_ch = load_surface(torso_path)
        vV = load_surface(f['valve'])[0] if f.get('valve') else None
        self.theta, self.z = vertex_coords(hV, tV, vV)
        # closed vest surface
        self.tV = tV.astype(np.float64); self.tF = _orient_outward(self.tV, tF.astype(np.int64))
        self.tN = vertex_normals(self.tV, self.tF).astype(np.float64)
        self.t_ch = t_ch.astype(np.int64)            # torso vertex i -> ECG channel
        # propagation graph on the epicardial vertices (edge length in cm)
        E = np.unique(np.sort(np.vstack([hF[:, [0, 1]], hF[:, [1, 2]], hF[:, [2, 0]]]), axis=1), axis=0)
        self.edges = E
        self.edge_len = np.linalg.norm(hV[E[:, 0]] - hV[E[:, 1]], axis=1)
        self.patient_dir = patient_dir
        self.torso_path = torso_path

    # ---- eikonal ------------------------------------------------------------
    def geodesic(self, sources, cv_vertex):
        """Arrival time (ms) from each source vertex, given a per-vertex
        conduction velocity (cm/ms); edge cost = length / mean CV of its ends."""
        E, L = self.edges, self.edge_len
        w = L / (0.5 * (cv_vertex[E[:, 0]] + cv_vertex[E[:, 1]]))
        G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                       shape=(self.n_epi, self.n_epi)).tocsr()
        return dijkstra(G, directed=False, indices=np.atleast_1d(sources))

    def geodesic_from(self, v0):
        return self.geodesic([v0], np.ones(self.n_epi))[0]      # cm

    def vertex_at(self, theta_deg, z, rng=None, jitter_deg=0.0, jitter_z=0.0):
        """Epicardial vertex closest to (theta, z) in the patient-independent frame."""
        th = np.deg2rad(theta_deg + (rng.normal(0, jitter_deg) if rng is not None and jitter_deg else 0.0))
        zz = z + (rng.normal(0, jitter_z) if rng is not None and jitter_z else 0.0)
        d = np.abs(np.angle(np.exp(1j * (self.theta - th)))) / np.pi + np.abs(self.z - zz) * 2.0
        return int(np.argmin(d))

    def _modes(self, n_modes=24):
        """Cached low-frequency graph-Laplacian eigenmodes (n_epi, n_modes),
        computed once per geometry (shift-invert Lanczos on the sparse Laplacian)."""
        if getattr(self, '_U', None) is not None and self._U.shape[1] >= n_modes:
            return self._U[:, :n_modes]
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        E = self.edges; n = self.n_epi
        A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                       shape=(n, n)).tocsr()
        Lp = diags(np.asarray(A.sum(1)).ravel()) - A
        w, U = eigsh(Lp.tocsc(), k=n_modes + 1, sigma=-1e-3, which='LM')
        order = np.argsort(w); U = U[:, order][:, 1:]                  # drop the constant mode
        self._U = U
        return U[:, :n_modes]

    def smooth_field(self, rng, n_modes=12, sd=1.0):
        """Random smooth scalar field (unit SD) on the epicardium: low-frequency
        graph-Laplacian eigenmodes with 1/sqrt(k)-damped Gaussian coefficients."""
        U = self._modes(max(n_modes, 12))[:, :n_modes]
        coef = rng.normal(size=U.shape[1]) / np.sqrt(np.arange(1, U.shape[1] + 1))
        f = U @ coef
        f = (f - f.mean()) / max(f.std(), 1e-9)
        return sd * f

    def extend_to_caps(self, vals):
        """Extend a per-epicardial-vertex array to the cap centroids (ring mean)."""
        out = np.empty(self.V.shape[0], vals.dtype)
        out[:self.n_epi] = vals
        for k, lp in enumerate(self.cap_rings):
            out[self.n_epi + k] = vals[lp].mean()
        return out


# --------------------------------------------------------------------------- #
# solid angles and the linear forward operator
# --------------------------------------------------------------------------- #
def solid_angles(P, V, F, chunk=96):
    """Omega[p, f] = signed solid angle of triangle f seen from point p
    (van Oosterom & Strackee 1983); POSITIVE when the face's outward normal
    points AWAY from the point. Sum over a closed outward-oriented surface:
    0 outside, +4 pi inside.  Branch-cut guard: tan(Omega/2) = num/den has a cut at
    num = 0, den < 0, i.e. for points (nearly) in the plane of the triangle;
    there the true angle is 0 when the point projects OUTSIDE the triangle
    (and +/-2 pi only when it projects inside), so such entries are set by an
    explicit projection test instead of atan2."""
    out = np.empty((P.shape[0], F.shape[0]))
    T = V[F]                                            # (n_f, 3, 3)
    e1 = T[:, 1] - T[:, 0]; e2 = T[:, 2] - T[:, 0]
    nrm = np.cross(e1, e2)
    nn = np.linalg.norm(nrm, axis=1); nrm = nrm / np.maximum(nn, 1e-30)[:, None]
    # barycentric helpers for the projection test
    d11 = np.einsum('fi,fi->f', e1, e1); d12 = np.einsum('fi,fi->f', e1, e2); d22 = np.einsum('fi,fi->f', e2, e2)
    invD = 1.0 / np.maximum(d11 * d22 - d12 * d12, 1e-30)
    for s in range(0, P.shape[0], chunk):
        Pc = P[s:s + chunk]
        R = T[None] - Pc[:, None, None, :]              # (c, n_f, 3, 3)
        R1, R2, R3 = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        l1, l2, l3 = (np.linalg.norm(R1, axis=2), np.linalg.norm(R2, axis=2),
                      np.linalg.norm(R3, axis=2))
        num = np.einsum('cfi,cfi->cf', R1, np.cross(R2, R3))
        den = (l1 * l2 * l3 + np.einsum('cfi,cfi->cf', R1, R2) * l3
               + np.einsum('cfi,cfi->cf', R2, R3) * l1
               + np.einsum('cfi,cfi->cf', R3, R1) * l2)
        om = 2.0 * np.arctan2(num, den)
        bad = den < 0
        if bad.any():
            ci, fi = np.nonzero(bad)
            w = Pc[ci] - T[fi, 0]                        # point relative to vertex 0
            h = np.einsum('ki,ki->k', w, nrm[fi])        # height above the plane
            wp = w - h[:, None] * nrm[fi]
            d1 = np.einsum('ki,ki->k', wp, e1[fi]); d2 = np.einsum('ki,ki->k', wp, e2[fi])
            u = (d22[fi] * d1 - d12[fi] * d2) * invD[fi]
            v = (d11[fi] * d2 - d12[fi] * d1) * invD[fi]
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)
            om[ci[~inside], fi[~inside]] = 0.0          # in-plane, outside -> 0
            # in-plane inside: exterior limit +/- 2 pi by the side of the plane
            om[ci[inside], fi[inside]] = np.sign(h[inside] + 1e-300) * 2.0 * np.pi * (np.abs(h[inside]) > 0)
        out[s:s + chunk] = om
    return out


def rigid_jitter(rng, max_deg=5.0, sd_cm=0.3):
    """Small random rotation (about a random axis) + translation."""
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    return Rm, rng.normal(0, sd_cm, 3)


class TorsoBEM:
    """Constant-element boundary-element solution of the insulated, homogeneous
    bounded conductor (Barr et al. 1977 / Geselowitz): for the potential on the
    closed torso surface T and inside it, given the infinite-medium source
    potential phi_inf,
        on T     : phi = 2 phi_inf + (1/2 pi) PV\oint_T phi dOmega
        inside   : phi = phi_inf   + (1/4 pi)   \oint_T phi dOmega
    with dOmega positive when the outward normal of T points away from the
    field point (solid_angles() convention: +2 pi over the PV surface, +4 pi
    from inside). Collocation at the vertices; the potential on
    a face is the mean of its vertices; faces adjacent to the collocation
    vertex contribute nothing (flat-face limit); the constant null space is
    removed by the pseudo-inverse and the solution is made zero-mean."""

    def __init__(self, tV, tF, rcond=1e-6):
        n_t, n_f = tV.shape[0], tF.shape[0]
        Om = solid_angles(tV, tV, tF)                                   # (n_t, n_f)
        for f in range(n_f):
            Om[tF[f], f] = 0.0
        self.A_f2v = np.zeros((n_f, n_t)); self.A_f2v[np.arange(n_f)[:, None], tF] = 1.0 / 3.0
        P = (Om @ self.A_f2v) / (2.0 * np.pi)                          # P 1 = 1 (closed surface)
        self.closure = float(np.abs(P.sum(1) - 1.0).max())
        B = np.eye(n_t) - P
        self.S = np.linalg.pinv(B, rcond=rcond) * 2.0                   # phi_T = S phi_inf(T)
        self.S -= self.S.mean(0, keepdims=True)                         # zero-mean solution
        self.tV, self.tF = tV, tF

    def interior(self, X):
        """Matrix K (n_x, n_t) such that phi(X) = phi_inf(X) + K phi_T."""
        Om = solid_angles(X, self.tV, self.tF)                          # (n_x, n_f)
        return (Om @ self.A_f2v) / (4.0 * np.pi)


class DoubleLayerForward:
    """Linear operator Vm_face(t) -> (BSP at the vest vertices, phi on the
    epicardium) for one geometry and one 'true' pose (jittered heart pose and
    electrode positions; the registered vendor geometry is what the dataset
    sees). Source = equivalent double layer on the closed epicardium, volume
    conductor = insulated homogeneous torso bounded by the closed vest surface
    (TorsoBEM)."""

    def __init__(self, geo, rng=None, jitter=True, kappa=KAPPA,
                 elec_sd_cm=0.2, heart_deg=5.0, heart_sd_cm=0.3):
        rng = np.random.default_rng(0) if rng is None else rng
        V, F, N = geo.V, geo.F, geo.N
        tV = geo.tV
        if jitter:
            Rm, tr = rigid_jitter(rng, heart_deg, heart_sd_cm)
            c = V.mean(0)
            V = (V - c) @ Rm.T + c + tr
            N = N @ Rm.T
            tV = tV + rng.normal(0, elec_sd_cm, tV.shape)
        self.V, self.F, self.N, self.tV = V, F, N, tV
        self.kappa = kappa
        # phi_inf = -(kappa/4pi) * sum_f Vm_f * Omega_f^{normal-facing}; with the
        # solid_angles() sign (normal pointing away = positive) this is +kappa/4pi
        k = kappa / (4.0 * np.pi)
        bem = TorsoBEM(tV, geo.tF)
        Om_e = solid_angles(tV, V, F)                                   # vest vertices
        Om_h = solid_angles(V + EPS_EPI * N, V, F)                      # exterior limit on S
        self.M_bsp_inf = k * Om_e                                       # unbounded-medium BSP
        self.M_bsp = bem.S @ self.M_bsp_inf                             # bounded BSP (n_t, n_f)
        self.M_epi = k * Om_h + bem.interior(V + EPS_EPI * N) @ self.M_bsp
        self.bem_closure = bem.closure
        self.closure_ext = float(np.abs(Om_e.sum(1)).max())            # ~0 for a closed source

    def face_values(self, vert_vals):
        return vert_vals[self.F].mean(1)

    def run(self, Vm_vertex):
        """Vm_vertex: (n_V, T) mV on the closed surface -> (BSP (n_t,T), phi_H (n_V,T))."""
        Vm_f = self.face_values(Vm_vertex)
        return self.M_bsp @ Vm_f, self.M_epi @ Vm_f


def sphere_test(n_sub=4, radius=10.0):
    """Analytic check of TorsoBEM: a current dipole at the centre of an
    insulated sphere has surface potential exactly 3x its infinite-medium
    value (and (1 + 2 r^3/R^3) x inside). Returns (relative error on the
    surface, relative error at r = R/2)."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None
    # icosphere-like: Fibonacci points + convex hull
    n = 40 * 4 ** (n_sub - 2)
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    P = radius * np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]
    F = ConvexHull(P).simplices.astype(np.int64)
    # hull simplices are NOT consistently oriented: orient each face outward
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = np.einsum('fi,fi->f', fn, P[F].mean(1)) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    bem = TorsoBEM(P, F)
    p = np.array([0.0, 0.0, 1.0])
    phi_inf = (P @ p) / (4 * np.pi * np.linalg.norm(P, axis=1) ** 3)
    phi_T = bem.S @ phi_inf
    exact = 3.0 * phi_inf
    err_T = np.linalg.norm(phi_T - exact) / np.linalg.norm(exact)
    X = 0.5 * P[:64]
    phi_in = (X @ p) / (4 * np.pi * np.linalg.norm(X, axis=1) ** 3) + bem.interior(X) @ phi_T
    exact_in = (X @ p) / (4 * np.pi * np.linalg.norm(X, axis=1) ** 3) * (1 + 2 * (0.5) ** 3)
    err_in = np.linalg.norm(phi_in - exact_in) / np.linalg.norm(exact_in)
    return float(err_T), float(err_in), float(bem.closure)


# --------------------------------------------------------------------------- #
# activation / repolarisation scenarios
# --------------------------------------------------------------------------- #
def _cv_field(geo, rng, base_cv_ms, hetero_sd=0.15):
    """Per-vertex conduction velocity (cm/ms) with smooth log-normal heterogeneity."""
    f = geo.smooth_field(rng, n_modes=10, sd=hetero_sd)
    return base_cv_ms * np.exp(f)


def make_scenario(geo, rng, kind=None, scar_prob=0.25):
    """Return dict(AT, RT, APD, amp (per epicardial vertex), meta)."""
    kinds = ['sinus', 'paced', 'lbbb', 'rbbb', 'double', 'vt']
    if kind is None:
        kind = rng.choice(kinds, p=[0.36, 0.28, 0.10, 0.08, 0.10, 0.08])
    n = geo.n_epi
    meta = {'kind': str(kind)}
    # heart rate first (clinical RR: median 668 ms, IQR 601-746); APD is rate-adapted below
    rr_ms = float(rng.uniform(330, 480)) if kind == 'vt' else float(rng.uniform(520, 900))
    if kind == 'sinus':
        # apparent epicardial CV is fast (Purkinje-driven transmural breakthroughs)
        cv = _cv_field(geo, rng, rng.uniform(*SINUS_CV))             # 1.0-2.2 m/s apparent
        sites = [(geo.vertex_at(-25, 0.50, rng, 15, 0.10), 0.0),      # RV anterior breakthrough
                 (geo.vertex_at(90, 0.40, rng, 25, 0.15), rng.uniform(5, 20)),   # LV anterolateral
                 (geo.vertex_at(rng.uniform(-180, 180), 0.12, rng, 0, 0.06), rng.uniform(8, 25))]  # apex
        if rng.uniform() < 0.5:
            sites.append((geo.vertex_at(rng.uniform(-60, 60), rng.uniform(0.3, 0.7), rng), rng.uniform(5, 25)))
    elif kind == 'paced':
        cv = _cv_field(geo, rng, rng.uniform(0.07, 0.12))            # 0.7-1.2 m/s
        sites = [(int(rng.integers(n)), 0.0)]
    elif kind == 'lbbb':
        cv = _cv_field(geo, rng, rng.uniform(0.07, 0.11))
        sites = [(geo.vertex_at(-25, 0.5, rng, 20, 0.12), 0.0)]        # RV only
    elif kind == 'rbbb':
        cv = _cv_field(geo, rng, rng.uniform(0.08, 0.13))
        sites = [(geo.vertex_at(95, 0.45, rng, 25, 0.15), 0.0),
                 (geo.vertex_at(rng.uniform(-180, 180), 0.15, rng, 0, 0.05), rng.uniform(0, 15))]
    elif kind == 'vt':   # monomorphic VT-like: fast single ectopic focus, myocardial spread
        cv = _cv_field(geo, rng, rng.uniform(0.08, 0.12))            # 0.8-1.2 m/s (QRS ~130-180 ms)
        sites = [(int(rng.integers(n)), 0.0)]
    else:  # double: two ectopic origins
        cv = _cv_field(geo, rng, rng.uniform(0.07, 0.13))
        sites = [(int(rng.integers(n)), 0.0), (int(rng.integers(n)), rng.uniform(10, 60))]
    amp = np.ones(n)
    scar = []
    if rng.uniform() < scar_prob:
        for _ in range(int(rng.integers(1, 3))):
            c0 = int(rng.integers(n)); r = rng.uniform(1.5, 3.5)
            d = geo.geodesic_from(c0)
            inside = d < r
            cv[inside] *= rng.uniform(0.2, 0.45)
            amp[inside] *= rng.uniform(0.3, 0.6)
            core = 0
            if rng.uniform() < 0.5:                                  # infarct: non-excitable core
                core_m = d < 0.5 * r
                amp[core_m] = 0.0; core = int(core_m.sum())
            scar.append((c0, float(r), int(inside.sum()), core))
    meta['scar'] = scar
    src_idx = [s for s, _ in sites]; delays = np.array([d for _, d in sites])
    D = geo.geodesic(src_idx, cv)                                    # (n_src, n)
    AT = (D + delays[:, None]).min(0)
    AT -= AT.min()
    meta['sites'] = [(int(s), float(d)) for s, d in sites]
    # APD: rate-adapted base (~200-260 ms at RR 600, ~250-320 at RR 900; VT ~150-200)
    # + apex->base gradient + anterior/posterior + AT coupling + smooth random
    apd0 = float(np.clip(rng.uniform(180, 250) + 0.18 * (rr_ms - 600.0), 140, 340))
    g_ab = rng.uniform(*APD_AB) * (geo.z - 0.5)                      # base longer
    g_ap = rng.uniform(-APD_AP, APD_AP) * np.cos(geo.theta)
    k_at = rng.uniform(*K_AT)
    APD = apd0 + g_ab + g_ap - k_at * (AT - AT.mean()) + geo.smooth_field(rng, 10, rng.uniform(*APD_HET_SD))
    APD = np.clip(APD, 150, 420)
    meta.update(apd0=float(apd0), rr_ms=rr_ms, k_at=float(k_at), cv_mean=float(cv.mean()),
                at_range=float(AT.max()), tau_r=float(rng.uniform(*TAU_R)),
                tau_u=float(rng.uniform(2.5, 4.5)))   # epicardial footprint of a transmural front
    return {'AT': AT, 'RT': AT + APD, 'APD': APD, 'amp': amp, 'meta': meta}


def action_potential(t, AT, RT, amp, tau_u=1.0, tau_r=18.0, v_rest=-85.0, v_amp=115.0,
                     plateau_drop=PLATEAU_DROP):
    """Template TMP (n_V, T): tanh upstroke at AT, linear phase-2 decline, tanh
    repolarisation centred on RT."""
    tt = t[None, :]
    up = 0.5 * (1.0 + np.tanh((tt - AT[:, None]) / tau_u))
    frac = np.clip((tt - AT[:, None]) / np.maximum(RT - AT, 1.0)[:, None], 0, 1)
    plateau = 1.0 - plateau_drop * frac
    rep = 0.5 * (1.0 + np.tanh((tt - RT[:, None]) / tau_r))
    return v_rest + v_amp * amp[:, None] * up * plateau * (1.0 - rep)


def simulate_beat(geo, fwd, scen, pre_ms=80.0, post_ms=90.0):
    """One beat: returns dict(t, bsp (n_t,T), phi (n_epi,T), Vm (n_epi,T))."""
    tau_r = scen['meta']['tau_r']
    T_end = scen['RT'].max() + 4 * tau_r + post_ms
    t = np.arange(0.0, T_end + pre_ms, 1000.0 / FS)
    AT = geo.extend_to_caps(scen['AT']) + pre_ms
    RT = geo.extend_to_caps(scen['RT']) + pre_ms
    amp = geo.extend_to_caps(scen['amp'])
    Vm = action_potential(t, AT, RT, amp, tau_u=scen['meta'].get('tau_u', 3.5), tau_r=tau_r)
    bsp, phi = fwd.run(Vm)
    return {'t': t, 'bsp': bsp, 'phi': phi[:geo.n_epi], 'Vm': Vm[:geo.n_epi], 'pre_ms': pre_ms}
