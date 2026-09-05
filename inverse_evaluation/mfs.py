"""
Method of Fundamental Solutions (MFS) inverse solver for ECGi.

References
----------
Wang Y, Rudy Y. "Application of the method of fundamental solutions to
potential-based inverse electrocardiography." Ann Biomed Eng. 2006;34(8):1272-88.

Setup
-----
The torso volume conductor is approximated as a homogeneous, isotropic medium
between the epicardium (heart surface, where φ is unknown) and the body surface
(torso, where φ is measured at electrode locations). The potential field φ(x)
in the volume satisfies Laplace's equation ∇²φ = 0.

MFS represents φ as a superposition of point sources placed *outside* the
field region:
    φ(x) = Σⱼ cⱼ · G(x, yⱼ),     G(x, y) = 1 / (4π ‖x − y‖)

Source placement (dual-source convention, classical Wang-Rudy):
  - "Inner" sources: heart vertices deflated toward heart centroid by α_in
                     (default 0.8 → sources sit inside the heart cavity,
                     outside the conductor volume)
  - "Outer" sources: torso vertices inflated outward from torso centroid
                     by α_out (default 1.5 → sources outside the body)

Boundary conditions (imposed on torso electrodes):
  - Dirichlet: φ(xₜᵢ) = Vᵢ (measured ECG)
  - Neumann:   ∂φ/∂n(xₜᵢ) = 0 (no current through skin)

Stack into:                A · c = b
where A = [G_dir; G_neu] is (2·n_torso, n_src) and b = [V; 0].
This is over-determined; we solve with zeroth-order Tikhonov:
    c(λ) = V · diag(σᵢ / (σᵢ² + λ)) · Uᵀ · b      (SVD-filtered form)

Then evaluate at heart vertices: φ_heart = G_h · c(λ).

Numerical-stability refactor
----------------------------
The naïve `(AᵀA + λI)⁻¹Aᵀb` form squares the condition number of A. For our
under-determined system (≈440 active equations vs ~1900 unknowns) this can
matter at small λ. Both `solve` and `l_curve_lambda` therefore go through a
shared `_ActiveSystem` that pre-computes one SVD of A and reuses it for both
the corner-finding sweep and the final reconstruction. For large meshes this
also drops the per-call cost of `solve` from O(N³) to O(N·T).

Missing channels are dropped from the BC system *before* assembling A and b.
This is the slot where MAE-imputed signals plug in: instead of dropping the
channel, supply its imputed value as Vᵢ.
"""

from dataclasses import dataclass
import warnings
import numpy as np


def green_kernel(X, Y):
    """
    G(xᵢ, yⱼ) = 1 / (4π ‖xᵢ − yⱼ‖) for all targets X (n_x, 3) and sources Y (n_y, 3).

    A `RuntimeWarning` is emitted if any target–source distance is below
    1e-9 — this should never happen with a sensible (α_in, α_out) choice
    and indicates a configuration error rather than a numerical artefact
    we want to silently absorb.
    """
    diff = X[:, None, :] - Y[None, :, :]            # (n_x, n_y, 3)
    r = np.linalg.norm(diff, axis=2)                # (n_x, n_y)
    if np.any(r < 1e-9):
        n_close = int(np.sum(r < 1e-9))
        warnings.warn(
            f'green_kernel: {n_close} target-source pair(s) are within '
            f'1e-9 — sources may be coinciding with targets. Check '
            f'(α_in, α_out) and mesh geometry.',
            RuntimeWarning,
            stacklevel=2,
        )
        r = np.where(r < 1e-12, 1.0, r)             # safety floor
    return 1.0 / (4.0 * np.pi * r)


def green_normal_deriv(X, Y, N_x):
    """
    ∂G/∂n_x at target xᵢ wrt source yⱼ along outward normal n_x at xᵢ.

        G(x, y)      = 1 / (4π ‖x − y‖)
        ∇_x G(x, y)  = −(x − y) / (4π ‖x − y‖³)
        ∂G/∂nₓ       = −((x − y) · n_x) / (4π ‖x − y‖³)
    """
    diff = X[:, None, :] - Y[None, :, :]            # (n_x, n_y, 3)
    r = np.linalg.norm(diff, axis=2)                # (n_x, n_y)
    if np.any(r < 1e-9):
        warnings.warn(
            'green_normal_deriv: target-source pair(s) within 1e-9 '
            '— check geometry.', RuntimeWarning, stacklevel=2,
        )
        r = np.where(r < 1e-12, 1.0, r)
    dot = np.einsum('ijk,ik->ij', diff, N_x)        # (n_x, n_y)
    return -dot / (4.0 * np.pi * r ** 3)


# ----------------------------------------------------------------------------
# Pre-factored active BC system (shared by solve and l_curve_lambda)
# ----------------------------------------------------------------------------

class _ActiveSystem:
    """
    Pre-factored Tikhonov problem for one specific (V_torso, active_mask).

    Holds the compact SVD of the active BC matrix A = [A_dir; A_neu]
    restricted to the active rows, plus the projection Uᵀb of the right-hand
    side onto A's left singular vectors. From these scalars we can compute
    the L-curve at any λ in O(r·T) and reconstruct c(λ) and φ_heart in
    O(N·T), without ever forming AᵀA or solving an N×N linear system.

    Identities used (with A = U·diag(σ)·Vᵀ, compact SVD; r = rank):
        c(λ)          = V · diag(σ/(σ² + λ)) · Uᵀ · b
        ‖c(λ)‖²       = Σᵢ (σᵢ/(σᵢ² + λ))² (Uᵀb)ᵢ²       (sum over i and time)
        ‖A·c(λ) − b‖² = Σᵢ (λ/(σᵢ² + λ))² (Uᵀb)ᵢ²
                        + ‖b‖² − ‖Uᵀb‖²              (perpendicular component
                                                       independent of λ)
    The perpendicular component is the part of b that lies outside col(A);
    it cannot be reduced by any choice of λ.
    """

    def __init__(self, solver, V_torso, active_mask):
        Ad = solver.A_dir[active_mask]
        An = solver.A_neu[active_mask]
        V = V_torso[active_mask].astype(np.float64)
        zero = np.zeros_like(V)

        A = np.vstack([Ad, An])                  # (2·n_active, n_src)
        b = np.vstack([V, zero])                 # (2·n_active, T)

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Utb = U.T @ b                            # (r, T)

        b_norm_sq   = float(np.linalg.norm(b) ** 2)
        Utb_norm_sq = float(np.linalg.norm(Utb) ** 2)
        # ‖(I − UUᵀ) b‖²  — λ-independent residual floor.
        perp_norm_sq = max(b_norm_sq - Utb_norm_sq, 0.0)

        self.solver = solver
        self.active_mask = active_mask
        self.A = A
        self.b = b
        self.U = U
        self.s = s
        self.Vt = Vt
        self.Utb = Utb
        self.perp_norm_sq = perp_norm_sq

    # ----- single-λ operations ----------------------------------------

    def coefficients(self, lam):
        """Source amplitudes c(λ). Shape (n_src, T)."""
        filt = self.s / (self.s ** 2 + lam)
        return self.Vt.T @ (filt[:, None] * self.Utb)

    def reconstruct(self, lam):
        """
        Return (phi_heart, c, residual_per_t):
          phi_heart    : (n_h, T)
          c            : (n_src, T)
          residual_per_t : (T,)  — ‖A·c(λ) − b‖₂ per timestep
        """
        c = self.coefficients(lam)
        phi_heart = self.solver.G_h @ c
        # Per-timestep residual is cheap from the SVD form too:
        # (A c − b)_t  has parallel norm² = Σᵢ (λ/(σᵢ² + λ))² (Utb)ᵢ,t²
        # plus per-timestep perp norm² = ‖b_t‖² − ‖Utb_t‖².
        s2 = self.s ** 2
        filt_r2 = (lam / (s2 + lam)) ** 2          # (r,)
        para = (filt_r2[:, None] * self.Utb ** 2).sum(axis=0)   # (T,)
        bt2 = (self.b ** 2).sum(axis=0)                          # (T,)
        utbt2 = (self.Utb ** 2).sum(axis=0)                      # (T,)
        perp = np.maximum(bt2 - utbt2, 0.0)
        residual = np.sqrt(para + perp)
        return phi_heart, c, residual

    # ----- L-curve operations -----------------------------------------

    def _norms_at_lam(self, lam):
        """Total Frobenius norms ‖A·c(λ) − b‖_F and ‖c(λ)‖_F."""
        s2 = self.s ** 2
        filt_r2 = (lam / (s2 + lam)) ** 2          # (r,)
        filt_s2 = (self.s / (s2 + lam)) ** 2       # (r,)
        utb2 = self.Utb ** 2                        # (r, T)
        para = float((filt_r2[:, None] * utb2).sum())
        sol2 = float((filt_s2[:, None] * utb2).sum())
        res_norm = (para + self.perp_norm_sq) ** 0.5
        sol_norm = sol2 ** 0.5
        return res_norm, sol_norm

    def l_curve(self, lams=None):
        """
        Return (lam_opt, info_dict). info_dict has 'lams', 'res_norm',
        'sol_norm', 'curvature' arrays.
        """
        if lams is None:
            lams = np.logspace(-12, 0, 25)
        res_norm = np.empty(len(lams))
        sol_norm = np.empty(len(lams))
        for k, lam in enumerate(lams):
            res_norm[k], sol_norm[k] = self._norms_at_lam(lam)
        log_r = np.log(res_norm + 1e-30)
        log_s = np.log(sol_norm + 1e-30)
        dr = np.gradient(log_r)
        ds = np.gradient(log_s)
        ddr = np.gradient(dr)
        dds = np.gradient(ds)
        curvature = (dr * dds - ddr * ds) / (dr ** 2 + ds ** 2 + 1e-30) ** 1.5
        # Exclude endpoints from corner detection.
        k_opt = int(np.argmax(curvature[1:-1])) + 1
        lam_opt = float(lams[k_opt])
        return lam_opt, {
            'lams': lams, 'res_norm': res_norm,
            'sol_norm': sol_norm, 'curvature': curvature,
        }


# ----------------------------------------------------------------------------
# MFSSolver
# ----------------------------------------------------------------------------

@dataclass
class MFSSolver:
    """
    Pre-compute the per-recording transfer matrices given the patient geometry.

    Attributes
    ----------
    heart_V    : (n_h, 3)   epicardial vertices  — evaluation targets
    torso_V    : (n_t, 3)   electrode positions  — BC targets
    torso_N    : (n_t, 3)   outward normals at torso vertices
    sources    : (n_s, 3)   stacked inner+outer auxiliary sources
    A_dir      : (n_t, n_s) Dirichlet kernel at torso
    A_neu      : (n_t, n_s) Neumann kernel at torso
    G_h        : (n_h, n_s) Dirichlet kernel at heart (evaluation)
    alpha_in   : float      heart deflation factor (centroid-anchored)
    alpha_out  : float      torso inflation factor (centroid-anchored)
    """
    heart_V: np.ndarray
    torso_V: np.ndarray
    torso_N: np.ndarray
    sources: np.ndarray
    A_dir: np.ndarray
    A_neu: np.ndarray
    G_h: np.ndarray
    alpha_in: float
    alpha_out: float

    @classmethod
    def build(cls, heart_V, torso_V, torso_N, alpha_in=0.8, alpha_out=1.5):
        h_c = heart_V.mean(axis=0)
        t_c = torso_V.mean(axis=0)
        sources_inner = h_c + alpha_in * (heart_V - h_c)
        sources_outer = t_c + alpha_out * (torso_V - t_c)
        sources = np.vstack([sources_inner, sources_outer]).astype(np.float64)
        torso_V64 = torso_V.astype(np.float64)
        torso_N64 = torso_N.astype(np.float64)
        heart_V64 = heart_V.astype(np.float64)
        A_dir = green_kernel(torso_V64, sources)
        A_neu = green_normal_deriv(torso_V64, sources, torso_N64)
        G_h   = green_kernel(heart_V64, sources)
        return cls(heart_V64, torso_V64, torso_N64, sources,
                   A_dir, A_neu, G_h, alpha_in, alpha_out)

    # ----- factorisation ----------------------------------------------

    def factorize(self, V_torso, active_mask=None):
        """
        Pre-factor the active BC system for a specific torso measurement and
        active-channel mask. Returns an `_ActiveSystem` whose `.l_curve()`
        and `.reconstruct(lam)` methods reuse the same SVD.

        The typical pipeline call is:
            sysA = solver.factorize(V_torso, active_mask)
            lam, _ = sysA.l_curve()
            phi, c, residual = sysA.reconstruct(lam)
        which performs exactly one SVD instead of one for L-curve + one for
        reconstruction.
        """
        if active_mask is None:
            active_mask = np.ones(self.A_dir.shape[0], dtype=bool)
        return _ActiveSystem(self, V_torso, active_mask)

    # ----- backwards-compatible single-call methods --------------------

    def solve(self, V_torso, lam, active_mask=None):
        """
        Reconstruct epicardial potentials at a specific λ. Backwards-compatible
        wrapper around `factorize().reconstruct(lam)`. For workflows that want
        L-curve + reconstruct, prefer the explicit two-step call to avoid
        re-factoring.

        Returns
        -------
        phi_heart : (n_h, T) float
        c         : (n_s, T) float
        residual  : (T,) float — ‖A·c − b‖₂ per time step
        """
        return self.factorize(V_torso, active_mask).reconstruct(lam)


def l_curve_lambda(solver, V_torso, active_mask=None,
                   lams=None, return_curve=False):
    """
    Pick λ via the L-curve corner. Backwards-compatible wrapper around
    `solver.factorize(...).l_curve(lams)`. For workflows that follow up with
    a `solver.solve(...)` call at the chosen λ, factor the system explicitly
    once and call both `l_curve()` and `reconstruct()` on it.
    """
    sysA = solver.factorize(V_torso, active_mask)
    lam_opt, info = sysA.l_curve(lams=lams)
    return (lam_opt, info) if return_curve else lam_opt
