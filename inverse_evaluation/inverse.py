"""
ECGi inverse (MFS + zeroth-order Tikhonov, per-recording L-curve lambda) for the
FixdElectrodsN_Recon downstream evaluation.

Adapted from ActivationTimeStudy/inverse/pipeline.py but kept self-contained:
it imports the LOCAL mesh_io / mfs (copied into this package) and THIS project's
preprocess (../bsp_reconstruction) — no cross-project imports.

Unlike the original solve_recording (which reads a file, detects missing, and
preprocesses internally), the entry point here — solve_from_field — takes an
already-assembled, already-preprocessed 256-channel body-surface field plus an
active-channel mask, so the SAME solver can be run on each of the four
reconstruction sources by only varying (V256, active256).
"""
import os
import sys
import numpy as np

from mesh_io import (load_surface, vertex_normals, find_patient_files,
                     import_pot, import_ecgdata)  # noqa: F401
from mfs import MFSSolver

_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
from preprocess import fir_highpass, savgol_smooth   # noqa: E402  local project preprocess

NCH_RAW = 256
# Calibrated MFS source-surface factors (from ActivationTimeStudy calibration).
ALPHA_IN = 0.95
ALPHA_OUT = 1.8
# The unconstrained L-curve corner (np.logspace(-12,0,25)) selects a
# pathologically small lambda on ~half the cohort B recordings -> catastrophic
# under-regularization (scale ratio 100-2000x, HSP-vs-commercial CC ~0). Bound
# the search to a physically-sensible floor: well-behaved recordings' L-curve
# lambda already sit in [1e-3, 1e-2], and flooring at 1e-3 keeps them while
# preventing the blow-up (validated: median HSP-vs-commercial CC 0.27 -> ~0.8).
LAM_FLOOR = 1e-3
LAM_CEIL = 1.0
LAM_N = 13

# If calibrate_inverse.py has been run, use its dataset-specific (alpha_in,
# alpha_out, lam_floor); otherwise fall back to the ActivationTimeStudy values.
_CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')
if os.path.exists(_CAL):
    try:
        import json as _json
        _c = _json.load(open(_CAL))
        ALPHA_IN = float(_c.get('alpha_in', ALPHA_IN))
        ALPHA_OUT = float(_c.get('alpha_out', ALPHA_OUT))
        LAM_FLOOR = float(_c.get('lam_floor', LAM_FLOOR))
    except Exception:
        pass


def common_average_reference(V, active_mask):
    """Zero the spatial mean across ACTIVE electrodes at each timestep. V:(n,T)."""
    if active_mask.sum() == 0:
        return V
    return V - V[active_mask].mean(axis=0, keepdims=True)


def assemble_solver(geom_dir, alpha_in=ALPHA_IN, alpha_out=ALPHA_OUT):
    """Build an MFSSolver from a patient's .heart + .torso meshes.
    Returns (solver, torso_ch_idx, heart_V, heart_F). torso_ch_idx[i] = the
    0..255 ECG channel that torso vertex i corresponds to."""
    files = find_patient_files(geom_dir)
    if files['heart'] is None or files['torso'] is None:
        raise FileNotFoundError(f'Missing .heart or .torso in {geom_dir}')
    heart_V, heart_F, _ = load_surface(files['heart'])
    torso_V, torso_F, torso_ch_idx = load_surface(files['torso'])
    if torso_ch_idx is None:
        raise ValueError(f'{files["torso"]}: missing ChannelIndices')
    torso_N = vertex_normals(torso_V, torso_F)
    solver = MFSSolver.build(heart_V, torso_V, torso_N,
                             alpha_in=alpha_in, alpha_out=alpha_out)
    return solver, torso_ch_idx, heart_V, heart_F


def solve_from_field(solver, torso_ch_idx, V256, active256, lam=None, car=True,
                     lam_floor=LAM_FLOOR):
    """Inverse-solve for a 256-channel PREPROCESSED body-surface field.

    V256      : (256, T) preprocessed potentials (physical units)
    active256 : (256,) bool — channels used as boundary conditions
    Returns (phi_heart (n_h,T), lam, residual (T,), active_mask_t, n_active).
    Returns phi=None if fewer than a handful of active torso electrodes.
    """
    V_torso = V256[torso_ch_idx].astype(np.float64)      # (n_tv, T)
    active_mask_t = active256[torso_ch_idx]
    n_active = int(active_mask_t.sum())
    if n_active < 4:
        return None, None, None, active_mask_t, n_active
    if car:
        V_torso = common_average_reference(V_torso, active_mask_t)
    sysA = solver.factorize(V_torso, active_mask=active_mask_t)
    if lam is None:
        # Floored L-curve: search only [lam_floor, LAM_CEIL] so the corner
        # detector cannot pick a pathologically small (under-regularizing) lambda.
        lam, _ = sysA.l_curve(lams=np.logspace(np.log10(lam_floor), np.log10(LAM_CEIL), LAM_N))
        lam = float(max(lam, lam_floor))
    phi, _c, residual = sysA.reconstruct(lam)
    if car:
        phi = phi - phi.mean(axis=0, keepdims=True)
    return phi, float(lam), residual, active_mask_t, n_active


def load_commercial(pot_path, n_h, time_window=None,
                    preprocess_reference=True, car=True):
    """Load the commercial .potential HSP, matched to our processing regime.
    Returns (n_h, T) float64, or None if absent / mesh-mismatched."""
    if not os.path.exists(pot_path):
        return None
    pot, _, n_pot = import_pot(pot_path)
    if n_pot != n_h:
        return None
    if time_window is not None:
        s, e = time_window
        pot = pot[:, s:e]
    pot = pot.astype(np.float32)
    if preprocess_reference:
        # Match the input-side FIR HP @1 Hz + Savitzky-Golay (skip powerline
        # notch — no physical signal there at the heart side).
        pot = fir_highpass(pot, fs=1000, cutoff_hz=1.0)
        pot = savgol_smooth(pot, window_length=61, polyorder=4)
    if car:
        pot = pot - pot.mean(axis=0, keepdims=True)
    return pot.astype(np.float64)
