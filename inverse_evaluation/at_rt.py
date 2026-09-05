"""
vest_at_rt.py — primitives for extracting activation time (AT), repolarization
time (RT), and activation-recovery interval (ARI) from multi-beat epicardial
potential signals (V vertices, T samples, fs=1000 Hz).

Mirrors the convention of CIV/CIV-master/vest/Vest_Analysis_Jij.py:
  AT = argmax(d(-pot)/dt) within [qrs_start, qrs_end]
  RT = argmax(d( pot)/dt) within [t_start,  t_end]      (Wyatt method)
  ARI = RT - AT                                         (positive ms)

But adapted to our case where the signal contains multiple cardiac cycles
within a single 5000-sample crop. Beat detection is performed on the
RMS-across-vertices trace, then beats are signal-averaged before extracting
one AT/RT per vertex (the noise-robust path recommended in section 6.1.2 of
"06 Automated strategies for improving ECGi repro v1.2").
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

def detect_beats(pot, fs=1000, min_rr_ms=400, prominence_frac=0.5):
    """Detect R-peak sample indices from a (V, T) potential array.

    Strategy: compute RMS across vertices, find local maxima with minimum
    spacing of `min_rr_ms` ms (defaults to 400 -> max HR 150 bpm) and
    prominence at least `prominence_frac * std(rms)`.

    Returns
    -------
    r_peaks : int array (n_beats,)
    """
    rms = np.sqrt((pot ** 2).mean(axis=0))
    peaks, _ = sps.find_peaks(rms,
                              distance=int(min_rr_ms * fs / 1000),
                              prominence=prominence_frac * rms.std())
    return peaks.astype(np.int64)


# ---------------------------------------------------------------------------
# Signal averaging
# ---------------------------------------------------------------------------

def signal_average(pot, r_peaks, pre_ms=120, post_ms=480, fs=1000):
    """Signal-average potential across detected beats.

    Each beat contributes a window of [R - pre_ms, R + post_ms]. Beats whose
    window falls outside the signal are dropped (NOT padded — the doc warns
    that 'edge' padding distorts argmax).

    Returns
    -------
    sigavg : (V, pre+post) array  -- the averaged beat
    used_peaks : list of int     -- R-peak indices that contributed
    rel_R : int                  -- index of R-peak within sigavg (== pre samples)
    """
    pre  = int(pre_ms  * fs / 1000)
    post = int(post_ms * fs / 1000)
    T = pot.shape[1]
    segs = []
    used = []
    for r in r_peaks:
        if r - pre < 0 or r + post > T:
            continue
        segs.append(pot[:, r - pre : r + post])
        used.append(int(r))
    if not segs:
        return None, [], pre
    sigavg = np.mean(np.stack(segs, axis=0), axis=0)
    return sigavg, used, pre


# ---------------------------------------------------------------------------
# AT / RT extraction
# ---------------------------------------------------------------------------

def calc_at(pot_window, qrs_start, qrs_end, sg_win=None, sg_order=3):
    """Activation time per vertex.

    AT = argmax( d(-pot)/dt )[qrs_start:qrs_end] + qrs_start.

    Optionally smooth with Savitzky-Golay before differentiation.

    Parameters
    ----------
    pot_window : (V, T) array
    qrs_start, qrs_end : ints, sample indices (qrs_start < qrs_end <= T-1)
    sg_win : odd int or None. If set, apply savgol_filter(window=sg_win, order=sg_order) first.

    Returns
    -------
    at : (V,) int array
    """
    if sg_win is not None:
        pot_window = sps.savgol_filter(pot_window, sg_win, sg_order, axis=1)
    pot_diff = np.diff(-pot_window, axis=1)
    # mask outside QRS window so argmax stays inside
    pot_diff[:, :qrs_start] = -np.inf
    pot_diff[:, qrs_end:]   = -np.inf
    at = np.argmax(pot_diff, axis=1)
    # match Vest convention: collapse 0 -> 1 (a 0 means argmax never moved)
    at[at == 0] = 1
    return at.astype(np.int64)


def calc_rt(pot_window, t_start, t_end, sg_win=51, sg_order=3):
    """Repolarization time per vertex (Wyatt method).

    RT = argmax( d(pot)/dt )[t_start:t_end] + t_start.

    Applies Savitzky-Golay smoothing (window=51, order=3 by default — matches
    Vest signal-averaged path).
    """
    pot_smooth = sps.savgol_filter(pot_window, sg_win, sg_order, axis=1)
    pot_diff = np.diff(pot_smooth, axis=1)
    pot_diff[:, :t_start] = -np.inf
    pot_diff[:, t_end:]   = -np.inf
    rt = np.argmax(pot_diff, axis=1)
    rt[rt == 0] = 1
    return rt.astype(np.int64)


def calc_at_rt_ari(pot_window, qrs_start, qrs_end, t_start, t_end,
                   at_sg_win=None, rt_sg_win=51, sg_order=3):
    """Convenience wrapper. Returns (at, rt, ari) — all (V,) arrays of int ms."""
    at  = calc_at(pot_window, qrs_start, qrs_end, sg_win=at_sg_win, sg_order=sg_order)
    rt  = calc_rt(pot_window, t_start, t_end, sg_win=rt_sg_win, sg_order=sg_order)
    ari = rt - at
    return at, rt, ari


# ---------------------------------------------------------------------------
# Default beat windows (relative to R-peak inside sigavg)
# ---------------------------------------------------------------------------

def default_windows(rel_R, fs=1000):
    """Return (qrs_start, qrs_end, t_start, t_end) sample indices relative to
    a signal-averaged beat window, given the index of R-peak inside it.

    Defaults (ms relative to R):
      QRS [-50, +80]
      T   [+150, +400]
    """
    ms_to_samp = lambda ms: int(ms * fs / 1000)
    return (rel_R + ms_to_samp(-50),
            rel_R + ms_to_samp(+80),
            rel_R + ms_to_samp(+150),
            rel_R + ms_to_samp(+400))


# ---------------------------------------------------------------------------
# Vertex suitability mask
# ---------------------------------------------------------------------------

def good_vertex_mask(pot_window, qrs_start, qrs_end, t_start, t_end,
                     qrs_amp_pctile=10, slope_pctile=10,
                     t_amp_pctile=10, amp_ratio_min=0.15, t_slope_ratio_min=0.10):
    """Per-vertex usability masks — SEPARATE for AT and RT.

    qrs (for AT): the vertex has a usable QRS — QRS peak-to-peak >= its
        `qrs_amp_pctile`th percentile AND steepest QRS gradient >= its
        `slope_pctile`th percentile.
    t   (for RT): qrs-good AND a REAL T-wave — T peak-to-peak >= its
        `t_amp_pctile`th percentile, T/QRS amplitude ratio >= `amp_ratio_min`
        (a meaningful T-wave, not a flat repolarization), AND a clear T-wave
        upslope: steepest T gradient >= `t_slope_ratio_min` * steepest QRS
        gradient (so RT is not read off a flat/noisy T-segment — important for
        the paced / wide-QRS / small-T beats in this dataset).

    Returns dict {'qrs': (V,) bool, 't': (V,) bool}.
    """
    qrs_seg = pot_window[:, qrs_start:qrs_end]
    t_seg   = pot_window[:, t_start:t_end]
    qrs_amp = qrs_seg.max(axis=1) - qrs_seg.min(axis=1)
    t_amp   = t_seg.max(axis=1)   - t_seg.min(axis=1)
    qrs_grad = np.abs(np.diff(qrs_seg, axis=1)).max(axis=1)
    t_grad   = np.abs(np.diff(t_seg,   axis=1)).max(axis=1) if t_seg.shape[1] > 1 \
        else np.zeros_like(t_amp)

    qrs_good = (qrs_amp >= np.percentile(qrs_amp, qrs_amp_pctile)) & \
               (qrs_grad >= np.percentile(qrs_grad, slope_pctile))
    t_good = qrs_good & \
        (t_amp >= np.percentile(t_amp, t_amp_pctile)) & \
        ((t_amp / (qrs_amp + 1e-12)) >= amp_ratio_min) & \
        (t_grad >= t_slope_ratio_min * qrs_grad)
    return {'qrs': qrs_good, 't': t_good}


# ---------------------------------------------------------------------------
# Convenience: extract AT/RT/ARI from a multi-beat (V, T) signal
# ---------------------------------------------------------------------------

def extract_sample(pot, r_peaks=None, fs=1000,
                   pre_ms=120, post_ms=480,
                   qrs_offsets_ms=(-50, 80),
                   t_offsets_ms=(150, 400),
                   t_rr_frac=(0.20, 0.55),
                   at_sg_win=None, rt_sg_win=51, sg_order=3):
    """Detect beats (if needed), signal-average, then extract per-vertex AT/RT/ARI.

    The T-wave scales with heart rate, so the RT search window is made
    RR-ADAPTIVE by default: [rel_R + t_rr_frac[0]*RR, rel_R + t_rr_frac[1]*RR]
    (RR = median beat interval). A FIXED window [R+150,R+400]ms clips the T-wave
    into the TP-segment baseline at fast HR, pinning ~half the RT values to the
    window edge and inflating RT error. Pass t_rr_frac=None to use the fixed
    t_offsets_ms instead. The window is always kept after the QRS and inside the
    beat.

    Returns dict with keys:
      sigavg, r_peaks, rel_R, windows (qrs_start,qrs_end,t_start,t_end),
      rr (median RR samples or None), AT, RT, ARI (V,) int arrays.
    """
    if r_peaks is None:
        r_peaks = detect_beats(pot, fs=fs)
    sigavg, used, rel_R = signal_average(pot, r_peaks,
                                          pre_ms=pre_ms, post_ms=post_ms, fs=fs)
    if sigavg is None:
        return None
    ms_to_samp = lambda ms: int(ms * fs / 1000)
    L = sigavg.shape[1]
    qrs_start = rel_R + ms_to_samp(qrs_offsets_ms[0])
    qrs_end   = rel_R + ms_to_samp(qrs_offsets_ms[1])
    # T-window: keep t_start fixed at R + t_offsets_ms[0] (after the QRS/S-wave,
    # important for the wide-QRS paced/LBBB beats here), and CAP t_end at a
    # fraction of RR so fast heart rates don't push the search into the
    # TP-segment baseline (which pins argmax(dV/dt) to the window edge).
    rr = float(np.median(np.diff(used))) if len(used) > 1 else None
    t_start = rel_R + ms_to_samp(t_offsets_ms[0])
    t_end = rel_R + ms_to_samp(t_offsets_ms[1])
    if t_rr_frac is not None and rr is not None:
        t_end = min(t_end, rel_R + int(t_rr_frac[1] * rr))
    t_start = max(t_start, qrs_end + ms_to_samp(20))
    t_end = min(max(t_end, t_start + ms_to_samp(40)), L - 2)
    AT, RT, ARI = calc_at_rt_ari(sigavg, qrs_start, qrs_end, t_start, t_end,
                                 at_sg_win=at_sg_win, rt_sg_win=rt_sg_win,
                                 sg_order=sg_order)
    return {
        'sigavg':  sigavg,
        'r_peaks': used,
        'rel_R':   int(rel_R),
        'windows': (int(qrs_start), int(qrs_end), int(t_start), int(t_end)),
        'rr':      (int(rr) if rr is not None else None),
        'AT':      AT,
        'RT':      RT,
        'ARI':     ARI,
    }
