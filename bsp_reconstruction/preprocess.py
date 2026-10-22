"""
ECGi signal preprocessing pipeline.

Mirrors the calc_maps pipeline from CIV/CIV-master/vest/Vest_Analysis_Jij.py:
  1. FIR highpass @ 1 Hz  (baseline wander removal)   — from FIR_filter_Jij
  2. Savitzky-Golay smoothing (window=61, poly=4)      — high-frequency noise
  3. Powerline notch filter  (50 Hz and 60 Hz)         — standard ECG protocol

All filters operate on shape (n_channels, L) with axis=1 (time axis).
Only active (non-missing) channels are filtered; missing channels stay zero.

Sampling rate assumed: 1000 Hz (matches the dataset hardware).
"""

import numpy as np
import scipy.signal as ss


# ---------------------------------------------------------------------------
# Individual filter stages
# ---------------------------------------------------------------------------

def fir_highpass(ecg, fs=1000, cutoff_hz=1.0, order=None):
    """
    FIR zero-phase highpass filter (baseline wander removal).
    Mirrors FIR_filter_Jij from Jij_utils.py.

    ecg:      (C, L) float32
    fs:       sampling rate in Hz
    cutoff:   highpass cutoff in Hz
    order:    filter order; default = int(0.6 * fs), forced odd
    """
    if order is None:
        order = int(0.6 * fs)
    if order % 2 == 0:
        order += 1

    wn = 2.0 * cutoff_hz / fs          # normalized cutoff (Nyquist = 1)
    b  = ss.firwin(numtaps=order, cutoff=wn, pass_zero=False)
    a  = np.array([1.0])

    padlen = min(3 * len(b), ecg.shape[1] - 1)
    return ss.filtfilt(b, a, ecg, axis=1, padlen=padlen)


def savgol_smooth(ecg, window_length=61, polyorder=4):
    """
    Savitzky-Golay smoothing (high-frequency noise suppression).
    Mirrors the SG step in calc_maps.

    ecg: (C, L) float32
    """
    # window_length must be odd and <= signal length
    wl = min(window_length, ecg.shape[1])
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 1:
        return ecg.copy()
    return ss.savgol_filter(ecg, window_length=wl, polyorder=polyorder, axis=1)


def notch_filter(ecg, fs=1000, freqs=(50.0, 60.0), Q=30.0):
    """
    IIR zero-phase notch filter for powerline interference.
    Standard clinical ECG preprocessing step.

    ecg:   (C, L) float32
    freqs: tuple of powerline frequencies to notch (Hz)
    Q:     quality factor (sharpness of notch)
    """
    out = ecg.copy()
    for f0 in freqs:
        if f0 >= fs / 2:
            continue
        b, a = ss.iirnotch(f0, Q, fs=fs)
        padlen = min(3 * max(len(a), len(b)), out.shape[1] - 1)
        out = ss.filtfilt(b, a, out, axis=1, padlen=padlen)
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_ecgi(ecg, missing_mask, fs=1000,
                    highpass_cutoff=1.0,
                    sg_window=61, sg_poly=4,
                    notch_freqs=(50.0, 60.0), notch_Q=30.0):
    """
    Apply the full preprocessing pipeline to an ECGi recording.

    Steps (applied only to active channels):
      1. FIR highpass @ highpass_cutoff Hz  — baseline wander removal
      2. Savitzky-Golay smoothing            — residual HF noise
      3. Notch filters @ notch_freqs         — powerline interference

    ecg:          (256, L) float32 — raw signal
    missing_mask: (256,)  bool    — True = missing channel (skip)
    Returns:      (256, L) float32 — preprocessed signal
                  (missing channels remain zero)
    """
    out = np.zeros_like(ecg)

    active_idx = np.where(~missing_mask)[0]
    if len(active_idx) == 0:
        return out

    active = ecg[active_idx].copy()   # (n_active, L)

    # 1. FIR highpass baseline removal
    active = fir_highpass(active, fs=fs, cutoff_hz=highpass_cutoff)

    # 2. Savitzky-Golay smoothing
    active = savgol_smooth(active, window_length=sg_window, polyorder=sg_poly)

    # 3. Powerline notch filters
    active = notch_filter(active, fs=fs, freqs=notch_freqs, Q=notch_Q)

    out[active_idx] = active.astype(np.float32)
    return out
