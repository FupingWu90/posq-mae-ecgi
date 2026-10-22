"""Improved missing/bad-channel detection for ECGi body-surface signals.

Motivation
----------
The legacy detector flagged a channel missing on the RAW signal with a fixed
amplitude rule (|x|max < 0.04 or > 2.0). Two problems, confirmed empirically:
  * baseline wander inflates raw amplitude of otherwise-clean channels;
  * the absolute upper bound (2.0) is not portable across acquisition systems
    with different gain — internal clean ECG rarely exceeds ~3 after high-pass,
    while clean external-cohort ECG reaches ~5, so any fixed cut either keeps
    internal noise or discards external signal.

This detector works on the 1 Hz HIGH-PASS-FILTERED signal (wander removed) and
combines GAIN-INVARIANT criteria so the same rule transfers across datasets:

  1. DEAD            : hp_max < dead_abs  OR  hp_std ~ 0     (≈0 at any gain)
  2. AMPLITUDE OUTLIER: hp_max > k_rel * median(hp_max)      (relative ⇒ gain-free;
                        catches drift/step transients, gross saturation, extreme noise)
  3. SPATIAL INCOHERENCE: max_j |corr(channel, neighbour_j)| < corr_thr over the
                        k nearest electrodes (scale- & powerline-invariant; catches
                        moderate-amplitude broadband noise that amplitude alone
                        misses). We take the MAX |corr| over INDIVIDUAL neighbours
                        rather than corr with their MEAN: body-surface potentials
                        are dipolar, so neighbours straddling a polarity-transition
                        line average to ~0 and a clean lead there would look
                        "incoherent" against the mean. Using |corr| (polarity-
                        invariant) to the best-matching single neighbour fixes that
                        — a real lead always strongly matches at least one same-side
                        neighbour; a bad lead matches none.

A channel is missing if ANY criterion fires. Criterion 3 needs electrode
coordinates; channels without geometry fall back to criteria 1–2.

`detect_missing` returns a boolean mask plus a dict of the per-channel features
so callers can log / threshold-sweep. The defaults were tuned on the filtered
amplitude/▒waveform analysis (see reports/fig_threshold_hp_*).
"""
import numpy as np
from preprocess import fir_highpass

# Tuned defaults (see analyze_threshold_filtered.py + validation)
DEAD_ABS  = 0.02     # hp_max below this ⇒ dead (≈0 regardless of gain)
DEAD_STD  = 1e-6     # flat/constant channel
K_REL     = 8.0      # hp_max > K_REL * median(hp_max over live ch) ⇒ amplitude outlier
MAXLAG    = 40       # samples (~40 ms @1 kHz): max time-shift when correlating with a
                     # neighbour, to absorb conduction delay between adjacent electrodes
CORR_THR  = 0.30     # max|corr| with nearest neighbours below this ⇒ spatially incoherent.
                     # Conservative: clean narrow-QRS leads that don't correlate with
                     # neighbours (sharp spikes / conduction-delay timing, e.g. ch87/ch96)
                     # sit at ~0.32-0.40 and must be KEPT; only clearest noise (<0.30,
                     # e.g. continuous broadband jitter) is flagged. Irreducible overlap
                     # with moderate noise (~0.27) is accepted in favour of not dropping
                     # real ECG. Tunable.
KNN       = 6        # number of nearest electrodes for spatial coherence


def _spatial_coherence(hp, coords, eligible, k=KNN, min_neigh=2, maxlag=MAXLAG):
    """Max LAG-TOLERANT |corr| of each channel with its k nearest ELIGIBLE
    electrodes, per neighbour. NaN where not assessable (no geometry /
    < min_neigh eligible neighbours). hp: (C,L); coords: (C,3) NaN=no-geometry;
    eligible: (C,) bool = candidates allowed as neighbours (known-good so far =
    not dead, not amplitude-outlier).

    Design choices:
      * MAX over INDIVIDUAL neighbours (not their mean): the mean cancels across
        the dipolar polarity transition; a real lead need only match >=1
        same-side neighbour. |.| handles polarity inversion.
      * LAGGED cross-correlation: for each neighbour take the max |corr| over
        time-shifts in [-maxlag, +maxlag] (~40 ms), so a small conduction delay
        between adjacent electrodes (sharp QRS arriving at slightly different
        times) does not artificially lower a clean lead's coherence. Computed
        via FFT, reusing one transform per channel.
      * Skip ALREADY-MISSING neighbours and search a WIDER candidate pool, so a
        clean lead near some bad electrodes is compared against the nearest GOOD
        ones rather than dropped. < min_neigh good neighbours -> NaN (kept)."""
    C, L = hp.shape
    coh = np.full(C, np.nan, dtype=np.float64)
    has = ~np.isnan(coords[:, 0])
    idx = np.where(has)[0]
    if len(idx) < min_neigh + 1:
        return coh
    mu = hp.mean(axis=1, keepdims=True)
    sd = hp.std(axis=1, keepdims=True) + 1e-12
    hpn = (hp - mu) / sd
    # FFTs for lagged cross-correlation (zero-pad to >=2L to avoid wraparound)
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    F = np.fft.rfft(hpn, n=nfft, axis=1)           # (C, nf)
    X = coords[idx]
    search = min(max(3 * k, 18) + 1, len(idx))
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        nbrs = tree.query(X, k=search)[1]
    except Exception:
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        nbrs = np.argsort(d2, axis=1)[:, :search]
    for ii, c in enumerate(idx):
        neigh = []
        for j in nbrs[ii]:
            cc = idx[j]
            if cc == c or not eligible[cc]:
                continue
            neigh.append(cc)
            if len(neigh) >= k:
                break
        if len(neigh) < min_neigh:
            continue
        # cross-correlation of c with each neighbour, all lags, via FFT
        xc = np.fft.irfft(F[neigh] * np.conj(F[c]), n=nfft, axis=1) / L   # (m, nfft)
        win = np.concatenate([xc[:, :maxlag + 1], xc[:, -maxlag:]], axis=1)  # lags [0..+T, -T..-1]
        coh[c] = float(np.max(np.abs(win)))
    return coh


def detect_missing(raw_ecg, coords=None, fs=1000,
                   dead_abs=DEAD_ABS, dead_std=DEAD_STD,
                   k_rel=K_REL, corr_thr=CORR_THR, knn=KNN,
                   return_features=False):
    """Detect missing/bad channels on the high-pass-filtered signal.

    Parameters
    ----------
    raw_ecg : (C, L) float array (raw, pre-filter)
    coords  : (C, 3) float array or None — electrode coords, NaN for no-geometry.
              If None, the spatial-coherence criterion is skipped.

    Returns
    -------
    missing : (C,) bool
    features (optional) : dict with hp_max, rel_amp, neigh_corr, and the
                          per-criterion boolean masks.
    """
    raw = np.asarray(raw_ecg, dtype=np.float64)
    C = raw.shape[0]
    hp = fir_highpass(raw, fs=fs, cutoff_hz=1.0)
    hp_max = np.max(np.abs(hp), axis=1)
    hp_std = hp.std(axis=1)

    dead = (hp_max < dead_abs) | (hp_std < dead_std)
    live = ~dead
    med = np.median(hp_max[live]) if live.any() else 1.0
    med = med if med > 1e-9 else 1.0
    rel_amp = hp_max / med
    amp_outlier = (rel_amp > k_rel) & ~dead

    neigh_corr = np.full(C, np.nan)     # holds max|corr| coherence (key kept for compat)
    incoherent = np.zeros(C, dtype=bool)
    if coords is not None:
        # neighbours must be KNOWN-GOOD so far: not dead, not amplitude-outlier.
        eligible = ~dead & ~amp_outlier
        neigh_corr = _spatial_coherence(hp, np.asarray(coords, dtype=np.float64), eligible, k=knn)
        incoherent = (neigh_corr < corr_thr) & ~np.isnan(neigh_corr) & ~dead & ~amp_outlier

    missing = dead | amp_outlier | incoherent
    if return_features:
        return missing, dict(hp_max=hp_max, hp_std=hp_std, rel_amp=rel_amp,
                             neigh_corr=neigh_corr, median_hp_max=med,
                             dead=dead, amp_outlier=amp_outlier, incoherent=incoherent)
    return missing
