"""
Realism layer for Route A: multi-beat assembly, amplitude calibration,
missing/bad channels drawn from the clinical fingerprint, and measurement
noise.  Everything is expressed in the RAW vendor units (mV) so that the
written .ecgdata files go through the SAME missing detector and preprocessing
chain as the clinical recordings (bsp_reconstruction/detect_missing.py, bsp_reconstruction/preprocess.py).

Realism levels (--realism):
  clean   : clean multi-beat BSP, no missing channels, no noise
  missing : + missing channels (zeros) from the empirical per-channel fingerprint
  noise   : + noise / wander / powerline / bad channels, NO missing channels
  full    : everything (default; the level the real cohorts are matched to)
"""
import numpy as np
from scipy import signal as sps

FS = 1000
NCH = 256


# --------------------------------------------------------------------------- #
# multi-beat assembly
# --------------------------------------------------------------------------- #
def assemble_beats(bsp_beat, phi_beat, rng, L=7000, rr_ms=None, rr_jitter=0.03,
                   amp_jitter=0.03, resp_gain=0.05, min_rr_frac=None):
    """Tile one simulated beat into an L-sample recording with RR variability,
    beat-to-beat amplitude jitter and a slow respiratory gain modulation.
    Returns (bsp (n_e, L), phi (n_h, L), beat_starts, rr_ms)."""
    n_e, T = bsp_beat.shape
    if rr_ms is None:
        rr_ms = rng.uniform(520, 900)          # clinical RR: median 668 ms, IQR 601-746
    # beats may not overlap, except that a fast rhythm (VT) may run its T-wave tail
    # into the next QRS (linear superposition of the tails): min_rr_frac * T
    rr_ms = max(rr_ms, (T + 30) / 0.97 if min_rr_frac is None else min_rr_frac * T)
    bsp = np.zeros((n_e, L)); phi = np.zeros((phi_beat.shape[0], L))
    t0 = rng.uniform(20, 200); starts = []
    while t0 + T < L:
        s = int(round(t0)); a = 1.0 + rng.normal(0, amp_jitter)
        bsp[:, s:s + T] += a * bsp_beat; phi[:, s:s + T] += a * phi_beat
        starts.append(s)
        t0 += rr_ms * (1.0 + rng.normal(0, rr_jitter))
    tt = np.arange(L) / FS
    resp = 1.0 + resp_gain * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * tt + rng.uniform(0, 2 * np.pi))
    return bsp * resp, phi, np.array(starts, int), float(rr_ms)


def calibrate_gain(bsp, rng, target_median_sd=0.19, log_sd=0.4):
    """Per-recording gain so that the median per-channel SD (over the record)
    matches the clinical preprocessed recordings (median 0.14 mV, IQR
    0.10-0.26; the target 0.19 compensates the ~25% loss through missing
    channels, high-pass and smoothing); the same gain must be applied to the
    epicardial potentials."""
    sd = np.median(bsp.std(1))
    target = target_median_sd * np.exp(rng.normal(0, log_sd))
    return float(target / max(sd, 1e-9))


# --------------------------------------------------------------------------- #
# missing / bad channels
# --------------------------------------------------------------------------- #
def sample_missing(rate252, coords252, rng, keep_idx, scale_log_sd=0.5, corr_len_cm=5.0):
    """256-wide boolean missing mask drawn from the per-channel clinical
    missing ratio (rate252 on the kept channels) with a per-recording
    severity factor and spatially correlated draws (Gaussian copula with an
    exponential kernel on the electrode positions), so drop-outs cluster the
    way loose vest regions do.  The 4 never-registered channels are always
    missing."""
    p = np.clip(rate252 * np.exp(rng.normal(0, scale_log_sd)), 0, 0.98)
    ok = ~np.isnan(coords252[:, 0])
    X = np.where(ok[:, None], coords252, 0.0)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    C = np.exp(-D / corr_len_cm) + 1e-6 * np.eye(len(p))
    z = np.linalg.cholesky(C) @ rng.normal(size=len(p))
    from scipy.stats import norm
    miss252 = z < norm.ppf(p)
    miss252 |= ~ok
    m = np.ones(NCH, bool); m[keep_idx] = miss252
    return m


def bad_channels(rng, n_active_idx, max_outlier=2, max_noisy=4):
    """Indices of channels that get detector-relevant artefacts."""
    k_out = int(rng.integers(0, max_outlier + 1)) if rng.uniform() < 0.5 else 0
    k_noi = int(rng.integers(0, max_noisy + 1)) if rng.uniform() < 0.6 else 0
    pool = rng.permutation(n_active_idx)
    return pool[:k_out], pool[k_out:k_out + k_noi]


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def _pink(n, rng):
    w = rng.normal(size=n)
    Fw = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / FS); f[0] = f[1]
    x = np.fft.irfft(Fw / np.sqrt(f), n)
    return x / (x.std() + 1e-12)


def add_noise(sig, active_idx, rng, level='full'):
    """In-place style: returns sig + wander + broadband + powerline + EMG bursts
    on the active channels; per-channel gain jitter. sig: (256, L) mV."""
    n, L = sig.shape; tt = np.arange(L) / FS
    out = sig.copy()
    if level in ('clean', 'missing'):
        return out
    # baseline wander (removed by the 1 Hz high-pass, present in the raw file)
    for c in active_idx:
        w = np.zeros(L)
        for _ in range(3):
            w += rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * tt + rng.uniform(0, 2 * np.pi))
        w += np.cumsum(rng.normal(0, 0.02 / np.sqrt(FS), L))
        out[c] += w
    # broadband: white + pink, per-channel SD log-normal around 8 uV
    sd_w = np.exp(rng.normal(np.log(0.008), 0.6, size=len(active_idx)))
    for c, s in zip(active_idx, sd_w):
        out[c] += s * rng.normal(size=L) + 0.5 * s * _pink(L, rng)
    # powerline
    a50 = np.exp(rng.normal(np.log(0.005), 0.8)); ph = rng.uniform(0, 2 * np.pi, size=n)
    out[active_idx] += a50 * np.sin(2 * np.pi * 50 * tt)[None] * (1 + 0.3 * rng.normal(size=(len(active_idx), 1)))
    if rng.uniform() < 0.2:
        out[active_idx] += 0.5 * a50 * np.sin(2 * np.pi * 60 * tt + ph[active_idx, None])
    # EMG bursts on ~10% of channels
    b, a = sps.butter(3, [20 / (FS / 2), 150 / (FS / 2)], 'band')
    for c in active_idx:
        if rng.uniform() < 0.10:
            for _ in range(int(rng.integers(1, 4))):
                dur = int(rng.uniform(0.3, 1.5) * FS); s0 = int(rng.integers(0, max(1, L - dur)))
                burst = sps.lfilter(b, a, rng.normal(size=dur)); burst *= rng.uniform(0.05, 0.3) / (burst.std() + 1e-12)
                out[c, s0:s0 + dur] += burst
    # electrode gain jitter
    out[active_idx] *= (1.0 + rng.normal(0, 0.08, size=(len(active_idx), 1)))
    return out


def apply_bad_channels(sig, outlier_idx, noisy_idx, rng):
    """Amplitude-outlier channels (saturation / step transients, x12-40) and
    noise-only incoherent channels (0.15-0.4 mV broadband) -- the artefact
    classes the clinical detector was built for."""
    out = sig.copy(); L = out.shape[1]
    for c in outlier_idx:
        if rng.uniform() < 0.5:
            out[c] *= rng.uniform(12, 40)
        else:
            s0 = int(rng.integers(0, L)); out[c, s0:] += rng.uniform(3, 10) * rng.choice([-1, 1])
    for c in noisy_idx:
        out[c] = rng.uniform(0.15, 0.4) * rng.normal(size=L)
    return out
