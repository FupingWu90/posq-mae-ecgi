"""
Fixed-count valid-channel (VC) selection for FixdElectrodsN_Recon.

Unlike the predecessor codebase, which masks a *ratio* of channels, this model keeps a
FIXED number of VISIBLE valid channels (VC, e.g. 16 / 32 / 64) per recording as
the encoder input and reconstructs every other channel from them plus the
physical electrode coordinates. Selection is always whole-channel.

Given per-sample electrode coordinates and an ACTIVE-channel mask (channels
that carry real signal — i.e. NOT flagged missing by the detector), pick
exactly `vc` electrodes to keep visible under one of three modes:

  cluster : electrodes spread over the torso so they are representative.
            Two sub-methods (cluster_method):
              greedy : farthest-point sampling (greedy k-center). Start from a
                       RANDOM active electrode, then repeatedly add the active
                       electrode whose minimum distance to the already-chosen
                       set is largest. Maximises the minimum pairwise
                       separation ("as far from each other as possible"); the
                       random start makes the set differ across epochs.
              kmean  : k-means (k = vc) on the active electrode coordinates
                       (k-means++ init seeded from `rng`, Lloyd iterations),
                       then take the medoid (the electrode nearest each final
                       centroid). k-means converges to similar partitions
                       regardless of the random init, so the selection is MORE
                       STABLE across epochs than greedy FPS while still covering
                       the torso.
  patch   : a contiguous cluster of electrodes. Pick a RANDOM active seed
            electrode, then its (vc-1) nearest active electrodes (by coord
            distance).
  random  : vc active electrodes chosen uniformly at random.

Optionally the CANDIDATE set is restricted to one half of the torso BEFORE any
mode runs (region, default 'all' = no restriction):

  front   : anterior half  -- electrodes on the side of the torso-electrode
            centroid the heart sits on (ANT_SIGN * coords[:, AP_AXIS] > 0).
  back    : posterior half -- the complement (registered electrodes only).

The anterior convention (AP_AXIS / ANT_SIGN) is the cohort-wide one fixed in
inverse_evaluation/heart_coords.py; it is RE-DECLARED here because bsp_reconstruction/ must not
import inverse_evaluation/. The selector receives NORMALISED coordinates (centroid-
subtracted, isotropically scaled), so the sign test applies directly. If a
region holds fewer than vc candidates, all of them are taken and the remainder
is filled with the nearest out-of-region active electrodes (nearest = smallest
distance to the dividing plane; deterministic, consumes no RNG); the number of
such fallback electrodes is reported via `return_info`. region='all' leaves the
candidate set AND the RNG stream exactly as before.

Everything runs in NumPy on small per-sample arrays (C ~ 252), so it is called
once per batch in the engine with the batch's coords/active mask moved to CPU —
no per-step GPU synchronisation, and the same code serves train / val / test.

Selection is capped at the number of candidate channels and always leaves at
least one active channel to reconstruct (so the loss is never empty).
"""
import numpy as np

MODES = ('cluster', 'patch', 'random')
CLUSTER_METHODS = ('greedy', 'kmean')
REGIONS = ('all', 'front', 'back')

# Anterior convention of the cohort -- MUST match inverse_evaluation/heart_coords.py
# (AP_AXIS, ANT_SIGN there: antero-posterior axis = coordinate column 1, the heart
# sits on the NEGATIVE side of the torso centre along it; validated on all 219
# test geometries). Re-declared here because bsp_reconstruction/ must not import inverse_evaluation/;
# inverse_evaluation/check_region_convention.py asserts the two copies agree.
AP_AXIS, ANT_SIGN = 1, -1.0


def _coord_valid(coords):
    """coords (C,3) -> (C,) bool; unregistered electrodes are all-zero."""
    return np.abs(coords).sum(axis=-1) > 1e-6


def region_mask(coords, region='all'):
    """(C,) bool -- electrodes lying in `region` ('all' | 'front' | 'back').

    coords must be the NORMALISED (centroid-subtracted) coordinates the model
    sees, so the anterior half is simply ANT_SIGN * coords[:, AP_AXIS] > 0 and
    the posterior half its complement. Unregistered (all-zero) electrodes have
    no position and belong to NEITHER half; for 'all' every electrode is in."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    C = coords.shape[0]
    if region == 'all':
        return np.ones(C, dtype=bool)
    ok = _coord_valid(coords)
    ap = ANT_SIGN * np.asarray(coords)[:, AP_AXIS]
    return ok & ((ap > 0) if region == 'front' else (ap <= 0))


def select_visible_channels(coords, active_mask, vc, mode='cluster',
                            cluster_method='greedy', rng=None, region='all',
                            return_info=False):
    """Select the visible channels for a SINGLE recording.

    coords         : (C, 3) float — normalized electrode coordinates
    active_mask    : (C,)  bool  — True where channel carries real signal
    vc             : int         — target number of visible channels
    mode           : 'cluster' | 'patch' | 'random'
    cluster_method : 'greedy' | 'kmean' (only used when mode == 'cluster')
    rng            : np.random.Generator (optional)
    region         : 'all' | 'front' | 'back' — restrict the candidate set to
                     one torso half (see module doc / region_mask) BEFORE the
                     mode runs. 'all' is a strict no-op (same RNG stream).
    return_info    : if True return (visible, info) with
                     info = {'n_region_cand': #active registered in-region,
                             'n_region_fallback': #out-of-region electrodes
                             used because the region ran short of k}

    Returns visible : (C,) bool with EXACTLY k = min(vc, n_active-1) True entries
    (n_active = number of active channels; the -1 leaves >=1 active channel to
    reconstruct). Keeping k a deterministic function of n_active is important:
    when the dataset is filtered so every recording has n_active >= vc+1 (see
    dataset.build_experiment_datasets(min_active=...)), k == vc for EVERY sample,
    so the batch has a uniform visible count and the MAE encoder's
    min-over-batch visible-token count never truncates any sample.

    cluster / patch need coordinates, so they select from the coordinate-
    REGISTERED active channels and, only if those run short of k, top up with
    coordinate-less active channels (shown to the encoder via the backbone's
    learned "unknown-position" embedding). random samples uniformly over all
    active channels.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == 'cluster' and cluster_method not in CLUSTER_METHODS:
        raise ValueError(f"cluster_method must be one of {CLUSTER_METHODS}, "
                         f"got {cluster_method!r}")
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")

    C = coords.shape[0]
    visible = np.zeros(C, dtype=bool)
    info = {'n_region_cand': None, 'n_region_fallback': 0}
    _ret = (lambda v: (v, info)) if return_info else (lambda v: v)

    coord_ok = _coord_valid(coords)
    cand = np.where(active_mask & coord_ok)[0]    # active & geometry-registered
    fill = np.where(active_mask & ~coord_ok)[0]   # active but coordinate-less
    n_active = len(cand) + len(fill)
    if n_active == 0:
        return _ret(visible)

    # Exactly k visible; leave >=1 active channel to reconstruct when possible.
    k = min(int(vc), n_active - 1) if n_active > 1 else 1
    k = max(1, k)

    # REGION restriction: candidates = active & registered & in-region. If the
    # region holds fewer than k, take all of them plus the nearest out-of-region
    # registered active electrodes (nearest to the dividing plane; deterministic).
    if region != 'all':
        inreg = region_mask(coords, region)
        cand = np.where(active_mask & coord_ok & inreg)[0]
        info['n_region_cand'] = int(len(cand))
        if len(cand) < k:
            out = np.where(active_mask & coord_ok & ~inreg)[0]
            order = np.argsort(np.abs(coords[out, AP_AXIS]), kind='stable')
            n_fb = min(k - len(cand), len(out))
            info['n_region_fallback'] = int(n_fb)
            visible[cand] = True
            visible[out[order[:n_fb]]] = True
            short = k - int(visible.sum())
            if short > 0 and len(fill) > 0:   # last resort: coordinate-less
                add = rng.choice(len(fill), size=min(short, len(fill)), replace=False)
                visible[fill[add]] = True
            return _ret(visible)

    # RANDOM: uniform over ALL active channels (coordinates irrelevant); under a
    # region restriction only the in-region registered candidates qualify.
    if mode == 'random':
        alla = np.concatenate([cand, fill]) if (len(fill) and region == 'all') else cand
        sel = rng.choice(len(alla), size=k, replace=False)
        visible[alla[sel]] = True
        return _ret(visible)

    # CLUSTER / PATCH: coordinate-based selection from registered candidates.
    n = len(cand)
    kc = min(k, n)
    if kc > 0:
        pts = coords[cand]                               # (n, 3)
        if n <= kc:
            sel = np.arange(n)
        elif mode == 'patch':
            seed = int(rng.integers(n))
            d = np.linalg.norm(pts - pts[seed], axis=1)
            sel = np.argsort(d, kind='stable')[:kc]      # seed (d=0) included
        elif cluster_method == 'kmean':
            sel = _kmeans_medoids(pts, kc, rng)
        else:  # cluster / greedy -> farthest-point sampling with random start
            sel = _farthest_point_sampling(pts, kc, rng)
        visible[cand[sel]] = True

    # Top up with coordinate-less active channels to reach exactly k.
    short = k - int(visible.sum())
    if short > 0 and len(fill) > 0:
        add = rng.choice(len(fill), size=min(short, len(fill)), replace=False)
        visible[fill[add]] = True
    return _ret(visible)


def _farthest_point_sampling(pts, k, rng):
    """Greedy k-center: return indices (into pts) of k maximally-spread points."""
    n = pts.shape[0]
    chosen = np.empty(k, dtype=np.int64)
    first = int(rng.integers(n))
    chosen[0] = first
    dmin = np.linalg.norm(pts - pts[first], axis=1)      # (n,)
    for i in range(1, k):
        nxt = int(np.argmax(dmin))
        chosen[i] = nxt
        d = np.linalg.norm(pts - pts[nxt], axis=1)
        dmin = np.minimum(dmin, d)
    return chosen


def _kmeans_medoids(pts, k, rng, n_iter=25):
    """k-means (k-means++ init) on `pts`, then return the index of the electrode
    nearest each final centroid (distinct medoids). Returns (k,) int indices."""
    n, dim = pts.shape

    # ---- k-means++ initialisation ----
    centers = np.empty((k, dim), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = pts[first]
    d2 = ((pts - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        tot = d2.sum()
        probs = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        centers[i] = pts[nxt]
        d2 = np.minimum(d2, ((pts - centers[i]) ** 2).sum(axis=1))

    # ---- Lloyd iterations ----
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(n_iter):
        dists = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)  # (n,k)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = pts[m].mean(axis=0)
            # empty cluster: keep previous centre

    # ---- distinct medoids: nearest electrode to each centre ----
    dists = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)      # (n,k)
    used = np.zeros(n, dtype=bool)
    chosen = []
    # let the tightest clusters (smallest nearest-point distance) claim first
    for c in np.argsort(dists.min(axis=0)):
        for idx in np.argsort(dists[:, c]):
            if not used[idx]:
                used[idx] = True
                chosen.append(int(idx))
                break
    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) < k:  # safety (only if duplicate points collapse clusters)
        remaining = np.setdiff1d(np.arange(n), chosen)
        chosen = np.concatenate([chosen, remaining[:k - len(chosen)]])
    return chosen[:k]


def select_visible_batch(coords, active_mask, vc, mode='cluster',
                         cluster_method='greedy', rng=None, region='all',
                         return_info=False):
    """Batched wrapper.

    coords      : (B, C, 3) float
    active_mask : (B, C)    bool
    region      : 'all' | 'front' | 'back' (see select_visible_channels)
    Returns visible : (B, C) bool  (or (visible, n_fallback_total) if return_info)
    """
    if rng is None:
        rng = np.random.default_rng()
    B, C, _ = coords.shape
    out = np.zeros((B, C), dtype=bool)
    n_fb = 0
    for b in range(B):
        out[b], inf = select_visible_channels(coords[b], active_mask[b], vc, mode,
                                               cluster_method, rng, region,
                                               return_info=True)
        n_fb += inf['n_region_fallback']
    return (out, n_fb) if return_info else out
