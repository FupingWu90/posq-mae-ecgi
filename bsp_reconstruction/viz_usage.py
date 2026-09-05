"""
Visualize how often each electrode was chosen as a VISIBLE valid channel (VC)
during training of FixdElectrodsN_Recon.

Produces the same unrolled-torso spatial map as the predecessor codebase's
figures/fig_missing_spatial.png (azimuth around the torso vs body height,
scatter coloured by a per-electrode scalar), so one can see whether the
selection concentrates on a few electrodes or spreads over the whole vest.

Outputs (into out_dir):
  usage_spatial.png        unrolled-torso scatter, colour = selection frequency
  usage_per_channel.png    per-kept-channel selection-frequency bar
  usage_hist.png           histogram of per-electrode selection frequency
  usage_counts.npy         raw counts (n_keep,)

`frequency` = counts / n_samples_seen = the fraction of recording-passes in
which the electrode was picked. Since each pass picks exactly VC electrodes,
the frequencies sum to ~VC and a perfectly uniform selector would give a flat
map at VC / n_keep.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torso_loader import load_torso_coords

NCH_RAW = 256
from dataset import COHORT_A_ROOT, COHORT_B_ROOT, EXTERNAL_ROOT
_TORSO_SEARCH_ROOTS = [COHORT_B_ROOT, COHORT_A_ROOT, EXTERNAL_ROOT]


def find_repr_torso(keep_idx, min_registered=250):
    """Return raw (256,3) coords of a torso registering >= min_registered of the
    kept electrodes (a 'complete' vest), or None."""
    for root in _TORSO_SEARCH_ROOTS:
        for t in glob.glob(os.path.join(root, '*', '**', '*.torso'), recursive=True):
            try:
                c = load_torso_coords(t)
            except Exception:
                continue
            if (~np.isnan(c[keep_idx, 0])).sum() >= min_registered:
                return c
    return None


def _unrolled_torso(coords_keep):
    """coords_keep: (n_keep,3) raw coords (NaN where unregistered).
    Returns (angle, height, valid) for the unrolled-torso projection: the axis
    with the largest spread is 'height', the other two give the azimuth."""
    valid = ~np.isnan(coords_keep[:, 0])
    vc = coords_keep[valid]
    rng_axis = int(np.argmax(np.nanmax(vc, 0) - np.nanmin(vc, 0)))
    other = [a for a in range(3) if a != rng_axis]
    height = coords_keep[:, rng_axis]
    ang = np.arctan2(coords_keep[:, other[1]] - np.nanmean(vc[:, other[1]]),
                     coords_keep[:, other[0]] - np.nanmean(vc[:, other[0]]))
    return ang, height, valid


def plot_usage(usage_counts, n_samples, keep_idx, out_dir, vc, mode_label,
               min_registered=250):
    """Write the spatial usage map + per-channel bar + histogram.

    usage_counts : (n_keep,) int   — times each kept channel was chosen visible
    n_samples    : int             — total recording-passes seen in training
    keep_idx     : (n_keep,) int   — original electrode indices of kept channels
    """
    os.makedirs(out_dir, exist_ok=True)
    usage_counts = np.asarray(usage_counts, dtype=np.float64)
    n_keep = len(usage_counts)
    np.save(os.path.join(out_dir, 'usage_counts.npy'), usage_counts.astype(np.int64))
    freq = usage_counts / max(int(n_samples), 1)          # (n_keep,) in [0,1]
    uniform = float(vc) / n_keep

    # ---- histogram of per-electrode selection frequency ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(freq, bins=40, color='tab:purple', alpha=0.8)
    ax.axvline(uniform, color='k', ls='--', lw=1.2,
               label=f'uniform = VC/n_keep = {uniform:.3f}')
    ax.set_xlabel('per-electrode selection frequency (fraction of passes chosen)')
    ax.set_ylabel('number of electrodes')
    ax.set_title(f'VC selection frequency — mode={mode_label}, VC={vc}\n'
                 f'({int((usage_counts>0).sum())}/{n_keep} electrodes ever used)')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'usage_hist.png'), dpi=130)
    plt.close()

    # ---- per-kept-channel bar ----
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(np.arange(n_keep), freq, width=1.0, color='tab:purple')
    ax.axhline(uniform, color='k', ls='--', lw=1.0, label=f'uniform {uniform:.3f}')
    ax.set_xlabel('kept-channel index (0..n_keep-1)')
    ax.set_ylabel('selection frequency')
    ax.set_title(f'Per-channel VC selection frequency — mode={mode_label}, VC={vc}')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'usage_per_channel.png'), dpi=130)
    plt.close()

    # ---- unrolled-torso spatial map ----
    coords = find_repr_torso(keep_idx, min_registered=min_registered)
    if coords is None:
        print('[viz_usage] no representative torso found; skipping spatial map')
        return
    ck = coords[keep_idx]                                  # (n_keep,3) raw
    ang, height, valid = _unrolled_torso(ck)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    vmax = max(float(freq[valid].max()), 1e-3)
    sc = ax.scatter(ang[valid], height[valid], c=freq[valid], cmap='viridis',
                    vmin=0, vmax=vmax, s=60, edgecolor='k', lw=0.3)
    ax.set_xlabel('azimuth around torso (rad)')
    ax.set_ylabel('body height')
    ax.set_title(f'Spatial distribution of VC selection frequency\n'
                 f'mode={mode_label}, VC={vc}  (scale 0–{vmax:.2f}, '
                 f'uniform={uniform:.3f})')
    plt.colorbar(sc, ax=ax, label='selection frequency')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'usage_spatial.png'), dpi=130)
    plt.close()
    print(f"[viz_usage] wrote usage_spatial.png ({int(valid.sum())} electrodes placed), "
          f"usage_per_channel.png, usage_hist.png to {out_dir}")
