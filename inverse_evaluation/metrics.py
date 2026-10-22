"""HSP comparison metrics: potential temporal CC, scale ratio, AT/RT map MAE+CC."""
import numpy as np


def temporal_cc(a, b):
    """(V,T) fields -> (median, mean) per-vertex temporal Pearson correlation."""
    am = a - a.mean(1, keepdims=True)
    bm = b - b.mean(1, keepdims=True)
    den = np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    cc = np.divide((am * bm).sum(1), den, out=np.full(a.shape[0], np.nan), where=den > 1e-12)
    cc = cc[np.isfinite(cc)]
    if cc.size == 0:
        return 0.0, 0.0
    return float(np.median(cc)), float(np.mean(cc))


def scale_ratio(a, b):
    """||a|| / ||b|| — CC-invariant sanity check (recon_complete inflates this)."""
    return float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-12))


def map_errors(x, ref, good=None):
    """AT/RT/ARI per-vertex maps -> (mean_abs_err, median_abs_err, spatial_CC),
    the per-vertex absolute error aggregated over `good` vertices by BOTH the
    spatial mean (conventional MAE) and the spatial median (robust to the
    outlier vertices that survive gating). Errors in samples/ms."""
    x = np.asarray(x, float); ref = np.asarray(ref, float)
    if good is not None and good.sum() > 2:
        x = x[good]; ref = ref[good]
    if x.size == 0:
        return float('nan'), float('nan'), 0.0
    ae = np.abs(x - ref)
    mae = float(ae.mean())
    mdae = float(np.median(ae))
    cc = 0.0 if (x.std() < 1e-8 or ref.std() < 1e-8) else float(np.corrcoef(x, ref)[0, 1])
    if not np.isfinite(cc):
        cc = 0.0
    return mae, mdae, cc


def map_mae_cc(x, ref, good=None):
    """Back-compat wrapper: (spatial-mean MAE, spatial CC)."""
    mae, _mdae, cc = map_errors(x, ref, good)
    return mae, cc


def hsp_metrics(a, b, good=None):
    """Heart-surface potential fields (V,T) source `a` vs reference `b`.
    Per-vertex TEMPORAL MAE / RMSE / CC, then aggregated over vertices by BOTH
    the spatial mean and the spatial median. Returns a dict with keys
    mae, mae_md, rmse, rmse_md, cc, cc_md  (mae/rmse in signal units; cc unitless).
    MAE/RMSE are amplitude-sensitive; CC is scale-invariant."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if good is not None and good.sum() > 2:
        a = a[good]; b = b[good]
    if a.size == 0:
        return dict(mae=float('nan'), mae_md=float('nan'), rmse=float('nan'),
                    rmse_md=float('nan'), cc=0.0, cc_md=0.0)
    d = a - b
    mae_v = np.abs(d).mean(axis=1)                    # (V,) per-vertex temporal MAE
    rmse_v = np.sqrt((d ** 2).mean(axis=1))           # (V,) per-vertex temporal RMSE
    am = a - a.mean(1, keepdims=True); bm = b - b.mean(1, keepdims=True)
    den = np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    cc_v = np.divide((am * bm).sum(1), den, out=np.full(a.shape[0], np.nan), where=den > 1e-12)
    cc_v = cc_v[np.isfinite(cc_v)]
    return dict(
        mae=float(mae_v.mean()), mae_md=float(np.median(mae_v)),
        rmse=float(rmse_v.mean()), rmse_md=float(np.median(rmse_v)),
        cc=float(cc_v.mean()) if cc_v.size else 0.0,
        cc_md=float(np.median(cc_v)) if cc_v.size else 0.0,
    )
