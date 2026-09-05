"""
Assemble the body-surface reconstruction sources for one recording.

Split into a MODEL-FREE part and a MODEL-DEPENDENT part so the references can be
cached independently of any trained model:

  build_context(...)         -> prep signal, missing/active/VC masks, coords, g,
                                and the two MODEL-FREE 256-ch fields:
                                  src1_vc       (real BSP, VC channels active)
                                  src2_allvalid (real BSP, all valid active)  [ref]
  reconstruct_sources(model,ctx) -> the two MODEL-DEPENDENT fields:
                                  src3_recon_whole (VC real + recon; all geometry)
                                  src4_recon_valid (VC real + recon; valid subset)

Each field is (V256 preprocessed physical, active256 bool). Scale handling:
g = median std of the VC channels (physical); model input = prep/g, output /g is
de-scaled by xg back to physical so reconstructed channels share the VC scale.
"""
import os
import sys
import json
import numpy as np
import torch

_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
import models_fixedn                                    # noqa: E402
from preprocess import preprocess_ecgi                  # noqa: E402
from detect_missing import detect_missing               # noqa: E402
from torso_loader import load_torso_coords, normalize_coords  # noqa: E402
from channel_select import select_visible_channels, _coord_valid  # noqa: E402
from mesh_io import import_ecgdata                       # noqa: E402

NCH_RAW = 256
CROP = 5000


def center_crop(x, n=CROP):
    L = x.shape[1]
    if L <= n:
        return x, 0
    s = (L - n) // 2
    return x[:, s:s + n], s


def load_model(run_dir, device):
    """Load a trained FixdElectrodsN_Recon model + its keep_idx + args."""
    args = json.load(open(os.path.join(run_dir, 'args.json')))
    keep_idx = np.load(os.path.join(run_dir, 'keep_idx.npy'))
    model = models_fixedn.__dict__[args['model']](
        signal_shape=(len(keep_idx), args['crop_len']),
        patch_size=tuple(args['patch_size']),
        pos_mode=args.get('pos_embed', 'ppe'))
    ck = torch.load(os.path.join(run_dir, 'checkpoints', 'checkpoint_best.pth'),
                    map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    return model, keep_idx.astype(np.int64), args


def _to256(field252, act_sub, keep_idx):
    V256 = np.zeros((NCH_RAW, CROP), np.float32); V256[keep_idx] = field252
    a256 = np.zeros(NCH_RAW, bool); a256[keep_idx[act_sub]] = True
    return V256, a256


def build_context(ecgdata_path, torso_path, keep_idx, args,
                  vc=16, mode='cluster', cluster_method='greedy', seed=42,
                  region='all'):
    """MODEL-FREE: prep signal + masks + VC selection + src1/src2 fields.
    `region` ('all' | 'front' | 'back') restricts the VC candidates to one torso
    half (channel_select.region_mask); 'all' is a strict no-op.
    Returns dict or None."""
    ecg256, L, n_e = import_ecgdata(ecgdata_path)
    if n_e != NCH_RAW:
        return None
    ecg252, s = center_crop(ecg256[keep_idx].astype(np.float32), CROP)
    if ecg252.shape[1] < CROP:
        return None

    raw = load_torso_coords(torso_path).astype(np.float32)
    coords = np.nan_to_num(normalize_coords(raw, pp_norm=args.get('pp_norm', 'centroid')),
                           nan=0.0).astype(np.float32)[keep_idx]
    raw_coords = raw[keep_idx]

    missing252 = detect_missing(ecg252, coords=raw_coords)
    active252 = ~missing252
    prep252 = preprocess_ecgi(ecg252, missing252).astype(np.float32)

    rng = np.random.default_rng(seed)
    visible252, sel_info = select_visible_channels(coords, active252, vc, mode,
                                                   cluster_method, rng, region,
                                                   return_info=True)
    if visible252.sum() == 0:
        return None
    # g = median std of the VC channels, computed EXACTLY as the training-time
    # engine.global_norm (torch std ddof=1 + torch.median) so the model sees the
    # same input scale it was normalized with.
    g = float(torch.from_numpy(prep252[visible252]).std(dim=1).median())
    if g < 1e-8:
        g = 1.0

    return dict(
        keep_idx=keep_idx, prep252=prep252, coords=coords,
        active252=active252, visible252=visible252, g=g, crop_start=s, L=L,
        n_active=int(active252.sum()), n_vc=int(visible252.sum()),
        n_recon=int((active252 & ~visible252).sum()),
        region=region, n_region_cand=sel_info['n_region_cand'],
        n_region_fallback=sel_info['n_region_fallback'],
        src1_vc=_to256(prep252, visible252, keep_idx),          # real, VC only
        src2_allvalid=_to256(prep252, active252, keep_idx),     # real, all valid (ref)
    )


def held_out_bsp_cc(ctx, recon252):
    """BSP CC of the model reconstruction on the HELD-OUT electrodes (active
    and NOT visible -- coordinate-less active channels included, exactly the
    engine's recon_mask = active & ~visible behind test_compare.json): per-
    electrode temporal Pearson CC with the measured preprocessed signal, mean
    over electrodes (std<1e-8 guard on either side). Returns (cc, n_electrodes);
    cc is NaN if no electrode qualifies."""
    prep252 = ctx['prep252']
    held = ctx['active252'] & ~ctx['visible252']
    ccs = []
    for c in np.where(held)[0]:
        a_, b_ = np.asarray(recon252[c], np.float64), np.asarray(prep252[c], np.float64)
        if a_.std() > 1e-8 and b_.std() > 1e-8:
            cc = float(np.corrcoef(a_, b_)[0, 1])
            if np.isfinite(cc):
                ccs.append(cc)
    return (float(np.mean(ccs)) if ccs else float('nan')), len(ccs)


@torch.no_grad()
def reconstruct_sources(model, ctx, device):
    """MODEL-DEPENDENT: run the model on the context, return src3/src4 fields."""
    keep_idx = ctx['keep_idx']; prep252 = ctx['prep252']; g = ctx['g']
    visible252 = ctx['visible252']; active252 = ctx['active252']
    x = torch.from_numpy(prep252 / g).unsqueeze(0).to(device)
    active_t = torch.from_numpy(active252).unsqueeze(0).to(device)
    coords_t = torch.from_numpy(ctx['coords']).unsqueeze(0).to(device)
    vis_t = torch.from_numpy(visible252).unsqueeze(0).to(device)
    _, pred = model.forward_fixedn(x, active_t, coords_t, vis_t,
                                   w_spatial=0.0, w_temporal=0.0)
    recon252 = model.unpatchify(pred).squeeze(0).squeeze(0).cpu().numpy() * g
    assembled252 = np.where(visible252[:, None], prep252, recon252).astype(np.float32)
    all_keep = np.ones(len(keep_idx), dtype=bool)
    return dict(
        src3_recon_whole=_to256(assembled252, all_keep, keep_idx),
        src4_recon_valid=_to256(assembled252, active252, keep_idx),
        recon252=recon252,          # raw model output (physical), 252-ch
    )
