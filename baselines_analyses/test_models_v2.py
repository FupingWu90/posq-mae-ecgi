"""Unit tests of models_v2.PosQMAE against the v1 model (real validation batch, GPU).

T1  encoder equivalence : with the v1 checkpoint's encoder weights, the v2 encoder path gives the same
                          visible-token representations as v1.forward_encoder (token order aside).
T2  loss equivalence    : the v2 reconstruction and temporal terms equal v1.forward_loss / smoothness_loss
                          evaluated on the same prediction.
T3  zero-init residuals : tokenizer tnt/conv and head subattn/convref reproduce the plain model at init.
T4  every variant       : train/eval forward + backward, finite loss, output layout, query independence.
"""
import os, sys, json, warnings
import numpy as np
import torch
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
warnings.filterwarnings('ignore')
import models_v2 as M
import models_fixedn
from paths import V1_RUNS
from dataset import build_experiment_datasets
from engine import global_norm
from channel_select import select_visible_batch

dev = torch.device('cuda')
RUN = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')
args = json.load(open(os.path.join(RUN, 'args.json')))
_, val_ds, _, _, info = build_experiment_datasets(crop_len=5000, preprocess=True, seed=42, aug_times=1, pp_norm='centroid', tag='exp',
                                                  min_active=17, prep_cache=True, norm_mode='global', verbose=False)
B = 4
items = [val_ds[i] for i in (0, 30, 60, 90)]
ecg = torch.stack([it[0] for it in items]).to(dev); miss = torch.stack([it[1] for it in items]).to(dev); coords = torch.stack([it[2] for it in items]).to(dev)
active = ~miss
vis = torch.from_numpy(select_visible_batch(coords.cpu().numpy(), active.cpu().numpy(), 16, 'cluster', 'kmean', np.random.default_rng(42))).to(dev)
x, _ = global_norm(ecg, vis)
C, L = x.shape[1:]; T = 100

v1 = models_fixedn.fixedn_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50)).to(dev)
ck = torch.load(os.path.join(RUN, 'checkpoints', 'checkpoint_best.pth'), map_location='cpu', weights_only=False)
v1.load_state_dict(ck['model']); v1.eval()

# ---------------- T1 ----------------
v2 = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50)).to(dev)
sd = {k: v for k, v in ck['model'].items() if not k.startswith('decoder_blocks')}
missing, unexpected = v2.load_state_dict(sd, strict=False)
assert all(k.startswith('decoder_blocks') for k in missing) and not unexpected, (missing[:3], unexpected[:3])
v2.eval()
with torch.no_grad():
    lat, mask, ids_restore = v1.forward_encoder(x.unsqueeze(1), ~vis, coords, mask_mode='fixed')
    n_tok = lat.shape[1] - 1
    # v1 token j of the shuffled visible list sits at patch position p with ids_restore[p] == j
    pos_of = torch.argsort(ids_restore, dim=1)[:, :n_tok]                # patch index of visible token j
    order = torch.argsort(pos_of, dim=1)                                 # canonical (electrode, time) order
    lat_sorted = torch.gather(lat[:, 1:], 1, order.unsqueeze(-1).expand(-1, -1, lat.shape[-1]))
    vis_idx = torch.argsort(vis.to(torch.uint8), dim=1, descending=True, stable=True)[:, :16]
    tok = v2._tok_series(M._gather(x, vis_idx)) + v2._pos(M._gather(v2.ppe(coords), vis_idx), v2.time_embed, T)
    enc = v2._encode(tok.reshape(B, 16 * T, -1))
d1 = (enc[:, 1:] - lat_sorted).abs().max().item(); d1c = (enc[:, 0] - lat[:, 0]).abs().max().item()
print(f'T1 encoder equivalence: max |diff| tokens {d1:.2e}, cls {d1c:.2e}  ->', 'PASS' if max(d1, d1c) < 1e-3 else 'FAIL')

# ---------------- T2 ----------------
with torch.no_grad():
    loss2, pred2, Lr2, Ls2, Lt2 = v2.forward_fixedn(x, active, coords, vis, w_spatial=0.0, w_temporal=3.0, return_components=True)
    L1_v1 = v1.forward_loss(x.unsqueeze(1), pred2, mask, ~active)
    ps = v1.unpatchify(pred2).squeeze(1)
    _, Lt_v1 = v1.smoothness_loss(ps, x, vis, active, coords)
print(f'T2 loss equivalence: L1 v2 {Lr2.item():.6f} vs v1 formula {L1_v1.item():.6f}; temporal v2 {Lt2.item():.6e} vs v1 {Lt_v1.item():.6e}  ->',
      'PASS' if abs(Lr2.item() - L1_v1.item()) < 1e-5 and abs(Lt2.item() - Lt_v1.item()) < 1e-6 * max(1, abs(Lt_v1.item())) + 1e-7 else 'FAIL')

# ---------------- T3 ----------------
ok3 = True
torch.manual_seed(0); base = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50)).to(dev).eval()
with torch.no_grad():
    _, pb = base.forward_fixedn(x, active, coords, vis)
for kw in (dict(tokenizer='tnt'), dict(tokenizer='conv'), dict(head='subattn'), dict(head='convref')):
    torch.manual_seed(0); m = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50), **kw).to(dev).eval()
    sdb = base.state_dict(); sdm = m.state_dict(); sdm.update({k: v for k, v in sdb.items() if k in sdm}); m.load_state_dict(sdm)
    with torch.no_grad():
        _, pm = m.forward_fixedn(x, active, coords, vis)
    d = (pm - pb).abs().max().item(); ok3 &= d < 1e-5
    print(f'T3 zero-init residual {kw}: max |diff| vs plain {d:.2e}')
print('T3 ->', 'PASS' if ok3 else 'FAIL')

# ---------------- T4 ----------------
ok4 = True
variants = [dict(), dict(dec_tsa=True), dict(dec_ssa=True), dict(dec_ssa=True, dec_tsa=True, head='subattn', inner_depth=1), dict(dec_depth=6), dict(query_frac=0.5), dict(tokenizer='tnt'), dict(tokenizer='conv'),
            dict(head='subattn'), dict(head='convref'), dict(time_embed='learned'), dict(ppe_type='fourier'),
            dict(tmask_mode='dropout', tmask_ratio=0.25), dict(tmask_mode='span', tmask_ratio=0.2), dict(tmask_mode='alternate'),
            dict(tmask_mode='alternate', tokenizer='conv'), dict(tmask_mode='dropout', tmask_ratio=0.5, dec_tsa=True, head='subattn', tokenizer='tnt', query_frac=0.5)]
for kw in variants:
    m = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50), **kw).to(dev)
    m.train(); losses = []
    for step in range(2):                       # call 0 = pure electrode step, call 1 = mixed step
        with torch.amp.autocast('cuda'):
            loss, pred, Lr, Ls, Lt = m.forward_fixedn(x, active, coords, vis, w_temporal=3.0, return_components=True)
        loss.backward(); losses.append(loss.item())
    gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    m.eval()
    with torch.no_grad():
        _, pe = m.forward_fixedn(x, active, coords, vis)
        # query independence: decoding a subset of the masked electrodes gives the same values
        sig = m.unpatchify(pe).squeeze(1)
    held = active & ~vis
    fin = all(np.isfinite(l) for l in losses) and np.isfinite(gn) and torch.isfinite(pe).all().item()
    shape_ok = tuple(pe.shape) == (B, C * T, 50) and (pe.view(B, C, T, 50)[vis].abs().max().item() == 0.0)
    ok4 &= fin and shape_ok and not nograd
    print(f'T4 {str(kw):110s} loss {losses[0]:.4f}/{losses[1]:.4f} finite={fin} layout={shape_ok} params-without-grad={len(nograd)}')
print('T4 ->', 'PASS' if ok4 else 'FAIL')

# ---------------- T5: arbitrary-position query == electrode query ----------------
m = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50), dec_tsa=True, head='subattn').to(dev).eval()
with torch.no_grad():
    _, pe = m.forward_fixedn(x, active, coords, vis)
    full = m.unpatchify(pe).squeeze(1)
    qi = torch.argsort(vis.to(torch.uint8), dim=1, stable=True)[:, :40]            # 40 masked electrodes
    out = m.complete_at(x, coords, vis, M._gather(coords, qi))
d5 = (out - M._gather(full, qi)).abs().max().item()
print(f'T5 arbitrary-position decoding equals the standard completion: max |diff| {d5:.2e}  ->', 'PASS' if d5 < 1e-4 else 'FAIL')
