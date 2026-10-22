"""What does the position-queried decoder attend to?  (read-only analysis of a trained posq_* run)

For validation recordings under the template layout, the cross-attention weights of every decoder block are
computed for a subsample of the queries and summarised per block and per head:
  * attention mass on memory tokens of the SAME time patch as the query, and within +-1 / +-2 patches
    (uniform attention would give 1.0% / 3.0% / 5.0% for N=16, T=100);
  * mass on the tokens of the spatially nearest visible electrode(s);
  * effective number of attended tokens (exp of the entropy; 1601 = uniform);
  * number of time-local heads (more than half of their mass on the same time patch) and the total width
    (in dimensions) of those heads = the width of the channel that carries time-specific information.
  python baselines_analyses/diag_cross_attention.py --run_dir <run> [--n 10] [--out json]
"""
import os, sys, json, argparse, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (os.path.join(V2, 'bsp_reconstruction'), os.path.join(V2, 'inverse_evaluation')):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import torch
import models_v2 as M
from build_sources import load_model, build_context
from dataset import collect_split


@torch.no_grad()
def analyse(run_dir, n=10, n_elec=24, split='val'):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(run_dir, dev)
    if not isinstance(model, M.PosQMAE):
        return None
    vc, T = args['vc'], model.T
    recs = [r for r in collect_split(seed=42)[0][split] if r[2] is not None]
    recs = recs[::max(1, len(recs) // n)][:n]
    nb = len(model.decoder_blocks)
    H = model.decoder_blocks[0].attn_x.h
    keys = ('cls', 'dt0', 'dt1', 'dt2', 'near1', 'near3', 'neff')
    acc = {k: np.zeros((nb, H)) for k in keys}
    cnt = 0
    rng = np.random.default_rng(0)
    for sid, ecg, tor in recs:
        ctx = build_context(ecg, tor, keep_idx, args, vc=vc, mode='template', cluster_method='greedy', seed=42)
        if ctx is None:
            continue
        x = torch.from_numpy(ctx['prep252'] / ctx['g']).unsqueeze(0).to(dev)
        coords = torch.from_numpy(ctx['coords']).unsqueeze(0).to(dev)
        vis = torch.from_numpy(ctx['visible252']).unsqueeze(0).to(dev)
        vis_idx = torch.argsort(vis.to(torch.uint8), dim=1, descending=True, stable=True)[:, :vc]
        tok = model._tok_series(M._gather(x, vis_idx)) + model._pos(M._gather(model.ppe(coords), vis_idx), model.time_embed, T)
        enc = model._encode(tok.reshape(1, vc * T, -1))
        ch_d = model.decoder_ppe(coords)
        mpos = model._pos(M._gather(ch_d, vis_idx), model.decoder_time_embed, T).reshape(1, vc * T, -1)
        mem = model.decoder_embed(enc)
        mem = torch.cat([mem[:, :1] + model.decoder_cls_pos_embed, mem[:, 1:] + mpos], dim=1)
        cand = np.where(ctx['active252'] & ~ctx['visible252'])[0]
        qe = np.sort(rng.choice(cand, size=min(n_elec, len(cand)), replace=False))
        q_idx = torch.from_numpy(qe).unsqueeze(0).to(dev)
        q = (model.mask_token + model._pos(M._gather(ch_d, q_idx), model.decoder_time_embed, T).reshape(1, len(qe) * T, -1))
        order = torch.cdist(coords[0, q_idx[0]], coords[0, vis_idx[0]]).argsort(dim=1)      # nearest visible electrode first
        tq = torch.arange(T, device=dev).repeat(len(qe)); eq = torch.arange(len(qe), device=dev).repeat_interleave(T)
        tm = torch.arange(T, device=dev).repeat(vc); em = torch.arange(vc, device=dev).repeat_interleave(T)
        dt = (tq[:, None] - tm[None, :]).abs()
        near1 = (em[None, :] == order[eq, 0][:, None])
        near3 = (em[None, :, None] == order[eq][:, None, :3]).any(-1)
        for b, blk in enumerate(model.decoder_blocks):
            at = blk.attn_x
            d = q.shape[-1] // at.h
            qq = at.q(blk.norm_q(q)).view(1, -1, at.h, d).transpose(1, 2)
            kk = at.k(blk.norm_m(mem)).view(1, -1, at.h, d).transpose(1, 2)
            A = torch.softmax(qq.float() @ kk.float().transpose(-1, -2) / (d ** 0.5), dim=-1)[0]      # (h, Lq, 1+vc*T)
            Ab = A[:, :, 1:]
            acc['cls'][b] += A[:, :, 0].mean(1).cpu().numpy()
            for k, msk in (('dt0', dt == 0), ('dt1', dt <= 1), ('dt2', dt <= 2), ('near1', near1), ('near3', near3)):
                acc[k][b] += (Ab * msk).sum(-1).mean(1).cpu().numpy()
            acc['neff'][b] += (-(A.clamp_min(1e-12).log() * A).sum(-1)).exp().mean(1).cpu().numpy()
            q = blk(q, mem, T)
        cnt += 1
    per_head = {k: (v / max(cnt, 1)) for k, v in acc.items()}
    Dd = model.decoder_embed.out_features
    res = dict(run=os.path.basename(run_dir.rstrip('/')), n_recordings=cnt, heads=H, head_dim=Dd // H,
               block_mean={k: v.mean(1).round(4).tolist() for k, v in per_head.items()},
               time_local_heads=[int((per_head['dt0'][b] > 0.5).sum()) for b in range(nb)],
               time_local_width=[int((per_head['dt0'][b] > 0.5).sum()) * (Dd // H) for b in range(nb)],
               per_head_same_time=[np.sort(per_head['dt0'][b])[::-1].round(3).tolist() for b in range(nb)],
               per_head_neff=[per_head['neff'][b][np.argsort(-per_head['dt0'][b])].round(0).tolist() for b in range(nb)],
               uniform=dict(dt0=round(vc / (vc * T + 1), 4), dt1=round(3 * vc / (vc * T + 1), 4), dt2=round(5 * vc / (vc * T + 1), 4),
                            near1=round(T / (vc * T + 1), 4), near3=round(3 * T / (vc * T + 1), 4), neff=vc * T + 1))
    return res


def report(res):
    bm = res['block_mean']
    for b in range(len(bm['dt0'])):
        print(f"block {b + 1}: same time patch {100 * bm['dt0'][b]:.1f}% | +-2 patches {100 * bm['dt2'][b]:.1f}% | nearest visible electrode "
              f"{100 * bm['near1'][b]:.1f}% | effective tokens {bm['neff'][b]:.0f} | time-local heads {res['time_local_heads'][b]} of {res['heads']} "
              f"(= {res['time_local_width'][b]} dims) | per-head same-time mass: " + ' '.join(f'{100 * v:.0f}' for v in res['per_head_same_time'][b]))
    print('uniform attention would give:', res['uniform'])


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--n_elec', type=int, default=24); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = analyse(a.run_dir, a.n, a.n_elec)
    if r is None:
        print('not a PosQ-MAE run: nothing to analyse')
    else:
        report(r)
        out = a.out or os.path.join(a.run_dir, 'cross_attention_val.json')
        json.dump(r, open(out, 'w'), indent=1); print('wrote', out)
