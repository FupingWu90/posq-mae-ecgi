"""Layout-shift stress test: physical-position (PPE) vs index embedding.

Scenario: the electrode-to-channel assignment deviates from the nominal
training layout -- the vest is worn rotated/shifted, a harness variant or a
different acquisition system wires channels in another order -- while the
geometry registration still records every electrode's TRUE position. Signals
and their measured positions are therefore correct but arrive on OTHER channel
indices. A physical-position model reads the measured geometry, so its
(signal, position) pairs are unchanged as a set (permutation-equivariant); an
index-embedding model has its positional prior baked into the slot statistics
of the training layout and is mislabelled on every channel.

Emulation per recording: rotate the electrode cloud by alpha about its first
principal axis (the torso long axis), match each rotated position to the
nearest true electrode (greedy bijection) -> permutation pi; channel i then
carries the signal of electrode pi(i) with measured coordinate p_{pi(i)}.
PPE input = permuted (signal, coord) pairs (identical set: results must equal
alpha=0, built-in sanity check); index input = permuted signals on the nominal
slots. Selection (k-means on the true active positions) picks the same
physical electrodes for both. Reported: held-out BSP MAE/CC vs alpha, plus the
mean displacement ||p_i - p_{pi(i)}|| in units of the electrode spacing.

Usage: python layout_shift_test.py [--n 60] [--alphas 0 5 10 20 40]
Out:   generated/layout_shift.json
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

from common import GEN, PROJ, run_dir

sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation'))
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from build_sources import load_model, build_context                       # noqa: E402
from channel_select import select_visible_channels                        # noqa: E402
from dataset import collect_split                                         # noqa: E402

warnings.filterwarnings('ignore')

BASE = 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'


def perm_from_rotation(coords, ok, alpha_deg, rng):
    """Greedy bijection pi: slot i <- electrode nearest to R(alpha) p_i."""
    idx = np.where(ok)[0]
    P = coords[idx]
    axis = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    a = np.deg2rad(alpha_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    Q = (P - P.mean(0)) @ R.T + P.mean(0)
    D = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=-1)
    pi = np.full(len(idx), -1)
    used = np.zeros(len(idx), bool)
    for i in np.argsort(D.min(1)):                 # greedy, best matches first
        j = np.argmin(np.where(used, np.inf, D[i]))
        pi[i] = j; used[j] = True
    full = np.arange(len(coords))
    full[idx] = idx[pi]
    disp = np.linalg.norm(coords[idx] - coords[idx[pi]], axis=-1)
    return full, float(disp.mean())


def recon_metrics(model, device, prep, coords, active, visible, g):
    x = torch.from_numpy(prep / g).unsqueeze(0).to(device)
    act = torch.from_numpy(active).unsqueeze(0).to(device)
    cd = torch.from_numpy(coords).unsqueeze(0).to(device)
    sel = torch.from_numpy(visible).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred = model.forward_fixedn(x, act, cd, sel,
                                       w_spatial=0.0, w_temporal=0.0)
        pred = model.unpatchify(pred).squeeze(0).squeeze(0).cpu().numpy()
    held = active & ~visible
    tgt = prep / g
    mae = float(np.abs(pred[held] - tgt[held]).mean())
    ccs = [float(np.corrcoef(pred[c], tgt[c])[0, 1]) for c in np.where(held)[0]
           if pred[c].std() > 1e-8 and tgt[c].std() > 1e-8]
    return mae, float(np.mean(ccs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--alphas', type=float, nargs='*', default=[0, 5, 10, 20, 40])
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    for lab, sfx in (('ppe', ''), ('index', '_pe-index')):
        m, keep_idx, args = load_model(
            os.path.join(PROJ, 'runs', 'data-cohortA+cohortB_vs_external', BASE + sfx), device)
        models[lab] = m
    buckets, _ = collect_split(seed=42)
    res = {}
    spacing = None
    for split in ('test_internal', 'test_external'):
        recs = [r for r in buckets[split] if r[2]]
        recs = [recs[i] for i in np.linspace(0, len(recs) - 1, a.n).astype(int)]
        acc = {(lab, al): [] for lab in models for al in a.alphas}
        disp_acc = {al: [] for al in a.alphas}
        for k, rec in enumerate(recs):
            ctx = build_context(rec[1], rec[2], keep_idx, args, vc=16,
                                mode='cluster', cluster_method='kmean', seed=42)
            if ctx is None:
                continue
            prep, coords = ctx['prep252'], ctx['coords']
            active = ctx['active252'].astype(bool)
            ok = np.abs(coords).sum(-1) > 1e-6
            if spacing is None:
                from scipy.spatial import cKDTree
                spacing = float(np.median(cKDTree(coords[ok]).query(coords[ok], k=2)[0][:, 1]))
            sd = prep[active].std(1); g = np.median(sd[sd > 1e-8])
            vis = select_visible_channels(coords, active, 16, 'cluster', 'kmean',
                                          np.random.default_rng(42))
            for al in a.alphas:
                pi, disp = perm_from_rotation(coords, ok & active, al,
                                              np.random.default_rng(0))
                disp_acc[al].append(disp)
                prep_p, act_p, vis_p = prep[pi], active[pi], vis[pi]
                coords_p = coords[pi]
                for lab, m in models.items():
                    cd = coords_p if lab == 'ppe' else coords
                    mae, cc = recon_metrics(m, device, prep_p, cd, act_p, vis_p, g)
                    acc[(lab, al)].append((mae, cc))
            if (k + 1) % 20 == 0:
                print(f'[{split}] {k+1}/{len(recs)}', flush=True)
        res[split] = {f'{lab}|{al}': {
            'mae': float(np.mean([v[0] for v in acc[(lab, al)]])),
            'cc': float(np.mean([v[1] for v in acc[(lab, al)]])),
            'n': len(acc[(lab, al)])} for lab in models for al in a.alphas}
        res[split]['displacement_spacings'] = {
            str(al): float(np.mean(disp_acc[al]) / spacing) for al in a.alphas}
    json.dump(res, open(os.path.join(GEN, 'layout_shift.json'), 'w'), indent=2)
    for split in res:
        print(f'--- {split} (mean displacement in electrode spacings per alpha) ---')
        print('   ', res[split]['displacement_spacings'])
        for lab in ('ppe', 'index'):
            print('   ', lab, {al: round(res[split][f'{lab}|{al}']['cc'], 4)
                               for al in a.alphas})


if __name__ == '__main__' and '--figs' not in sys.argv:
    main()


def make_figs():
    """CC vs coherent layout displacement: PPE flat, index degrades."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(GEN, 'layout_shift.json')))
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        r = res[split]
        alphas = sorted({k.split('|')[1] for k in r if '|' in k}, key=float)
        x = [r['displacement_spacings'][a] for a in alphas]
        fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        ax.plot(x, [r[f'ppe|{a}']['cc'] for a in alphas], 's-', color='#0072B2',
                lw=2, ms=6, label='Ours (physical positions)')
        ax.plot(x, [r[f'index|{a}']['cc'] for a in alphas], 'o--', color='#D55E00',
                lw=1.8, ms=6, label='index embedding (ST-MEM-style)')
        ax.set_xlabel('coherent layout shift (electrode spacings)')
        ax.set_ylabel('BSP reconstruction CC (held-out electrodes)')
        ax.set_title(f'{lab} dataset', fontsize=10)
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc='lower left')
        out = f'fig_layout_shift_{lab}.png'
        fig.savefig(os.path.join(FIGS, out), dpi=150); plt.close(fig)
        print('wrote', out)


if __name__ == '__main__' and '--figs' in sys.argv:
    from common import FIGS  # noqa: F811
    make_figs()
