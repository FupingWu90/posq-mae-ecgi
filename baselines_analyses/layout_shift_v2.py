"""Layout-shift stress test on the final architecture (v2 port of the frozen v1 layout_shift_test.py):
physical-position queries (Ours, PosQ-MAE) vs the index-embedding variant (--pos_embed index), both arms with
the SAME backbone / training protocol and up to three training seeds each.

Scenario (unchanged from v1): the electrode-to-channel assignment deviates coherently from the nominal training
layout (vest worn rotated / shifted, harness variant, another system's channel order) while the geometry
registration still records every electrode's TRUE position.  Emulation per recording: rotate the electrode cloud
by alpha about its first principal axis, match each rotated position to the nearest true electrode (greedy
bijection) -> permutation pi; channel i then carries the signal of electrode pi(i) with measured coordinate
p_pi(i).  Position model input = permuted (signal, coordinate) pairs (the same set: must equal alpha = 0 up to
float noise, built-in check); index model input = permuted signals on the nominal slots.  The visible electrodes
are the deployment layout's PHYSICAL electrodes (fixed template with nearest-valid substitution, or per-recording
k-means, both on the true geometry), so both arms see the same measured electrodes.  Reported per cohort, layout,
model and alpha: held-out BSP MAE / CC (engine definition: active & ~visible), plus the mean displacement
||p_i - p_pi(i)|| in electrode spacings (median nearest-neighbour distance of each recording's own geometry).
  python baselines_analyses/layout_shift_v2.py [--ppe LOC3 P3S43 P3S44] [--index P3IDX42 P3IDX43 P3IDX44] [--n 60]
                                  [--alphas 0 5 10 20 40] [--layouts template kmeans] [--figs]
Out: baselines_analyses/generated/layout_shift_v2.json ; --figs: figures/fig_layout_shift_{internal,external}
     {,_km}_v2.png (seed-mean curves).  Runs whose checkpoint is missing are skipped.  Prints aggregates only."""
import os, sys, json, glob, argparse, warnings
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import numpy as np
import torch
from scipy.spatial import cKDTree
from paths import V2_GENERATED, V2_RUNS, FIG_DIR_V2, assert_v2
from build_sources import load_model, build_context
from channel_select import select_visible_channels
from dataset import collect_split

warnings.filterwarnings('ignore')
OUT = os.path.join(V2_GENERATED, 'layout_shift_v2.json')
VC = 16


def run_dir(pid):
    c = sorted(glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', f'{pid}__*')))
    return c[0] if c else None


def perm_from_rotation(coords, ok, alpha_deg):
    """Greedy bijection pi over the registered electrodes: slot i <- electrode nearest to R(alpha) p_i."""
    idx = np.where(ok)[0]
    P = coords[idx]
    axis = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    a = np.deg2rad(alpha_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    Q = (P - P.mean(0)) @ R.T + P.mean(0)
    D = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=-1)
    pi = np.full(len(idx), -1); used = np.zeros(len(idx), bool)
    for i in np.argsort(D.min(1)):                 # greedy, best matches first
        j = int(np.argmin(np.where(used, np.inf, D[i])))
        pi[i] = j; used[j] = True
    full = np.arange(len(coords)); full[idx] = idx[pi]
    disp = np.linalg.norm(coords[idx] - coords[idx[pi]], axis=-1)
    return full, float(disp.mean())


@torch.no_grad()
def recon_metrics(model, device, prep, coords, active, visible, g):
    x = torch.from_numpy(np.ascontiguousarray(prep / g)).unsqueeze(0).to(device)
    act = torch.from_numpy(np.ascontiguousarray(active)).unsqueeze(0).to(device)
    cd = torch.from_numpy(np.ascontiguousarray(coords)).unsqueeze(0).to(device)
    sel = torch.from_numpy(np.ascontiguousarray(visible)).unsqueeze(0).to(device)
    _, pred = model.forward_fixedn(x, act, cd, sel, w_spatial=0.0, w_temporal=0.0)
    pred = model.unpatchify(pred).squeeze(0).squeeze(0).float().cpu().numpy()
    held = active & ~visible
    tgt = prep / g
    mae = float(np.abs(pred[held] - tgt[held]).mean())
    ccs = [float(np.corrcoef(pred[c], tgt[c])[0, 1]) for c in np.where(held)[0]
           if pred[c].std() > 1e-8 and tgt[c].std() > 1e-8]
    return mae, float(np.mean(ccs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppe', nargs='*', default=['LOC3', 'P3S43', 'P3S44'])
    ap.add_argument('--index', nargs='*', default=['P3IDX42', 'P3IDX43', 'P3IDX44'])
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--alphas', type=float, nargs='*', default=[0, 5, 10, 20, 40])
    ap.add_argument('--layouts', nargs='*', default=['template', 'kmeans'])
    ap.add_argument('--figs', action='store_true', help='only draw the figures from the existing JSON')
    a = ap.parse_args()
    if a.figs:
        make_figs(); return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, keep_idx, args0 = {}, None, None
    for arm, pids in (('ppe', a.ppe), ('index', a.index)):
        for pid in pids:
            rd = run_dir(pid)
            if rd is None or not os.path.exists(os.path.join(rd, 'test_compare.json')):   # written at the end of training
                print(f'[skip] {pid}: no finished run'); continue
            m, ki, args = load_model(rd, device)
            exp = 'index' if arm == 'index' else 'ppe'
            assert args.get('pos_embed', 'ppe') == exp, (pid, args.get('pos_embed'))
            if keep_idx is None:
                keep_idx, args0 = ki, args
            assert np.array_equal(keep_idx, ki), pid
            models[(arm, pid)] = m
    print('models:', sorted(f'{arm}:{pid}' for arm, pid in models))
    buckets, _ = collect_split(seed=42)
    res = {'config': {'vc': VC, 'n': a.n, 'alphas': a.alphas, 'layouts': a.layouts,
                      'models': {arm: [p for (ar, p) in models if ar == arm] for arm in ('ppe', 'index')}}}
    for split in ('test_internal', 'test_external'):
        recs = [r for r in buckets[split] if r[2]]
        recs = [recs[i] for i in np.linspace(0, len(recs) - 1, a.n).astype(int)]
        acc = {(lay, key, al): [] for lay in a.layouts for key in models for al in a.alphas}
        disp_acc = {(lay, al): [] for lay in a.layouts for al in a.alphas}
        n_done = 0
        for k, rec in enumerate(recs):
            ctx = build_context(rec[1], rec[2], keep_idx, args0, vc=VC, mode='template', cluster_method='greedy', seed=42)
            if ctx is None:
                continue
            prep, coords = ctx['prep252'], ctx['coords']
            active = ctx['active252'].astype(bool)
            ok = np.abs(coords).sum(-1) > 1e-6
            spacing = float(np.median(cKDTree(coords[ok]).query(coords[ok], k=2)[0][:, 1]))
            vis_by_layout = {'template': ctx['visible252'].astype(bool)}
            if 'kmeans' in a.layouts:
                vis_by_layout['kmeans'] = select_visible_channels(coords, active, VC, 'cluster', 'kmean',
                                                                  np.random.default_rng(42)).astype(bool)
            perms = {al: perm_from_rotation(coords, ok & active, al) for al in a.alphas}
            for lay in a.layouts:
                vis = vis_by_layout[lay]
                g = float(torch.from_numpy(prep[vis]).std(dim=1).median()); g = g if g > 1e-8 else 1.0
                for al in a.alphas:
                    pi, disp = perms[al]
                    disp_acc[(lay, al)].append(disp / spacing)
                    prep_p, act_p, vis_p, coords_p = prep[pi], active[pi], vis[pi], coords[pi]
                    for key, m in models.items():
                        cd = coords_p if key[0] == 'ppe' else coords
                        acc[(lay, key, al)].append(recon_metrics(m, device, prep_p, cd, act_p, vis_p, g))
            n_done += 1
            if (k + 1) % 20 == 0:
                print(f'[{split}] {k+1}/{len(recs)}', flush=True)
        R = {'n_recordings': n_done}
        for lay in a.layouts:
            R[lay] = {'displacement_spacings': {str(al): float(np.mean(disp_acc[(lay, al)])) for al in a.alphas}}
            for key in models:
                arm, pid = key
                R[lay][f'{arm}|{pid}'] = {str(al): {'mae': float(np.mean([v[0] for v in acc[(lay, key, al)]])),
                                                    'cc': float(np.mean([v[1] for v in acc[(lay, key, al)]]))}
                                          for al in a.alphas}
            for arm in ('ppe', 'index'):
                pids = [p for (ar, p) in models if ar == arm]
                if pids:
                    R[lay][f'{arm}|mean'] = {str(al): {'cc': float(np.mean([R[lay][f'{arm}|{p}'][str(al)]['cc'] for p in pids])),
                                                       'mae': float(np.mean([R[lay][f'{arm}|{p}'][str(al)]['mae'] for p in pids])),
                                                       'cc_min': float(np.min([R[lay][f'{arm}|{p}'][str(al)]['cc'] for p in pids])),
                                                       'cc_max': float(np.max([R[lay][f'{arm}|{p}'][str(al)]['cc'] for p in pids])),
                                                       'n_seeds': len(pids)} for al in a.alphas}
        res[split] = R
    json.dump(res, open(assert_v2(OUT), 'w'), indent=1)
    for split in ('test_internal', 'test_external'):
        for lay in a.layouts:
            R = res[split][lay]
            print(f'--- {split} / {lay} (n={res[split]["n_recordings"]}); displacement (spacings):',
                  {k: round(v, 2) for k, v in R['displacement_spacings'].items()})
            for key in sorted(k for k in R if '|' in k):
                ccs = {al: round(R[key][str(al)]['cc'], 4) for al in a.alphas}
                flat = max(abs(R[key][str(al)]['cc'] - R[key][str(a.alphas[0])]['cc']) for al in a.alphas)
                print(f'   {key:14s} CC {ccs}  max|dCC| vs alpha0 = {flat:.5f}')
    print('wrote', OUT)


def make_figs():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = json.load(open(OUT))
    alphas = [str(x) for x in res['config']['alphas']]
    for lay, sfx in (('template', ''), ('kmeans', '_km')):
        for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
            if split not in res or lay not in res[split]:
                continue
            R = res[split][lay]
            x = [R['displacement_spacings'][al] for al in alphas]
            fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
            ns = R['ppe|mean'][alphas[0]]['n_seeds']
            ax.plot(x, [R['ppe|mean'][al]['cc'] for al in alphas], 's-', color='#0072B2', lw=2, ms=6,
                    label='Ours (physical positions)')
            ax.plot(x, [R['index|mean'][al]['cc'] for al in alphas], 'o--', color='#D55E00', lw=1.8, ms=6,
                    label='PosQ-MAE (index embedding)')
            ax.set_xlabel('coherent layout shift (electrode spacings)')
            ax.set_ylabel('BSP completion CC (held-out electrodes)')
            ax.set_title(f'{lab} dataset' + (' (per-recording $k$-means layout)' if lay == 'kmeans' else ''), fontsize=10)
            ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc='lower left')
            out = f'fig_layout_shift_{lab}{sfx}_v2.png'
            fig.savefig(assert_v2(os.path.join(FIG_DIR_V2, out)), dpi=150); plt.close(fig)
            print('wrote', out, f'(seed-mean over {ns} / {R["index|mean"][alphas[0]]["n_seeds"]} models)')


if __name__ == '__main__':
    main()
