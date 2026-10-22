"""
E0: ground-truth evaluation on the SIMULATED test cohort (simulation/make_dataset.py,
--split test --store_phi).  Every simulated test recording goes through the
clinical pipeline unchanged (vendor reader -> missing detector -> preprocessing
-> VC selection -> model reconstruction -> MFS inverse on the VENDOR geometry),
and the epicardial AT/RT of every source are scored against the ground truth:

  * 'gt'      : Wyatt markers extracted from the TRUE epicardial potential
                (same extract_sample / gating as every other source -> no
                marker-definition error; the gate comes from the GT field)
  * 'gt_par'  : the simulation parameter maps AT(v), RT(v) themselves
                (all epicardial vertices, no gate)

Sources: src1_vc (raw N electrodes), src2_allvalid (all valid electrodes),
src4_recon_valid (Ours: model reconstruction), plus the model's BSP CC against
the CLEAN simulated field (bsp_cc_clean) and against the noisy written signal
(bsp_cc).  Output: one CSV row per (recording, source).

  python eval_sim_gt.py --run_dir <run> --n 1000000 --vc 16 --out ../baselines_analyses/generated/sim_gt/<tag>.csv
"""
import os, sys, csv, json, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)
import torch
from build_sources import load_model, build_context, reconstruct_sources, CROP, held_out_bsp_cc
from inverse import assemble_solver, solve_from_field
from metrics import temporal_cc, scale_ratio, map_errors, hsp_metrics
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split

COMPARE = ['src1_vc', 'src2_allvalid', 'src4_recon_valid']


def _atrt(phi):
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    try:
        return extract_sample(phi)
    except Exception:
        return None


def gt_epicardial(gt, L):
    """Tile the stored GT beat at the stored beat starts -> (n_epi, L) potential
    in the same time base as the written recording."""
    phi_b = gt['phi_beat'].astype(np.float64); T = phi_b.shape[1]
    out = np.zeros((phi_b.shape[0], L))
    for s in gt['beat_starts']:
        if s + T <= L:
            out[:, s:s + T] += phi_b
    return out


def evaluate(run_dir, n=20, vc=16, mode='cluster', cluster_method='kmean', seed=42,
             out_csv=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(run_dir, device)
    buckets, _ = collect_split(seed=args.get('seed', 42))
    recs = buckets['sim_test'][:n]
    if verbose:
        print(f"model={os.path.basename(run_dir)}  sim_test n={len(recs)}  vc={vc} {mode}/{cluster_method}")
    solver_cache, rows = {}, []
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        try:
            gtp = ecgpath.replace('.ecgdata', '.gt.npz')
            with np.load(gtp, allow_pickle=False) as z:
                gt = {k: z[k] for k in z.files}
            if 'phi_beat' not in gt:
                if verbose: print(f"  [{i+1}] {sid}: no phi_beat (generate the test split with --store_phi)")
                continue
            geom_dir = os.path.dirname(torsopath)
            if geom_dir not in solver_cache:
                solver_cache[geom_dir] = assemble_solver(geom_dir)
            solver, torso_ch_idx, heart_V, heart_F = solver_cache[geom_dir]
            n_h = heart_V.shape[0]
            assert n_h == int(gt['n_epi']), (n_h, gt['n_epi'])
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=vc, mode=mode,
                                cluster_method=cluster_method, seed=seed)
            if ctx is None:
                continue
            s0 = ctx['crop_start']; L = ctx['L']
            # ---- ground truth in the crop window ----
            PHI = gt_epicardial(gt, L)[:, s0:s0 + CROP]
            a_gt = _atrt(PHI)
            if a_gt is None:
                continue
            gate = good_vertex_mask(a_gt['sigavg'], *a_gt['windows'])
            # parameter maps -> same time base as the markers: relative timing only
            par_AT = gt['AT'].astype(float); par_RT = gt['RT'].astype(float)
            # ---- sources ----
            phi = {'src1_vc': solve_from_field(solver, torso_ch_idx, *ctx['src1_vc'])[0],
                   'src2_allvalid': solve_from_field(solver, torso_ch_idx, *ctx['src2_allvalid'])[0]}
            recon = reconstruct_sources(model, ctx, device)
            phi['src4_recon_valid'] = solve_from_field(solver, torso_ch_idx, *recon['src4_recon_valid'])[0]
            bsp_cc, _ = held_out_bsp_cc(ctx, recon['recon252'])
            # BSP CC against the CLEAN field on the held-out electrodes
            clean = gt['bsp_clean'].astype(np.float32)[:, s0:s0 + CROP]
            held = ctx['active252'] & ~ctx['visible252']
            ccs = []
            for c in np.where(held)[0]:
                a_, b_ = recon['recon252'][c].astype(np.float64), clean[c].astype(np.float64)
                if a_.std() > 1e-8 and b_.std() > 1e-8:
                    ccs.append(float(np.corrcoef(a_, b_)[0, 1]))
            bsp_cc_clean = float(np.mean(ccs)) if ccs else float('nan')
            meta = json.loads(str(gt['meta']))
            for sname in COMPARE:
                ps = phi.get(sname)
                row = dict(sid=sid, source=sname, kind=meta.get('kind'), n_active=ctx['n_active'],
                           n_vc=ctx['n_vc'], n_recon=ctx['n_recon'], n_h=int(n_h),
                           gt_qrs_frac=float(gate['qrs'].mean()), gt_t_frac=float(gate['t'].mean()),
                           at_range=meta.get('at_range'), scar=int(len(meta.get('scar', []))))
                if sname == 'src4_recon_valid':
                    row['bsp_cc'] = bsp_cc; row['bsp_cc_clean'] = bsp_cc_clean
                ok = ps is not None and np.all(np.isfinite(ps))
                row['success'] = int(ok)
                if ok:
                    row['pot_cc'], _ = temporal_cc(ps, PHI)
                    row['scale_ratio'] = scale_ratio(ps, PHI)
                    h = hsp_metrics(ps, PHI)
                    row['hsp_mae'], row['hsp_cc'], row['hsp_cc_md'] = h['mae'], h['cc'], h['cc_md']
                    a_s = _atrt(ps)
                    if a_s is not None:
                        # vs Wyatt markers of the TRUE potential, GT-gated (same rule as the paper)
                        row['AT_mae'], row['AT_mdae'], row['AT_cc'] = map_errors(a_s['AT'], a_gt['AT'], gate['qrs'])
                        row['RT_mae'], row['RT_mdae'], row['RT_cc'] = map_errors(a_s['RT'], a_gt['RT'], gate['t'])
                        ari_s = np.asarray(a_s['RT'], float) - np.asarray(a_s['AT'], float)
                        ari_g = np.asarray(a_gt['RT'], float) - np.asarray(a_gt['AT'], float)
                        row['ARI_mae'], row['ARI_mdae'], row['ARI_cc'] = map_errors(ari_s, ari_g, gate['t'])
                        # vs the PARAMETER maps (all vertices): remove the common time offset
                        # (markers are relative to the detected R peak) by matching the means
                        d_at = np.asarray(a_s['AT'], float) - par_AT; d_rt = np.asarray(a_s['RT'], float) - par_RT
                        row['AT_mae_par'] = float(np.mean(np.abs(d_at - np.median(d_at))))
                        row['RT_mae_par'] = float(np.mean(np.abs(d_rt - np.median(d_rt))))
                        row['AT_cc_par'] = float(np.corrcoef(a_s['AT'], par_AT)[0, 1])
                        row['RT_cc_par'] = float(np.corrcoef(a_s['RT'], par_RT)[0, 1])
                rows.append(row)
            if verbose:
                r4 = rows[-1]
                print(f"  [{i+1}/{len(recs)}] {sid[:40]:40s} {meta.get('kind'):6s} n_active={ctx['n_active']:3d} "
                      f"Ours AT {r4.get('AT_mae', float('nan')):5.2f} RT {r4.get('RT_mae', float('nan')):5.2f} "
                      f"bspCC {r4.get('bsp_cc', float('nan')):.3f}/{r4.get('bsp_cc_clean', float('nan')):.3f}")
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(recs)}] {sid[:40]} FAILED: {type(e).__name__}: {e}")
    _summarize(rows)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('sid', 'source', 'kind'), k))
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in keys})
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def _summarize(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r['source']].append(r)
    print("\n=== SIM GROUND-TRUTH SUMMARY (mean over recordings) ===")
    print("source             n   HSP-CC  AT_mae  RT_mae  ARI_mae | AT_par RT_par | bspCC bspCC_clean")
    for s in COMPARE:
        rs = [r for r in by[s] if 'AT_mae' in r]
        if not rs:
            continue
        m = lambda k: np.nanmean([r.get(k, np.nan) for r in rs])
        print(f"{s:18s} {len(rs):3d}   {m('hsp_cc'):.3f}  {m('AT_mae'):6.2f}  {m('RT_mae'):6.2f}  {m('ARI_mae'):6.2f} | "
              f"{m('AT_mae_par'):6.2f} {m('RT_mae_par'):6.2f} | {m('bsp_cc'):.3f} {m('bsp_cc_clean'):.3f}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--vc', type=int, default=16)
    p.add_argument('--mode', default='cluster')
    p.add_argument('--cluster_method', default='kmean')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    evaluate(a.run_dir, n=a.n, vc=a.vc, mode=a.mode, cluster_method=a.cluster_method,
             seed=a.seed, out_csv=a.out)
