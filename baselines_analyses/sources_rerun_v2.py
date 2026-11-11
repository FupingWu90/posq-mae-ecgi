"""Re-run the N-electrode sources of a trained run under modified evaluation conditions (reviewer requests,
2026-10-23) and score them against the cached references.  Per-recording rows are written to
generated/percase_rerun/<tag>_vc<N>_<split>_<layout>.csv; only aggregates are printed.

Variants (optional, combinable; the defaults reproduce the evaluation of the paper):
  --neumann_all   zero-flux (Neumann) rows at ALL torso nodes of the geometry for every inverse, while the
                  Dirichlet rows stay restricted to the active electrodes (reviewer point 3)
  --hp CUTOFF     high-pass cut-off in Hz of the preprocessing chain, applied to the body-surface field and to
                  the commercial reference potentials alike (paper: 1.0)
  --reref MODE    re-reference the visible channels: 'car' = their own common average, 'ch0' = the first
                  visible electrode (a sparse device cannot share the vest's reference; reviewer point 4c)
  --nosub         a missing template electrode is dropped instead of replaced by its nearest valid neighbour,
                  so the model receives N_eff <= N electrodes (reviewer point 4a)
  --qrs LO HI     QRS search window in ms relative to the R peak (paper: -50 80; reviewer point 7a)
  --rpeaks bsp    detect the beats once on the measured body-surface field and use the same R peaks for every
                  source and for the reference (paper: each source's own epicardial RMS; reviewer point 7c)
  --sources ...   subset of src1_vc (VC-only), src2_allvalid (full vest), src4_recon_valid (completion),
                  src3_recon_whole (completed field with ALL 252 channels entering the inverse)
  --visible_all   every valid, registered electrode is visible: the model completes only the missing channels, and
                  src3_recon_whole is then 'full vest + completion of the missing electrodes' (reviewer point 2c)
  --oracle_lam    for every source, choose the regularisation weight per recording on the 13-point grid that
                  minimises the AT error against the commercial reference (an upper bound; reviewer point 3)

Scoring follows inverse_evaluation/eval_downstream.py: AT on the reference's QRS-gated vertices, RT/ARI on its T-gated
vertices (>= 3 vertices), mean/median absolute error per recording; the commercial map is the primary reference and
the cached standard all-valid inverse the secondary one.  Extra columns: AT_bias (mean signed difference), AT_mae_sc
(MAE after removing the per-recording mean shift), edge fractions (gated vertices whose AT sits on a QRS-window
boundary) and, for the completion, the held-out BSP CC.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')   # BLAS threads per process (2026-10-23: parallel reruns oversubscribed the 32 cores)
import os, sys, argparse, warnings, csv, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V2, 'inverse_evaluation')); sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
import torch
from build_sources import load_model, build_context, reconstruct_sources, held_out_bsp_cc, CROP, _to256
from inverse import assemble_solver, solve_from_field, load_commercial, LAM_FLOOR, LAM_CEIL, LAM_N
from at_rt import extract_sample, good_vertex_mask, detect_beats
from metrics import map_errors, scale_ratio, temporal_cc
from cache_references import cache_lookup
from dataset import collect_split
from dedup import ok as dedup_ok
from paths import V2_GENERATED
import channel_select

LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}
SOURCES = ('src1_vc', 'src2_allvalid', 'src4_recon_valid', 'src3_recon_whole', 'src5_svd', 'src6_mdn')   # src5/src6: 2026-10-23 denoising variants


def _edge_fracs(AT, win, gate):
    """fraction of gated vertices whose activation time sits on the lower / upper boundary of the QRS window."""
    a = np.asarray(AT)[gate]
    if a.size == 0:
        return float('nan'), float('nan')
    qs, qe = int(win[0]), int(win[1])
    return float(np.mean(a <= qs + 1)), float(np.mean(a >= qe - 2))


def _score(AT_s, RT_s, AT_r, RT_r, qmask, tmask, win_s):
    row = {}
    row['AT_mae'], row['AT_mdae'], row['AT_cc'] = map_errors(AT_s, AT_r, qmask)
    if tmask is not None and int(np.asarray(tmask, bool).sum()) >= 3:
        row['RT_mae'], row['RT_mdae'], _ = map_errors(RT_s, RT_r, tmask)
        ari_s = np.asarray(RT_s, float) - np.asarray(AT_s, float); ari_r = np.asarray(RT_r, float) - np.asarray(AT_r, float)
        row['ARI_mae'], _, _ = map_errors(ari_s, ari_r, tmask)
    else:
        row['RT_mae'] = row['RT_mdae'] = row['ARI_mae'] = float('nan')
    d = (np.asarray(AT_s, float) - np.asarray(AT_r, float))[np.asarray(qmask, bool)]
    row['AT_bias'] = float(d.mean()) if d.size else float('nan')
    row['AT_mae_sc'] = float(np.abs(d - d.mean()).mean()) if d.size else float('nan')
    row['edge_lo_frac'], row['edge_hi_frac'] = _edge_fracs(AT_s, win_s, np.asarray(qmask, bool))
    return row


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True); ap.add_argument('--split', required=True)
    ap.add_argument('--vc', type=int, default=16); ap.add_argument('--layout', default='template', choices=sorted(LAYOUTS))
    ap.add_argument('--tag', required=True); ap.add_argument('--n', type=int, default=10 ** 9); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--neumann_all', action='store_true'); ap.add_argument('--hp', type=float, default=1.0)
    ap.add_argument('--reref', default='none', choices=['none', 'car', 'ch0']); ap.add_argument('--nosub', action='store_true')
    ap.add_argument('--qrs', type=float, nargs=2, default=[-50.0, 80.0]); ap.add_argument('--rpeaks', default='source', choices=['source', 'bsp', 'vis'])
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--visible_all', action='store_true'); ap.add_argument('--oracle_lam', action='store_true')
    ap.add_argument('--noise', nargs=2, default=None, metavar=('KIND', 'SNR_DB'), help='inject noise on the active channels: white|emg|bw and SNR in dB (point 2b)')
    ap.add_argument('--noise_seed', type=int, default=0)
    ap.add_argument('--fixed_lam_vc', type=float, default=None, help='VC-only inverse with this fixed regularisation weight (calibrated per N on the validation split, 2026-10-25)')
    ap.add_argument('--lam_grid_vc', action='store_true', help='VC-only inverse at every weight of the 13-point grid (one row per weight, column lam_grid=1; for the calibration)')
    ap.add_argument('--svd_k', type=int, default=0, help='rank-k SVD denoising of the full-vest field -> source src5_svd (point 2d)')
    ap.add_argument('--svd_input', action='store_true', help='with --svd_k: the visible channels given to the model (and the VC-only inverse) are the denoised ones')
    ap.add_argument('--model_denoise', nargs='?', const='cross', default=None, choices=['cross', 'qall'],
                    help="the completion model as a denoiser -> source src6_mdn: 'cross' = the valid electrodes in two random halves, each completed "
                         "from the other; 'qall' = all valid electrodes visible and every electrode position queried (model.complete_at)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    channel_select.TEMPLATE_SUBSTITUTE = not a.nosub
    default_markers = (a.hp == 1.0 and tuple(a.qrs) == (-50.0, 80.0) and a.rpeaks == 'source')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(a.run_dir, device)
    hold = set(l.strip() for l in open(args['holdout_file']) if l.strip()) if args.get('holdout_file') else None
    buckets, _ = collect_split(seed=args.get('seed', 42), holdout=hold)
    recs = [r for r in buckets[a.split] if r[2] and dedup_ok(r[0])][:a.n]
    mode, cm = LAYOUTS[a.layout]
    out_dir = os.path.join(V2_GENERATED, 'percase_rerun'); os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'{a.tag}_vc{a.vc}_{a.split}_{a.layout}.csv')
    need_model = ('src4_recon_valid' in a.sources) or ('src3_recon_whole' in a.sources) or a.svd_input
    noise = (a.noise[0], float(a.noise[1]), a.noise_seed) if a.noise else None
    rows, solver_cache, n_fail, t0 = [], {}, 0, time.time()
    print(f'[rerun] tag={a.tag} split={a.split} vc={a.vc} layout={a.layout} n={len(recs)} neumann_all={a.neumann_all} '
          f'hp={a.hp} reref={a.reref} nosub={a.nosub} qrs={a.qrs} rpeaks={a.rpeaks} noise={noise} svd_k={a.svd_k} svd_input={a.svd_input} '
          f'model_denoise={a.model_denoise} sources={a.sources}', flush=True)
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        sid_safe = sid.replace('/', '_')
        try:
            gdir = os.path.dirname(torsopath)
            if gdir not in solver_cache:
                if len(solver_cache) > 3:
                    solver_cache.clear()
                solver_cache[gdir] = assemble_solver(gdir)
            solver, tci, hV, hF = solver_cache[gdir]; n_h = hV.shape[0]
            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=a.vc, mode=mode, cluster_method=cm, seed=a.seed,
                                highpass_cutoff=a.hp, reref=a.reref, noise=noise)
            if ctx is None:
                n_fail += 1; continue
            if a.visible_all:   # all valid, registered electrodes visible; the model fills the missing channels only
                vis = ctx['active252'] & channel_select._coord_valid(ctx['coords'])
                ctx['visible252'] = vis; ctx['n_vc'] = int(vis.sum()); ctx['n_recon'] = int((ctx['active252'] & ~vis).sum())
                g = float(torch.from_numpy(ctx['prep252'][vis]).std(dim=1).median()); ctx['g'] = g if g > 1e-8 else 1.0
                ctx['src1_vc'] = (ctx['src1_vc'][0] * 0 + ctx['src2_allvalid'][0], ctx['src2_allvalid'][1])
            tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
            rp = cache_lookup(a.split, 'refs', sid_safe + '.npz')
            ref = dict(np.load(rp, allow_pickle=False)) if rp else {}
            # ---- commercial reference potentials (re-filtered when the cut-off changes) ----
            if a.hp == 1.0 and 'phicom' in ref:
                phicom = np.asarray(ref['phicom'], np.float64)
            else:
                phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw, highpass_cutoff=a.hp)
            if phicom is None:
                n_fail += 1; continue
            # ---- beats: per source (paper) or once on the measured body-surface field ----
            r_peaks = None
            if a.rpeaks in ('bsp', 'vis'):   # 'vis' (2026-10-25): beats detected on the N visible electrodes only, shared by every source
                meas = ctx['prep252_meas'][ctx['active252'] if a.rpeaks == 'bsp' else ctx['visible252']]
                r_peaks = detect_beats(meas.astype(np.float64))
                if r_peaks.size < 2:
                    n_fail += 1; continue
            ex = lambda phi: extract_sample(phi, r_peaks=r_peaks, qrs_offsets_ms=tuple(a.qrs)) if (phi is not None and np.all(np.isfinite(phi))) else None
            # ---- reference markers and gates ----
            refs = {}; refphi = {'commercial': phicom, 'allvalid': (np.asarray(ref['phi2'], np.float64) if 'phi2' in ref else None)}
            if default_markers and all(k in ref for k in ('ATcom', 'RTcom', 'qrs_goodcom', 't_goodcom')):
                refs['commercial'] = (ref['ATcom'], ref['RTcom'], ref['qrs_goodcom'].astype(bool), ref['t_goodcom'].astype(bool), (70, 200))
            else:
                rc = ex(phicom)
                if rc is None:
                    n_fail += 1; continue
                g = good_vertex_mask(rc['sigavg'], *rc['windows'])
                refs['commercial'] = (rc['AT'], rc['RT'], g['qrs'].astype(bool), g['t'].astype(bool), rc['windows'][:2])
            if a.hp == 1.0 and default_markers and all(k in ref for k in ('AT2', 'RT2', 'qrs_good2', 't_good2')):
                refs['allvalid'] = (ref['AT2'], ref['RT2'], ref['qrs_good2'].astype(bool), ref['t_good2'].astype(bool), (70, 200))
            elif a.hp == 1.0 and 'phi2' in ref:
                r2 = ex(np.asarray(ref['phi2'], np.float64))
                if r2 is not None:
                    g = good_vertex_mask(r2['sigavg'], *r2['windows'])
                    refs['allvalid'] = (r2['AT'], r2['RT'], g['qrs'].astype(bool), g['t'].astype(bool), r2['windows'][:2])
            # ---- sources ----
            fields = {'src1_vc': ctx['src1_vc'], 'src2_allvalid': ctx['src2_allvalid']}
            if a.svd_k > 0:   # classical spatial low-rank denoising of the full-vest field (reviewer point 2d)
                act = ctx['active252']; X = ctx['prep252'][act].astype(np.float64)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                k = min(int(a.svd_k), len(S))
                f252 = ctx['prep252'].copy(); f252[act] = ((U[:, :k] * S[:k]) @ Vt[:k]).astype(np.float32)
                fields['src5_svd'] = _to256(f252, act, keep_idx)
                if a.svd_input:   # denoise, then complete: the model (and the VC-only inverse) see the denoised visible channels
                    ctx['prep252'] = f252
                    g = float(torch.from_numpy(f252[ctx['visible252']]).std(dim=1).median()); ctx['g'] = g if g > 1e-8 else 1.0
                    fields['src1_vc'] = _to256(f252, ctx['visible252'], keep_idx); ctx['src1_vc'] = fields['src1_vc']
            bsp_cc = float('nan')
            if need_model:
                with torch.no_grad():
                    recon = reconstruct_sources(model, ctx, device)
                fields['src4_recon_valid'] = recon['src4_recon_valid']; fields['src3_recon_whole'] = recon['src3_recon_whole']
                bsp_cc, _ = held_out_bsp_cc(ctx, recon['recon252'])
            if a.model_denoise == 'qall':
                # the completion model as a denoiser by QUERYING EVERY POSITION with all valid registered electrodes visible
                # (model.complete_at: the decoder can be queried at any position, including the visible ones, 2026-10-23)
                vis_q = ctx['active252'] & channel_select._coord_valid(ctx['coords'])
                g_q = float(torch.from_numpy(ctx['prep252'][vis_q]).std(dim=1).median()); g_q = g_q if g_q > 1e-8 else 1.0
                xq = torch.from_numpy(ctx['prep252'] / g_q).unsqueeze(0).to(device); cq = torch.from_numpy(ctx['coords']).unsqueeze(0).to(device)
                with torch.no_grad():
                    pq = model.complete_at(xq, cq, torch.from_numpy(vis_q).unsqueeze(0).to(device), cq)
                f252 = ctx['prep252'].copy(); f252[vis_q] = (pq.squeeze(0).cpu().numpy() * g_q)[vis_q].astype(np.float32)
                fields['src6_mdn'] = _to256(f252, ctx['active252'], keep_idx)
            if a.model_denoise == 'cross':
                # the completion model as a denoiser by CROSS-COMPLETION (2026-10-23, reviewer point 2d): the valid registered
                # electrodes are split into two random halves; each half is completed from the other, so every active channel
                # receives the model's re-synthesis from ~half of the vest (the model does not predict at visible positions).
                import zlib
                rng_h = np.random.default_rng(a.seed * 1000003 + zlib.crc32(os.path.basename(ecgpath).encode()))
                reg_act = np.where(ctx['active252'] & channel_select._coord_valid(ctx['coords']))[0]
                perm = rng_h.permutation(len(reg_act)); halves = (reg_act[perm[: len(perm) // 2]], reg_act[perm[len(perm) // 2:]])
                f252 = ctx['prep252'].copy()
                for h_vis, h_hid in ((halves[0], halves[1]), (halves[1], halves[0])):
                    c = dict(ctx); vis_h = np.zeros(len(ctx['active252']), bool); vis_h[h_vis] = True
                    c['visible252'] = vis_h; c['n_vc'] = int(vis_h.sum())
                    g_h = float(torch.from_numpy(c['prep252'][vis_h]).std(dim=1).median()); c['g'] = g_h if g_h > 1e-8 else 1.0
                    with torch.no_grad():
                        r_h = reconstruct_sources(model, c, device)
                    f252[h_hid] = r_h['recon252'][h_hid].astype(np.float32)
                fields['src6_mdn'] = _to256(f252, ctx['active252'], keep_idx)
            for s in [s for s in a.sources if s in fields]:
                if a.oracle_lam:   # per-recording oracle regularisation weight (minimum AT error vs the commercial reference)
                    AT_c, _, qm_c, _, _ = refs['commercial']; best = None
                    for lam_i in np.logspace(np.log10(LAM_FLOOR), np.log10(LAM_CEIL), LAM_N):
                        phi_i, lam_i = solve_from_field(solver, tci, *fields[s], lam=float(lam_i), neumann_all=a.neumann_all)[:2]
                        r_i = ex(phi_i)
                        if r_i is None:
                            continue
                        e_i = map_errors(r_i['AT'], AT_c, qm_c)[0]
                        if best is None or e_i < best[0]:
                            best = (e_i, phi_i, lam_i, r_i)
                    if best is None:
                        phi, lam, r = None, float('nan'), None
                    else:
                        _, phi, lam, r = best
                elif a.lam_grid_vc and s == 'src1_vc':   # calibration mode: every grid weight, one row per weight
                    cands = []
                    for lam_i in np.logspace(np.log10(LAM_FLOOR), np.log10(LAM_CEIL), LAM_N):
                        phi_i, lam_i = solve_from_field(solver, tci, *fields[s], lam=float(lam_i), neumann_all=a.neumann_all)[:2]
                        cands.append((phi_i, lam_i, ex(phi_i)))
                    phi, lam, r = cands[0]
                elif a.fixed_lam_vc is not None and s == 'src1_vc':
                    phi, lam = solve_from_field(solver, tci, *fields[s], lam=float(a.fixed_lam_vc), neumann_all=a.neumann_all)[:2]
                    r = ex(phi)
                else:
                    phi, lam = solve_from_field(solver, tci, *fields[s], neumann_all=a.neumann_all)[:2]
                    r = ex(phi)
                cand_list = cands if (a.lam_grid_vc and s == 'src1_vc') else [(phi, lam, r)]
                for phi, lam, r in cand_list:
                    base = dict(sid=sid, source=s, n_active=ctx['n_active'], n_vc=ctx['n_vc'], n_template_sub=ctx.get('n_template_sub', 0),
                                lam=lam, rr=(r['rr'] if r else None), success=int(r is not None), lam_grid=int(a.lam_grid_vc and s == 'src1_vc'))
                    if s == 'src4_recon_valid':
                        base['bsp_cc'] = bsp_cc
                    for rname, (AT_r, RT_r, qm, tm, win_r) in refs.items():
                        row = dict(base, ref=rname)
                        if r is not None:
                            row.update(_score(r['AT'], r['RT'], AT_r, RT_r, qm, tm, r['windows'][:2]))
                            # absolute marker means (gated) and amplitude/potential agreement (2026-10-23: filter and boundary-row analyses)
                            row['AT_mean_src'] = float(np.mean(np.asarray(r['AT'], float)[np.asarray(qm, bool)])); row['AT_mean_ref'] = float(np.mean(np.asarray(AT_r, float)[np.asarray(qm, bool)]))
                            if tm is not None and int(np.asarray(tm, bool).sum()) >= 3:
                                row['RT_mean_src'] = float(np.mean(np.asarray(r['RT'], float)[np.asarray(tm, bool)])); row['RT_mean_ref'] = float(np.mean(np.asarray(RT_r, float)[np.asarray(tm, bool)]))
                            pr = refphi.get(rname)
                            if pr is not None and phi is not None and pr.shape == phi.shape:
                                row['scale_ratio'] = float(scale_ratio(phi, pr)); row['pot_cc'] = float(temporal_cc(phi, pr)[0])
                            row['ref_edge_lo_frac'], row['ref_edge_hi_frac'] = _edge_fracs(AT_r, win_r, qm)
                            row['ref_qrs_frac'] = float(np.mean(qm)); row['ref_t_frac'] = float(np.mean(tm))
                        rows.append(row)
        except Exception as e:
            n_fail += 1; print(f'  [{i + 1}] FAILED: {type(e).__name__}: {str(e)[:80]}', flush=True)
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(recs)} done ({time.time() - t0:.0f} s, {n_fail} failed)', flush=True)
    keys = ['sid', 'source', 'ref', 'n_active', 'n_vc', 'n_template_sub', 'lam', 'lam_grid', 'rr', 'success', 'bsp_cc', 'AT_mae', 'AT_mdae', 'AT_cc',
            'RT_mae', 'RT_mdae', 'ARI_mae', 'AT_bias', 'AT_mae_sc', 'edge_lo_frac', 'edge_hi_frac', 'ref_edge_lo_frac', 'ref_edge_hi_frac',
            'ref_qrs_frac', 'ref_t_frac', 'AT_mean_src', 'AT_mean_ref', 'RT_mean_src', 'RT_mean_ref', 'scale_ratio', 'pot_cc']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)
    # ---- aggregates only ----
    for s in a.sources:
        for rname in ('commercial', 'allvalid'):
            sub = [r for r in rows if r['source'] == s and r['ref'] == rname and r.get('success')]
            if not sub:
                continue
            m = lambda k: np.nanmean([r[k] for r in sub if r.get(k) is not None])
            print(f'  {s:16s} vs {rname:10s} n={len(sub):3d}  AT {m("AT_mae"):6.2f}  RT {m("RT_mae"):6.2f}  ARI {m("ARI_mae"):6.2f}  '
                  f'AT_sc {m("AT_mae_sc"):6.2f}  bias {m("AT_bias"):+5.2f}  edge lo/hi {100 * m("edge_lo_frac"):4.1f}/{100 * m("edge_hi_frac"):4.1f}%  '
                  f'ref edge {100 * m("ref_edge_lo_frac"):4.1f}/{100 * m("ref_edge_hi_frac"):4.1f}%  n_vc {m("n_vc"):5.1f}', flush=True)
    print(f'[rerun] wrote {out_csv} ({len(rows)} rows; {n_fail} failed; {time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
