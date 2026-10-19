"""
Downstream HSP evaluation for a trained FixdElectrodsN_Recon model.

Loads the MODEL-INDEPENDENT references from cache_hsp/ (source (1) VC-only,
(2) all-valid, commercial .potential — precomputed by cache_references.py), then
computes ONLY the model-dependent sources (3) recon-whole and (4) recon-valid,
solves their inverse, and scores sources (1)/(3)/(4) against references
(2) and commercial on: potential temporal CC, scale ratio, AT & RT map MAE (ms)
+ spatial CC. If a recording is not cached, its references are computed on the
fly (slower).

  python eval_downstream.py --run_dir <trained run> [--split ...] [--n N] [--out csv]
                            [--region all|front|back]

--region restricts the deployment VC selection to one torso half (see
channel_select.region_mask); the src1 cache tag then carries `_reg-<region>`.
Per-case rows also carry `region` and, for the src4 rows, `bsp_cc` = mean
temporal Pearson CC of the model reconstruction vs the measured signal on the
held-out (active, registered, not visible) electrodes.
"""
import os, sys, csv, argparse, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

import torch
from build_sources import (load_model, build_context, reconstruct_sources, CROP,
                           held_out_bsp_cc)
from inverse import assemble_solver, solve_from_field, load_commercial
from metrics import temporal_cc, scale_ratio, map_mae_cc, map_errors, hsp_metrics
from at_rt import extract_sample, good_vertex_mask
from cache_references import CACHE_ROOT, CACHE_ROOT_V1, vc_tag, cache_lookup
from dataset import collect_split

COMPARE = ['src1_vc', 'src3_recon_whole', 'src4_recon_valid']
REFS = ['src2_allvalid', 'commercial']


def _atrt(phi):
    if phi is None or not np.all(np.isfinite(phi)):
        return None
    try:
        return extract_sample(phi)
    except Exception:
        return None


def _load_npz(path):
    if path is None or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def evaluate(run_dir, split='test_internal', n=20, vc=16, mode='cluster',
             cluster_method='greedy', seed=42, out_csv=None, verbose=True,
             region='all', template_npz=None):
    """template_npz (v2): alternative fixed layout installed for mode='template'; the src1 cache tag then
    carries '_tpl-<file stem>' so that no VC-only solution of the main template is reused."""
    tag_suffix = ''
    if template_npz:
        import channel_select as _CS
        with np.load(template_npz) as z:
            _CS._TEMPLATES[int(vc)] = (z['idx'].astype(np.int64), z['mean_xyz'].astype(np.float32))
        tag_suffix = '_tpl-' + os.path.splitext(os.path.basename(template_npz))[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, keep_idx, args = load_model(run_dir, device)
    buckets, _ = collect_split(seed=args.get('seed', 42))
    recs = buckets[split][:n]
    tag = vc_tag(vc, mode, cluster_method, seed, region) + tag_suffix
    refs_dir = os.path.join(CACHE_ROOT, split, 'refs')
    vc_dir = os.path.join(CACHE_ROOT, split, tag)
    if verbose:
        print(f"model={os.path.basename(run_dir)}  split={split}  n={len(recs)}  vc={vc} "
              f"{mode}/{cluster_method}  region={region}  tag={tag}  "
              f"cache={'yes' if (os.path.isdir(vc_dir) or os.path.isdir(os.path.join(CACHE_ROOT_V1, split, tag))) else 'no'}")
    if (args.get('select_region') or 'all') != region:
        print(f"[warn] model was trained with select_region="
              f"{args.get('select_region') or 'all'} but is deployed with region={region}")

    solver_cache = {}
    rows = []
    for i, (sid, ecgpath, torsopath) in enumerate(recs):
        if torsopath is None:
            continue
        sid_safe = sid.replace('/', '_')
        try:
            geom_dir = os.path.dirname(torsopath)
            if geom_dir not in solver_cache:
                solver_cache[geom_dir] = assemble_solver(geom_dir)
            solver, torso_ch_idx, heart_V, heart_F = solver_cache[geom_dir]
            n_h = heart_V.shape[0]

            ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=vc,
                                mode=mode, cluster_method=cluster_method, seed=seed,
                                region=region)
            if ctx is None:
                continue
            tw = (ctx['crop_start'], ctx['crop_start'] + CROP)

            # ---- references (from cache, else compute) ----
            ref = _load_npz(cache_lookup(split, 'refs', sid_safe + '.npz'))
            c1 = _load_npz(cache_lookup(split, tag, sid_safe + '.npz'))
            if ref is None:
                phi2 = solve_from_field(solver, torso_ch_idx, *ctx['src2_allvalid'])[0]
                phicom0 = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
                a2, acom = _atrt(phi2), _atrt(phicom0)
                ref = _pack_ref(phi2, a2, phicom0, acom)
            # commercial HSP is not cached (it is the .potential on disk) -> re-read
            # for potential-CC; commercial AT/RT come from the cache.
            phicom = _f(ref.get('phicom'))
            if phicom is None:
                phicom = load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw)
            phi = {'src2_allvalid': _f(ref.get('phi2')), 'commercial': phicom}
            at = {'src2_allvalid': _unpack_at(ref, '2'), 'commercial': _unpack_at(ref, 'com')}
            good = {'src2_allvalid': {'qrs': ref.get('qrs_good2'), 't': ref.get('t_good2')},
                    'commercial': {'qrs': ref.get('qrs_goodcom'), 't': ref.get('t_goodcom')}}

            if c1 is None:
                phi['src1_vc'] = solve_from_field(solver, torso_ch_idx, *ctx['src1_vc'])[0]
            else:
                phi['src1_vc'] = _f(c1.get('phi1'))

            # ---- model-dependent sources (always computed) ----
            recon = reconstruct_sources(model, ctx, device)
            for name in ('src3_recon_whole', 'src4_recon_valid'):
                phi[name] = solve_from_field(solver, torso_ch_idx, *recon[name])[0]
            # BSP CC on the held-out electrodes (active & registered & ~visible)
            bsp_cc, _n_bsp = held_out_bsp_cc(ctx, recon['recon252'])

            # AT/RT and each source's OWN good-vertex mask (mask reported only, never
            # used to gate the comparison -> the eval vertex set stays source-independent)
            good_src = {}
            for name in COMPARE:
                a = _atrt(phi.get(name))
                at[name] = {'AT': a['AT'], 'RT': a['RT']} if a else None
                good_src[name] = good_vertex_mask(a['sigavg'], *a['windows']) if a else None

            # ---- metrics ----
            for s in COMPARE:
                for r in REFS:
                    row = dict(sid=sid, source=s, ref=r, n_active=ctx['n_active'],
                               n_vc=ctx['n_vc'], n_recon=ctx['n_recon'],
                               region=region,
                               n_region_fallback=ctx.get('n_region_fallback', 0),
                               n_template_sub=ctx.get('n_template_sub', 0))
                    if s == 'src4_recon_valid' and np.isfinite(bsp_cc):
                        row['bsp_cc'] = bsp_cc
                    ps, pr = phi.get(s), phi.get(r)
                    ok = ps is not None and pr is not None and np.all(np.isfinite(ps)) and np.all(np.isfinite(pr))
                    row['success'] = int(ps is not None and np.all(np.isfinite(ps)))
                    if ok:
                        row['pot_cc'], _ = temporal_cc(ps, pr)
                        row['scale_ratio'] = scale_ratio(ps, pr)
                        # --- HSP potential MAE/RMSE/CC (per-vertex temporal -> spatial mean & median) ---
                        h = hsp_metrics(ps, pr)
                        row['hsp_mae'], row['hsp_mae_md'] = h['mae'], h['mae_md']
                        row['hsp_rmse'], row['hsp_rmse_md'] = h['rmse'], h['rmse_md']
                        row['hsp_cc'], row['hsp_cc_md'] = h['cc'], h['cc_md']
                        a_s, a_r = at.get(s), at.get(r); gd = good.get(r, {})
                        if a_s is not None and a_r is not None:
                            qmask, tmask = gd.get('qrs'), gd.get('t')
                            # (i) REFERENCE-GATED set (source-independent: same vertices for
                            #     every source vs this reference). spatial-MEAN(_mae)/MEDIAN(_mdae)/CC
                            row['AT_mae'], row['AT_mdae'], row['AT_cc'] = \
                                map_errors(a_s['AT'], a_r['AT'], qmask)
                            row['RT_mae'], row['RT_mdae'], row['RT_cc'] = \
                                map_errors(a_s['RT'], a_r['RT'], tmask)         # tightened T-gate
                            row['RT_mae_all'], row['RT_mdae_all'], row['RT_cc_all'] = \
                                map_errors(a_s['RT'], a_r['RT'], qmask)         # un-T-gated (QRS-good)
                            ari_s = np.asarray(a_s['RT'], float) - np.asarray(a_s['AT'], float)
                            ari_r = np.asarray(a_r['RT'], float) - np.asarray(a_r['AT'], float)
                            row['ARI_mae'], row['ARI_mdae'], row['ARI_cc'] = \
                                map_errors(ari_s, ari_r, tmask)
                            # (ii) ALL-VERTICES set (no exclusion) -> the unarguable fair comparison
                            row['AT_mae_allv'], row['AT_mdae_allv'], _ = map_errors(a_s['AT'], a_r['AT'], None)
                            row['RT_mae_allv'], row['RT_mdae_allv'], _ = map_errors(a_s['RT'], a_r['RT'], None)
                            row['ARI_mae_allv'], row['ARI_mdae_allv'], _ = map_errors(ari_s, ari_r, None)
                            # (iii) coverage: reference-gate retained fraction (same for all sources)
                            #       + THIS source's own valid-beat fraction (reported, not gated)
                            row['n_h'] = int(n_h)
                            row['ref_qrs_frac'] = float(qmask.mean()) if qmask is not None else None
                            row['ref_t_frac'] = float(tmask.mean()) if tmask is not None else None
                            gs = good_src.get(s)
                            row['src_qrs_frac'] = float(gs['qrs'].mean()) if gs is not None else None
                            row['src_t_frac'] = float(gs['t'].mean()) if gs is not None else None
                    rows.append(row)
            if verbose:
                print(f"  [{i+1}/{len(recs)}] {sid_safe[:44]:44s} n_active={ctx['n_active']:3d} done"
                      + (f"  region_fallback={ctx['n_region_fallback']}"
                         if ctx.get('n_region_fallback', 0) else ""))
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(recs)}] {sid[:40]} FAILED: {type(e).__name__}: {e}")

    _summarize(rows)
    if out_csv:
        _write_csv(rows, out_csv); print(f"\nwrote {out_csv} ({len(rows)} rows)")
    return rows


# ---- cache (un)packing helpers ----
def _f(x):
    return None if x is None else np.asarray(x, np.float64)


def _unpack_at(d, suf):
    k = f'AT{suf}'
    if d is None or k not in d:
        return None
    return {'AT': d[k], 'RT': d[f'RT{suf}']}


def _pack_ref(phi2, a2, phicom, acom):
    d = {}
    if phi2 is not None:
        d['phi2'] = phi2
        if a2:
            gm = good_vertex_mask(a2['sigavg'], *a2['windows'])
            d.update(AT2=a2['AT'], RT2=a2['RT'], qrs_good2=gm['qrs'], t_good2=gm['t'])
    if phicom is not None:
        d['phicom'] = phicom
        if acom:
            gm = good_vertex_mask(acom['sigavg'], *acom['windows'])
            d.update(ATcom=acom['AT'], RTcom=acom['RT'], qrs_goodcom=gm['qrs'], t_goodcom=gm['t'])
    return d


def _summarize(rows):
    def agg(vals):
        v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
        return np.mean(v) if v else float('nan')
    print("\n=== SUMMARY (mean over recordings; lower MAE, higher CC better) ===")
    print("  RT_mae = tightened T-wave gate (real-T vertices);  RT_mae_all = all QRS-good vertices")
    print(f"{'source':18s} {'ref':14s} {'succ':>5s} {'pot_cc':>7s} {'scale':>6s} "
          f"{'AT_mae':>7s} {'RT_mae':>7s} {'RT_mae_all':>10s}  n")
    for s in COMPARE:
        for r in REFS:
            sub = [x for x in rows if x['source'] == s and x['ref'] == r]
            if not sub:
                continue
            print(f"{s:18s} {r:14s} {np.mean([x.get('success',0) for x in sub]):5.2f} "
                  f"{agg([x.get('pot_cc') for x in sub]):7.3f} {agg([x.get('scale_ratio') for x in sub]):6.2f} "
                  f"{agg([x.get('AT_mae') for x in sub]):7.2f} {agg([x.get('RT_mae') for x in sub]):7.2f} "
                  f"{agg([x.get('RT_mae_all') for x in sub]):10.2f}  {len(sub)}")


def _write_csv(rows, path):
    keys = ['sid', 'source', 'ref', 'n_active', 'n_vc', 'n_recon', 'success',
            'pot_cc', 'scale_ratio',
            # HSP potential accuracy (per-vertex temporal, spatial mean/median)
            'hsp_mae', 'hsp_mae_md', 'hsp_rmse', 'hsp_rmse_md', 'hsp_cc', 'hsp_cc_md',
            # AT/RT/ARI on the REFERENCE-gated vertex set (spatial mean/median + CC)
            'AT_mae', 'AT_mdae', 'AT_cc',
            'RT_mae', 'RT_mdae', 'RT_cc',
            'RT_mae_all', 'RT_mdae_all', 'RT_cc_all',
            'ARI_mae', 'ARI_mdae', 'ARI_cc',
            # AT/RT/ARI on ALL vertices (no exclusion)
            'AT_mae_allv', 'AT_mdae_allv',
            'RT_mae_allv', 'RT_mdae_allv',
            'ARI_mae_allv', 'ARI_mdae_allv',
            # coverage: reference-gate retained fraction + this source's own valid fraction
            'n_h', 'ref_qrs_frac', 'ref_t_frac', 'src_qrs_frac', 'src_t_frac',
            # deployment region of the VC selection + held-out BSP CC (src4 rows)
            'region', 'n_region_fallback', 'bsp_cc',
            # v2: number of substituted electrodes of the fixed template layout
            'n_template_sub']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--split', default='test_internal',
                   choices=['test_internal', 'test_external', 'val'])
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--vc', type=int, default=16)
    p.add_argument('--mode', default='cluster', help="cluster | patch | random | template (v2 fixed layout)")
    p.add_argument('--cluster_method', default='greedy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None)
    p.add_argument('--template_npz', default=None,
                   help="v2: alternative fixed layout (npz with idx / mean_xyz, e.g. baselines_analyses/deploy_robustness.py "
                        "--build_templates) installed in place of bsp_reconstruction/templates/template_vc<N>.npz for mode=template")
    p.add_argument('--region', default='all', choices=['all', 'front', 'back'],
                   help='restrict the deployment VC selection to one torso half')
    a = p.parse_args()
    evaluate(a.run_dir, split=a.split, n=a.n, vc=a.vc, mode=a.mode,
             cluster_method=a.cluster_method, seed=a.seed, out_csv=a.out,
             region=a.region, template_npz=a.template_npz)
