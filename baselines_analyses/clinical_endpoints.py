"""Clinical endpoints of the epicardial maps and their agreement between the 16-electrode completion and the full vest
(collaborators' items A1-A3, 2026-10-21).  Everything is read from the cached per-vertex maps: the completion
(generated/atmaps/<ID>/<split>_<layout>/<sid>.npz: AT4/RT4), the all-valid inverse (refs cache AT2/RT2), the
commercial reference (ATcom/RTcom) and the raw 16-electrode inverse (vc cache AT1/RT1).  Vertices: the commercial
QRS-gated set for activation endpoints, the commercial T-gated set for repolarisation endpoints (same set for every
source).  Left/right ventricle = angular sectors of the patient-independent heart frame of inverse_evaluation/heart_coords.py
(LV free wall theta in (60, 180] deg, RV free wall theta in (-120, 30) deg; the vendor mesh carries no ventricle labels,
so these are geometric approximations).  Endpoints per map (ms unless stated), all relative to the map's own onset
(5th percentile of the gated activation times):
  tat      total activation time (95th - 5th percentile)        at_mean / at_sd   mean / SD of the activation times
  lvat95   LV activation time (95th percentile of the LV sector) lvat_max          latest LV activation
  rvat95   RV activation time                                     lvdi              LV dyssynchrony index (SD over the LV sector)
  veu      ventricular electrical uncoupling (mean LV - mean RV)  rt_mean / rt_sd   repolarisation time mean / SD
  ari_mean / ari_sd   activation-recovery interval mean / SD (RT - AT on the T-gated vertices)
A1  agreement of each source with the full vest (all-valid inverse) and with the commercial reference, recording and
    patient level: bias, SD, limits of agreement, ICC(2,1), correlations; floor = full vest vs commercial.
A3  equivalence: patient-cluster bootstrap 90% CI of the mean difference (TOST at alpha = 0.05) against clinically
    anchored margins (MARGINS), plus the smallest margin at which equivalence would hold.
A2  label-free pacing-decision concordance on cohorts with several recordings per patient (internal test, hold-out):
    ICC of the within-patient deviations, best-recording agreement and regret, pairwise sign agreement of
    within-patient differences larger than the margin.
  python baselines_analyses/clinical_endpoints.py [--id LOC3] [--holdout_id CARTOHO] [--layouts template kmeans] [--splits ...]
  -> generated/clinical_endpoints/<split>_<layout>.csv (per recording), generated/clinical_endpoints_v2.json,
     generated/tab_clinical_endpoints_v2.tex (+_km), tab_pacing_concordance_v2.tex (+_km).  Aggregates only."""
import os, sys, csv, json, glob, argparse, warnings
import numpy as np
from scipy.stats import wilcoxon, spearmanr, pearsonr, kendalltau
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('bsp_reconstruction', 'baselines_analyses', 'inverse_evaluation'):
    sys.path.insert(0, os.path.join(V2, d))
from paths import V2_GENERATED, V1_ROOT, assert_v2
from dataset import collect_split
from cache_references import cache_lookup, vc_tag
from patient_ids import patient_map
from heart_coords import coords_for_geometry, LV_SECTOR, RV_SECTOR
from dedup import ok as dedup_ok

GEOM_DIR_V1 = os.path.join(V1_ROOT, 'inverse_evaluation', 'spatial_out', 'geom')      # frozen per-geometry (theta, z) cache
GEOM_DIR_V2 = os.path.join(V2, 'inverse_evaluation', 'cache_hsp', 'geom')               # new geometries (hold-out patients)
MIN_GATE, MIN_SECTOR, MIN_T = 20, 10, 20
ENDPOINTS = ['tat', 'at_mean', 'at_sd', 'lvat95', 'lvat_max', 'rvat95', 'lvdi', 'veu', 'rt_mean', 'rt_sd', 'ari_mean', 'ari_sd']
LABEL = {'tat': 'total activation time', 'at_mean': 'mean activation time', 'at_sd': 'activation dispersion (SD)',
         'lvat95': 'LV activation time (95th pct)', 'lvat_max': 'latest LV activation', 'rvat95': 'RV activation time (95th pct)',
         'lvdi': 'LV dyssynchrony index (SD)', 'veu': 'ventricular electrical uncoupling', 'rt_mean': 'mean repolarisation time',
         'rt_sd': 'repolarisation dispersion (SD)', 'ari_mean': 'mean ARI', 'ari_sd': 'ARI dispersion (SD)'}
# clinically anchored equivalence margins (ms): activation times +-10 ms, dispersion indices +-5 ms (see the Methods of the paper)
MARGINS = {'tat': 10, 'at_mean': 10, 'at_sd': 5, 'lvat95': 10, 'lvat_max': 10, 'rvat95': 10, 'lvdi': 5, 'veu': 10,
           'rt_mean': 10, 'rt_sd': 5, 'ari_mean': 10, 'ari_sd': 5}
DECISION = ['lvat95', 'tat', 'lvdi', 'veu', 'at_sd']          # lower = better (dyssynchrony), |veu| for the uncoupling
SOURCES = ['ours', 'vconly', 'allvalid', 'commercial']
PAIRS = [('ours', 'allvalid'), ('ours', 'commercial'), ('vconly', 'allvalid'), ('vconly', 'commercial'), ('allvalid', 'commercial')]
SHORT = {'test_internal': 'internal', 'test_external': 'external', 'holdout': 'hold-out'}
_GEOM = {}


def geom_coords(gdir):
    if gdir in _GEOM:
        return _GEOM[gdir]
    key = gdir.strip('/').replace('/', '_') + '.npz'
    for root in (GEOM_DIR_V1, GEOM_DIR_V2):
        g = os.path.join(root, key)
        if os.path.exists(g):
            with np.load(g, allow_pickle=False) as z:
                _GEOM[gdir] = (np.degrees(z['theta'].astype(float)), z['z'].astype(float))
            return _GEOM[gdir]
    th, zc, _ = coords_for_geometry(gdir)
    os.makedirs(GEOM_DIR_V2, exist_ok=True); np.savez_compressed(os.path.join(GEOM_DIR_V2, key), theta=th, z=zc)
    _GEOM[gdir] = (np.degrees(np.asarray(th, float)), np.asarray(zc, float))
    return _GEOM[gdir]


def sectors(theta_deg):
    lv = (theta_deg > LV_SECTOR[0]) & (theta_deg <= LV_SECTOR[1])
    rv = (theta_deg > RV_SECTOR[0]) & (theta_deg < RV_SECTOR[1])
    return lv, rv


def endpoints(at, rt, qgate, tgate, lv, rv):
    out = {k: np.nan for k in ENDPOINTS}
    at = np.asarray(at, float); rt = np.asarray(rt, float)
    q = qgate & np.isfinite(at)
    if q.sum() < MIN_GATE:
        return out
    v = at[q]; onset = float(np.quantile(v, 0.05))
    out['tat'] = float(np.quantile(v, 0.95) - onset); out['at_mean'] = float(v.mean() - onset); out['at_sd'] = float(v.std())
    lvv, rvv = at[q & lv], at[q & rv]
    if lvv.size >= MIN_SECTOR:
        out['lvat95'] = float(np.quantile(lvv, 0.95) - onset); out['lvat_max'] = float(lvv.max() - onset); out['lvdi'] = float(lvv.std())
    if rvv.size >= MIN_SECTOR:
        out['rvat95'] = float(np.quantile(rvv, 0.95) - onset)
    if lvv.size >= MIN_SECTOR and rvv.size >= MIN_SECTOR:
        out['veu'] = float(lvv.mean() - rvv.mean())
    t = tgate & np.isfinite(rt)
    if t.sum() >= MIN_T:
        w = rt[t]; out['rt_mean'] = float(w.mean() - onset); out['rt_sd'] = float(w.std())
        tq = t & q
        if tq.sum() >= MIN_T:
            a = rt[tq] - at[tq]; out['ari_mean'] = float(a.mean()); out['ari_sd'] = float(a.std())
    return out


def recordings(split, holdout_id):
    """[(sid, torso_dir)], {sid: patient} after the paper's de-duplication rules."""
    if split == 'holdout':
        from holdout_summary import holdout_patient_map, dedup
        pm, cm = holdout_patient_map(holdout_id); sids = dedup(list(pm), cm)
        hf = None
        runs = glob.glob(os.path.join(V2, 'runs', '*', f'{holdout_id}__*'))
        if runs and os.path.exists(os.path.join(runs[0], 'args.json')):
            hf = json.load(open(os.path.join(runs[0], 'args.json'))).get('holdout_file')
        hold = set(l.strip() for l in open(hf) if l.strip()) if hf else None
        b, _ = collect_split(seed=42, holdout=hold)
        torso = {r[0]: r[2] for r in b['holdout']}
        pm = {s: ('|'.join(map(str, v)) if isinstance(v, (tuple, list)) else str(v)) for s, v in pm.items()}   # patient keys as plain strings (never printed)
        return [(s, os.path.dirname(torso[s])) for s in sids if torso.get(s)], pm
    b, _ = collect_split(seed=42); pm = patient_map(split)
    recs = [(r[0], os.path.dirname(r[2])) for r in b[split] if r[2] and dedup_ok(r[0]) and r[0] in pm]
    return recs, pm


def load_maps(split, layout, sid, rid):
    """{source: (AT, RT)} plus gates, or None if the completion map / reference is missing."""
    sid_safe = sid.replace('/', '_')
    fa = os.path.join(V2_GENERATED, 'atmaps', rid, f'{split}_{layout}', sid_safe + '.npz')
    fr = cache_lookup(split, 'refs', sid_safe + '.npz')
    vtag = 'vc16_template_s42' if layout == 'template' else vc_tag(16, 'cluster', 'kmean', 42)
    fv = cache_lookup(split, vtag, sid_safe + '.npz')
    if not os.path.exists(fa) or fr is None:
        return None
    with np.load(fa, allow_pickle=False) as za:
        if not bool(za['valid']):
            return None
        maps = {'ours': (za['AT4'].astype(float), za['RT4'].astype(float))}
    with np.load(fr, allow_pickle=False) as zr:
        if 'ATcom' not in zr.files:
            return None
        maps['allvalid'] = (zr['AT2'].astype(float), zr['RT2'].astype(float)); maps['commercial'] = (zr['ATcom'].astype(float), zr['RTcom'].astype(float))
        gates = (zr['qrs_goodcom'].astype(bool), zr['t_goodcom'].astype(bool))
    if fv is not None:
        with np.load(fv, allow_pickle=False) as zv:
            maps['vconly'] = (zv['AT1'].astype(float), zv['RT1'].astype(float))
    return maps, gates


# ----------------------------------------------------------------------------------------------------- statistics ---
def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measures (two raters)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = x.size
    if n < 3:
        return np.nan
    M = np.stack([x, y], 1); k = 2
    gm = M.mean(); mr = M.mean(1); mc = M.mean(0)
    ssr = k * ((mr - gm) ** 2).sum(); ssc = n * ((mc - gm) ** 2).sum(); sse = ((M - mr[:, None] - mc[None, :] + gm) ** 2).sum()
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def cluster_boot_ci(d, pat, B=2000, seed=0, alpha=0.10, x=None, y=None):
    """percentile CI of the mean of d under patient-cluster resampling (TOST at alpha/2 per side <-> (1-alpha) CI);
    with x, y also the percentile CI of ICC(2,1) under the same resampling."""
    rng = np.random.default_rng(seed); pats = np.array(sorted(set(pat))); idx = {p: np.where(np.asarray(pat) == p)[0] for p in pats}
    means = np.empty(B); iccs = np.full(B, np.nan)
    for b in range(B):
        pick = rng.choice(pats, size=pats.size, replace=True)
        sel = np.concatenate([idx[p] for p in pick]); means[b] = d[sel].mean()
        if x is not None:
            iccs[b] = icc21(x[sel], y[sel])
    ci = (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
    if x is None:
        return ci
    ok_ = np.isfinite(iccs)
    return ci, ((float(np.quantile(iccs[ok_], alpha / 2)), float(np.quantile(iccs[ok_], 1 - alpha / 2))) if ok_.sum() > 10 else (np.nan, np.nan))


def agreement(src, ref, pat, margin):
    m = np.isfinite(src) & np.isfinite(ref); x, y, p = src[m], ref[m], np.asarray(pat)[m]
    if x.size < 3:
        return None
    d = x - y; pats = sorted(set(p))
    xp = np.array([x[p == q].mean() for q in pats]); yp = np.array([y[p == q].mean() for q in pats])
    (lo, hi), (ilo, ihi) = cluster_boot_ci(d, p, x=x, y=y)
    try:
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        wp = np.nan
    return dict(n_rec=int(x.size), n_pat=len(pats), mean_ref=float(y.mean()), sd_ref=float(y.std(ddof=1)), mean_src=float(x.mean()),
                bias=float(d.mean()), sd_diff=float(d.std(ddof=1)), loa_lo=float(d.mean() - 1.96 * d.std(ddof=1)), loa_hi=float(d.mean() + 1.96 * d.std(ddof=1)),
                mae=float(np.abs(d).mean()), median_diff=float(np.median(d)), wilcoxon_p=wp, r=float(pearsonr(x, y)[0]), rho=float(spearmanr(x, y)[0]),
                icc=icc21(x, y), icc_ci90_lo=ilo, icc_ci90_hi=ihi, bias_pat=float((xp - yp).mean()), sd_diff_pat=float((xp - yp).std(ddof=1)) if len(pats) > 1 else np.nan,
                icc_pat=icc21(xp, yp), ci90_lo=lo, ci90_hi=hi, margin=margin, equivalent=bool(lo > -margin and hi < margin),
                equivalence_bound=float(max(abs(lo), abs(hi))))


def concordance(src, ref, pat, margin, abs_value=False):
    """within-patient decisions (lower = better): deviations ICC, best-recording agreement / regret, pairwise sign agreement."""
    m = np.isfinite(src) & np.isfinite(ref); x, y, p = src[m], ref[m], np.asarray(pat)[m]
    if abs_value:
        x, y = np.abs(x), np.abs(y)
    dev_x, dev_y, agree, regret, pairs_ok, pairs_n, taus = [], [], [], [], 0, 0, []
    for q in sorted(set(p)):
        i = np.where(p == q)[0]
        if i.size < 2:
            continue
        xs, ys = x[i], y[i]
        dev_x += list(xs - xs.mean()); dev_y += list(ys - ys.mean())
        bx, by = int(np.argmin(xs)), int(np.argmin(ys)); agree.append(bx == by); regret.append(float(ys[bx] - ys[by]))
        if i.size >= 3:
            t = kendalltau(xs, ys)[0]
            if np.isfinite(t):
                taus.append(float(t))
        for a in range(i.size):
            for b in range(a + 1, i.size):
                dy = ys[a] - ys[b]
                if abs(dy) >= margin:
                    pairs_n += 1; pairs_ok += int(np.sign(xs[a] - xs[b]) == np.sign(dy))
    if not agree:
        return None
    dev_x, dev_y, regret = np.array(dev_x), np.array(dev_y), np.array(regret)
    return dict(n_pat=len(agree), n_rec=int(dev_x.size), icc_dev=icc21(dev_x, dev_y), r_dev=float(pearsonr(dev_x, dev_y)[0]) if dev_x.size > 2 else np.nan,
                best_agree=int(sum(agree)), best_agree_frac=float(np.mean(agree)), regret_median=float(np.median(regret)), regret_max=float(regret.max()),
                regret_mean=float(regret.mean()), regret_le5=int((regret <= 5).sum()), regret_le10=int((regret <= 10).sum()),
                pairs_n=int(pairs_n), pairs_agree=int(pairs_ok), pairs_agree_frac=float(pairs_ok / pairs_n) if pairs_n else np.nan,
                kendall_tau_mean=float(np.mean(taus)) if taus else np.nan, margin=margin)


# ----------------------------------------------------------------------------------------------------------- main ---
def f1(x, nd=1):
    return '--' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def stars(p):
    return '' if not np.isfinite(p) else '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='LOC3'); ap.add_argument('--holdout_id', default='CARTOHO')
    ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans']); ap.add_argument('--splits', nargs='+', default=['test_internal', 'test_external', 'holdout'])
    a = ap.parse_args(); warnings.filterwarnings('ignore')
    od = assert_v2(os.path.join(V2_GENERATED, 'clinical_endpoints')); os.makedirs(od, exist_ok=True)
    res = {'model': a.id, 'holdout_model': a.holdout_id, 'margins_ms': MARGINS, 'sectors_deg': {'LV': LV_SECTOR, 'RV': RV_SECTOR},
           'gates': 'commercial QRS-gated vertices (AT), commercial T-gated vertices (RT/ARI); min %d / %d vertices, %d per sector' % (MIN_GATE, MIN_T, MIN_SECTOR),
           'A1': {}, 'A2': {}, 'n': {}}
    for layout in a.layouts:
        for split in a.splits:
            rid = a.holdout_id if split == 'holdout' else a.id
            try:
                recs, pm = recordings(split, a.holdout_id)
            except Exception as e:
                print(f'[{layout} {split}] skipped ({type(e).__name__})'); continue
            rows, n_miss = [], {'maps': 0, 'geom': 0}
            for sid, gdir in recs:
                lm = load_maps(split, layout, sid, rid)
                if lm is None:
                    n_miss['maps'] += 1; continue
                maps, (qg, tg) = lm
                try:
                    th, _z = geom_coords(gdir)
                except Exception:
                    n_miss['geom'] += 1; continue
                if th.shape[0] != qg.shape[0]:
                    n_miss['geom'] += 1; continue
                lv, rv = sectors(th)
                row = {'sid': sid, 'patient': pm[sid], 'n_q': int(qg.sum()), 'n_t': int(tg.sum()), 'n_lv': int((qg & lv).sum()), 'n_rv': int((qg & rv).sum())}
                for s in SOURCES:
                    if s in maps:
                        e = endpoints(maps[s][0], maps[s][1], qg, tg, lv, rv)
                        row.update({f'{s}_{k}': v for k, v in e.items()})
                rows.append(row)
            if not rows:
                print(f'[{layout} {split}] no recordings (missing maps {n_miss})'); continue
            cols = list(rows[0].keys())
            for r in rows:
                for c in cols:
                    r.setdefault(c, np.nan)
            with open(os.path.join(od, f'{split}_{layout}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
            pat = [r['patient'] for r in rows]
            key = f'{layout}|{split}'; res['n'][key] = {'recordings': len(rows), 'patients': len(set(pat)), 'missing': n_miss,
                                                        'median_vertices': {k: float(np.median([r[k] for r in rows])) for k in ('n_q', 'n_t', 'n_lv', 'n_rv')}}
            A1 = {}
            for src, ref in PAIRS:
                for ep in ENDPOINTS:
                    xs = np.array([r.get(f'{src}_{ep}', np.nan) for r in rows], float); ys = np.array([r.get(f'{ref}_{ep}', np.nan) for r in rows], float)
                    st = agreement(xs, ys, pat, MARGINS[ep])
                    if st:
                        A1[f'{src}-{ref}|{ep}'] = st
            res['A1'][key] = A1
            if split in ('test_internal', 'holdout'):
                A2 = {}
                for src, ref in (('ours', 'allvalid'), ('ours', 'commercial'), ('vconly', 'allvalid'), ('vconly', 'commercial'), ('allvalid', 'commercial')):
                    for ep in DECISION:
                        xs = np.array([r.get(f'{src}_{ep}', np.nan) for r in rows], float); ys = np.array([r.get(f'{ref}_{ep}', np.nan) for r in rows], float)
                        st = concordance(xs, ys, pat, MARGINS[ep], abs_value=(ep == 'veu'))
                        if st:
                            A2[f'{src}-{ref}|{ep}'] = st
                res['A2'][key] = A2
            print(f'[{layout} {split}] {len(rows)} recordings, {len(set(pat))} patients (missing {n_miss}); median gated vertices {res["n"][key]["median_vertices"]}')
            for ep in ('lvat95', 'lvdi', 'tat', 'veu', 'ari_mean'):
                st = A1.get(f'ours-allvalid|{ep}'); fl = A1.get(f'allvalid-commercial|{ep}')
                if st:
                    print(f'   {ep:9s} ref {st["mean_ref"]:6.1f}+-{st["sd_ref"]:5.1f}  Ours-full vest bias {st["bias"]:+5.1f} [LoA {st["loa_lo"]:+5.1f},{st["loa_hi"]:+5.1f}] ICC {st["icc"]:.2f} '
                          f'(pat {st["icc_pat"]:.2f}) 90% CI [{st["ci90_lo"]:+.1f},{st["ci90_hi"]:+.1f}] margin {st["margin"]} -> {"EQUIV" if st["equivalent"] else "not"}'
                          + (f' | floor full-vest vs commercial: bias {fl["bias"]:+.1f}, ICC {fl["icc"]:.2f}' if fl else ''))
    json.dump(res, open(assert_v2(os.path.join(V2_GENERATED, 'clinical_endpoints_v2.json')), 'w'), indent=1)
    tables(res, a.layouts)


def tables(res, layouts):
    for layout in layouts:
        sfx = '' if layout == 'template' else '_km'
        L = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'Cohort & Endpoint & Full vest & \multicolumn{3}{c}{Ours ($16$) $-$ full vest} & \multicolumn{2}{c}{Ours $-$ commercial} & Floor \\',
             r'\cmidrule(lr){4-6}\cmidrule(lr){7-8}',
             r' & & mean (SD) & bias [LoA] & ICC & 90\% CI ($\delta$) & bias & ICC & bias / ICC \\']
        for split in ('test_internal', 'test_external', 'holdout'):
            A1 = res['A1'].get(f'{layout}|{split}')
            if not A1:
                continue
            L.append(r'\midrule'); first = True
            for ep in ('tat', 'lvat95', 'rvat95', 'lvdi', 'veu', 'at_sd', 'ari_mean', 'ari_sd'):
                s = A1.get(f'ours-allvalid|{ep}'); c = A1.get(f'ours-commercial|{ep}'); fl = A1.get(f'allvalid-commercial|{ep}')
                if not s:
                    continue
                verdict = r'$\checkmark$' if s['equivalent'] else '--'
                L.append(f"{SHORT[split] if first else ''} & {LABEL[ep]} & {f1(s['mean_ref'])} ({f1(s['sd_ref'])}) & ${s['bias']:+.1f}$ [${s['loa_lo']:+.1f}$, ${s['loa_hi']:+.1f}$] & {f1(s['icc'], 2)} & "
                         f"[${s['ci90_lo']:+.1f}$, ${s['ci90_hi']:+.1f}$] ($\\pm{s['margin']}$) {verdict} & "
                         + (f"${c['bias']:+.1f}$ & {f1(c['icc'], 2)}" if c else '-- & --') + ' & ' + (f"${fl['bias']:+.1f}$ / {f1(fl['icc'], 2)}" if fl else '--') + r' \\')
                first = False
        L += [r'\bottomrule', r'\end{tabular}']
        open(assert_v2(os.path.join(V2_GENERATED, f'tab_clinical_endpoints{sfx}_v2.tex')), 'w').write('\n'.join(L) + '\n')
        L = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Cohort & Endpoint & Reference & within-patient ICC & best recording & regret (ms) & pairs $\ge\delta$ & sign agreement \\']
        for split in ('test_internal', 'holdout'):
            A2 = res['A2'].get(f'{layout}|{split}')
            if not A2:
                continue
            L.append(r'\midrule'); first = True
            for ep in ('lvat95', 'tat', 'lvdi', 'veu'):
                for src, ref, lab in (('ours', 'allvalid', 'full vest'), ('ours', 'commercial', 'commercial'), ('vconly', 'allvalid', 'full vest (VC-only)'), ('allvalid', 'commercial', 'floor: full vest vs commercial')):
                    s = A2.get(f'{src}-{ref}|{ep}')
                    if not s:
                        continue
                    L.append(f"{SHORT[split] if first else ''} & {LABEL[ep] if src == 'ours' and ref == 'allvalid' else ''} & {lab} & {f1(s['icc_dev'], 2)} & {s['best_agree']}/{s['n_pat']} & "
                             f"{f1(s['regret_median'])} [max {f1(s['regret_max'])}] & {s['pairs_n']} & {f1(100 * s['pairs_agree_frac'], 0)}\\% \\\\")
                    first = False
        L += [r'\bottomrule', r'\end{tabular}']
        open(assert_v2(os.path.join(V2_GENERATED, f'tab_pacing_concordance{sfx}_v2.tex')), 'w').write('\n'.join(L) + '\n')
    # condensed main-text table (fixed template layout; five endpoints; both test cohorts)
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Cohort & Endpoint & Full vest, mean (SD) & \multicolumn{3}{c}{completion from $16$ electrodes $-$ full vest} & Two full-vest solutions \\',
         r'\cmidrule(lr){4-6}\cmidrule(lr){7-7}', r' & & & bias [LoA] & ICC & 90\% CI of the bias & bias / ICC \\']
    for split in ('test_internal', 'test_external'):
        A1 = res['A1'].get(f'template|{split}')
        if not A1:
            continue
        L.append(r'\midrule'); first = True
        for ep in ('tat', 'lvat95', 'lvdi', 'veu', 'ari_mean'):
            s = A1.get(f'ours-allvalid|{ep}'); fl = A1.get(f'allvalid-commercial|{ep}')
            if not s:
                continue
            eq = r'$^{\dagger}$' if s['equivalent'] else ''
            L.append(f"{SHORT[split] if first else ''} & {LABEL[ep]} & {f1(s['mean_ref'])} ({f1(s['sd_ref'])}) & ${s['bias']:+.1f}$ [${s['loa_lo']:+.1f}$, ${s['loa_hi']:+.1f}$] & {f1(s['icc'], 2)} & "
                     f"[${s['ci90_lo']:+.1f}$, ${s['ci90_hi']:+.1f}$]{eq} & " + (f"${fl['bias']:+.1f}$ / {f1(fl['icc'], 2)}" if fl else '--') + r' \\'); first = False
    L += [r'\bottomrule', r'\end{tabular}']
    open(assert_v2(os.path.join(V2_GENERATED, 'tab_clinical_endpoints_npjdm_v2.tex')), 'w').write('\n'.join(L) + '\n')
    print('wrote clinical_endpoints_v2.json, tab_clinical_endpoints{,_km}_v2.tex, tab_pacing_concordance{,_km}_v2.tex, tab_clinical_endpoints_npjdm_v2.tex')


if __name__ == '__main__':
    main()
