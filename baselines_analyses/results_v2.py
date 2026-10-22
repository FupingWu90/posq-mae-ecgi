"""Main-results tables and figures of the v2 manuscript (port of the frozen v1 merged_tables.py / learned_baselines.py
with the PosQ-MAE runs and both deployment layouts).  Read-only w.r.t. v1; writes generated/ and figures/.

  python baselines_analyses/results_v2.py --layout template --ref commercial       # main-text tables + figures
  python baselines_analyses/results_v2.py --layout template --ref src2_allvalid    # S7 (_ref2) tables
  python baselines_analyses/results_v2.py --layout kmeans   --ref commercial       # supplement (_km) tables + figures

Outputs (sfx = '' | '_ref2', lay = '' | '_km'):
  tab_all_means_{internal,external}{sfx}{lay}_v2.tex, tab_all_sig{sfx}{lay}_v2.tex, tab_ari_sig{lay}_v2.tex (commercial only),
  figures/fig_baselines_{at,rt,bsp,hsp}_{internal,external}{lay}_v2.png (commercial only), results{sfx}{lay}_v2.json
"""
import os, sys, csv, glob, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'baselines_analyses')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
from paths import V1_RUNS, V1_GENERATED, V2_GENERATED, V2_RUNS, FIG_DIR_V2, assert_v2
from compare_classical_layouts import load_template as cl_template, load_kmeans as cl_kmeans
from cache_references import cache_lookup, CACHE_ROOT, CACHE_ROOT_V1

VCS = [4, 8, 16, 32, 64]
SPLITS = ['test_internal', 'test_external']
METS = ['AT', 'RT', 'ARI']
OURS_ID = {4: 'P3N4', 8: 'P3N8', 16: 'LOC3', 32: 'P3N32', 64: 'P3N64'}
STD = 'cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'
COLS = [('ours', 'Ours'), ('stmem', 'ST-MEM'), ('mcma', 'MCMA'), ('lmmse', 'LMMSE'), ('laplacian', 'Laplace'), ('tps', 'TPS'), ('vconly', 'VC-only')]
REF, LAYOUT = 'commercial', 'template'
FIG_DPI = int(os.environ.get('ECGI_FIG_DPI', '150'))   # 2026-10-20: 300 for the npj figure files
_PMAP = {}


def pmap_of(split):
    if split not in _PMAP:
        from patient_ids import patient_map
        _PMAP[split] = patient_map(split)
    return _PMAP[split]


def boot_ci(vals, pmap, B=1000, seed=0):
    """(mean, lo, hi): cohort mean over recordings and its 95% CI from a bootstrap over PATIENTS (the recordings of a
    resampled patient enter together; recordings of one patient are repeated measures).  2026-10-20, Fig. 2 error bars."""
    groups = {}
    for sid, v in vals.items():
        if sid in pmap and v is not None and np.isfinite(v):
            groups.setdefault(pmap[sid], []).append(float(v))
    if not groups:
        return np.nan, np.nan, np.nan
    psum = np.array([np.sum(g) for g in groups.values()]); pcnt = np.array([len(g) for g in groups.values()], float)
    mean = float(psum.sum() / pcnt.sum())
    idx = np.random.default_rng(seed).integers(0, len(psum), size=(B, len(psum)))
    boots = psum[idx].sum(axis=1) / pcnt[idx].sum(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
fmt = lambda x, p=2: (f'{x:.{p}f}' if isinstance(x, (int, float)) and np.isfinite(x) else '--')


def stars(p):
    if not np.isfinite(p):
        return ''
    return '$^{***}$' if p < 1e-3 else '$^{**}$' if p < 1e-2 else '$^{*}$' if p < 5e-2 else '$^{\\mathrm{ns}}$'


def percase(path, source):
    """{sid: {AT, RT, ARI, HSP, BSP?}} from an eval_downstream CSV, rows of `source` vs the current REF.
    RT / ARI are kept only where the reference T-gate holds >= 3 vertices (baselines_analyses/tgate_rule.py; 2026-10-20)."""
    from tgate_rule import ok as _tgate_ok
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    split = next((sp for sp in ('test_internal', 'test_external', 'holdout', 'val') if sp in os.path.basename(path)), None)
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r['source'] != source or r['ref'] != REF or not _dedup_ok(r['sid']):
            continue
        d = {}
        for met in METS:
            if met in ('RT', 'ARI') and split and not _tgate_ok(split, r['sid'], REF):
                continue
            try:
                d[met] = float(r[f'{met}_mae'])
            except (TypeError, ValueError):
                pass
        for k, col in (('HSP', 'hsp_cc'), ('BSP', 'bsp_cc')):
            try:
                v = float(r[col])
                if np.isfinite(v):
                    d[k] = v
            except (TypeError, ValueError, KeyError):
                pass
        out[r['sid']] = d
    return out


def ours_csv(vc, split):
    return os.path.join(V2_GENERATED, 'percase_candidate', f'{OURS_ID[vc]}_vc{vc}_{split}_{LAYOUT}.csv')


def learned_csv(key, vc, split):
    if LAYOUT == 'template':
        return os.path.join(V2_GENERATED, 'percase_template', f'{key}_vc{vc}_{split}.csv')
    return os.path.join(V1_GENERATED, 'vc_percase', f"{'peindex' if key == 'stmem' else 'mcma'}_vc{vc}_{split}.csv")


def classical_percase(entries, method, vc, split=None):
    from tgate_rule import ok as _tgate_ok
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    suf = 'com' if REF == 'commercial' else 'src2'
    out = {}
    for (sid, m, v), e in entries.items():
        if m != method or v != vc or not _dedup_ok(sid):
            continue
        d = {}
        for met, col in (('AT', f'at_{suf}'), ('RT', f'rt_{suf}'), ('ARI', f'ari_{suf}')):
            if met in ('RT', 'ARI') and split and not _tgate_ok(split, sid, REF):
                continue
            if e.get(col) is not None and np.isfinite(e[col]):
                d[met] = float(e[col])
        h = e.get('hsp_cc' if REF == 'commercial' else 'hsp_cc_src2')
        if h is not None and np.isfinite(h):
            d['HSP'] = float(h)
        if e.get('bsp_cc') is not None and np.isfinite(e['bsp_cc']):
            d['BSP'] = float(e['bsp_cc'])
        out[sid] = d
    return out


def load_allvalid(split, met):
    """The OTHER reference as a competing source: per-recording |AT2-ATcom| on the gated vertices (model-free)."""
    out = {}
    if REF == 'commercial':
        src_sfx, ref_sfx = '2', 'com'; gate = 'qrs_goodcom' if met == 'AT' else 't_goodcom'
    else:
        src_sfx, ref_sfx = 'com', '2'; gate = 'qrs_good2' if met == 'AT' else 't_good2'
    files = {}
    for root in (CACHE_ROOT, CACHE_ROOT_V1):
        for f in glob.glob(os.path.join(root, split, 'refs', '*.npz')):
            files.setdefault(os.path.basename(f), f)
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for name, f in files.items():
        if not _dedup_ok(name[:-4]):
            continue
        with np.load(f, allow_pickle=False) as z:
            if 'AT2' not in z.files or 'ATcom' not in z.files:
                continue
            g = z[gate].astype(bool)
            if g.sum() < 3:
                continue
            if met == 'ARI':
                a = z[f'RT{src_sfx}'].astype(float) - z[f'AT{src_sfx}'].astype(float)
                b = z[f'RT{ref_sfx}'].astype(float) - z[f'AT{ref_sfx}'].astype(float)
            else:
                a, b = z[f'{met}{src_sfx}'].astype(float), z[f'{met}{ref_sfx}'].astype(float)
            out[name[:-4]] = float(np.abs(a - b)[g].mean())
    return out


def engine_bsp(run_dir, split):
    f = os.path.join(run_dir, 'test_compare.json')
    if not os.path.exists(f):
        return None
    key = 'template' if LAYOUT == 'template' else 'cluster-kmean'
    return json.load(open(f))['results'].get(split, {}).get(key, {}).get('cc')


def mean_of(d, key):
    v = [r[key] for r in d.values() if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(v)) if v else None


def load_everything():
    P, M = {}, {}
    for split in SPLITS:
        T = cl_template(split) if LAYOUT == 'template' else cl_kmeans(split)
        for vc in VCS:
            P[('ours', vc, split)] = percase(ours_csv(vc, split), 'src4_recon_valid')
            P[('vconly', vc, split)] = percase(ours_csv(vc, split), 'src1_vc')
            for key in ('stmem', 'mcma'):
                P[(key, vc, split)] = percase(learned_csv(key, vc, split), 'src4_recon_valid')
            for m in ('lmmse', 'laplacian', 'tps'):
                P[(m, vc, split)] = classical_percase(T, m, vc, split)
            for m, _ in COLS:
                for q in METS + ['HSP']:
                    M[(m, vc, split, q)] = mean_of(P[(m, vc, split)], q)
            # BSP CC: engine value for the trained models (per-recording CSV mean where the engine value is not available)
            g = glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', f'{OURS_ID[vc]}__*'))
            M[('ours', vc, split, 'BSP')] = engine_bsp(g[0], split) if g else None
            if LAYOUT == 'kmeans':
                M[('stmem', vc, split, 'BSP')] = engine_bsp(os.path.join(V1_RUNS, f'fixedn_ecgi_wide_vc{vc}_{STD}_pe-index'), split)
                M[('mcma', vc, split, 'BSP')] = engine_bsp(os.path.join(V1_RUNS, f'mcma_unet_vc{vc}_{STD}'), split)
            else:
                for key in ('stmem', 'mcma'):
                    M[(key, vc, split, 'BSP')] = mean_of(P[(key, vc, split)], 'BSP')
            for m in ('lmmse', 'laplacian', 'tps'):
                M[(m, vc, split, 'BSP')] = mean_of(P[(m, vc, split)], 'BSP')
            M[('vconly', vc, split, 'BSP')] = None
    return P, M


def sfx_lay():
    return ('' if REF == 'commercial' else '_ref2') + ('' if LAYOUT == 'template' else '_km')


def means_tables(M):
    blocks = [('BSP', 'BSP CC $\\uparrow$', 3), ('HSP', 'HSP CC $\\uparrow$', 3), ('AT', 'AT MAE (ms) $\\downarrow$', 2),
              ('RT', 'RT MAE (ms) $\\downarrow$', 2), ('ARI', 'ARI MAE (ms) $\\downarrow$', 2)]
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        L = ['\\begin{tabular}{lccccccc}', '\\toprule', '$N$ & ' + ' & '.join(n for _, n in COLS) + ' \\\\']
        for q, head, prec in blocks:
            L.append('\\midrule\n\\multicolumn{8}{l}{\\emph{' + head + '}} \\\\')
            for vc in VCS:
                L.append(f'{vc} & ' + ' & '.join(fmt(M.get((m, vc, split, q)), prec) for m, _ in COLS) + ' \\\\')
        L.append('\\bottomrule\n\\end{tabular}')
        p = assert_v2(os.path.join(V2_GENERATED, f'tab_all_means_{lab}{sfx_lay()}_v2.tex'))
        open(p, 'w').write('\n'.join(L) + '\n'); print('wrote', os.path.basename(p))


def paired(a, b, met):
    sids = sorted(set(a) & set(b))
    x = np.array([a[s].get(met, np.nan) if isinstance(a[s], dict) else a[s] for s in sids], float)
    y = np.array([b[s].get(met, np.nan) if isinstance(b[s], dict) else b[s] for s in sids], float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 10:
        return np.nan, np.nan, 0
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return float(np.median(x - y)), float(p), int(len(x))


def sig_table(P):
    other_lab = 'all-valid' if REF == 'commercial' else 'commercial'
    comps = [c for c in COLS if c[0] != 'ours'] + [('other', other_lab)]
    header = ['\\begin{tabular}{lccccccc}', '\\toprule', '$N$ & ' + ' & '.join('vs.\\ ' + n for _, n in comps) + ' \\\\']
    L, L_ari, res = list(header), list(header), {}
    for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
        av = {met: load_allvalid(split, met) for met in METS}
        for met in METS:
            T = L_ari if (REF == 'commercial' and met == 'ARI') else L
            T.append('\\midrule\n\\multicolumn{8}{l}{\\emph{' + lab + ' --- ' + met + '}} \\\\')
            for vc in VCS:
                ours = {k.replace('/', '_'): v for k, v in P[('ours', vc, split)].items()}
                cells = []
                for m, _ in comps:
                    alt = av[met] if m == 'other' else {k.replace('/', '_'): v for k, v in P[(m, vc, split)].items()}
                    if not alt or not ours:
                        cells.append('--'); continue
                    d, p, n = paired(ours, alt, met)
                    res[f'{split}|{met}|{vc}|{m}'] = dict(med_diff=d, p=p, n=n)
                    cells.append(f'${d:+.1f}${stars(p)}' if np.isfinite(d) else '--')
                T.append(f'{vc} & ' + ' & '.join(cells) + ' \\\\')
    L.append('\\bottomrule\n\\end{tabular}')
    p = assert_v2(os.path.join(V2_GENERATED, f'tab_all_sig{sfx_lay()}_v2.tex')); open(p, 'w').write('\n'.join(L) + '\n'); print('wrote', os.path.basename(p))
    if REF == 'commercial':
        L_ari.append('\\bottomrule\n\\end{tabular}')
        p = assert_v2(os.path.join(V2_GENERATED, f'tab_ari_sig{"" if LAYOUT == "template" else "_km"}_v2.tex')); open(p, 'w').write('\n'.join(L_ari) + '\n'); print('wrote', os.path.basename(p))
    return res


STY = {'ours': ('#0072B2', 's-', 'Ours'), 'stmem': ('#E69F00', 'o-', 'ST-MEM'), 'mcma': ('#56B4E9', 'D-', 'MCMA'),
       'lmmse': ('#000000', 'v-', 'LMMSE'), 'laplacian': ('#CC79A7', '^--', 'Laplace'), 'tps': ('#7F7F7F', 'P--', 'TPS'),
       'vconly': ('#D55E00', 'X:', 'VC-only')}


def figures(M, P=None):
    lay = '' if LAYOUT == 'template' else '_km'
    os.makedirs(FIG_DIR_V2, exist_ok=True)
    offs = np.linspace(-0.16, 0.16, len(STY))   # dodge on the log2 axis so that error bars do not overlap
    for met in ('AT', 'RT'):
        for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
            allv = load_allvalid(split, met); allv_mean = float(np.mean(list(allv.values())))
            pm = pmap_of(split)
            fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
            for i, (m, (c, mk, name)) in enumerate(STY.items()):
                y = [M.get((m, vc, split, met)) if M.get((m, vc, split, met)) is not None else np.nan for vc in VCS]
                if P is not None:
                    ci = [boot_ci({sid: d.get(met) for sid, d in P.get((m, vc, split), {}).items()}, pm) for vc in VCS]
                    yerr = [[max(yy - lo, 0) if np.isfinite(lo) else 0 for yy, (_, lo, _h) in zip(y, ci)],
                            [max(hi - yy, 0) if np.isfinite(hi) else 0 for yy, (_, _l, hi) in zip(y, ci)]]
                    ax.errorbar(np.array(VCS) * 2.0 ** offs[i], y, yerr=yerr, fmt=mk, color=c, lw=2.0 if m == 'ours' else 1.4,
                                ms=6 if m == 'ours' else 4.5, capsize=2, elinewidth=0.9, capthick=0.9, label=name)
                else:
                    ax.plot(VCS, y, mk, color=c, lw=2.0 if m == 'ours' else 1.4, ms=6 if m == 'ours' else 4.5, label=name)
            if P is not None:
                _, alo, ahi = boot_ci(allv, pm)
                if np.isfinite(alo):
                    ax.axhspan(alo, ahi, color='#009E73', alpha=0.12, lw=0)
            ax.axhline(allv_mean, color='#009E73', ls='--', lw=1.4, label='all-valid inverse (252)')   # legend entry instead of an in-plot label (2026-10-20)
            ax.set_xscale('log', base=2); ax.set_xticks(VCS); ax.set_xticklabels(VCS)
            ax.set_xlabel('number of visible electrodes $N$'); ax.set_ylabel(f'epicardial {met} MAE vs commercial (ms)')
            ax.set_title(f'{lab} dataset', fontsize=10); ax.grid(alpha=0.25, lw=0.5)
            ax.legend(frameon=False, fontsize=7.5, ncol=2 if met == 'RT' else 1)
            out = os.path.join(FIG_DIR_V2, f'fig_baselines_{met.lower()}_{lab}{lay}_v2.png'); fig.savefig(out, dpi=FIG_DPI); plt.close(fig); print('wrote', os.path.basename(out))
    panels = [('bsp', 'BSP completion CC (held-out electrodes)', 'BSP', ('ours', 'stmem', 'mcma', 'lmmse', 'laplacian', 'tps')),
              ('hsp', 'epicardial HSP CC vs commercial', 'HSP', ('ours', 'stmem', 'mcma', 'lmmse', 'laplacian', 'tps', 'vconly'))]
    for kind, ylab, q, meths in panels:
        for split, lab in (('test_internal', 'internal'), ('test_external', 'external')):
            fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
            pm = pmap_of(split)
            for i, m in enumerate(meths):
                c, mk, name = STY[m]
                y = [M.get((m, vc, split, q)) if M.get((m, vc, split, q)) is not None else np.nan for vc in VCS]
                if P is not None:
                    ci = [boot_ci({sid: d.get(q) for sid, d in P.get((m, vc, split), {}).items()}, pm) for vc in VCS]
                    yerr = [[max(yy - lo, 0) if np.isfinite(lo) else 0 for yy, (_, lo, _h) in zip(y, ci)],
                            [max(hi - yy, 0) if np.isfinite(hi) else 0 for yy, (_, _l, hi) in zip(y, ci)]]
                    ax.errorbar(np.array(VCS) * 2.0 ** offs[i], y, yerr=yerr, fmt=mk, color=c, lw=2.0 if m == 'ours' else 1.4,
                                ms=6 if m == 'ours' else 4.5, capsize=2, elinewidth=0.9, capthick=0.9, label=name)
                else:
                    ax.plot(VCS, y, mk, color=c, lw=2.0 if m == 'ours' else 1.4, ms=6 if m == 'ours' else 4.5, label=name)
            ax.set_xscale('log', base=2); ax.set_xticks(VCS); ax.set_xticklabels(VCS)
            ax.set_xlabel('number of visible electrodes $N$'); ax.set_ylabel(ylab); ax.set_title(f'{lab} dataset', fontsize=10)
            ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower right' if kind == 'bsp' else 'best')
            out = os.path.join(FIG_DIR_V2, f'fig_baselines_{kind}_{lab}{lay}_v2.png'); fig.savefig(out, dpi=FIG_DPI); plt.close(fig); print('wrote', os.path.basename(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='commercial', choices=['commercial', 'src2_allvalid'])
    ap.add_argument('--layout', default='template', choices=['template', 'kmeans'])
    a = ap.parse_args(); REF, LAYOUT = a.ref, a.layout
    P, M = load_everything()
    means_tables(M)
    res = sig_table(P)
    if REF == 'commercial':
        figures(M, P)
    json.dump({'means': {f'{m}|{vc}|{s}|{q}': v for (m, vc, s, q), v in M.items()}, 'sig': res},
              open(assert_v2(os.path.join(V2_GENERATED, f'results{sfx_lay()}_v2.json')), 'w'), indent=1)
    print('done', REF, LAYOUT)
