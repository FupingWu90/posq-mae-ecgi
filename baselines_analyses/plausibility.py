"""Reference-free physiological plausibility of epicardial activation maps: apparent conduction velocity on
the epicardial mesh edges, CV_ij = d_ij / |AT_i - AT_j| (cm / ms * 10 = m/s), and the fraction of edges whose
apparent CV is below a physiological lower bound (0.2 m/s; 0.1 m/s as a stricter variant).  Computed for the
VC-only inverse (AT1, cache), the completed field of a run (AT4, inverse_evaluation/at_maps.py), the all-valid inverse
(AT2, refs cache) and the commercial maps (ATcom), on (i) all mesh edges and (ii) the edges inside the
commercial QRS-good vertex set (the reference-gated set used for the error metrics).  Paired Wilcoxon tests
between sources on the same recordings.  Aggregates only.
  python baselines_analyses/plausibility.py --id LOC3 [--splits test_internal test_external] [--layouts template kmeans]
  -> baselines_analyses/generated/plausibility_<ID>.json ; results/plausibility_<ID>.md
"""
import os, sys, json, argparse, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
from paths import V1_ROOT, V2_GENERATED, assert_v2
from cache_references import cache_lookup, vc_tag
from mesh_io import load_surface, find_patient_files
from dataset import collect_split

VMIN = (0.2, 0.1)      # m/s
SOURCES = ['vconly', 'ours', 'v1', 'allvalid', 'commercial']
V1_PROP = os.path.join(V1_ROOT, 'baselines_analyses', 'generated', 'propagation')   # previous release: per-vertex AT of the v1 model, k-means layout


def edges_of(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e.sort(axis=1)
    return np.unique(e, axis=0)


def rings(E, n, k=2):
    """k-ring neighbourhoods (lists of vertex indices) from the edge list."""
    adj = [set() for _ in range(n)]
    for i, j in E:
        adj[i].add(j); adj[j].add(i)
    out = []
    for v in range(n):
        front, seen = {v}, {v}
        for _ in range(k):
            front = set().union(*(adj[u] for u in front)) - seen
            seen |= front
        out.append(np.fromiter(seen - {v}, dtype=np.int64))
    return out


def vertex_cv(T, V, nbrs, keep=None, min_nb=5):
    """Apparent CV per vertex from a least-squares plane fit of AT over the neighbourhood (m/s)."""
    T = np.asarray(T, float); n = len(T); cv = np.full(n, np.nan)
    for v in range(n):
        if keep is not None and not keep[v]:
            continue
        nb = nbrs[v]
        nb = nb[np.isfinite(T[nb])] if keep is None else nb[np.isfinite(T[nb]) & keep[nb]]
        if nb.size < min_nb or not np.isfinite(T[v]):
            continue
        X = V[nb] - V[v]; y = T[nb] - T[v]
        g, *_ = np.linalg.lstsq(X, y, rcond=None)                      # ms per cm (3-D gradient of the local fit)
        gn = np.linalg.norm(g)
        cv[v] = np.inf if gn < 1e-9 else 10.0 / gn                     # m/s
    return cv


def edge_stats(T, E, d_cm, keep=None):
    """T: (n_h,) ms; keep: optional vertex mask. Returns dict of fractions / medians over the kept edges."""
    T = np.asarray(T, float)
    ok = np.isfinite(T[E[:, 0]]) & np.isfinite(T[E[:, 1]])
    if keep is not None:
        ok &= keep[E[:, 0]] & keep[E[:, 1]]
    if ok.sum() < 20:
        return None
    dt = np.abs(T[E[ok, 0]] - T[E[ok, 1]]); dd = d_cm[ok]
    with np.errstate(divide='ignore'):
        cv = np.where(dt > 0, 10.0 * dd / np.maximum(dt, 1e-9), np.inf)       # m/s
    out = {'n_edges': int(ok.sum()), 'frac_simultaneous': float(np.mean(dt == 0))}
    for v in VMIN:
        out[f'frac_slow{v:g}'] = float(np.mean(cv < v))
    fin = np.isfinite(cv)
    out['median_cv'] = float(np.median(cv[fin])) if fin.any() else float('nan')
    out['frac_jump10'] = float(np.mean(dt > 10.0))      # AT jump > 10 ms across one edge
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='LOC3'); ap.add_argument('--vc', type=int, default=16)
    ap.add_argument('--splits', nargs='+', default=['test_internal', 'test_external'])
    ap.add_argument('--layouts', nargs='+', default=['template', 'kmeans'])
    a = ap.parse_args()
    buckets, _ = collect_split(seed=42)
    rows = []; mesh_cache = {}; V_cache = {}; ring_cache = {}
    for split in a.splits:
        recs = {r[0]: r for r in buckets[split] if r[2]}
        for lay in a.layouts:
            tag = vc_tag(a.vc, 'template', 'greedy', 42) if lay == 'template' else vc_tag(a.vc, 'cluster', 'kmean', 42)
            adir = os.path.join(V2_GENERATED, 'atmaps', a.id, f'{split}_{lay}')
            for f in sorted(glob.glob(os.path.join(adir, '*.npz'))):
                z = np.load(f, allow_pickle=False)
                if not bool(z['valid']):
                    continue
                sid_safe = os.path.basename(f)[:-4]
                sid = next((s for s in recs if s.replace('/', '_') == sid_safe), None)
                if sid is None or not _dedup_ok(sid):
                    continue
                gdir = os.path.dirname(recs[sid][2])
                if gdir not in mesh_cache:
                    V, F, _ = load_surface(find_patient_files(gdir)['heart'])
                    V = np.asarray(V, float); E = edges_of(np.asarray(F)); d = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
                    mesh_cache[gdir] = (E, d); V_cache[gdir] = V
                E, d = mesh_cache[gdir]
                rp = cache_lookup(split, 'refs', sid_safe + '.npz'); vp = cache_lookup(split, tag, sid_safe + '.npz')
                if rp is None or vp is None:
                    continue
                ref = np.load(rp, allow_pickle=False); vcz = np.load(vp, allow_pickle=False)
                if 'ATcom' not in ref.files or 'AT2' not in ref.files or 'AT1' not in vcz.files:
                    continue
                maps = {'vconly': vcz['AT1'], 'ours': z['AT4'], 'allvalid': ref['AT2'], 'commercial': ref['ATcom']}
                if lay == 'kmeans':
                    vp1 = os.path.join(V1_PROP, split, sid_safe + '.npz')
                    if os.path.exists(vp1):
                        z1 = np.load(vp1, allow_pickle=True)
                        if 'at_ours_v' in z1.files:
                            maps['v1'] = z1['at_ours_v']
                gate = ref['qrs_goodcom'].astype(bool)
                if gdir not in ring_cache:
                    ring_cache[gdir] = (rings(E, V_cache[gdir].shape[0], 2), rings(E, V_cache[gdir].shape[0], 4))
                for src, T in maps.items():
                    for setname, keep in (('all', None), ('gated', gate)):
                        st = edge_stats(T, E, d, keep)
                        if st:
                            for ringk, nbrs in (('', ring_cache[gdir][0]), ('4', ring_cache[gdir][1])):
                                cvv = vertex_cv(T, V_cache[gdir], nbrs, keep)
                                fin = np.isfinite(cvv)
                                for v in VMIN:
                                    st[f'vfrac{ringk}_slow{v:g}'] = float(np.mean(cvv[fin] < v)) if fin.any() else float('nan')
                                st[f'vmedian{ringk}_cv'] = float(np.median(cvv[fin & (cvv < np.inf)])) if (fin & (cvv < np.inf)).any() else float('nan')
                                st[f'vfrac{ringk}_plateau'] = float(np.mean(np.isinf(cvv[fin]))) if fin.any() else float('nan')
                            rows.append(dict(split=split, layout=lay, sid=sid, source=src, vset=setname, **st))
    df = pd.DataFrame(rows)
    out = {'meta': dict(id=a.id, vc=a.vc, vmin=VMIN, definition='apparent CV per mesh edge = 10*d_cm/|dAT_ms| (m/s); '
                        'frac_slow = fraction of edges with CV below the bound; gated = both vertices in the commercial QRS-good set'),
           'cohort': {}, 'paired': {}}
    md = [f'# Physiological plausibility of activation maps ({a.id}, N={a.vc})', '',
          'Fraction of epicardial mesh edges whose apparent conduction velocity is below 0.2 m/s (mean over recordings; lower is more plausible),',
          'median apparent CV, and the fraction of edges with an AT jump > 10 ms. Paired Wilcoxon on the same recordings.', '']
    for split in a.splits:
        for lay in a.layouts:
            for vset in ('gated', 'all'):
                sub = df[(df.split == split) & (df.layout == lay) & (df.vset == vset)]
                if sub.empty:
                    continue
                key = f'{split}|{lay}|{vset}'
                g = sub.groupby('source')[['vfrac_slow0.2', 'vfrac_plateau', 'vmedian_cv', 'vfrac4_slow0.2', 'vfrac4_plateau', 'vmedian4_cv', 'frac_jump10', 'frac_simultaneous']].mean()
                out['cohort'][key] = {s: {k: float(v) for k, v in g.loc[s].items()} for s in g.index}
                out['cohort'][key]['n'] = int(sub.sid.nunique())
                md += [f'## {split}, {lay} layout, {vset} edges (n = {sub.sid.nunique()} recordings)', '',
                       '| source | 2-ring (~1 cm): CV < 0.2 | 2-ring: plateau | 2-ring median CV | 4-ring (~2 cm): CV < 0.2 | 4-ring: plateau | 4-ring median CV | edge jump > 10 ms | simultaneous edges |', '|---|---|---|---|---|---|---|---|---|']
                for s in SOURCES:
                    if s in g.index:
                        r = g.loc[s]
                        md.append(f"| {s} | {r['vfrac_slow0.2']*100:.1f}% | {r['vfrac_plateau']*100:.1f}% | {r['vmedian_cv']:.2f} | {r['vfrac4_slow0.2']*100:.1f}% | {r['vfrac4_plateau']*100:.1f}% | {r['vmedian4_cv']:.2f} | {r['frac_jump10']*100:.1f}% | {r['frac_simultaneous']*100:.1f}% |")
                for col, scale in (('vfrac_slow0.2', '2-ring, about 1 cm'), ('vfrac4_slow0.2', '4-ring, about 2 cm'), ('frac_jump10', 'edge jumps > 10 ms')):
                    piv = sub.pivot_table(index='sid', columns='source', values=col)
                    lines = []
                    for x, y in (('ours', 'vconly'), ('ours', 'allvalid'), ('ours', 'commercial'), ('ours', 'v1'), ('allvalid', 'commercial')):
                        if x in piv.columns and y in piv.columns:
                            d_ = (piv[x] - piv[y]).dropna()
                            p = float(wilcoxon(d_).pvalue) if (d_ != 0).any() else 1.0
                            out['paired'][f'{key}|{col}|{x}-{y}'] = dict(med_diff=float(d_.median()), p=p, n=int(d_.size))
                            lines.append(f'{x} − {y}: median {d_.median()*100:+.2f} pp, p = {p:.2g}')
                    md += ['', f'Paired ({scale}; slow = CV < 0.2 m/s): ' + '; '.join(lines)]
                md.append('')
    jp = assert_v2(os.path.join(V2_GENERATED, f'plausibility_{a.id}.json')); json.dump(out, open(jp, 'w'), indent=1)
    mp = assert_v2(os.path.join(V2, 'results', f'plausibility_{a.id}.md')); open(mp, 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md)); print('wrote', jp, mp)


if __name__ == '__main__':
    main()
