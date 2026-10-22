"""Classical baselines (Laplace / TPS / LMMSE): fixed TEMPLATE layout (v2 cache) versus the
per-recording K-MEANS layout (frozen v1 shards + fidelity caches), doubling grid, both test
cohorts.  Cohort means, paired per-recording median difference and Wilcoxon p on the common
recordings.  Prints aggregates only; writes generated/classical_template/compare_layouts.json.
"""
import csv, glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
from paths import V1_GENERATED, V2_GENERATED   # noqa: E402

VCS = [4, 8, 16, 32, 64]; METHODS = ['laplacian', 'tps', 'lmmse']
SPLITS = ['test_internal', 'test_external']
KEYS = ['at_com', 'rt_com', 'ari_com', 'at_src2', 'rt_src2', 'ari_src2', 'bsp_cc', 'hsp_cc', 'hsp_mae', 'hsp_cc_src2']
OUT = os.path.join(V2_GENERATED, 'classical_template')


def load_template(split):
    d = {}
    for f in glob.glob(os.path.join(OUT, 'cache_template', split, '*.json')):
        j = json.load(open(f))
        if 'failed' in j:
            continue
        for m in METHODS:
            for vc, e in j['results'].get(m, {}).items():
                d[(j['sid'], m, int(vc))] = {k: e.get(k) for k in KEYS}
    return d


def load_kmeans(split):
    d = {}
    for f in glob.glob(os.path.join(V1_GENERATED, 'interp', f'{split}_shard*.csv')):
        for r in csv.DictReader(open(f)):
            key = (r['sid'], r['method'], int(r['vc']))
            if key[2] not in VCS:
                continue
            d.setdefault(key, {})
            for k in KEYS[:6]:
                try:
                    d[key][k] = float(r[k])
                except (KeyError, ValueError, TypeError):
                    pass
    for sub, keys in (('cache', ('bsp_cc', 'hsp_cc', 'hsp_mae')), ('cache_src2', ('hsp_cc_src2',))):
        for f in glob.glob(os.path.join(V1_GENERATED, 'fidelity_baselines', sub, split, '*.json')):
            j = json.load(open(f))
            if 'failed' in j:
                continue
            for m in METHODS:
                for vc, e in j['results'].get(m, {}).items():
                    key = (j['sid'], m, int(vc))
                    d.setdefault(key, {})
                    for k in keys:
                        if e.get(k) is not None:
                            d[key][k] = e[k]
    return d


def main():
    out = {}
    for split in SPLITS:
        T, K = load_template(split), load_kmeans(split)
        print(f'\n== {split}: template entries {len(T)}, k-means entries {len(K)} ==')
        print(f"{'method':10s} {'N':>3s} | {'AT tmpl':>8s} {'AT kmean':>8s} {'med.diff':>8s} {'p':>8s} | "
              f"{'RT tmpl':>8s} {'RT kmean':>8s} {'p':>8s} | {'HSPcc t':>8s} {'HSPcc k':>8s} {'p':>8s} | {'BSPcc t':>8s} {'BSPcc k':>8s} | n")
        for m in METHODS:
            for vc in VCS:
                sids = sorted({s for (s, mm, v) in T if mm == m and v == vc} & {s for (s, mm, v) in K if mm == m and v == vc})
                cell = {'n_common': len(sids)}
                for k in KEYS:
                    a = np.array([T[(s, m, vc)].get(k) for s in sids], float)
                    b = np.array([K[(s, m, vc)].get(k) for s in sids], float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 5:
                        continue
                    dd = a[ok] - b[ok]
                    try:
                        p = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                    except ValueError:
                        p = float('nan')
                    cell[k] = dict(template=round(float(a[ok].mean()), 4), kmeans=round(float(b[ok].mean()), 4),
                                   median_diff=round(float(np.median(dd)), 4), p=p, n=int(ok.sum()))
                out[f'{split}|{m}|{vc}'] = cell
                g = lambda k, f: cell.get(k, {}).get(f, float('nan'))
                print(f"{m:10s} {vc:3d} | {g('at_com','template'):8.2f} {g('at_com','kmeans'):8.2f} {g('at_com','median_diff'):8.2f} {g('at_com','p'):8.2g} | "
                      f"{g('rt_com','template'):8.2f} {g('rt_com','kmeans'):8.2f} {g('rt_com','p'):8.2g} | "
                      f"{g('hsp_cc','template'):8.4f} {g('hsp_cc','kmeans'):8.4f} {g('hsp_cc','p'):8.2g} | "
                      f"{g('bsp_cc','template'):8.4f} {g('bsp_cc','kmeans'):8.4f} | {cell['n_common']}")
    json.dump(out, open(os.path.join(OUT, 'compare_layouts.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'compare_layouts.json'))


if __name__ == '__main__':
    main()
