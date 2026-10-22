"""Repolarisation gating rule, applied uniformly (2026-10-20, user decision): a recording contributes RT / ARI values
against a reference only if that reference's T-gate holds at least three vertices (T_MIN).  The all-valid-vs-commercial
values (results_v2.load_allvalid) always used this rule; the per-method evaluation computed RT on 1-2 vertices for
three external recordings, which made the paired n differ between rows.  This tool blanks RT/ARI cells of the existing
per-recording CSVs where the rule fails (idempotent; originals backed up in generated/_backup_pre_tgate_20261020/),
and results_v2 / eval_downstream / classical_baselines_template now apply the same rule at load / evaluation time.
  python baselines_analyses/tgate_rule.py            -> patches the CSVs under baselines_analyses/generated and results/val_percase
Aggregates only."""
import os, sys, csv, glob, re
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'inverse_evaluation')); sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from cache_references import CACHE_ROOT, CACHE_ROOT_V1
T_MIN = 3
GEN = os.path.join(V2, 'baselines_analyses', 'generated')
_CACHE = {}


def gate_counts(split):
    """{sid_safe: (n_T_commercial, n_T_allvalid)} from the reference cache (v2 first, then the frozen v1 cache)."""
    if split in _CACHE:
        return _CACHE[split]
    files = {}
    for root in (CACHE_ROOT, CACHE_ROOT_V1):
        for f in glob.glob(os.path.join(root, split, 'refs', '*.npz')):
            files.setdefault(os.path.basename(f), f)
    out = {}
    from dedup import ok as _dedup_ok   # one recording per content key (bsp_reconstruction/dedup.py; 2026-10-21)
    for name, f in files.items():
        if not _dedup_ok(name[:-4]):
            continue
        with np.load(f, allow_pickle=False) as z:
            nc = int(z['t_goodcom'].astype(bool).sum()) if 't_goodcom' in z.files else -1
            n2 = int(z['t_good2'].astype(bool).sum()) if 't_good2' in z.files else -1
        out[name[:-4]] = (nc, n2)
    _CACHE[split] = out
    return out


def ok(split, sid, ref='commercial'):
    """True if the recording's T-gate for `ref` holds >= T_MIN vertices (unknown recordings pass)."""
    g = gate_counts(split).get(sid.replace('/', '_'))
    if g is None:
        return True
    n = g[0] if ref == 'commercial' else g[1]
    return n < 0 or n >= T_MIN


EVAL_COLS = ['RT_mae', 'RT_mdae', 'RT_cc', 'ARI_mae', 'ARI_mdae', 'ARI_cc']
SPLIT_RE = re.compile(r'(test_internal|test_external|holdout|val)')


def patch_eval_csv(path, split):
    rows = list(csv.DictReader(open(path))); n = 0
    if not rows or 'RT_mae' not in rows[0]:
        return 0, len(rows)
    for r in rows:
        if r.get('ref') in ('commercial', 'src2_allvalid') and not ok(split, r['sid'], r['ref']):
            if any(r.get(c) not in ('', None, 'nan') for c in EVAL_COLS):
                n += 1
            for c in EVAL_COLS:
                if c in r:
                    r[c] = ''
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n, len(rows)


def patch_classical_csv(path):
    rows = list(csv.DictReader(open(path))); n = 0
    for r in rows:
        for cols, ref in ((('rt_com', 'ari_com'), 'commercial'), (('rt_src2', 'ari_src2'), 'src2_allvalid')):
            if not ok(r['split'], r['sid'], ref):
                if any(r.get(c) not in ('', None, 'nan') for c in cols):
                    n += 1
                for c in cols:
                    if c in r:
                        r[c] = ''
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n, len(rows)


def patch_dense_csv(path, split):
    rows = list(csv.DictReader(open(path))); n = 0
    for r in rows:
        if not ok(split, r['sid'], 'commercial'):
            if r.get('rt_mae') not in ('', None, 'nan'):
                n += 1
            for c in ('rt_mae', 'rt_cc'):
                if c in r:
                    r[c] = ''
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n, len(rows)


def main():
    report = {}
    for sub in ('percase_candidate', 'percase_template', 'percase_loss', 'percase_region', 'percase_holdout'):
        tot = cnt = 0
        for p in sorted(glob.glob(os.path.join(GEN, sub, '*.csv'))):
            m = SPLIT_RE.search(os.path.basename(p))
            if not m:
                continue
            n, N = patch_eval_csv(p, m.group(1)); tot += n; cnt += 1
        report[sub] = (cnt, tot)
    tot = cnt = 0
    for p in sorted(glob.glob(os.path.join(V2, 'results', 'val_percase', '*.csv'))):
        n, N = patch_eval_csv(p, 'val'); tot += n; cnt += 1
    report['results/val_percase'] = (cnt, tot)
    tot = cnt = 0
    for p in sorted(glob.glob(os.path.join(GEN, 'classical_template', 'percase_*.csv'))):
        n, N = patch_classical_csv(p); tot += n; cnt += 1
    report['classical_template'] = (cnt, tot)
    tot = cnt = 0
    for sub in ('dense_field_template', 'dense_field_kmeans'):
        for p in sorted(glob.glob(os.path.join(GEN, sub, '*.csv'))):
            m = SPLIT_RE.search(os.path.basename(p))
            if m:
                n, N = patch_dense_csv(p, m.group(1)); tot += n; cnt += 1
    report['dense_field'] = (cnt, tot)
    for split in ('test_internal', 'test_external', 'holdout', 'val'):
        g = gate_counts(split)
        print(f'{split}: {len(g)} reference entries; T-gate < {T_MIN} vertices: commercial {sum(1 for a, b in g.values() if 0 <= a < T_MIN)}, all-valid {sum(1 for a, b in g.values() if 0 <= b < T_MIN)}')
    for k, (c, n) in report.items():
        print(f'{k}: {c} files, {n} rows blanked')


if __name__ == '__main__':
    main()
