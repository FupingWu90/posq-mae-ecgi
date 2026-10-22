"""Content de-duplication of the internal test set (2026-10-21; collaborators' review: the paper counts 275 unique internal
test recordings, the raw list holds 276 files).  Rule: bsp_reconstruction/dedup.py (one recording per content key, the first in sorted
sample-id order); identity from the frozen v1 baselines_analyses/common._identity (md5 per file).
  python baselines_analyses/dedup_rule.py --build   -> bsp_reconstruction/dedup_exclude.json (md5 digests of the dropped ids; counts per split)
  python baselines_analyses/dedup_rule.py           -> builds the file if missing, then removes the rows of the dropped recordings from
                                          every per-recording CSV under baselines_analyses/generated (percase_*, classical
                                          percase_*.csv, dense_field_*, lam_recal_*, site_error) and results/val_percase;
                                          originals backed up once in generated/_backup_pre_dedup_20261021/ together
                                          with a snapshot of the generated tables / JSON (for the before/after diff).
Idempotent.  Aggregates only: no sample id, folder or file name of a recording is printed."""
import os, sys, csv, glob, json, re, shutil, argparse, time
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'baselines_analyses'))
from paths import V2_GENERATED, V1_ROOT
import dedup
BACKUP = os.path.join(V2_GENERATED, '_backup_pre_dedup_20261021')
SPLIT_RE = re.compile(r'(test_internal|test_external|holdout|val)')


def build():
    from dataset import collect_split
    sys.path.insert(0, os.path.join(V1_ROOT, 'baselines_analyses'))
    from common import _identity
    b, _ = collect_split(seed=42)
    md5, _pid = _identity()
    out = {'rule': 'one recording per content key (first in sorted sample-id order); keys are md5 digests of the sample id '
                   'and of its "/"-safe form', 'built': time.strftime('%Y-%m-%d'), 'exclude': {}, 'counts': {}}
    subsets = {}
    for split in ('test_internal', 'val'):
        recs = sorted(b[split], key=lambda r: r[0])
        seen, drop = set(), []
        for r in recs:
            k = md5.get(r[1], r[1])
            (drop.append(r[0]) if k in seen else seen.add(k))
        out['exclude'][split] = sorted({dedup.key(s) for s in drop} | {dedup.key(s.replace('/', '_')) for s in drop})
        out['counts'][split] = {'raw': len(recs), 'unique': len(seen), 'dropped': len(drop)}
        # membership of the dropped recordings in the evenly spaced sub-samples used by the subset experiments
        with_t = [r for r in b[split] if r[2]]
        for n in (60, 80):
            idx = np.linspace(0, len(with_t) - 1, n).astype(int)
            subsets[f'{split}:n{n}'] = sum(1 for i in idx if with_t[i][0] in drop)
        subsets[f'{split}:example_a390db55'] = sum(1 for s in drop if 'a390db55' in s)
    json.dump(out, open(dedup._FILE, 'w'), indent=1)
    print('counts:', json.dumps(out['counts']))
    print('dropped recordings inside the evenly spaced sub-samples / the paper example:', json.dumps(subsets))
    dedup._KEYS = None


def _rows(path):
    return list(csv.DictReader(open(path)))


def _write(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def _backup(path):
    rel = os.path.relpath(path, V2); dst = os.path.join(BACKUP, rel)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(path, dst)


def patch_csv(path, sid_col='sid'):
    rows = _rows(path)
    if not rows or sid_col not in rows[0]:
        return 0, len(rows)
    keep = [r for r in rows if dedup.ok(r[sid_col])]
    if len(keep) != len(rows):
        _backup(path); _write(path, keep, list(rows[0].keys()))
    return len(rows) - len(keep), len(rows)


def snapshot():
    """one-time copy of the generated tables / JSON / CSV / xlsx (top level) for the before/after diff."""
    n = 0
    for f in sorted(glob.glob(os.path.join(V2_GENERATED, '*.*'))):
        if os.path.isfile(f) and f.rsplit('.', 1)[-1] in ('tex', 'json', 'csv', 'xlsx', 'md'):
            dst = os.path.join(BACKUP, os.path.relpath(f, V2))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(f, dst); n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); a = ap.parse_args()
    if a.build or not os.path.exists(dedup._FILE):
        build()
    if a.build:
        return
    if not dedup.excluded():
        print('nothing excluded'); return
    print('snapshot of generated files:', snapshot())
    report = {}
    groups = {'percase_eval': [p for sub in ('percase_candidate', 'percase_template', 'percase_loss', 'percase_region', 'percase_holdout')
                               for p in sorted(glob.glob(os.path.join(V2_GENERATED, sub, '*.csv')))],
              'results/val_percase': sorted(glob.glob(os.path.join(V2, 'results', 'val_percase', '*.csv'))),
              'classical_template': sorted(glob.glob(os.path.join(V2_GENERATED, 'classical_template', 'percase_*.csv'))),
              'dense_field': [p for sub in ('dense_field_template', 'dense_field_kmeans') for p in sorted(glob.glob(os.path.join(V2_GENERATED, sub, '*.csv')))],
              'lam_recal': sorted(glob.glob(os.path.join(V2_GENERATED, 'lam_recal_*', '*.csv'))),
              'site_error': sorted(glob.glob(os.path.join(V2_GENERATED, 'site_error', '*.csv')))}
    for g, paths in groups.items():
        files = removed = 0
        for p in paths:
            n, N = patch_csv(p); files += 1 if n else 0; removed += n
        report[g] = (len(paths), files, removed)
    for g, (n_all, n_mod, n_rem) in report.items():
        print(f'{g}: {n_all} files, {n_mod} modified, {n_rem} rows removed')
    # residual: engine BSP metrics (test_compare.json of each run) are means over the raw file list; quantify on LOC3
    p = os.path.join(V2_GENERATED, '_backup_pre_dedup_20261021', 'baselines_analyses', 'generated', 'percase_candidate', 'LOC3_vc16_test_internal_template.csv')
    if os.path.exists(p):
        rows = [r for r in _rows(p) if r['source'] == 'src4_recon_valid' and r['ref'] == 'commercial']
        cc_all = np.array([float(r['bsp_cc']) for r in rows if r.get('bsp_cc') not in ('', None)])
        cc_keep = np.array([float(r['bsp_cc']) for r in rows if r.get('bsp_cc') not in ('', None) and dedup.ok(r['sid'])])
        print(f'LOC3 internal test BSP CC (per-recording, template): mean over raw {len(cc_all)} = {cc_all.mean():.5f}; over kept {len(cc_keep)} = {cc_keep.mean():.5f}; shift {cc_keep.mean() - cc_all.mean():+.5f}')


if __name__ == '__main__':
    main()
