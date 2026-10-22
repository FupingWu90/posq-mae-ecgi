"""Before/after comparison of the generated aggregates (JSON values and table lines) against a backup snapshot, e.g.
  python baselines_analyses/diff_generated.py --backup baselines_analyses/generated/_backup_pre_dedup_20261021
Prints counts per file, the largest absolute changes and the changed table lines.  Key paths are printed only for the
aggregate-only JSON files (whitelist) and never when a path segment looks like a sample id."""
import os, sys, json, glob, argparse, difflib
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(V2, 'baselines_analyses', 'generated')
SAFE = ('results', 'patient_level', 'n16_paired', 'gate_sens', 'seed_tables', 'region_deploy', 'dense_field', 'heavy_loss', 'holdout_replication',
        'sim_tables', 'data_scaling', 'index_jitter', 'lam_recal', 'baselines_two_layouts', 'deploy_robustness', 'amplitude_tables',
        'inverse_agreement', 'site_error', 'propagation', 'maps_numbers', 'external_strata', 'cohort_clinical', 'layout_shift', 'ours16_vs_allvalid')


def flat(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f'{path}[{i}]')
    else:
        yield path, o


def sid_like(path):
    return any(('__' in seg and len(seg) > 12) for seg in path.split('/'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--backup', required=True); ap.add_argument('--top', type=int, default=6); a = ap.parse_args()
    bk = a.backup if os.path.isabs(a.backup) else os.path.join(V2, a.backup)
    bk_gen = os.path.join(bk, 'baselines_analyses', 'generated') if os.path.isdir(os.path.join(bk, 'baselines_analyses')) else bk
    tot_changed = 0; n_files_changed = 0
    for f in sorted(glob.glob(os.path.join(bk_gen, '*.json'))):
        cur = os.path.join(GEN, os.path.basename(f))
        if not os.path.exists(cur):
            print(f'{os.path.basename(f)}: missing now'); continue
        try:
            A = dict(flat(json.load(open(f)))); B = dict(flat(json.load(open(cur))))
        except Exception as e:
            print(f'{os.path.basename(f)}: unreadable ({e})'); continue
        changed = []
        for k in set(A) | set(B):
            va, vb = A.get(k), B.get(k)
            if va == vb:
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                changed.append((abs(vb - va), k, va, vb))
            else:
                changed.append((float('inf'), k, va, vb))
        if not changed:
            continue
        n_files_changed += 1; tot_changed += len(changed)
        name = os.path.basename(f); safe = name.startswith(SAFE)
        num = [c for c in changed if c[0] != float('inf')]
        print(f'{name}: {len(changed)} values changed ({len(A)} -> {len(B)} entries); max |d| = {max([c[0] for c in num], default=0):.4g}')
        if safe:
            for d, k, va, vb in sorted(changed, key=lambda c: -c[0] if c[0] != float('inf') else -1e300)[:a.top]:
                if sid_like(k):
                    continue
                fa = f'{va:.5g}' if isinstance(va, float) else str(va); fb = f'{vb:.5g}' if isinstance(vb, float) else str(vb)
                print(f'    {k[:90]}: {fa[:40]} -> {fb[:40]}')
    print(f'JSON: {n_files_changed} files, {tot_changed} values changed')
    n_tab = 0
    for f in sorted(glob.glob(os.path.join(bk_gen, '*.tex'))):
        cur = os.path.join(GEN, os.path.basename(f))
        if not os.path.exists(cur):
            continue
        a_l, b_l = open(f).read().split('\n'), open(cur).read().split('\n')
        if a_l == b_l:
            continue
        n_tab += 1
        d = [l for l in difflib.unified_diff(a_l, b_l, lineterm='', n=0) if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        print(f'{os.path.basename(f)}: {sum(1 for l in d if l[0] == "-")} lines changed')
        for l in d[:2 * a.top]:
            print('    ' + l[:170])
    print(f'TEX: {n_tab} tables changed')


if __name__ == '__main__':
    main()
