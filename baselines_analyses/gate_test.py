"""Phase-2 adoption gate (the architecture plan §6, G1/G2) for a candidate configuration against the v1 final
model, SAME training seed, on BOTH test cohorts and under BOTH deployment layouts (user rule 2026-10-06).
  python baselines_analyses/gate_test.py LOC3 [LOC3B ...]
Inputs: candidate per-recording CSVs  baselines_analyses/generated/percase_candidate/<ID>_vc16_<split>_<layout>.csv
        v1 per-recording CSVs         template: generated/percase_template/ours-v1_vc16_<split>.csv
                                      k-means : v1 generated/vc_percase/vc16_<split>.csv
        engine validation MAE         <run>/val_engine.json  vs  results/v1_reference/wide_val_engine.json
        cost                          training_stats.json of both runs
Outputs: results/gate_<ID>.md + baselines_analyses/generated/gate_<ID>.json ; prints aggregates only.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V1_GENERATED, V2_GENERATED, V1_RUNS, V2_RUNS, assert_v2

SPLITS = ['test_internal', 'test_external']; LAYOUTS = ['template', 'kmeans']
METRICS = ['AT_mae', 'RT_mae', 'ARI_mae', 'hsp_cc', 'AT_cc', 'bsp_cc']
V1_RUN = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')
TOL_MAE, G2_MAE, TOL_MS = 0.002, 0.003, 0.3


def v1_csv(split, layout):
    return (os.path.join(V2_GENERATED, 'percase_template', f'ours-v1_vc16_{split}.csv') if layout == 'template'
            else os.path.join(V1_GENERATED, 'vc_percase', f'vc16_{split}.csv'))


def load(path, ref):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df[(df.source == 'src4_recon_valid') & (df.ref == ref)].set_index('sid')


def paired(a, b):
    """a, b: DataFrames indexed by sid (candidate, v1). Returns metric -> stats."""
    out = {}
    common = a.index.intersection(b.index); a, b = a.loc[common], b.loc[common]
    pat = pd.Series([s.split('__')[1] for s in common], index=common)
    for m in METRICS:
        if m not in a.columns or m not in b.columns:
            continue
        x = pd.to_numeric(a[m], errors='coerce'); y = pd.to_numeric(b[m], errors='coerce'); ok = x.notna() & y.notna()
        if ok.sum() < 10:
            continue
        d = (x - y)[ok]
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        per_pat = d.groupby(pat[ok]).mean()
        better = int((per_pat < 0).sum()) if m.endswith('mae') else int((per_pat > 0).sum())
        out[m] = dict(cand=float(x[ok].mean()), v1=float(y[ok].mean()), med_diff=float(d.median()), p=p, n=int(ok.sum()),
                      patients_better=better, n_patients=int(per_pat.size))
    return out


def main(ids):
    v1e = json.load(open(os.path.join(V2, 'results', 'v1_reference', 'wide_val_engine.json')))
    v1s = json.load(open(os.path.join(V1_RUN, 'training_stats.json')))
    for cid in ids:
        run = glob.glob(os.path.join(V2_RUNS, 'data-cohortA+cohortB_vs_external', f'{cid}__*'))[0]
        ce = json.load(open(os.path.join(run, 'val_engine.json'))); cs = json.load(open(os.path.join(run, 'training_stats.json')))
        res = {'id': cid, 'run': os.path.basename(run), 'val_mae': {}, 'cost': {}, 'test': {}}
        md = [f'# Adoption gate: {cid} vs v1 (same seed 42; both test cohorts; both deployment layouts)', '',
              '## Validation completion MAE (engine; selection metric)', '', '| layout | candidate | v1 | diff | G1 (<= +0.002) | G2 (<= -0.003) |', '|---|---|---|---|---|---|']
        g1_val, g2_val = True, True
        for lay in LAYOUTS:
            c, v = ce[f'val_{lay}']['mae'], v1e[f'val_{lay}']['mae']; d = c - v
            res['val_mae'][lay] = dict(cand=c, v1=v, diff=d)
            g1_val &= d <= TOL_MAE; g2_val &= d <= -G2_MAE
            md.append(f'| {lay} | {c:.4f} | {v:.4f} | {d:+.4f} | {"yes" if d <= TOL_MAE else "NO"} | {"yes" if d <= -G2_MAE else "no"} |')
        tr, ti = cs.get('elapsed_train_hours'), cs.get('infer_ms_per_sample'); vtr, vti = v1s.get('elapsed_train_hours'), v1s.get('infer_ms_per_sample')
        res['cost'] = dict(train_h=tr, v1_train_h=vtr, infer_ms=ti, v1_infer_ms=vti, params=cs.get('params'), v1_params=v1s.get('params'))
        speed = (vtr / tr if tr and vtr else None, vti / ti if ti and vti else None)
        md += ['', '## Cost', '', f'training {tr} h vs {vtr} h ({speed[0]:.1f}x), inference {ti} ms vs {vti} ms per recording ({speed[1]:.1f}x), '
               f'parameters {cs.get("params")} vs {v1s.get("params")}', '', '## Test cohorts, per-recording paired comparison (candidate minus v1; commercial reference)', '']
        g1_test, g2_at, g2_rt, worse = True, True, True, []
        for lay in LAYOUTS:
            md += [f'### layout: {lay}', '', '| cohort | metric | candidate | v1 | median diff | p | patients better | n |', '|---|---|---|---|---|---|---|---|']
            for split in SPLITS:
                cp = os.path.join(V2_GENERATED, 'percase_candidate', f'{cid}_vc16_{split}_{lay}.csv')
                a, b = load(cp, 'commercial'), load(v1_csv(split, lay), 'commercial')
                if a is None or b is None:
                    md.append(f'| {split} | (missing: {"candidate" if a is None else "v1"} CSV) | | | | | | |'); g1_test = False; continue
                st = paired(a, b); res['test'][f'{split}|{lay}'] = st
                a2, b2 = load(cp, 'src2_allvalid'), load(v1_csv(split, lay), 'src2_allvalid')
                if a2 is not None and b2 is not None:
                    res['test'][f'{split}|{lay}|vs_allvalid'] = paired(a2, b2)
                for m, s in st.items():
                    md.append(f'| {split} | {m} | {s["cand"]:.3f} | {s["v1"]:.3f} | {s["med_diff"]:+.3f} | {s["p"]:.2g} | {s["patients_better"]}/{s["n_patients"]} | {s["n"]} |')
                for m in ('AT_mae', 'RT_mae'):
                    s = st.get(m)
                    if s is None:
                        continue
                    sig_worse = s['p'] < 0.05 and s['med_diff'] > 0 and s['med_diff'] > TOL_MS
                    if sig_worse:
                        g1_test = False; worse.append(f'{split}/{lay}/{m} (+{s["med_diff"]:.2f} ms, p={s["p"]:.2g})')
                    sig_better = s['p'] < 0.05 and s['med_diff'] < 0
                    if m == 'AT_mae': g2_at &= sig_better
                    else: g2_rt &= sig_better
            md.append('')
        g2_speed = g1_val and g1_test and speed[0] is not None and (speed[0] >= 3 or speed[1] >= 3)
        g1 = g1_val and g1_test; g2 = g2_val or g2_at or g2_rt or g2_speed
        res['gate'] = dict(G1_val=g1_val, G1_test=g1_test, G1=g1, G2_val=g2_val, G2_AT_both_cohorts=g2_at, G2_RT_both_cohorts=g2_rt, G2_speed=g2_speed, G2=g2,
                           adopt=g1 and g2, worse=worse)
        md += ['## Gate', '', f'- G1 not worse: validation MAE {"yes" if g1_val else "NO"} (both layouts), test AT/RT {"yes" if g1_test else "NO"}'
               + (f' (worse: {"; ".join(worse)})' if worse else ''),
               f'- G2 beneficial: validation MAE improved >= 0.003 under both layouts: {"yes" if g2_val else "no"}; AT significantly better on both cohorts under both layouts: {"yes" if g2_at else "no"}; '
               f'RT: {"yes" if g2_rt else "no"}; >= 3x faster given G1: {"yes" if g2_speed else "no"}',
               f'- **Verdict: {"ADOPT" if (g1 and g2) else "do not adopt"}** (G1 {"pass" if g1 else "fail"}, G2 {"pass" if g2 else "fail"})', '']
        jp = assert_v2(os.path.join(V2_GENERATED, f'gate_{cid}.json')); json.dump(res, open(jp, 'w'), indent=1)
        mp = assert_v2(os.path.join(V2, 'results', f'gate_{cid}.md')); open(mp, 'w').write('\n'.join(md))
        print('\n'.join(md)); print('wrote', jp, mp)


if __name__ == '__main__':
    main(sys.argv[1:])
