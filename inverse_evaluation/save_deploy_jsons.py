"""Aggregate the deploy-matrix CSVs (out_matrix/) into per-(model, deploy-mode)
JSON files written INTO each model's run dir, in the same shape as
at_rt_results.json:  at_rt_results_deploy-<mode>.json

So for each trained model you get one file per deployment selection, discoverable
next to the model's own at_rt_results.json.
"""
import os, csv, json, glob
import numpy as np

RUNS = os.path.join(os.environ.get("ECGI_V1_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v1")), "runs", "data-cohortA+cohortB_vs_external")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_matrix')
SPLITS = ['test_internal', 'test_external']
SOURCES = ['src1_vc', 'src3_recon_whole', 'src4_recon_valid']
REFS = ['src2_allvalid', 'commercial']
METRICS = ['success', 'pot_cc', 'scale_ratio',
           'hsp_mae', 'hsp_mae_md', 'hsp_rmse', 'hsp_rmse_md', 'hsp_cc', 'hsp_cc_md',
           'AT_mae', 'AT_mdae', 'AT_cc',
           'RT_mae', 'RT_mdae', 'RT_cc',
           'RT_mae_all', 'RT_mdae_all', 'RT_cc_all',
           'ARI_mae', 'ARI_mdae', 'ARI_cc',
           'AT_mae_allv', 'AT_mdae_allv',
           'RT_mae_allv', 'RT_mdae_allv',
           'ARI_mae_allv', 'ARI_mdae_allv',
           'ref_qrs_frac', 'ref_t_frac', 'src_qrs_frac', 'src_t_frac']
TRAIN_DIR = {
    'greedy': 'cluster-greedy',
    'kmean':  'cluster-kmean',
}


def rundir(train):
    h = glob.glob(os.path.join(RUNS, f"fixedn_ecgi_base_vc16_{TRAIN_DIR[train]}_nm-gl_*"))
    return h[0] if h else None


def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def agg_split(csv_path):
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    out = {}
    for s in SOURCES:
        for r in REFS:
            sub = [x for x in rows if x['source'] == s and x['ref'] == r]
            if not sub:
                continue
            d = {'n': len(sub)}
            for m in METRICS:
                v = [float(x[m]) for x in sub if _finite(x.get(m))]
                arr = np.asarray(v, float)
                # cohort mean / std / median under explicit keys (never a bare key)
                d[f"{m}_mean"] = float(arr.mean()) if v else None
                d[f"{m}_std"] = float(arr.std(ddof=1)) if len(v) > 1 else (0.0 if v else None)
                d[f"{m}_median"] = float(np.median(arr)) if v else None
            out[f"{s}__vs__{r}"] = d
    return out


def main():
    written = []
    for train, deploy in [(t, d) for t in TRAIN_DIR for d in TRAIN_DIR]:
        rd = rundir(train)
        if not rd:
            continue
        splits = {}
        for split in SPLITS:
            a = agg_split(os.path.join(OUT, f"train-{train}_deploy-{deploy}_{split}.csv"))
            if a is not None:
                splits[split] = a
        if not splits:
            continue
        doc = {'run': os.path.basename(rd), 'train_mode': f"cluster-{train}",
               'deploy_mode': f"cluster-{deploy}", 'vc': 16,
               'agg': ('per-recording spatial aggregation: _mae=mean-over-vertices, '
                       '_mdae=median-over-vertices; cohort aggregation suffixes: '
                       '_mean/_std/_median over recordings. e.g. AT_mae_mean=mean-mean, '
                       'AT_mdae_median=median-median, AT_mae_std=SD of per-rec spatial-mean'),
               'splits': splits}
        out_path = os.path.join(rd, f"at_rt_results_deploy-cluster-{deploy}.json")
        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2)
        written.append(out_path)
        # when deploying with the model's OWN training selection, this cell IS the
        # canonical result -> also refresh at_rt_results.json (new mean+median+ARI schema)
        if deploy == train:
            canon = dict(doc, select_mode=f"cluster-{train}")
            cpath = os.path.join(rd, 'at_rt_results.json')
            with open(cpath, 'w') as f:
                json.dump(canon, f, indent=2)
            written.append(cpath)
    print(f"wrote {len(written)} per-(model,deploy) JSONs:")
    for p in written:
        print("  " + p)


if __name__ == '__main__':
    main()
