"""Engine held-out completion MAE / RMSE / CC of a run on the TEST cohorts under both deployment layouts
(fixed template, per-recording k-means) -- the quantity test_compare.json holds for v2 runs, computed here for
the frozen v1 baseline runs (their test_compare.json has k-means entries only).  Output JSON goes to v2.
  python baselines_analyses/test_engine_metrics.py --run_dir <run> --out <json> [--batch_size 4]
"""
import os, sys, json, argparse, warnings
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import torch
from torch.utils.data import DataLoader
from paths import assert_v2
from dataset import build_experiment_datasets
from engine import run_eval
from build_sources import load_model

LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    a = ap.parse_args()
    run = a.run_dir.rstrip('/'); out = assert_v2(a.out)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = json.load(open(os.path.join(run, 'args.json')))
    _, _, ti_ds, te_ds, _ = build_experiment_datasets(crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1,
                                                      pp_norm=args['pp_norm'], tag='exp', min_active=args['vc'] + args['min_recon'],
                                                      prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
    model, _, _ = load_model(run, dev)
    res = {'run': os.path.basename(run), 'vc': args['vc'], 'results': {}}
    for split, ds in (('test_internal', ti_ds), ('test_external', te_ds)):
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
        res['results'][split] = {}
        for lab, (mode, cm) in LAYOUTS.items():
            with torch.no_grad():
                r = run_eval(model, dl, dev, args['vc'], mode, cm, seed=args['seed'], norm_mode=args['norm_mode'])
            res['results'][split][lab] = {k: r[k] for k in ('mae', 'rmse', 'cc', 'n_samples')}
            print(split, lab, json.dumps(res['results'][split][lab]), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1); print('wrote', out)


if __name__ == '__main__':
    main()
