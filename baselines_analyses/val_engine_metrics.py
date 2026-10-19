"""Engine validation MAE / CC of a run under BOTH deployment layouts (fixed template, per-recording
k-means) -- the pilot selection metric, reported for both layouts (user decision 2026-10-06).
  python baselines_analyses/val_engine_metrics.py --run_dir <run> [--out <json>]      -> <run>/val_engine.json
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
    ap.add_argument('--run_dir', required=True); ap.add_argument('--out', default=None)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    a = ap.parse_args()
    run = a.run_dir.rstrip('/')
    out = assert_v2(a.out or os.path.join(run, 'val_engine.json'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = json.load(open(os.path.join(run, 'args.json')))
    _, val_ds, _, _, _ = build_experiment_datasets(crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1,
                                                   pp_norm=args['pp_norm'], tag='exp', min_active=args['vc'] + args['min_recon'],
                                                   prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
    model, _, _ = load_model(run, dev)
    dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    res = {'run': os.path.basename(run), 'split': 'val', 'vc': args['vc']}
    for lab, (mode, cm) in LAYOUTS.items():
        with torch.no_grad():
            r = run_eval(model, dl, dev, args['vc'], mode, cm, seed=args['seed'], norm_mode=args['norm_mode'])
        res[f'val_{lab}'] = {k: r[k] for k in ('mae', 'rmse', 'cc', 'n_samples')}
        print(lab, json.dumps(res[f'val_{lab}']), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
