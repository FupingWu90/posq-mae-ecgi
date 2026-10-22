"""Reference numbers of the v1 final model (and optionally other v1 runs) under the v2 protocol
(fixed template layout): engine validation MAE/CC (the pilot selection metric) and the downstream
validation report.  Read-only w.r.t. v1; everything is written to results/v1_reference/."""
import os, sys, json, argparse, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import torch
from torch.utils.data import DataLoader
from paths import V1_RUNS
from dataset import build_experiment_datasets
from engine import run_eval
from build_sources import load_model
from report_at_rt import report

RUNS = {'wide': 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800'}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--name', default='wide'); ap.add_argument('--skip_downstream', action='store_true')
    a = ap.parse_args()
    run = os.path.join(V1_RUNS, RUNS[a.name])
    out_dir = os.path.join(V2, 'results', 'v1_reference'); os.makedirs(out_dir, exist_ok=True)
    dev = torch.device('cuda')
    args = json.load(open(os.path.join(run, 'args.json')))
    _, val_ds, _, _, info = build_experiment_datasets(crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1,
                                                      pp_norm=args['pp_norm'], tag='exp', min_active=args['vc'] + args['min_recon'],
                                                      prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
    model, _, _ = load_model(run, dev)
    dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8)
    res = {'run': RUNS[a.name]}
    for lab, (mode, cm) in {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}.items():
        r = run_eval(model, dl, dev, args['vc'], mode, cm, seed=args['seed'], norm_mode=args['norm_mode'])
        res[f'val_{lab}'] = {k: r[k] for k in ('mae', 'rmse', 'cc', 'n_samples')}
        print(lab, res[f'val_{lab}'])
    json.dump(res, open(os.path.join(out_dir, f'{a.name}_val_engine.json'), 'w'), indent=1)
    if not a.skip_downstream:
        report(run, splits=('val',), deploy_mode='template', out_json=os.path.join(out_dir, f'{a.name}_at_rt_val_template.json'))


if __name__ == '__main__':
    main()
