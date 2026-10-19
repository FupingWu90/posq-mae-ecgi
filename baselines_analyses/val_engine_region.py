"""In-region engine validation MAE / CC of a region-trained run (P3REGF / P3REGB): the half-torso template
(generated/deploy_robustness/templates/tpl_<region>.npz, installed in channel_select exactly as
inverse_evaluation/eval_downstream.py --template_npz does) and the half-restricted per-recording k-means -- the in-region
counterparts of the two layouts of baselines_analyses/val_engine_metrics.py.  Aggregates only.
  python baselines_analyses/val_engine_region.py --run_dir <run> [--region front|back] [--out <json>]   -> <run>/val_engine_region.json
"""
import os, sys, json, argparse, warnings
import numpy as np, torch
from torch.utils.data import DataLoader
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
from paths import assert_v2, V2_GENERATED                                   # noqa: E402
from dataset import build_experiment_datasets                               # noqa: E402
from engine import run_eval                                                 # noqa: E402
from build_sources import load_model                                        # noqa: E402
import channel_select as CS                                                 # noqa: E402


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--region', default=None, choices=[None, 'front', 'back'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    a = ap.parse_args()
    run = a.run_dir.rstrip('/')
    out = assert_v2(a.out or os.path.join(run, 'val_engine_region.json'))
    args = json.load(open(os.path.join(run, 'args.json')))
    region = a.region or args.get('select_region')
    assert region in ('front', 'back'), f'not a region-trained run (select_region={args.get("select_region")})'
    tpl = os.path.join(V2_GENERATED, 'deploy_robustness', 'templates', f'tpl_{region}.npz')
    with np.load(tpl) as z:                                                   # half-torso template -> mode='template'
        CS._TEMPLATES[int(args['vc'])] = (z['idx'].astype(np.int64), z['mean_xyz'].astype(np.float32))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_ds, _, _, _ = build_experiment_datasets(crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1,
                                                   pp_norm=args['pp_norm'], tag='exp', min_active=args['vc'] + args['min_recon'],
                                                   prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
    model, _, _ = load_model(run, dev)
    dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    res = {'run': os.path.basename(run), 'split': 'val', 'vc': args['vc'], 'region': region, 'template_npz': os.path.basename(tpl)}
    with torch.no_grad():
        r = run_eval(model, dl, dev, args['vc'], 'template', 'greedy', seed=args['seed'], norm_mode=args['norm_mode'])
        res['val_template'] = {k: r[k] for k in ('mae', 'rmse', 'cc', 'n_samples')}        # the installed half-torso template
        print('template (in-region)', json.dumps(res['val_template']), flush=True)
        r = run_eval(model, dl, dev, args['vc'], 'cluster', 'kmean', seed=args['seed'], norm_mode=args['norm_mode'], region=region)
        res['val_kmeans'] = {k: r[k] for k in ('mae', 'rmse', 'cc', 'n_samples')}          # half-restricted k-means
        print('kmeans (in-region)', json.dumps(res['val_kmeans']), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
