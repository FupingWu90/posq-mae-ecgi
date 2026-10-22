"""Are v1 and the v2 pilots compared under the SAME electrode layout?

Every model is evaluated with the same engine code on the same recordings under BOTH layouts:
  template : fixed predetermined layout (deterministic)
  k-means  : per-recording k-means on the valid electrodes (engine protocol, seed 42 -> identical electrode
             sets for every model, because the selection consumes the RNG in the same order)
Validation: v1 best checkpoint (chosen by k-means validation during v1 training), v1 last checkpoint, and the
best checkpoint of each finished pilot (chosen by template validation).  Test: BSP CC / MAE under both layouts.
Read-only w.r.t. v1; writes results/v1_reference/layout_fairness.json.
"""
import os, sys, json, glob, warnings
import numpy as np
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import torch
from torch.utils.data import DataLoader
from paths import V1_RUNS
from dataset import build_experiment_datasets
from engine import run_eval
from build_sources import load_model

warnings.filterwarnings('ignore')
dev = torch.device('cuda')
V1 = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')
_, val_ds, ti_ds, te_ds, info = build_experiment_datasets(crop_len=5000, preprocess=True, seed=42, aug_times=1, pp_norm='centroid', tag='exp',
                                                          min_active=17, prep_cache=True, norm_mode='global', verbose=False)
dl = lambda ds: DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
LAY = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}


def ev(model, ds):
    return {lab: {k: float(v) for k, v in run_eval(model, dl(ds), dev, 16, m, cm, seed=42, norm_mode='global').items() if k in ('mae', 'cc')}
            for lab, (m, cm) in LAY.items()}


res = {}
model, _, _ = load_model(V1, dev)
res['v1 (best checkpoint, chosen by k-means validation)'] = {'val': ev(model, val_ds), 'test_internal': ev(model, ti_ds), 'test_external': ev(model, te_ds)}
ck = torch.load(os.path.join(V1, 'checkpoints', 'checkpoint_last.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model']); model.to(dev).eval()
res['v1 (last checkpoint, epoch %d)' % ck.get('epoch', -1)] = {'val': ev(model, val_ds)}
del model; torch.cuda.empty_cache()
for pid in sys.argv[1:] or ['DEC1', 'DECH', 'DEC2']:
    g = glob.glob(os.path.join(V2, 'runs', 'data-cohortA+cohortB_vs_external', pid + '__*'))
    if not g or not os.path.exists(os.path.join(g[0], 'test_compare.json')):
        continue
    model, _, _ = load_model(g[0], dev)
    t = json.load(open(os.path.join(g[0], 'test_compare.json')))['results']
    res[pid] = {'val': ev(model, val_ds),
                'test_internal': {'template': {k: t['test_internal']['template'][k] for k in ('mae', 'cc')}, 'kmeans': {k: t['test_internal']['cluster-kmean'][k] for k in ('mae', 'cc')}},
                'test_external': {'template': {k: t['test_external']['template'][k] for k in ('mae', 'cc')}, 'kmeans': {k: t['test_external']['cluster-kmean'][k] for k in ('mae', 'cc')}}}
    del model; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(V2, 'results', 'v1_reference', 'layout_fairness.json'), 'w'), indent=1)
print('%-52s | %-21s | %-21s | %-27s | %-27s' % ('model', 'val MAE  templ / kmeans', 'val CC   templ / kmeans', 'test CC internal templ / km', 'test CC external templ / km'))
for name, r in res.items():
    v = r['val']; ti = r.get('test_internal'); te = r.get('test_external')
    print('%-52s | %.4f / %.4f       | %.4f / %.4f       | %s | %s' % (
        name, v['template']['mae'], v['kmeans']['mae'], v['template']['cc'], v['kmeans']['cc'],
        ('%.4f / %.4f            ' % (ti['template']['cc'], ti['kmeans']['cc'])) if ti else ' ' * 27,
        ('%.4f / %.4f' % (te['template']['cc'], te['kmeans']['cc'])) if te else ''))
