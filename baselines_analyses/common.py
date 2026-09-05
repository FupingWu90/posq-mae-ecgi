"""Shared helpers for the manuscript tables/figures.

All scripts in this directory read ONLY finished run artefacts
(test_compare.json / at_rt_results.json / training_stats.json) plus the
downstream caches -- they never retrain anything.  Outputs:
    LaTeX tables -> baselines_analyses/generated/*.tex   (\input from manuscript.tex)
    figures      -> <ECGI_FIG_DIR>/*.png
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RUNS = os.path.join(PROJ, 'runs', 'data-cohortA+cohortB_vs_external')
GEN = os.path.join(HERE, 'generated')
FIGS = os.environ.get('ECGI_FIG_DIR', os.path.join(PROJ, 'figures'))   # manuscript figures
os.makedirs(GEN, exist_ok=True)

TAIL = 'sc-recon_k6s10_ebs64_crop5000_ep800'


def run_dir(model='fixedn_ecgi_base', vc=16, train_sel='cluster-greedy',
            val_tag='val-cluster-kmean', ws='0', wt='0', rw=None, scope='recon',
            ebs=64):
    """Build a run-dir path from the naming convention."""
    v = f'_{val_tag}' if val_tag else ''
    r = f'-rw{rw}' if rw else ''
    e = f'ebs{ebs}' + ('-aug2' if ebs == 128 else '')
    name = (f'{model}_vc{vc}_{train_sel}{v}_nm-gl_ws{ws}-wt{wt}{r}'
            f'_sc-{scope}_k6s10_{e}_crop5000_ep800')
    return os.path.join(RUNS, name)


def bsp(run, deploy='cluster-kmean'):
    """BSP reconstruction metrics under one deployment selection."""
    f = os.path.join(run, 'test_compare.json')
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))['results']
    return {s: d[s].get(deploy, {}) for s in ('test_internal', 'test_external')}


def hsp(run, source='src4_recon_valid', ref='commercial', json_name=None):
    """Downstream (HSP/AT/RT/ARI) aggregates for one source vs one reference."""
    f = os.path.join(run, json_name or 'at_rt_results.json')
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))['splits']
    return {s: d[s].get(f'{source}__vs__{ref}', {}) for s in d}


def val_mae(run):
    f = os.path.join(run, 'training_stats.json')
    return json.load(open(f)).get('val_best_mae') if os.path.exists(f) else None


def fmt(x, p=2):
    return f'{x:.{p}f}' if isinstance(x, (int, float)) else '--'
