"""Phase 3 registry (adopted architecture = LOC3: PosQ-MAE, query decoder + one sub-token attention output head,
w_t = 3; the architecture plan §6 Phase 3).  Every run trains the FINAL configuration with one factor changed and
is evaluated on both test cohorts under both deployment layouts (training_sweeps/eval_candidate_test.sh).

  python training_sweeps/phase3.py next       # "ID<TAB>run_dir<TAB>args" of the next run, or DONE / WAIT:<reason>
  python training_sweeps/phase3.py cmd ID     # print the train.py command of one run
  python training_sweeps/phase3.py summary    # rewrite results/phase3_log.md
Training length: training_sweeps/final_epochs.json (written by `python training_sweeps/phase3.py epochs` from the EPL3 result:
1600 epochs only if EPL3 beats LOC3 by > 0.002 validation MAE under BOTH layouts and its validation AT is not
worse under either layout; else 800).
"""
import os, sys, json, glob, argparse
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(V2, 'bsp_reconstruction'); RUNS = os.path.join(V2, 'runs', 'data-cohortA+cohortB_vs_external')
STATE = os.path.join(V2, 'training_sweeps'); FAILED = os.path.join(STATE, 'failed')
LOG_MD = os.path.join(V2, 'results', 'phase3_log.md')
if CODE not in sys.path:
    sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(V2, 'training_sweeps'))
from pilots import COMMON, _load, percase_means, result as pilot_result, flags_for as pilot_flags   # noqa: E402
from paths import V2_GENERATED   # noqa: E402

FINAL = ['--head', 'subattn', '--inner_depth', '1']          # adopted configuration on top of COMMON (LOC3)
PERCASE_TEST = os.path.join(V2_GENERATED, 'percase_candidate')
REGION_RUNS = {'P3REGF': 'front', 'P3REGB': 'back'}   # scored in their half: val_engine_region.json, percase_region/*_{tpl,km}_<half>.csv

# ID -> (extra flags, description, group, needs)   needs: ID whose best checkpoint initialises this run
RUNS_P3 = {
    'P3N4':    (['--vc', '4'],  'final config at N = 4',  'N grid', None),
    'P3N8':    (['--vc', '8'],  'final config at N = 8',  'N grid', None),
    'P3N32':   (['--vc', '32'], 'final config at N = 32', 'N grid', None),
    'P3N64':   (['--vc', '64'], 'final config at N = 64', 'N grid', None),
    'P3INT2':  (['--dec_tsa', '--batch_size', '16', '--accum_iter', '4'], 'query interaction: temporal (queries of one electrode attend to one another; on the final config)', 'interaction', None),
    'P3INT1':  (['--dec_ssa', '--batch_size', '16', '--accum_iter', '4'], 'query interaction: spatial (queries of one time patch attend across electrodes)', 'interaction', None),
    'P3INT3':  (['--dec_ssa', '--dec_tsa', '--batch_size', '16', '--accum_iter', '4'], 'query interaction: spatial + temporal (axial)', 'interaction', None),
    'P3S43':   (['--rng_seed', '43'], 'training seed 43 (N = 16)', 'seeds', None),
    'P3S44':   (['--rng_seed', '44'], 'training seed 44 (N = 16)', 'seeds', None),
    'P3IDX42': (['--pos_embed', 'index'], 'index embedding instead of physical positions, seed 42 (new framework)', 'index', None),
    'P3IDX43': (['--pos_embed', 'index', '--rng_seed', '43'], 'index embedding, seed 43', 'index', None),
    'P3IDX44': (['--pos_embed', 'index', '--rng_seed', '44'], 'index embedding, seed 44', 'index', None),
    'P3TF05':  (['--train_frac', '0.5', '--aug_times', '2'], 'training set 1/2 (compute matched)', 'data', None),
    'P3TF025': (['--train_frac', '0.25', '--aug_times', '3'], 'training set 1/4', 'data', None),
    'P3TF0125': (['--train_frac', '0.125', '--aug_times', '7'], 'training set 1/8', 'data', None),
    'P3SELKM': (['--select_mode', 'cluster', '--cluster_method', 'kmean'], 'training selection: k-means (instead of farthest point)', 'selection', None),
    'P3SELPT': (['--select_mode', 'patch'], 'training selection: patch', 'selection', None),
    'P3SELRD': (['--select_mode', 'random'], 'training selection: random', 'selection', None),
    'P3REGF':  (['--select_region', 'front', '--val_select_mode', 'cluster', '--val_cluster_method', 'kmean'], 'training restricted to the anterior half (validation: in-region k-means)', 'selection', None),
    'P3REGB':  (['--select_region', 'back', '--val_select_mode', 'cluster', '--val_cluster_method', 'kmean'], 'training restricted to the posterior half (validation: in-region k-means)', 'selection', None),
    'P3SIM':   (['--train_sources', 'sim', '--val_source', 'sim'], 'simulated data only', 'simulation', None),
    'P3SIM2R': (['--init_tag', 'sim'], 'simulation pre-training then clinical training', 'simulation', 'P3SIM'),
    'P3SIMR':  (['--train_sources', 'sim+both', '--real_repeat', '1'], 'simulated + clinical training', 'simulation', None),
    'P3FT0125': (['--train_frac', '0.125', '--aug_times', '7', '--init_tag', 'sim'], 'low data 1/8 after simulation pre-training', 'simulation', 'P3SIM'),
    'P3FT025': (['--train_frac', '0.25', '--aug_times', '3', '--init_tag', 'sim'], 'low data 1/4 after simulation pre-training', 'simulation', 'P3SIM'),
    'P3FT05':  (['--train_frac', '0.5', '--aug_times', '2', '--init_tag', 'sim'], 'low data 1/2 after simulation pre-training', 'simulation', 'P3SIM'),
}
ORDER = list(RUNS_P3)


def final_epochs():
    j = _load(os.path.join(STATE, 'final_epochs.json'))
    return int(j['epochs']) if j else 800


def decide_epochs():
    """EPL3 vs LOC3 rule -> training_sweeps/final_epochs.json (idempotent; only decides once EPL3 is finished)."""
    out = os.path.join(STATE, 'final_epochs.json')
    if os.path.exists(out):
        return _load(out)
    r3, re_ = pilot_result('LOC3', pilot_flags('LOC3', decide=False)), pilot_result('EPL3', pilot_flags('EPL3', decide=False))
    if r3 is None or re_ is None:
        return None
    d_t = (re_.get('val_mae') or 9) - (r3.get('val_mae') or 0)
    d_k = (re_.get('val_mae_km') or 9) - (r3.get('val_mae_km') or 0)
    at_ok = True
    for lay in ('template', 'kmeans'):
        j = _load(os.path.join(V2, 'results', 'val_percase', 'paired_LOC3_vs_EPL3' + ('' if lay == 'template' else '_kmeans') + '.json'))
        s = (j or {}).get('EPL3|commercial|AT_mae')
        if s is None:
            at_ok = False
        elif s['p'] < 0.05 and s['med_diff'] > 0:
            at_ok = False
    use1600 = d_t < -0.002 and d_k < -0.002 and at_ok
    dec = dict(epochs=1600 if use1600 else 800, epl3_minus_loc3_val_mae=dict(template=d_t, kmeans=d_k), at_not_worse_both_layouts=at_ok,
               reason=('EPL3 better than LOC3 by > 0.002 under both layouts and AT not worse' if use1600 else
                       'EPL3 not better than LOC3 by > 0.002 under both layouts (or AT worse) -> 800 epochs kept'))
    json.dump(dec, open(out, 'w'), indent=1)
    return dec


def run_dir_of(cmd):
    import train
    a = train.get_args().parse_args(cmd)
    return os.path.join(RUNS, train.build_run_name(a, a.batch_size * a.accum_iter))


def args_of(pid):
    extra, desc, group, needs = RUNS_P3[pid]
    cmd = list(COMMON) + list(FINAL) + ['--epochs', str(final_epochs())] + list(extra)
    if needs:
        ck = os.path.join(run_dir_of(args_of(needs)), 'checkpoints', 'checkpoint_best.pth')
        cmd += ['--init_from', ck]
    return cmd + ['--tag', pid]


def finished(pid):
    rd = run_dir_of(args_of(pid)); st = _load(os.path.join(rd, 'training_stats.json'))
    return bool(st and os.path.exists(os.path.join(rd, 'test_compare.json')) and st.get('epochs_done') == st.get('epochs_total'))


def is_failed(pid):
    return os.path.exists(os.path.join(FAILED, pid))


def next_run():
    if not os.path.exists(os.path.join(STATE, 'final_epochs.json')):
        if decide_epochs() is None:
            return 'WAIT: EPL3 not finished (training length undecided)'
    for pid in ORDER:
        if finished(pid) or is_failed(pid):
            continue
        needs = RUNS_P3[pid][3]
        if needs and not finished(needs):
            if is_failed(needs):
                continue
            return f'WAIT: {pid} needs {needs}'
        return pid
    return 'DONE'


def summary():
    ep = _load(os.path.join(STATE, 'final_epochs.json'))
    L = ['# Phase 3 runs (adopted configuration LOC3; one factor changed per run; both test cohorts, both layouts)', '',
         f"Training length: {ep['epochs'] if ep else 'undecided (waiting for EPL3)'}" + (f" ({ep['reason']})" if ep else ''), '',
         '| ID | group | change | val MAE tmpl / km | int AT tmpl / km | ext AT tmpl / km | int RT tmpl / km | ext RT tmpl / km | train h | status |',
         '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v, n=2: ('%.*f' % (n, v)) if isinstance(v, (int, float)) else '--'
    for pid, (extra, desc, group, needs) in RUNS_P3.items():
        try:
            rd = run_dir_of(args_of(pid))
        except Exception:
            rd = None
        st = _load(os.path.join(rd, 'training_stats.json')) if rd else None
        done = finished(pid) if rd else False
        reg = REGION_RUNS.get(pid)
        ve = _load(os.path.join(rd, 'val_engine_region.json' if reg else 'val_engine.json')) if rd else None
        vm = (f(ve['val_template']['mae'], 4) + ' / ' + f(ve['val_kmeans']['mae'], 4)) if ve else ('%s / --' % f(st.get('val_best_mae'), 4) if st else '-- / --')
        cells = []
        for split, tag in (('test_internal', 'int'), ('test_external', 'ext')):
            if reg:
                m = {lay: percase_means(os.path.join(V2_GENERATED, 'percase_region', f'{pid}_vc{_vc(pid)}_{split}_{pre}_{reg}.csv')) or {}
                     for lay, pre in (('template', 'tpl'), ('kmeans', 'km'))}
            else:
                m = {lay: percase_means(os.path.join(PERCASE_TEST, f'{pid}_vc{_vc(pid)}_{split}_{lay}.csv')) or {} for lay in ('template', 'kmeans')}
            cells.append(f"{f(m['template'].get('AT_mae'))} / {f(m['kmeans'].get('AT_mae'))}")
            cells.append(f"{f(m['template'].get('RT_mae'))} / {f(m['kmeans'].get('RT_mae'))}")
        status = 'failed' if is_failed(pid) else ('done' if done else ('running / pending' if st else 'pending'))
        L.append(f"| {pid} | {group} | {desc} | {vm} | {cells[0]} | {cells[2]} | {cells[1]} | {cells[3]} | {f(st.get('elapsed_train_hours'), 1) if st else '--'} | {status} |")
    open(LOG_MD, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def _vc(pid):
    ex = RUNS_P3[pid][0]
    return ex[ex.index('--vc') + 1] if '--vc' in ex else '16'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['next', 'cmd', 'summary', 'epochs']); ap.add_argument('pid', nargs='?')
    a = ap.parse_args()
    if a.what == 'next':
        n = next_run()
        if n.startswith(('DONE', 'WAIT')):
            print(n)
        else:
            c = args_of(n); print(f'{n}\t{run_dir_of(c)}\t' + ' '.join(c))
    elif a.what == 'cmd':
        c = args_of(a.pid); print('python train.py ' + ' '.join(c)); print(run_dir_of(c))
    elif a.what == 'epochs':
        print(json.dumps(decide_epochs(), indent=1))
    else:
        summary()
