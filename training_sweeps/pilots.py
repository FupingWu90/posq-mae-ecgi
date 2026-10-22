"""Architecture pilots of PosQ-MAE (v2): registry, command builder, unattended queue logic, summary.

Design (validation only; fixed template layout; one training seed; clinical recordings only):
  1. DEC1  : v1 with the position-queried cross-attention decoder (same widths / heads as v1).
  2. DECH  : DEC1 with 4 decoder attention heads of 32 dimensions instead of 16 heads of 8 dimensions.
             Added 2026-10-04 after inspecting DEC1's attention: one single 8-dimensional head carries all
             time-specific retrieval.  -> HEADS decision (DECH adopted if it lowers the validation MAE by
             more than TOL).
  3. DEC2  : head winner + temporal self-attention among the queries of one electrode.
             -> BASE decoder decision.
  4. every other pilot changes ONE factor on that base (star design).
  5. EPO   : base with a 1600-epoch schedule; run only if the base run is still converging at 800 epochs
             (baselines_analyses/convergence_report.py).  -> EPOCHS decision; the capacity checks (CAPD/CAPW/CAPE), which
             interact with the training length, then use the chosen schedule.
Selection metric: best validation completion MAE (training_stats.json: val_best_mae); secondary: downstream
AT error on the validation recordings.  Seed noise of the primary metric (v1, three training seeds):
SD ~0.001 -> only differences > TOL = 0.002 count; ties keep the simpler / v1-like setting.

  python training_sweeps/pilots.py next       # next pilot:  ID <TAB> run_dir <TAB> train.py args   | DONE | STOP: ...
  python training_sweeps/pilots.py summary    # rewrite results/results_log.md
  python training_sweeps/pilots.py cmd ID     # print the train.py command of one pilot
"""
import os, sys, json, argparse
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(V2, 'bsp_reconstruction')
RUNS = os.path.join(V2, 'runs', 'data-cohortA+cohortB_vs_external')
REF = os.path.join(V2, 'results', 'v1_reference')
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from paths import V1_RUNS, V1_GENERATED, V2_GENERATED   # noqa: E402
V1_RUN = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')
PERCASE_TEST = os.path.join(V2_GENERATED, 'percase_candidate')      # <ID>_vc16_<split>_<layout>.csv (tools: training_sweeps/eval_candidate_test.sh)
STATE = os.path.join(V2, 'training_sweeps')                 # decision files live here (overridden in the logic tests)
FAILED = os.path.join(V2, 'training_sweeps', 'failed')
RESULTS_MD = os.path.join(V2, 'results', 'results_log.md')
TOL = 0.002          # ~2 SD of the seed noise of the validation MAE
BAD = 0.004          # a decoder trailing v1 by more than this is not adopted without a deeper retry

COMMON = ['--model', 'posq_ecgi_wide', '--batch_size', '32', '--accum_iter', '2', '--patch_size', '1', '50',
          '--vc', '16', '--norm_mode', 'global', '--w_spatial', '0', '--w_temporal', '3',
          '--reg_warmup_epochs', '100', '--reg_scope', 'recon',
          '--select_mode', 'cluster', '--cluster_method', 'greedy', '--val_select_mode', 'template',
          '--aug_times', '1', '--epochs', '800', '--use_amp', '--num_workers', '8',
          '--save_every', '40', '--eval_every', '20', '--downstream_splits', 'val']

# id -> (extra args, on_base, what it tests)
PILOTS = {
    'DEC1': ([], False, 'position-queried cross-attention decoder (3 blocks); everything else as v1'),
    'DECH': (['--dec_heads', '4'], False, 'DEC1 with 4 decoder attention heads of 32 dimensions (instead of 16 heads of 8)'),
    'DEC2': (['--dec_tsa'], False, 'head winner + temporal self-attention among the queries of one electrode'),
    'DEC3': (['--dec_depth', '6'], False, 'query decoder with 6 blocks (run only if the decoders above trail v1)'),
    'LOC3': (['--head', 'subattn', '--inner_depth', '1'], True,
             'output head: sub-token (5-sample) local attention across patch boundaries'),
    'LOC2': (['--tokenizer', 'tnt'], True, 'tokenizer: self-attention within each patch (5-sample sub-tokens, 2 blocks)'),
    'LOC1': (['--tokenizer', 'conv'], True, 'tokenizer: 1-D convolutional stem along time'),
    'MSK1': (['--tmask_mode', 'dropout', '--tmask_ratio', '0.25'], True,
             'mixed masking: 25% of the visible electrodes\' time patches hidden on every second step'),
    'MSK3': (['--tmask_mode', 'alternate'], True,
             'mixed masking: random token masking over all active electrodes on every second step'),
    'EMB1': (['--time_embed', 'learned'], True, 'learned instead of sinusoidal time embedding'),
    # ---- follow-ups of LOC3, the first module with a clear benefit (user-approved 2026-10-05) ----
    'LOC3B': (['--head', 'convref'], True, 'output head: 1-D convolutional refinement of the assembled series (conv counterpart of LOC3)'),
    'LOC4': (['--head', 'subattn', '--inner_depth', '1', '--w_temporal', '0'], True,
             'LOC3 without the temporal-smoothness loss (w_t = 0): does the output head replace it?'),
    'LOC3D': (['--head', 'subattn', '--inner_depth', '2'], True, 'LOC3 with two sub-token blocks in the output head'),
    'LOC5': (['--head', 'subattn', '--inner_depth', '1', '--w_temporal', '1'], True,
             'LOC3 with w_t = 1: compromise between the smoothness loss (LOC3, w_t = 3) and none (LOC4)'),
    # ---- hyper-parameter checks on the base decoder (run after the pilots above) ----
    'DEC4': (['--query_frac', '0.5'], True, 'training cost: half of the masked electrodes decoded per training step'),
    'EPL3': (['--head', 'subattn', '--inner_depth', '1', '--epochs', '1600'], True,
             'LOC3 with a 1600-epoch schedule (LOC3 was still converging at 800 epochs: best epoch 780, last-200-epoch gain 0.0023)'),
    'EPO': (['--epochs', '1600'], True, 'training length: 1600 instead of 800 epochs (only if the base run still converges at 800)'),
    'CAPD': (['--dec_depth', '6'], True, 'capacity: decoder depth 6 instead of 3'),
    'CAPW': (['--dec_dim', '256'], True, 'capacity: decoder width 256 instead of 128'),
    'CAPE': (['--enc_depth', '9'], True, 'capacity: encoder depth 9 instead of 6'),
}
# order revised 2026-10-05 (user-approved): LOC3 follow-ups added; decoder depth / width checks moved to the end
ORDER_AFTER_BASE = ['LOC3', 'LOC2', 'LOC1', 'MSK1', 'MSK3', 'LOC3B', 'LOC4', 'EMB1', 'LOC3D', 'LOC5', 'DEC4', 'EPL3', 'EPO',
                    'CAPE', 'CAPD', 'CAPW']
HEADS_ALT = ['--dec_heads', '4']
EPOCHS_ALT = ['--epochs', '1600']
EPOCH_DEPENDENT = ('CAPD', 'CAPW', 'CAPE')      # run with the schedule chosen by the EPOCHS decision


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _state(name):
    return os.path.join(STATE, name)


def v1_ref():
    j = _load(os.path.join(REF, 'wide_val_engine.json'))
    return j['val_template']['mae'] if j else None


def is_failed(pid):
    return os.path.exists(os.path.join(FAILED, pid))


def _val(flags, key, default):
    return int(flags[flags.index(key) + 1]) if key in flags else default


def _fit_memory(flags):
    """Temporal self-attention together with a deeper or wider decoder exceeds 32 GB at batch 32:
    use batch 16 x 4 accumulation steps (same effective batch of 64)."""
    if '--dec_tsa' in flags and (_val(flags, '--dec_depth', 3) >= 5 or _val(flags, '--dec_dim', 128) >= 192):
        return flags + ['--batch_size', '16', '--accum_iter', '4']
    return flags


def heads_flags(decide=True):
    """Decoder-head setting chosen from DEC1 vs DECH (decided once, then frozen in heads_decision.json)."""
    j = _load(_state('heads_decision.json'))
    if j:
        return j['flags']
    if not decide:
        return None
    r1, rh = result('DEC1', []), result('DECH', [])
    if r1 is None or (rh is None and not is_failed('DECH')):
        return None
    if rh is None:
        out = dict(pick='DEC1', flags=[], reason='DECH failed twice; the 16 heads of v1 are kept')
    elif rh['val_mae'] < r1['val_mae'] - TOL:
        out = dict(pick='DECH', flags=list(HEADS_ALT), reason='validation MAE lower by more than %.3f with 4 heads of 32 dimensions' % TOL)
    else:
        out = dict(pick='DEC1', flags=[], reason='4 wider heads not better by more than %.3f; the 16 heads of v1 are kept' % TOL)
    out.update(val_mae={'DEC1': r1['val_mae'], 'DECH': (rh or {}).get('val_mae')},
               val_AT={'DEC1': r1.get('val_AT'), 'DECH': (rh or {}).get('val_AT')})
    json.dump(out, open(_state('heads_decision.json'), 'w'), indent=1)
    return out['flags']


def base_flags(decide=True):
    """Flags of the base decoder for the 'on_base' pilots (decided once, then frozen in base_decision.json)."""
    j = _load(_state('base_decision.json'))
    if j:
        return j['flags']
    if not decide:
        return None
    hf = heads_flags()
    if hf is None:
        return None
    win = 'DECH' if hf else 'DEC1'
    rw, r2, r3 = result(win, []), result('DEC2', []), result('DEC3', [])
    if rw is None or r2 is None:
        return None
    m1, m2 = rw['val_mae'], r2['val_mae']
    pick, flags, why = win, list(hf), 'temporal self-attention not better than %s by more than %.3f in validation MAE' % (win, TOL)
    if m2 < m1 - TOL:
        pick, flags, why = 'DEC2', list(hf) + ['--dec_tsa'], 'validation MAE lower by more than %.3f with temporal self-attention' % TOL
    elif abs(m2 - m1) <= TOL and None not in (rw.get('val_AT'), r2.get('val_AT')) and r2['val_AT'] < rw['val_AT'] - 0.5:
        pick, flags, why = 'DEC2', list(hf) + ['--dec_tsa'], 'validation MAE tied; downstream validation AT lower by more than 0.5 ms'
    if r3 and r3.get('val_mae') is not None and r3['val_mae'] < min(m1, m2) - TOL:
        pick, flags, why = 'DEC3', [x for x in r3['extra'] if x not in ('--batch_size', '--accum_iter', '16', '4')], 'six-block decoder lower by more than %.3f' % TOL
    out = dict(base=pick, flags=flags, reason=why, val_mae={win: m1, 'DEC2': m2, 'DEC3': (r3 or {}).get('val_mae')},
               val_AT={win: rw.get('val_AT'), 'DEC2': r2.get('val_AT')}, v1_val_mae_template=v1_ref())
    json.dump(out, open(_state('base_decision.json'), 'w'), indent=1)
    return flags


def base_still_converging():
    """Convergence rule of baselines_analyses/convergence_report.py applied to the run of the base decoder."""
    j = _load(_state('base_decision.json'))
    if not j:
        return None
    r = result(j['base'], [])
    if r is None:
        return None
    tools = os.path.join(V2, 'baselines_analyses')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from convergence_report import analyse
    c = analyse(r['run_dir'])
    return None if c is None else dict(still=bool(c['still_converging']), best_epoch=c['best_epoch'], gain_last_200=c['gain_last_200'],
                                       gain_second_half=c['gain_second_half'], base_val_mae=r['val_mae'])


def epochs_flags(decide=True):
    """Training length for the schedule-dependent pilots (decided once, frozen in epochs_decision.json):
    1600 epochs only if the base run was still converging at 800 AND the 1600-epoch run (EPO) lowers the
    validation MAE by more than TOL."""
    j = _load(_state('epochs_decision.json'))
    if j and 'flags' in j:
        return j['flags']
    if not decide:
        return None
    base = base_flags(decide=False)
    conv = base_still_converging()
    if base is None or conv is None:
        return None
    if not conv['still']:
        out = dict(flags=[], epo_run=False, reason='base run not converging any more at 800 epochs by the rule; 800 epochs kept')
    else:
        r = result('EPO', base)
        if r is None and not is_failed('EPO'):
            return None                       # EPO still has to run
        if r is None:
            out = dict(flags=[], epo_run=True, reason='EPO failed twice; 800 epochs kept')
        elif r['val_mae'] < conv['base_val_mae'] - TOL:
            out = dict(flags=list(EPOCHS_ALT), epo_run=True, reason='validation MAE lower by more than %.3f with 1600 epochs' % TOL,
                       val_mae={'base_800': conv['base_val_mae'], 'EPO_1600': r['val_mae']})
        else:
            out = dict(flags=[], epo_run=True, reason='1600 epochs not better by more than %.3f; 800 epochs kept' % TOL,
                       val_mae={'base_800': conv['base_val_mae'], 'EPO_1600': r['val_mae']})
    out['convergence_of_base_run'] = conv
    json.dump(out, open(_state('epochs_decision.json'), 'w'), indent=1)
    return out['flags']


def epo_needed():
    conv = base_still_converging()
    return bool(conv and conv['still'])


def flags_for(pid, decide=True):
    """Flags a pilot is built on: [] for the decoder pilots, the base decoder for the single-factor pilots,
    base + chosen training length for the schedule-dependent capacity checks.  None = not decidable yet."""
    if not PILOTS[pid][1]:
        return []
    base = base_flags(decide=decide)
    if base is None:
        return None
    if pid in EPOCH_DEPENDENT:
        ef = epochs_flags(decide=decide)
        if ef is None:
            return None
        return list(base) + list(ef)
    return list(base)


def args_of(pid, base=None):
    extra, on_base, _ = PILOTS[pid]
    extra = list(extra)
    if pid == 'DEC2':
        extra = list(heads_flags(decide=False) or []) + extra
    if pid == 'DEC3':
        d = _load(_state('dec3_flags.json'))
        extra = d['flags'] if d else extra
    flags = _fit_memory((list(base) if (on_base and base) else []) + extra)
    cmd = COMMON + flags + ['--tag', pid]
    # pilots train and validate on CLINICAL recordings only (user decision 2026-10-04: simulated-data
    # experiments are repeated only after the final model is fixed, as a Discussion item)
    assert not any(x in cmd for x in ('--train_sources', '--init_from', '--real_repeat', '--val_source')), cmd
    return cmd


def run_dir_of(cmd_args):
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import train
    a = train.get_args().parse_args(cmd_args)
    return os.path.join(RUNS, train.build_run_name(a, a.batch_size * a.accum_iter))


def result(pid, base):
    """Metrics of a finished pilot (None if not finished)."""
    if PILOTS[pid][1] and base is None:
        return None
    cmd = args_of(pid, base)
    rd = run_dir_of(cmd)
    st = _load(os.path.join(rd, 'training_stats.json'))
    if not (st and os.path.exists(os.path.join(rd, 'test_compare.json')) and st.get('epochs_done') == st.get('epochs_total')):
        return None
    out = dict(id=pid, run_dir=rd, extra=cmd[len(COMMON):-2],
               val_mae=st.get('val_best_mae'), best_epoch=st.get('best_epoch'),
               hours=st.get('elapsed_train_hours'), params=st.get('params'), ms_per_iter=st.get('ms_per_iter'),
               infer_ms=st.get('infer_ms_per_sample'), gpu_gb=st.get('gpu_peak_alloc_gb'))
    d = _load(os.path.join(rd, 'at_rt_results_val.json'))
    if d:
        s = d['splits']['val'].get('src4_recon_valid__vs__commercial', {})
        out.update(val_AT=s.get('AT_mae_mean'), val_RT=s.get('RT_mae_mean'), val_HSPcc=s.get('hsp_cc_mean'))
    pm = _load(os.path.join(rd, 'pilot_metrics_val.json'))
    if pm:
        out.update(val_cc=pm['template']['bsp_cc'], dcc=pm['template']['dcc'], boundary=pm['template']['boundary_ratio'],
                   val_cc_km=pm['kmeans']['bsp_cc'], dcc_km=pm['kmeans']['dcc'], boundary_km=pm['kmeans']['boundary_ratio'])
    # second deployment layout (per-recording k-means; user decision 2026-10-06: both layouts for every comparison)
    ve = _load(os.path.join(rd, 'val_engine.json'))
    if ve:
        out.update(val_mae_km=ve.get('val_kmeans', {}).get('mae'), val_mae_engine_tmpl=ve.get('val_template', {}).get('mae'))
    km = percase_means(os.path.join(V2, 'results', 'val_percase', f'{pid}_vc16_val_kmeans.csv'))
    if km:
        out.update(val_AT_km=km.get('AT_mae'), val_RT_km=km.get('RT_mae'), val_HSPcc_km=km.get('hsp_cc'))
    return out


def percase_means(csv_path, source='src4_recon_valid', ref='commercial'):
    """Cohort means of a per-recording validation CSV (inverse_evaluation/eval_downstream.py) for one source
    vs one reference; None if the file is missing."""
    if not os.path.exists(csv_path):
        return None
    import csv as _csv
    acc = {}
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            if r.get('source') != source or r.get('ref') != ref:
                continue
            for k in ('AT_mae', 'RT_mae', 'hsp_cc', 'bsp_cc'):
                try:
                    v = float(r[k])
                except (KeyError, TypeError, ValueError):
                    continue
                if v == v:
                    acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()} if acc else None


def test_metrics(rd, pid):
    """TEST-cohort metrics of a run under both layouts (user request 2026-10-08: shown next to the validation
    results; pilot decisions were taken on validation before these were looked at): engine BSP MAE / CC from
    test_compare.json, downstream AT / RT from the per-recording CSVs when they exist."""
    out = {}
    tc = _load(os.path.join(rd, 'test_compare.json')) if rd else None
    for split, tag in (('test_internal', 'int'), ('test_external', 'ext')):
        for lay, key, csvlay in (('tmpl', 'template', 'template'), ('km', 'cluster-kmean', 'kmeans')):
            e = (tc or {}).get('results', {}).get(split, {}).get(key, {})
            out[f'{tag}_mae_{lay}'], out[f'{tag}_cc_{lay}'] = e.get('mae'), e.get('cc')
            if pid == 'v1':
                f = (os.path.join(V2_GENERATED, 'percase_template', f'ours-v1_vc16_{split}.csv') if lay == 'tmpl'
                     else os.path.join(V1_GENERATED, 'vc_percase', f'vc16_{split}.csv'))
            else:
                f = os.path.join(PERCASE_TEST, f'{pid}_vc16_{split}_{csvlay}.csv')
            m = percase_means(f)
            if m:
                out[f'{tag}_AT_{lay}'], out[f'{tag}_RT_{lay}'] = m.get('AT_mae'), m.get('RT_mae')
                if out[f'{tag}_cc_{lay}'] is None and m.get('bsp_cc') is not None:
                    out[f'{tag}_cc_{lay}'] = m['bsp_cc']
    return out


def v1_ref_km():
    """v1 final model: engine validation MAE under per-recording k-means (results/v1_reference)."""
    j = _load(os.path.join(REF, 'wide_val_engine.json'))
    return j['val_kmeans']['mae'] if j else None


def next_pilot():
    if result('DEC1', []) is None:
        return 'STOP: DEC1 failed twice; see training_sweeps/logs/DEC1.out' if is_failed('DEC1') else 'DEC1'
    if result('DECH', []) is None and not is_failed('DECH'):
        return 'DECH'
    hf = heads_flags()
    if result('DEC2', []) is None:
        return 'STOP: DEC2 failed twice; see training_sweeps/logs/DEC2.out' if is_failed('DEC2') else 'DEC2'
    ref = v1_ref()
    win = 'DECH' if hf else 'DEC1'
    m1, m2 = result(win, [])['val_mae'], result('DEC2', [])['val_mae']
    if _load(_state('base_decision.json')) is None and ref is not None and min(m1, m2) > ref + BAD:
        r3 = result('DEC3', [])
        if r3 is None:
            if is_failed('DEC3'):
                return 'STOP: DEC3 failed twice; see training_sweeps/logs/DEC3.out'
            flags = ['--dec_depth', '6'] + list(hf) + (['--dec_tsa'] if m2 < m1 - TOL else [])
            json.dump({'flags': flags}, open(_state('dec3_flags.json'), 'w'))
            return 'DEC3'
        if r3['val_mae'] > ref + BAD:
            return ('STOP: the query decoder trails v1 on validation (v1 %.4f; %s %.4f, DEC2 %.4f, DEC3 %.4f); '
                    'decision needed before the remaining pilots' % (ref, win, m1, m2, r3['val_mae']))
    base_flags()
    for pid in ORDER_AFTER_BASE:
        if pid == 'EPO' and not epo_needed():
            continue                          # base run converged at 800 epochs -> no longer schedule
        fl = flags_for(pid)
        if fl is None:
            continue
        if result(pid, fl) is None and not is_failed(pid):
            return pid
    return 'DONE'


def summary():
    base = base_flags(decide=False)
    ref = v1_ref()
    ra = _load(os.path.join(REF, 'wide_at_rt_val_template.json'))
    rp = _load(os.path.join(REF, 'wide_pilot_metrics_val.json'))
    ref_km = v1_ref_km()
    L = ['# Pilot results (validation only; N=16; clinical recordings; one training seed; BOTH deployment layouts)', '',
         'tmpl = fixed template layout (checkpoint selection), km = per-recording k-means layout (user decision 2026-10-06: every',
         'comparison is made under both layouts). Selection metric = engine validation completion MAE (lower is better;',
         'seed-noise SD about 0.001; the tmpl value is the best-epoch value of training, the km value is the same checkpoint',
         're-evaluated under k-means). Downstream columns = Ours vs. commercial reference on the 106 validation recordings',
         '(5 patients). Boundary ratio = step across patch boundaries / step inside patches (1.0 for measured signals; tmpl).', '',
         '| ID | change | val MAE tmpl | val MAE km | vs v1 tmpl / km | val CC tmpl | dCC tmpl | boundary | AT tmpl | AT km | RT tmpl | RT km | HSP CC tmpl | HSP CC km | train h | infer ms | params M | status |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v, n=4: ('%.*f' % (n, v)) if isinstance(v, (int, float)) else '--'
    if ref is not None:
        s = (ra or {}).get('splits', {}).get('val', {}).get('src4_recon_valid__vs__commercial', {})
        t = (rp or {}).get('template', {})
        km = percase_means(os.path.join(V2, 'results', 'val_percase', 'v1_vc16_val_kmeans.csv')) or {}
        L.append(f"| v1 | self-attention decoder (final v1 model, same layouts) | {f(ref)} | {f(ref_km)} | 0 / 0 | {f(t.get('bsp_cc'))} | {f(t.get('dcc'))} | "
                 f"{f(t.get('boundary_ratio'), 2)} | {f(s.get('AT_mae_mean'), 2)} | {f(km.get('AT_mae'), 2)} | {f(s.get('RT_mae_mean'), 2)} | {f(km.get('RT_mae'), 2)} | "
                 f"{f(s.get('hsp_cc_mean'), 3)} | {f(km.get('hsp_cc'), 3)} | 28.6 | 376 | 5.40 | reference |")
    for pid, (extra, on_base, desc) in PILOTS.items():
        fl = flags_for(pid, decide=False)
        r = result(pid, fl) if fl is not None else None
        if r is None:
            st = 'failed' if is_failed(pid) else ('waiting for earlier decisions' if (on_base and fl is None) else 'pending / running')
            if pid == 'DEC3' and not os.path.exists(_state('dec3_flags.json')):
                st = 'not needed so far'
            ej = _load(_state('epochs_decision.json'))
            if pid == 'EPO' and ej and not ej.get('epo_run', True):
                st = 'not needed (base run converged at 800 epochs)'
            L.append(f"| {pid} | {desc} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | {st} |")
            continue
        d = (r['val_mae'] - ref) if ref is not None else None
        dk = (r['val_mae_km'] - ref_km) if (ref_km is not None and isinstance(r.get('val_mae_km'), float)) else None
        L.append(f"| {pid} | {desc} | {f(r['val_mae'])} | {f(r.get('val_mae_km'))} | {('%+.4f' % d) if d is not None else '--'} / {('%+.4f' % dk) if dk is not None else '--'} | "
                 f"{f(r.get('val_cc'))} | {f(r.get('dcc'))} | {f(r.get('boundary'), 2)} | {f(r.get('val_AT'), 2)} | {f(r.get('val_AT_km'), 2)} | "
                 f"{f(r.get('val_RT'), 2)} | {f(r.get('val_RT_km'), 2)} | {f(r.get('val_HSPcc'), 3)} | {f(r.get('val_HSPcc_km'), 3)} | {f(r.get('hours'), 1)} | "
                 f"{f(r.get('infer_ms'), 0)} | {f((r.get('params') or 0) / 1e6, 2)} | done (best epoch {r.get('best_epoch')}) |")
    h, j, e = _load(_state('heads_decision.json')), _load(_state('base_decision.json')), _load(_state('epochs_decision.json'))
    L += ['', ('Decoder heads: **%s** (%s).' % (h['pick'], h['reason'])) if h else 'Decoder heads: not decided yet (after DEC1 and DECH).',
          ('Base decoder for the single-factor pilots: **%s**, flags `%s` (%s).' % (j['base'], ' '.join(j['flags']) or 'none', j['reason'])) if j else
          'Base decoder for the single-factor pilots: not decided yet (after DEC2).',
          ('Training length for the capacity checks: **%s** (%s).' % ('1600 epochs' if e['flags'] else '800 epochs', e['reason'])) if e else
          'Training length: 800 epochs for all pilots; a 1600-epoch run (EPO) follows if the base run is still converging at 800.', '']
    # ---- second table: TEST cohorts (both layouts) ------------------------------------------------------
    f2 = lambda a, b, n: f"{f(a, n)} / {f(b, n)}"
    L += ['## Test cohorts (same runs; both deployment layouts). Pilot decisions were taken on the validation table above before these were looked at.', '',
          'BSP CC = engine held-out completion CC (test_compare.json); AT / RT = downstream MAE vs. commercial (ms, mean over recordings) from the',
          'per-recording CSVs (`training_sweeps/eval_candidate_test.sh`, `baselines_analyses/generated/percase_candidate/`); "--" = not evaluated yet.', '',
          '| ID | int BSP CC tmpl / km | ext BSP CC tmpl / km | int AT tmpl / km | ext AT tmpl / km | int RT tmpl / km | ext RT tmpl / km |',
          '|---|---|---|---|---|---|---|']
    rows = [('v1', V1_RUN)]
    for pid, (extra, on_base, desc) in PILOTS.items():
        fl = flags_for(pid, decide=False)
        r = result(pid, fl) if fl is not None else None
        if r is not None:
            rows.append((pid, r['run_dir']))
    for pid, rd in rows:
        t = test_metrics(rd, pid)
        L.append(f"| {pid} | {f2(t.get('int_cc_tmpl'), t.get('int_cc_km'), 3)} | {f2(t.get('ext_cc_tmpl'), t.get('ext_cc_km'), 3)} | "
                 f"{f2(t.get('int_AT_tmpl'), t.get('int_AT_km'), 2)} | {f2(t.get('ext_AT_tmpl'), t.get('ext_AT_km'), 2)} | "
                 f"{f2(t.get('int_RT_tmpl'), t.get('int_RT_km'), 2)} | {f2(t.get('ext_RT_tmpl'), t.get('ext_RT_km'), 2)} |")
    L.append('')
    open(RESULTS_MD, 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['next', 'summary', 'cmd']); ap.add_argument('pid', nargs='?')
    a = ap.parse_args()
    if a.what == 'summary':
        summary()
    elif a.what == 'cmd':
        c = args_of(a.pid, flags_for(a.pid, decide=False))
        print('python train.py ' + ' '.join(c)); print(run_dir_of(c))
    else:
        nxt = next_pilot()
        if nxt.startswith(('DONE', 'STOP')):
            print(nxt)
        else:
            c = args_of(nxt, flags_for(nxt))
            print('\t'.join([nxt, run_dir_of(c), ' '.join(c)]))
