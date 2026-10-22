"""Inference time per sample of a finished run, re-measured on an idle GPU with the protocol of the training script's
test step: engine.run_eval over both clinical test sets under every selection config of engine.TEST_CONFIGS (the fixed
template is skipped for region-trained runs, as in engine.test_all_modes), batch size of the run, GPU-synced forward +
unpatchify; the mean over the configs is what training_stats.json::infer_ms_per_sample holds.  Used for the runs whose
automatic test step was re-run next to another training (P3REGF / P3REGB, 2026-10-16: 'infer' cells of tab:ablation)
and, as a check of the protocol, for LOC3.  Prints aggregates only.
  python baselines_analyses/infer_time_idle.py --pid LOC3 P3REGF P3REGB [--max_batches N] [--out_dir DIR]   -> <run>/infer_idle.json
"""
import os, sys, json, time, glob, argparse, warnings, subprocess
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
from dataset import build_experiment_datasets                               # noqa: E402
from engine import run_eval, TEST_CONFIGS, cfg_label                        # noqa: E402
from build_sources import load_model                                        # noqa: E402
RUNS = os.path.join(V2, 'runs', 'data-cohortA+cohortB_vs_external')


def run_dir(pid):
    g = glob.glob(os.path.join(RUNS, f'{pid}__*'))
    assert len(g) == 1, (pid, g)
    return g[0]


def gpu_state():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.strip().splitlines()[0]
        u, m = o.split(',')
        return int(u), int(m)
    except Exception:
        return None, None


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', nargs='+', required=True)
    ap.add_argument('--max_batches', type=int, default=0, help='smoke test: only the first N batches per split')
    ap.add_argument('--out_dir', default=None, help='write <pid>_infer_idle.json here instead of into the run directory')
    ap.add_argument('--num_workers', type=int, default=8)
    a = ap.parse_args()
    dev = torch.device('cuda')
    util, mem = gpu_state()
    print(f'GPU before: utilisation {util} %, memory used {mem} MiB', flush=True)
    for pid in a.pid:
        rd = run_dir(pid)
        args = json.load(open(os.path.join(rd, 'args.json')))
        region = args.get('select_region') or 'all'
        _, _, ti_ds, te_ds, _ = build_experiment_datasets(
            crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1, pp_norm=args['pp_norm'], tag='exp',
            min_active=args['vc'] + args['min_recon'], prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
        model, _, _ = load_model(rd, dev)
        res = {'run': os.path.basename(rd), 'pid': pid, 'batch_size': args['batch_size'], 'region': region,
               'gpu': torch.cuda.get_device_name(0), 'gpu_util_before': util, 'gpu_mem_before_mib': mem,
               'when': time.strftime('%F %T'), 'smoke': bool(a.max_batches), 'configs': {}}
        for split, ds in (('test_internal', ti_ds), ('test_external', te_ds)):
            if a.max_batches:
                ds = Subset(ds, range(min(len(ds), a.max_batches * args['batch_size'])))
            dl = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=a.num_workers, pin_memory=True)
            for mode, cm in TEST_CONFIGS:
                if mode == 'template' and region != 'all':      # as in engine.test_all_modes
                    continue
                lab = cfg_label(mode, cm)
                r = run_eval(model, dl, dev, args['vc'], mode, cm, seed=args['seed'], norm_mode=args['norm_mode'], region=region)
                res['configs'][f'{split}/{lab}'] = {'infer_ms_per_sample': r['infer_ms_per_sample'],
                                                    'select_ms_per_sample': r['select_ms_per_sample'], 'n': r['n_samples'], 'mae': r['mae']}
                print(f"  {pid} {split} {lab:16s} infer {r['infer_ms_per_sample']:.2f} ms/sample  mae {r['mae']:.4f}  (n={r['n_samples']})", flush=True)
        vals = [c['infer_ms_per_sample'] for c in res['configs'].values()]
        res['infer_ms_per_sample'] = round(float(np.mean(vals)), 2)
        st = json.load(open(os.path.join(rd, 'training_stats.json'))) if os.path.exists(os.path.join(rd, 'training_stats.json')) else {}
        res['infer_ms_training_stats'] = st.get('infer_ms_per_sample')
        out = os.path.join(a.out_dir, f'{pid}_infer_idle.json') if a.out_dir else os.path.join(rd, 'infer_idle.json')
        json.dump(res, open(out, 'w'), indent=1)
        print(f'{pid}: idle-GPU inference {res["infer_ms_per_sample"]} ms/sample (training_stats: {res["infer_ms_training_stats"]}) -> {out}', flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
