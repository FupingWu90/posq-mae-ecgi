"""Is a run still converging at the end of its schedule?

Reads the validation history (eval_test.log: 'VAL  epNNNN ... mae X') and the per-epoch training loss
(train.log: '[epoch_avg] loss X') of a run and reports the validation MAE at fixed epochs, the best epoch,
the gain over the last 100 / 200 / 400 epochs and the same numbers for a reference run (default: v1).
Note: the learning rate follows a cosine decay to zero at the last epoch, so the curve flattens at the end
by construction; the gain of the second half of the schedule is the more telling number.

Rule used for 'still converging': the best validation MAE is reached in the last 10% of the schedule AND
the validation MAE improved by more than 0.002 (about 2 SD of the seed noise) over the last 200 epochs.
  python baselines_analyses/convergence_report.py --run_dir <run> [--ref_dir <run>]
"""
import os, re, sys, json, argparse
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from paths import V1_RUNS

V1_WIDE = os.path.join(V1_RUNS, 'fixedn_ecgi_wide_vc16_cluster-greedy_val-cluster-kmean_nm-gl_ws0-wt3-rw100_sc-recon_k6s10_ebs64_crop5000_ep800')


def history(run_dir):
    val, tr = {}, {}
    p = os.path.join(run_dir, 'eval_test.log')
    if os.path.exists(p):
        for l in open(p):
            m = re.search(r'VAL\s+ep(\d+).*?\bmae ([0-9.]+)', l)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    p = os.path.join(run_dir, 'train.log')
    if os.path.exists(p):
        for l in open(p):
            m = re.search(r'epoch (\d+)\s+\[epoch_avg\]\s+loss ([0-9.]+)', l)
            if m:
                tr[int(m.group(1))] = float(m.group(2))
    return val, tr


def analyse(run_dir):
    val, tr = history(run_dir)
    if not val:
        return None
    eps = sorted(val)
    last = eps[-1]
    best_ep = min(eps, key=lambda e: val[e])
    at = lambda e: val[min(eps, key=lambda x: abs(x - e))]
    avg = lambda e0, e1: (sum(tr[e] for e in tr if e0 <= e <= e1) / max(1, sum(1 for e in tr if e0 <= e <= e1))) if tr else None
    total = last + 1
    r = dict(run=os.path.basename(run_dir.rstrip('/')), last_eval_epoch=last, best_epoch=best_ep, best=val[best_ep], final=val[last],
             val_at={e: at(e) for e in (200, 400, 600, 700) if e < last},
             gain_last_100=at(last - 100) - val[last], gain_last_200=at(last - 200) - val[last], gain_second_half=at(total // 2) - val[last],
             train_loss={'%d-%d' % (a, b): avg(a, b) for a, b in ((total - 220, total - 200), (total - 120, total - 100), (total - 20, total - 1))})
    r['still_converging'] = bool(best_ep >= 0.9 * last and r['gain_last_200'] > 0.002)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); ap.add_argument('--ref_dir', default=V1_WIDE)
    a = ap.parse_args()
    for name, d in (('run', a.run_dir), ('reference', a.ref_dir)):
        r = analyse(d)
        if r is None:
            print(name, ': no validation history'); continue
        print(f"{name}: {r['run'][:60]}")
        print(f"   validation MAE at epochs {r['val_at']} | final (ep {r['last_eval_epoch']}) {r['final']:.4f} | best {r['best']:.4f} at epoch {r['best_epoch']}")
        print(f"   gain over the last 100 epochs {r['gain_last_100']:+.4f} | last 200 {r['gain_last_200']:+.4f} | second half of the schedule {r['gain_second_half']:+.4f}")
        print(f"   mean training loss in epoch windows: " + ', '.join(f'{k}: {v:.4f}' for k, v in r['train_loss'].items() if v is not None))
        print(f"   still converging by the rule: {r['still_converging']}")
        if name == 'run':
            json.dump(r, open(os.path.join(a.run_dir, 'convergence.json'), 'w'), indent=1)
