"""Install the released weights of a PosQ-MAE model (N = 16, training seed 42) as a run directory.

Two models are released:
  final   results/final_model/   the final model of the paper (vest reference, no noise augmentation)    -> FINAL__...
  robust  results/robust_model/  the same recipe with input and targets re-referenced to the mean of the visible
                                 electrodes and noise augmentation of the visible channels (reference-invariant,
                                 robust to in-band noise; Supplementary Note 19 of the paper)                -> ROBUST__...
The evaluation scripts resolve a model through the run directory that train.py would have created, so this
script builds runs/data-<dataset_tag>/<TAG>__<configuration>/ from results/<model>_model/ (args.json,
keep_idx.npy, checkpoints/checkpoint_best.pth) and copies, or with --link symlinks, the three files.  Then:

    bash training_sweeps/eval_test_runs.sh FINAL      # (or ROBUST) both test cohorts x both deployment layouts

Usage: python bsp_reconstruction/install_final_model.py [--model final|robust] [--runs <dir>] [--link]
"""
import argparse, json, os, shutil, sys, types

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train  # noqa: E402  (build_run_name; importing train.py has no side effects)

SRCS = {'final': os.path.join(ROOT, 'results', 'final_model'), 'robust': os.path.join(ROOT, 'results', 'robust_model')}
FILES = ('args.json', 'keep_idx.npy', os.path.join('checkpoints', 'checkpoint_best.pth'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default=os.path.join(ROOT, 'runs'), help='runs directory (default: <repo>/runs)')
    p.add_argument('--link', action='store_true', help='symlink the files instead of copying them')
    p.add_argument('--model', default='final', choices=sorted(SRCS), help='which released model to install (default: final)')
    a = p.parse_args()
    SRC = SRCS[a.model]
    args = json.load(open(os.path.join(SRC, 'args.json')))
    run = train.build_run_name(types.SimpleNamespace(**args), args['batch_size'] * args['accum_iter'])
    dst = os.path.join(a.runs, f"data-{args['dataset_tag']}", run)
    os.makedirs(os.path.join(dst, 'checkpoints'), exist_ok=True)
    for rel in FILES:
        s, d = os.path.abspath(os.path.join(SRC, rel)), os.path.join(dst, rel)
        if os.path.lexists(d):
            os.remove(d)
        (os.symlink if a.link else shutil.copyfile)(s, d)
    print(f'installed the {a.model} model in', dst)


if __name__ == '__main__':
    main()
