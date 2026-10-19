"""Query-interaction diagnostics for the Discussion paragraph "Why independent queries suffice" (paper plan §7).

For a run with decoder query interaction (--dec_ssa spatial / --dec_tsa temporal) and for the final model:
  A. test-time switch-off : engine val MAE / CC under both deployment layouts with the interaction sub-layers
                            active and bypassed (the residual branch is skipped, all other trained weights kept)
  B. attention diagnostics: self mass (diagonal), normalised entropy, exp(entropy) (= effective number of attended
                            queries) and sequence-neighbour mass (|i-j| = 1; meaningful for the temporal attention)
                            of every interaction attention, per decoder block, on validation batches (template layout)
  C. consistency test     : with spatial interaction the completed signal at one position depends on which other
                            positions are queried in the same pass.  Joint decoding of all masked electrodes vs.
                            decoding in two random halves vs. decoding single electrodes alone (PosQMAE.complete_at);
                            max |delta| and RMS(delta) / RMS(signal) on the common positions (0 for independent queries).
Outputs baselines_analyses/generated/interaction_diag_<ID>.json; --md rewrites results/interaction_diag.md from all such
files.  Prints aggregates only (no recording IDs)."""
import os, sys, json, glob, argparse, warnings, math
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction')); sys.path.insert(0, os.path.join(V2, 'inverse_evaluation'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from paths import assert_v2
from dataset import build_experiment_datasets
from engine import run_eval, select_visible_batch, global_norm
from build_sources import load_model
from models_v2 import QueryBlock

LAYOUTS = {'template': ('template', 'greedy'), 'kmeans': ('cluster', 'kmean')}
GEN = os.path.join(V2, 'baselines_analyses', 'generated')
MD = os.path.join(V2, 'results', 'interaction_diag.md')


def run_dir(pid):
    c = sorted(glob.glob(os.path.join(V2, 'runs', 'data-cohortA+cohortB_vs_external', f'{pid}__*')))
    assert len(c) == 1, (pid, c)
    return c[0]


def qblocks(model):
    return [m for m in model.modules() if isinstance(m, QueryBlock)]


def set_interaction(model, spatial=None, temporal=None):
    for b in qblocks(model):
        if spatial is not None and hasattr(b, 'attn_s'):
            b.spatial_sa = bool(spatial)
        if temporal is not None and hasattr(b, 'attn_t'):
            b.temporal_sa = bool(temporal)


class AttnStats:
    """Forward pre-hook on an Attention module used as self-attention: recompute the attention probabilities from the
    module's own q / k projections and accumulate self mass, normalised entropy, exp(entropy) and neighbour mass."""

    def __init__(self, chunk=48):
        self.chunk = chunk
        self.n = 0; self.self_mass = 0.0; self.ent = 0.0; self.eff = 0.0; self.nb_mass = 0.0; self.seq_len = None

    def __call__(self, mod, inp):
        x = inp[0].detach()
        B, n, D = x.shape; h = mod.h; d = D // h
        self.seq_len = n
        idx = torch.arange(n, device=x.device)
        nb = ((idx[None, :] - idx[:, None]).abs() == 1).float()
        for s in range(0, B, self.chunk):
            xs = x[s:s + self.chunk]
            q = mod.q(xs).view(-1, n, h, d).transpose(1, 2)
            k = mod.k(xs).view(-1, n, h, d).transpose(1, 2)
            a = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(d), dim=-1).float()     # (b, h, n, n)
            diag = torch.diagonal(a, dim1=-2, dim2=-1)                                       # (b, h, n)
            ent = -(a * (a + 1e-12).log()).sum(-1)
            self.self_mass += diag.sum().item(); self.ent += (ent / math.log(n)).sum().item()
            self.eff += ent.exp().sum().item(); self.nb_mass += (a * nb).sum(-1).sum().item()
            self.n += diag.numel()

    def result(self):
        if self.n == 0:
            return None
        return dict(seq_len=self.seq_len, self_mass=self.self_mass / self.n, norm_entropy=self.ent / self.n,
                    eff_n=self.eff / self.n, neighbour_mass=self.nb_mass / self.n, n_rows=self.n)


def _prep(batch, dev, vc, norm_mode, rsel):
    ecg, missing_mask, coords, mu, sigma, sids = batch
    ecg = ecg.to(dev); missing = missing_mask.to(dev); coords = coords.to(dev); active = ~missing
    vis_np = select_visible_batch(coords.cpu().numpy(), active.cpu().numpy(), vc, 'template', 'greedy', rsel)
    visible = torch.from_numpy(vis_np).to(dev)
    if norm_mode == 'global':
        ecg, _ = global_norm(ecg, visible)
    if ecg.dim() == 4:
        ecg = ecg.squeeze(1)
    return ecg, active, coords, visible


@torch.no_grad()
def attention_stats(model, batches, dev, vc, seed, norm_mode):
    stats, handles = {}, []
    for bi, blk in enumerate(qblocks(model)):
        if hasattr(blk, 'attn_s'):
            st = AttnStats(); handles.append(blk.attn_s.register_forward_pre_hook(st)); stats[f'spatial_block{bi}'] = st
        if hasattr(blk, 'attn_t'):
            st = AttnStats(); handles.append(blk.attn_t.register_forward_pre_hook(st)); stats[f'temporal_block{bi}'] = st
    rsel = np.random.default_rng(seed)
    for batch in batches:
        ecg, active, coords, visible = _prep(batch, dev, vc, norm_mode, rsel)
        model.forward_fixedn(ecg, active, coords, visible, w_spatial=0.0, w_temporal=0.0)
    for h in handles:
        h.remove()
    return {k: v.result() for k, v in stats.items()}


@torch.no_grad()
def consistency(model, batches, dev, vc, seed, norm_mode, n_single=8):
    rng = np.random.default_rng(seed + 1); rsel = np.random.default_rng(seed)
    half_max, half_rel, single_max, single_rel, srms_all = [], [], [], [], []
    for batch in batches:
        ecg, active, coords, visible = _prep(batch, dev, vc, norm_mode, rsel)
        for b in range(ecg.shape[0]):
            xb, cb, vb = ecg[b:b + 1], coords[b:b + 1], visible[b:b + 1]
            m_idx = (~vb[0]).nonzero(as_tuple=False).squeeze(1)              # every masked electrode (as in eval)
            qc = cb[:, m_idx]
            joint = model.complete_at(xb, cb, vb, qc)[0]                      # (Q, L)
            perm = torch.from_numpy(rng.permutation(len(m_idx))).to(dev)
            h1, h2 = perm[: len(m_idx) // 2], perm[len(m_idx) // 2:]
            halves = torch.zeros_like(joint)
            halves[h1] = model.complete_at(xb, cb, vb, qc[:, h1])[0]
            halves[h2] = model.complete_at(xb, cb, vb, qc[:, h2])[0]
            srms = max(joint.pow(2).mean().sqrt().item(), 1e-8)
            d = halves - joint
            half_max.append(d.abs().max().item()); half_rel.append(d.pow(2).mean().sqrt().item() / srms)
            sel = rng.choice(len(m_idx), size=min(n_single, len(m_idx)), replace=False)
            dmax, dsq, cnt = 0.0, 0.0, 0
            for s in sel:
                single = model.complete_at(xb, cb, vb, qc[:, [int(s)]])[0, 0]
                ds = single - joint[int(s)]
                dmax = max(dmax, ds.abs().max().item()); dsq += ds.pow(2).sum().item(); cnt += ds.numel()
            single_max.append(dmax); single_rel.append(math.sqrt(dsq / max(cnt, 1)) / srms); srms_all.append(srms)
    return dict(n_rec=len(half_max), signal_rms=float(np.mean(srms_all)),
                half_rel_rms=float(np.mean(half_rel)), half_max_abs=float(np.max(half_max)),
                single_rel_rms=float(np.mean(single_rel)), single_max_abs=float(np.max(single_max)))


def write_md():
    files = sorted(glob.glob(os.path.join(GEN, 'interaction_diag_*.json')))
    L = ['# Query-interaction diagnostics (baselines_analyses/interaction_diag.py)', '',
         'A: engine val MAE (both layouts) with the interaction sub-layers on / bypassed at test time.  '
         'B: interaction attention on validation batches (template layout; mean over heads, queries and blocks): self mass, '
         'normalised entropy, effective number of attended queries exp(H), sequence-neighbour mass.  '
         'C: continuous-field consistency (complete_at): RMS(delta)/RMS(signal) and max |delta| between joint decoding of all '
         'masked electrodes and decoding in two halves / single electrodes.', '',
         '| run | interaction | val MAE template on / off | val MAE k-means on / off | attn self mass | norm. entropy | eff. n | neighbour mass | C half rel / max | C single rel / max |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for f in files:
        r = json.load(open(f))
        kind = '+'.join([k for k, v in (('spatial', r['spatial']), ('temporal', r['temporal'])) if v]) or 'none'
        so = r.get('switch_off', {})
        def oo(lab):
            if lab not in so: return '--'
            return f"{so[lab]['on']['mae']:.4f} / {so[lab]['off']['mae']:.4f}"
        at = [v for v in (r.get('attention') or {}).values() if v]
        def am(k):
            return f"{np.mean([v[k] for v in at]):.3f}" if at else '--'
        c = r.get('consistency')
        cs = (f"{c['half_rel_rms']:.2e} / {c['half_max_abs']:.2e}", f"{c['single_rel_rms']:.2e} / {c['single_max_abs']:.2e}") if c else ('--', '--')
        L.append(f"| {r['id']} | {kind} | {oo('template')} | {oo('kmeans')} | {am('self_mass')} | {am('norm_entropy')} | {am('eff_n')} | {am('neighbour_mass')} | {cs[0]} | {cs[1]} |")
    L.append('')
    for f in files:
        r = json.load(open(f))
        if r.get('attention'):
            L.append(f"Per-block attention, {r['id']}: " + '; '.join(
                f"{k}: self {v['self_mass']:.3f}, H/logn {v['norm_entropy']:.3f}, eff {v['eff_n']:.1f}/{v['seq_len']}, nb {v['neighbour_mass']:.3f}"
                for k, v in r['attention'].items() if v))
    open(assert_v2(MD), 'w').write('\n'.join(L) + '\n'); print('wrote', MD)


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', action='append', default=[])
    ap.add_argument('--n_batches', type=int, default=6); ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4); ap.add_argument('--skip_eval', action='store_true')
    ap.add_argument('--md', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pid in a.id:
        run = run_dir(pid); args = json.load(open(os.path.join(run, 'args.json')))
        _, val_ds, _, _, _ = build_experiment_datasets(crop_len=args['crop_len'], preprocess=True, seed=args['seed'], aug_times=1,
                                                       pp_norm=args['pp_norm'], tag='exp', min_active=args['vc'] + args['min_recon'],
                                                       prep_cache=True, norm_mode=args['norm_mode'], verbose=False)
        model, _, _ = load_model(run, dev)
        has_s = any(hasattr(b, 'attn_s') for b in qblocks(model)); has_t = any(hasattr(b, 'attn_t') for b in qblocks(model))
        res = dict(id=pid, run=os.path.basename(run), spatial=has_s, temporal=has_t, n_batches=a.n_batches)
        dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
        if not a.skip_eval and (has_s or has_t):
            res['switch_off'] = {}
            for lab, (mode, cm) in LAYOUTS.items():
                out = {}
                for state in ('on', 'off'):
                    set_interaction(model, spatial=(state == 'on') if has_s else None, temporal=(state == 'on') if has_t else None)
                    with torch.no_grad():
                        r = run_eval(model, dl, dev, args['vc'], mode, cm, seed=args['seed'], norm_mode=args['norm_mode'])
                    out[state] = {k: float(r[k]) for k in ('mae', 'rmse', 'cc')}
                    print(f'[{pid}] A {lab} interaction {state}: mae {r["mae"]:.4f} cc {r["cc"]:.4f}', flush=True)
                set_interaction(model, spatial=True if has_s else None, temporal=True if has_t else None)
                res['switch_off'][lab] = out
        batches = []
        for i, b in enumerate(dl):
            if i >= a.n_batches:
                break
            batches.append(b)
        if has_s or has_t:
            res['attention'] = attention_stats(model, batches, dev, args['vc'], args['seed'], args['norm_mode'])
            for k, v in res['attention'].items():
                print(f'[{pid}] B {k}: ' + json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}), flush=True)
        if getattr(model, 'pos_mode', 'physical') != 'index':
            res['consistency'] = consistency(model, batches, dev, args['vc'], args['seed'], args['norm_mode'])
            print(f'[{pid}] C ' + json.dumps({k: (f'{v:.3e}' if isinstance(v, float) else v) for k, v in res['consistency'].items()}), flush=True)
        json.dump(res, open(assert_v2(os.path.join(GEN, f'interaction_diag_{pid}.json')), 'w'), indent=1)
        print('wrote', f'interaction_diag_{pid}.json', flush=True)
    if a.md:
        write_md()


if __name__ == '__main__':
    main()
