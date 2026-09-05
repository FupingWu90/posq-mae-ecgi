"""Polarity / morphology diagnostic for the single-case HSP figure.

Question: at the example vertex the commercial beat shows a negative QRS while
the MFS sources show a positive bump -- is the inverse sign-flipped?
This script compares, for one recording, the commercial epicardial potential
with the all-valid, VC-only and reconstruction-based MFS inverses:
  * R-peak alignment of the four signal-averaged beats
  * per-vertex temporal CC over the whole beat / the QRS window / the T window
  * spatial agreement of the QRS polarity (sign of the dominant deflection)
  * the numbers at the figure vertex, and a "typical" (median-CC) vertex
"""
import os, sys, json, warnings
import numpy as np
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, 'inverse_evaluation')); sys.path.insert(0, os.path.join(PROJ, 'bsp_reconstruction'))
sys.path.insert(0, os.path.join(PROJ, 'baselines_analyses'))
from common import run_dir
from build_sources import load_model, build_context, reconstruct_sources, CROP
from inverse import assemble_solver, solve_from_field, load_commercial
from at_rt import extract_sample, good_vertex_mask
from dataset import collect_split
warnings.filterwarnings('ignore')
import torch

SID_KEY = sys.argv[1] if len(sys.argv) > 1 else '06b5ef63'
VFIG = int(sys.argv[2]) if len(sys.argv) > 2 else 1873
device = torch.device('cpu')          # GPU busy with training; single forward is cheap
rd = run_dir(model='fixedn_ecgi_wide', vc=16, ws='0', wt='3', rw=100)
model, keep_idx, args = load_model(rd, device)
buckets, _ = collect_split(seed=42)
sid, ecgpath, torsopath = next(r for r in buckets['test_internal'] if r[2] and SID_KEY in r[0])
ctx = build_context(ecgpath, torsopath, keep_idx, args, vc=16, mode='cluster', cluster_method='kmean', seed=42)
rec = reconstruct_sources(model, ctx, device)
solver, tci, hV, hF = assemble_solver(os.path.dirname(torsopath))
n_h = hV.shape[0]; tw = (ctx['crop_start'], ctx['crop_start'] + CROP)
phi = {'commercial': load_commercial(os.path.splitext(ecgpath)[0] + '.potential', n_h, time_window=tw),
       'allvalid': solve_from_field(solver, tci, *ctx['src2_allvalid'])[0],
       'recon': solve_from_field(solver, tci, *rec['src4_recon_valid'])[0],
       'vconly': solve_from_field(solver, tci, *ctx['src1_vc'])[0]}
ex = {k: extract_sample(v) for k, v in phi.items()}
print('== R-peaks (ms, first 6) and rel_R per source ==')
for k in ex: print(f'  {k:10s} relR={ex[k]["rel_R"]}  beats={len(ex[k]["r_peaks"])}  R={list(map(int, ex[k]["r_peaks"][:6]))}')

com = ex['commercial']; sc = com['sigavg'].astype(float)
qs, qe, ts_, te_ = com['windows']
gcom = good_vertex_mask(sc, *com['windows']); gav = good_vertex_mask(ex['allvalid']['sigavg'], *ex['allvalid']['windows'])
def cc_rows(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)
def dom_sign(s, lo, hi):
    w = s[:, lo:hi]; return np.sign(w[np.arange(len(w)), np.abs(w).argmax(1)])
print('\n== per-vertex CC vs commercial (signal-averaged beat; QRS = commercial QRS window, T = commercial T window) ==')
res = {}
for k in ('allvalid', 'recon', 'vconly'):
    s = ex[k]['sigavg'].astype(float)
    L = min(s.shape[1], sc.shape[1])
    full = cc_rows(s[:, :L], sc[:, :L]); q = cc_rows(s[:, qs:qe], sc[:, qs:qe]); t = cc_rows(s[:, ts_:te_], sc[:, ts_:te_])
    sgn_agree = (dom_sign(s, qs, qe) == dom_sign(sc, qs, qe))
    res[k] = dict(full=full, q=q, t=t)
    g = gcom['qrs']
    print(f'  {k:9s} full: median {np.median(full):+.3f} (<0: {(full<0).mean()*100:4.1f}%) | '
          f'QRS-win: median {np.median(q):+.3f} (<0: {(q<0).mean()*100:4.1f}%) | T-win: median {np.median(t):+.3f} (<0: {(t<0).mean()*100:4.1f}%) | '
          f'QRS dominant-deflection sign agrees: {sgn_agree.mean()*100:4.1f}% (QRS-good: {sgn_agree[g].mean()*100:4.1f}%)')
print(f'\n== figure vertex {VFIG} (selected as argmax of commercial T/QRS amplitude ratio among T-good vertices) ==')
ratio = np.ptp(sc[:, ts_:te_], axis=1) / (np.ptp(sc[:, qs:qe], axis=1) + 1e-9)
print(f'  commercial T/QRS ratio at vertex = {ratio[VFIG]:.2f}; median over T-good vertices = {np.median(ratio[gcom["t"]&gav["t"]]):.2f}; '
      f'QRS p2p percentile of this vertex among all = {(np.ptp(sc[:,qs:qe],1) < np.ptp(sc[VFIG,qs:qe])).mean()*100:.0f}%')
for k in ('allvalid', 'recon', 'vconly'):
    print(f'  {k:9s} CC full {res[k]["full"][VFIG]:+.3f}  QRS {res[k]["q"][VFIG]:+.3f}  T {res[k]["t"][VFIG]:+.3f}')
# nearest neighbours on the mesh: does the polarity flip locally?
d = np.linalg.norm(hV - hV[VFIG], axis=1); nn = np.argsort(d)[1:9]
print(f'  8 nearest vertices: allvalid QRS-CC = {np.round(res["allvalid"]["q"][nn],2).tolist()}')
print(f'  commercial dominant QRS sign at vertex/nn: {dom_sign(sc,qs,qe)[VFIG]:+.0f} / {dom_sign(sc,qs,qe)[nn].tolist()}')
print(f'  allvalid   dominant QRS sign at vertex/nn: {dom_sign(ex["allvalid"]["sigavg"].astype(float),qs,qe)[VFIG]:+.0f} / {dom_sign(ex["allvalid"]["sigavg"].astype(float),qs,qe)[nn].tolist()}')
# typical vertex: T-good in both, median full-beat CC(allvalid vs commercial)
tg = np.where(gcom['t'] & gav['t'])[0]
order = tg[np.argsort(res['allvalid']['full'][tg])]
vmed = int(order[len(order)//2])
print(f'\n== candidate representative vertex (median all-valid-vs-commercial CC among {len(tg)} T-good vertices): {vmed} ==')
for k in ('allvalid', 'recon', 'vconly'):
    print(f'  {k:9s} CC full {res[k]["full"][vmed]:+.3f}  QRS {res[k]["q"][vmed]:+.3f}  T {res[k]["t"][vmed]:+.3f}')
print(f'  T/QRS ratio there = {ratio[vmed]:.2f}')
json.dump({'vfig': VFIG, 'vmed': vmed}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_polarity.json'), 'w'))
