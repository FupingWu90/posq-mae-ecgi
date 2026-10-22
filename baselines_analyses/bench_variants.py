"""Throughput / memory of every pilot variant (batch 32, N=16, AMP, forward + backward, synthetic data)."""
import os, sys, time, json
import torch
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
import models_v2 as M
dev = torch.device('cuda')
B, C, L, N = int(os.environ.get('BENCH_B', 32)), 252, 5000, 16
torch.manual_seed(0)
x = torch.randn(B, C, L, device=dev); coords = torch.randn(B, C, 3, device=dev)
active = torch.rand(B, C, device=dev) > 0.2
vis = torch.zeros(B, C, dtype=torch.bool, device=dev)
for b in range(B):
    idx = torch.nonzero(active[b]).squeeze(1); vis[b, idx[torch.randperm(len(idx), device=dev)[:N]]] = True
V = {'DEC1': {}, 'DEC2': dict(dec_tsa=True), 'DEC3': dict(dec_depth=6), 'DEC3t': dict(dec_depth=6, dec_tsa=True), 'DEC4': dict(query_frac=0.5),
     'LOC1': dict(tokenizer='conv'), 'LOC2': dict(tokenizer='tnt'), 'LOC3': dict(head='subattn'),
     'LOC3-s10': dict(head='subattn', sub_len=10), 'LOC3+tsa': dict(head='subattn', dec_tsa=True), 'LOC3b': dict(head='convref'),
     'LOC3-d1': dict(head='subattn', inner_depth=1), 'LOC3-s10d1': dict(head='subattn', sub_len=10, inner_depth=1), 'LOC3-c16': dict(head='subattn', inner_dim=16),
     'LOC3-qf': dict(head='subattn', query_frac=0.5),
     'CAPD6': dict(dec_depth=6), 'CAPD5t': dict(dec_depth=5, dec_tsa=True), 'CAPD6t': dict(dec_depth=6, dec_tsa=True),
     'CAPW256': dict(dec_dim=256), 'CAPW256t': dict(dec_dim=256, dec_tsa=True), 'CAPW192t': dict(dec_dim=192, dec_tsa=True),
     'CAPE9': dict(enc_depth=9), 'CAPE9t': dict(enc_depth=9, dec_tsa=True), 'CAPH8': dict(dec_heads=8),
     'MSK1': dict(tmask_mode='dropout', tmask_ratio=0.25), 'MSK3': dict(tmask_mode='alternate'), 'EMB1': dict(time_embed='learned'),
     'ALL': dict(dec_tsa=True, tokenizer='tnt', head='subattn', tmask_mode='dropout', tmask_ratio=0.25)}
only = sys.argv[1:] or list(V)
out = {}
for name in only:
    kw = V[name]
    try:
        m = M.posq_ecgi_wide(signal_shape=(C, L), patch_size=(1, 50), **kw).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4); sc = torch.amp.GradScaler('cuda')
        torch.cuda.reset_peak_memory_stats(); ts = []
        for it in range(6):
            torch.cuda.synchronize(); t0 = time.time()
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = m.forward_fixedn(x, active, coords, vis, w_temporal=3.0, return_components=True)[0]
            sc.scale(loss).backward(); sc.step(opt); sc.update()
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        dt = sum(ts[2:]) / len(ts[2:]); mem = torch.cuda.max_memory_allocated() / 1e9
        out[name] = dict(s_per_iter=round(dt, 3), peak_gb=round(mem, 1), params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
        print(f'{name:9s} {str(kw):95s} {dt:.2f} s/iter  peak {mem:.1f} GB  est 800 ep {dt * 24000 / 3600:.1f} h', flush=True)
        del m, opt, sc, loss; torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        print(f'{name:9s} {str(kw):95s} OUT OF MEMORY', flush=True); out[name] = 'OOM'; torch.cuda.empty_cache()
json.dump(out, open(os.path.join(V2, 'results', 'bench_variants.json' if B == 32 and not sys.argv[1:] else f'bench_variants_B{B}_partial.json'), 'w'), indent=1)
