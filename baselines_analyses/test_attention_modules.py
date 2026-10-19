"""Structural tests of the attention modules of models_v2 (synthetic tensors, CPU/GPU).

A1  sub-token output head : the completed samples of patch t depend on the decoder tokens of patches
                            t-1, t, t+1 only (depth 1, window +-1), and the ends of the recording are not
                            contaminated by padding.
A2  temporal self-attention: the queries of one electrode never influence another electrode (both orders),
                            and with order 'post' the first block's temporal attention receives content.
A3  cross-attention        : the output does not depend on the order of the memory tokens.
A4  intra-patch tokenizer  : a token depends on the samples of its own patch only.
A5  same weights, two orders: 'pre' and 'post' share the same parameters (state-dict compatible).
"""
import os, sys
import torch
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
import models_v2 as M
torch.manual_seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ok = True

# ---- A1
h = M.SubPatchHead(dec_dim=128, patch_len=50, sub_len=5, inner_dim=32, depth=1, heads=4, ctx=1).to(dev).eval()
torch.nn.init.normal_(h.out.weight, std=0.5); torch.nn.init.normal_(h.pos, std=0.5)
z = torch.randn(3, 12, 128, device=dev)
with torch.no_grad():
    y0 = h(z)
    z1 = z.clone(); z1[:, 6] += 1.0
    y1 = h(z1)
d = (y1 - y0).abs().amax(dim=(0, 2))                       # per patch
a1 = bool((d[[0, 1, 2, 3, 4, 8, 9, 10, 11]] < 1e-6).all() and (d[[5, 6, 7]] > 1e-4).all())
with torch.no_grad():                                      # ends: only patches 0,1 / 10,11 react to the end tokens
    z2 = z.clone(); z2[:, 0] += 1.0; d2 = (h(z2) - y0).abs().amax(dim=(0, 2))
    z3 = z.clone(); z3[:, 11] += 1.0; d3 = (h(z3) - y0).abs().amax(dim=(0, 2))
a1 &= bool((d2[2:] < 1e-6).all() and (d2[:2] > 1e-4).all() and (d3[:10] < 1e-6).all() and (d3[10:] > 1e-4).all())
print('A1 sub-token head: receptive field +-1 patch, clean recording ends ->', 'PASS' if a1 else 'FAIL', '| change per patch:', [round(float(v), 4) for v in d]); ok &= a1

# ---- A2 / A3
B, nE, T, D = 2, 5, 12, 128
mem = torch.randn(B, 40, D, device=dev)
q = torch.randn(B, nE * T, D, device=dev)
for order in ('post', 'pre'):
    blk = M.QueryBlock(D, 16, 4.0, temporal_sa=True, tsa_order=order).to(dev).eval()
    with torch.no_grad():
        o0 = blk(q, mem, T)
        q1 = q.clone(); q1[:, 2 * T:3 * T] += 1.0           # perturb electrode 2
        o1 = blk(q1, mem, T)
        dd = (o1 - o0).abs().view(B, nE, T, D).amax(dim=(0, 2, 3))
        perm = torch.randperm(40, device=dev)
        o2 = blk(q, mem[:, perm], T)
    a2 = bool((dd[[0, 1, 3, 4]] < 1e-6).all() and dd[2] > 1e-3)
    a3 = bool((o2 - o0).abs().max() < 1e-4)
    print(f"A2 temporal self-attention ({order}): electrodes independent -> {'PASS' if a2 else 'FAIL'} | A3 memory-order invariance -> {'PASS' if a3 else 'FAIL'}")
    ok &= a2 and a3
# with position-only queries (as in the first decoder block) the 'pre' temporal attention is the same for
# every recording, i.e. it carries no information about the signal; 'post' depends on the memory
pos_only = torch.randn(1, nE * T, D, device=dev).expand(B, -1, -1)
for order in ('post', 'pre'):
    blk = M.QueryBlock(D, 16, 4.0, temporal_sa=True, tsa_order=order).to(dev).eval()
    with torch.no_grad():
        if order == 'pre':
            t_out = blk._tsa(pos_only, T)
        else:
            qq = pos_only + blk.attn_x(blk.norm_q(pos_only), blk.norm_m(mem)); t_out = blk._tsa(qq, T)
    same = float((t_out[0] - t_out[1]).abs().max())
    print(f"   first-block temporal attention, order {order}: difference between two recordings {same:.3e} ({'content-free' if same < 1e-6 else 'content-dependent'})")

# ---- A4
tok = M.TNTTokenizer(50, 256).to(dev).eval(); torch.nn.init.normal_(tok.proj.weight, std=0.1)
p = torch.randn(8, 50, device=dev)
with torch.no_grad():
    t0 = tok(p); p1 = p.clone(); p1[3] += 1.0; t1 = tok(p1)
dd = (t1 - t0).abs().amax(dim=1)
a4 = bool((dd[[0, 1, 2, 4, 5, 6, 7]] < 1e-6).all() and dd[3] > 1e-4)
print('A4 intra-patch tokenizer: token depends on its own patch only ->', 'PASS' if a4 else 'FAIL'); ok &= a4

# ---- A5
ka = set(M.posq_ecgi_wide(dec_tsa=True, tsa_order='post').state_dict().keys()); kb = set(M.posq_ecgi_wide(dec_tsa=True, tsa_order='pre').state_dict().keys())
a5 = ka == kb
print('A5 identical parameter sets for both orders ->', 'PASS' if a5 else 'FAIL'); ok &= a5
print('RESULT:', 'PASS' if ok else 'FAIL')
