"""
PosQ-MAE (v2): position-queried masked autoencoder for body-surface potential completion.

Same task, data, normalisation, position code and loss as the v1 model (models_fixedn.FixedNReconECGi),
with the following architectural options (each one a single switch, so that every change can be ablated
against the v1 behaviour):

  decoder   : POSITION-QUERIED CROSS-ATTENTION.  A query = mask token + position code [PPE(p) || TE(t)]
              of one (electrode position, time patch).  Each decoder block = (optional) temporal
              self-attention among the T queries of the SAME electrode -> cross-attention to the encoder
              memory (the visible tokens) -> MLP.  Queries of different electrodes never interact, so the
              completed potential at a torso position does not depend on which other positions are
              queried, and the cost is linear in the number of queried positions.
              (v1: mask tokens are concatenated with the visible tokens and all C*T tokens go through
              full self-attention.)
  dec_tsa   : temporal self-attention per electrode inside every decoder block (patch-level temporal
              attention on the completed signals); placed AFTER the cross-attention (tsa_order='post').
  dec_ssa   : SPATIAL query interaction: inside every decoder block the queries of the same time patch
              (all queried electrodes) attend to one another, after the cross-attention (and after the
              temporal self-attention if both are on).  Ablation of 'queries do not interact'.
  tokenizer : 'linear' (v1) | 'tnt' : + self-attention WITHIN each 50-sample patch over sub-tokens of
              `sub_len` samples (Transformer-in-Transformer style), added to the linear embedding as a
              zero-initialised residual | 'conv' : + a 1-D convolutional stem along time (receptive
              field 79 samples, i.e. across patch boundaries), zero-initialised residual.
  head      : 'linear' (v1) | 'subattn' : + sub-token local attention at the OUTPUT: every decoded patch
              is expanded into sub-tokens that attend to the sub-tokens of the same and the two adjacent
              patches of the same electrode before being mapped to samples (how the completed waveform
              moves within a patch and across patch boundaries) | 'convref' : + 1-D conv refinement of
              the assembled series.  Both are zero-initialised residuals on the linear head.
  time_embed: 'sincos' (v1, fixed) | 'learned'.
  ppe_type  : 'mlp' (v1) | 'fourier' (Fourier features of the 3-D position before the MLP).
  tmask_mode: time-patch masking mixed with the whole-electrode masking (training only; applied on every
              second forward call so that each optimiser update with accum_iter=2 contains one pure
              deployment-task step and one mixed step):
                'dropout'   : a fraction `tmask_ratio` of the visible electrodes' time patches is hidden
                              from the encoder and added to the completion targets;
                'span'      : the same, with one contiguous span per visible electrode;
                'alternate' : the encoder sees N*T tokens drawn at random from ALL active electrodes
                              (random token masking with the same visible-token budget).
  query_frac: fraction of the masked electrodes decoded per training step (electrodes with ground truth
              are preferred); evaluation always decodes every masked electrode.

`forward_fixedn` keeps the v1 signature and return values, so the v1 engine, evaluation and downstream
code run unchanged.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block, Mlp

from models_fixedn import FixedNReconECGi


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Multi-head attention of queries `x` over a context (self-attention if ctx is None)."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, attn_mask=None):
        ctx = x if ctx is None else ctx
        B, nq, D = x.shape
        nk = ctx.shape[1]
        q = self.q(x).view(B, nq, self.h, D // self.h).transpose(1, 2)
        k = self.k(ctx).view(B, nk, self.h, D // self.h).transpose(1, 2)
        v = self.v(ctx).view(B, nk, self.h, D // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, nq, D))


class QueryBlock(nn.Module):
    """Decoder block of the position-queried decoder (pre-norm).

    Order of the sub-layers with temporal self-attention (tsa_order):
      'post' (default): cross-attention -> temporal self-attention -> MLP.  The queries first retrieve
              content from the encoder memory and are then made temporally consistent, so every block's
              temporal attention operates on content, and the last operation before the output head is
              temporal.
      'pre'  : temporal self-attention -> cross-attention -> MLP (the usual Transformer-decoder order).
              In the first block the queries hold only their position code, so that block's temporal
              attention has no content to work on.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, norm_layer=nn.LayerNorm, temporal_sa=False,
                 tsa_order='post', spatial_sa=False):
        super().__init__()
        assert tsa_order in ('pre', 'post')
        self.tsa_order = tsa_order
        self.temporal_sa = bool(temporal_sa)
        self.spatial_sa = bool(spatial_sa)      # queries of the SAME time patch attend to one another (across electrodes)
        if self.temporal_sa:
            self.norm_t = norm_layer(dim)
            self.attn_t = Attention(dim, num_heads)
        if self.spatial_sa:
            self.norm_s = norm_layer(dim)
            self.attn_s = Attention(dim, num_heads)
        self.norm_q = norm_layer(dim)
        self.norm_m = norm_layer(dim)
        self.attn_x = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

    def _tsa(self, q, T):
        B, Lq, D = q.shape
        x = self.norm_t(q).reshape(B * (Lq // T), T, D)          # one sequence of T queries per electrode
        return self.attn_t(x).reshape(B, Lq, D)

    def _ssa(self, q, T):
        """Spatial query interaction: for every time patch, the n queried electrodes attend to one another."""
        B, Lq, D = q.shape
        n = Lq // T
        x = self.norm_s(q).reshape(B, n, T, D).transpose(1, 2).reshape(B * T, n, D)   # one sequence of n queries per time patch
        return self.attn_s(x).reshape(B, T, n, D).transpose(1, 2).reshape(B, Lq, D)

    def forward(self, q, mem, T):
        """q: (B, n_elec*T, D) queries grouped by electrode; mem: (B, M, D) encoder memory."""
        if self.temporal_sa and self.tsa_order == 'pre':
            q = q + self._tsa(q, T)
        q = q + self.attn_x(self.norm_q(q), self.norm_m(mem))
        if self.temporal_sa and self.tsa_order == 'post':
            q = q + self._tsa(q, T)
        if self.spatial_sa:
            q = q + self._ssa(q, T)
        q = q + self.mlp(self.norm2(q))
        return q


class TNTTokenizer(nn.Module):
    """Intra-patch self-attention (Transformer-in-Transformer style): a patch of S samples is split into
    K = S / sub_len sub-tokens which attend to each other; the result is projected to the token width.
    Used as a zero-initialised residual on top of the linear patch embedding."""

    def __init__(self, patch_len, out_dim, sub_len=5, inner_dim=32, depth=2, heads=4, mlp_ratio=2.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert patch_len % sub_len == 0
        self.K, self.s = patch_len // sub_len, sub_len
        self.embed = nn.Linear(sub_len, inner_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.K, inner_dim))
        self.blocks = nn.ModuleList([Block(inner_dim, heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                                     for _ in range(depth)])
        self.norm = norm_layer(inner_dim)
        self.proj = nn.Linear(self.K * inner_dim, out_dim)

    def forward(self, p):                      # p: (M, S) -> (M, out_dim)
        u = self.embed(p.reshape(p.shape[0], self.K, self.s)) + self.pos
        for blk in self.blocks:
            u = blk(u)
        return self.proj(self.norm(u).flatten(1))


class ConvStemTokenizer(nn.Module):
    """1-D convolutional stem along time (total stride = patch length 50; receptive field 79 samples).
    Zero-initialised residual on top of the linear patch embedding."""

    def __init__(self, out_dim, ch=(32, 64)):
        super().__init__()
        c1, c2 = ch
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, 9, stride=5, padding=4), nn.GroupNorm(1, c1), nn.GELU(),
            nn.Conv1d(c1, c2, 5, stride=5, padding=2), nn.GroupNorm(1, c2), nn.GELU(),
            nn.Conv1d(c2, out_dim, 3, stride=2, padding=1))

    def forward(self, x):                      # x: (M, L) -> (M, T, out_dim)
        return self.net(x.unsqueeze(1)).transpose(1, 2)


class WindowBlock(nn.Module):
    """Sub-token block of the output head: the K sub-tokens of a patch attend to the sub-tokens of the
    same patch and of its `ctx` neighbours on each side (same electrode)."""

    def __init__(self, dim, heads, mlp_ratio=2.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

    def forward(self, u, pos, ctx):
        """u: (M, T, K, c); pos: (1, 1, (2ctx+1)K, c) slot embedding of the window."""
        M, T, K, c = u.shape
        W = (2 * ctx + 1) * K
        x = self.norm1(u)
        xp = F.pad(x, (0, 0, 0, 0, ctx, ctx))                                   # pad the patch (time) axis
        win = torch.cat([xp[:, i:i + T] for i in range(2 * ctx + 1)], dim=2) + pos   # (M, T, W, c)
        q = x + pos[:, :, ctx * K:(ctx + 1) * K]
        # neighbours beyond the ends of the recording are padding -> masked out
        valid = torch.ones(T, 2 * ctx + 1, dtype=torch.bool, device=u.device)
        for i in range(ctx):
            valid[:ctx - i, i] = False
            valid[T - (ctx - i):, 2 * ctx - i] = False
        mask = valid.repeat_interleave(K, dim=1)                                 # (T, W)
        mask = mask.unsqueeze(0).expand(M, T, W).reshape(M * T, 1, 1, W)
        a = self.attn(q.reshape(M * T, K, c), win.reshape(M * T, W, c), attn_mask=mask)
        u = u + a.view(M, T, K, c)
        return u + self.mlp(self.norm2(u))


class SubPatchHead(nn.Module):
    """Output head with sub-token local attention (residual on the linear head; zero-initialised)."""

    def __init__(self, dec_dim, patch_len, sub_len=5, inner_dim=32, depth=2, heads=4, ctx=1, mlp_ratio=2.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert patch_len % sub_len == 0
        self.K, self.s, self.c, self.ctx = patch_len // sub_len, sub_len, inner_dim, ctx
        self.expand = nn.Linear(dec_dim, self.K * inner_dim)
        self.pos = nn.Parameter(torch.zeros(1, 1, (2 * ctx + 1) * self.K, inner_dim))
        self.blocks = nn.ModuleList([WindowBlock(inner_dim, heads, mlp_ratio, norm_layer) for _ in range(depth)])
        self.norm = norm_layer(inner_dim)
        self.out = nn.Linear(inner_dim, sub_len)

    def forward(self, z):                      # z: (M, T, dec_dim) -> (M, T, S)
        M, T, _ = z.shape
        u = self.expand(z).view(M, T, self.K, self.c)
        for blk in self.blocks:
            u = blk(u, self.pos, self.ctx)
        return self.out(self.norm(u)).reshape(M, T, self.K * self.s)


class ConvRefineHead(nn.Module):
    """1-D convolutional refinement of the assembled per-electrode series (zero-initialised residual)."""

    def __init__(self, ch=16, k=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, ch, k, padding=k // 2), nn.GELU(),
                                 nn.Conv1d(ch, ch, k, padding=k // 2), nn.GELU(),
                                 nn.Conv1d(ch, 1, k, padding=k // 2))

    def forward(self, y):                      # y: (M, L) -> (M, L)
        return self.net(y.unsqueeze(1)).squeeze(1)


class FourierPosEmbed(nn.Module):
    """Fourier features of the normalised 3-D position followed by the v1 MLP."""

    def __init__(self, out_dim, hidden_dim=64, n_freq=6):
        super().__init__()
        self.register_buffer('freq', 2.0 ** torch.arange(n_freq).float(), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * n_freq, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, coords):                 # (B, C, 3) -> (B, C, out_dim)
        a = coords.unsqueeze(-1) * self.freq                                     # (B, C, 3, F)
        f = torch.cat([coords, a.sin().flatten(2), a.cos().flatten(2)], dim=-1)
        return self.mlp(f)


def _gather(t, idx):
    """t: (B, n, ...), idx: (B, m) -> (B, m, ...)"""
    B, m = idx.shape
    return torch.gather(t, 1, idx.view(B, m, *([1] * (t.dim() - 2))).expand(B, m, *t.shape[2:]))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class PosQMAE(FixedNReconECGi):

    def __init__(self, *args, dec_tsa=False, tsa_order='post', dec_ssa=False, tokenizer='linear', head='linear',
                 time_embed='sincos', ppe_type='mlp', tmask_mode='none', tmask_ratio=0.0, query_frac=1.0,
                 sub_len=5, inner_dim=32, inner_depth=2, inner_heads=4, head_ctx=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.patch_size[0] == 1
        assert tokenizer in ('linear', 'tnt', 'conv') and head in ('linear', 'subattn', 'convref')
        assert time_embed in ('sincos', 'learned') and ppe_type in ('mlp', 'fourier')
        assert tmask_mode in ('none', 'dropout', 'span', 'alternate')
        self.S = int(self.patch_size[1])
        self.T = int(self.signal_shape[1]) // self.S
        self.tokenizer, self.head = tokenizer, head
        self.tmask_mode, self.tmask_ratio = tmask_mode, float(tmask_ratio)
        self.query_frac = float(query_frac)
        self._calls = 0
        self.sub_chunk = 1024                   # electrode sequences per chunk of the sub-token head
        D = self.patch_embed.proj.out_channels
        Dd = self.decoder_embed.out_features
        heads_d = self.decoder_blocks[0].attn.num_heads
        mlp_ratio = self.decoder_blocks[0].mlp.fc1.out_features / Dd
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # ---- position-queried decoder replaces the v1 self-attention decoder ----
        self.decoder_blocks = nn.ModuleList([
            QueryBlock(Dd, heads_d, mlp_ratio, norm_layer, temporal_sa=dec_tsa, tsa_order=tsa_order, spatial_sa=dec_ssa)
            for _ in range(len(self.decoder_blocks))])
        self.decoder_blocks.apply(self._init_weights)

        # ---- local temporal modelling on the input side ----
        if tokenizer == 'tnt':
            self.tok_local = TNTTokenizer(self.S, D, sub_len, inner_dim, inner_depth, inner_heads, norm_layer=norm_layer)
            self.tok_local.apply(self._init_weights)
            nn.init.normal_(self.tok_local.pos, std=0.02)
            nn.init.zeros_(self.tok_local.proj.weight); nn.init.zeros_(self.tok_local.proj.bias)
        elif tokenizer == 'conv':
            self.tok_local = ConvStemTokenizer(D)
            nn.init.zeros_(self.tok_local.net[-1].weight); nn.init.zeros_(self.tok_local.net[-1].bias)

        # ---- local temporal modelling on the output side ----
        if head == 'subattn':
            self.sub_head = SubPatchHead(Dd, self.S, sub_len, inner_dim, inner_depth, inner_heads, head_ctx,
                                         norm_layer=norm_layer)
            self.sub_head.apply(self._init_weights)
            nn.init.normal_(self.sub_head.pos, std=0.02)
            nn.init.zeros_(self.sub_head.out.weight); nn.init.zeros_(self.sub_head.out.bias)
        elif head == 'convref':
            self.ref_head = ConvRefineHead()
            nn.init.zeros_(self.ref_head.net[-1].weight); nn.init.zeros_(self.ref_head.net[-1].bias)

        # ---- embedding variants ----
        if time_embed == 'learned':
            del self.time_embed, self.decoder_time_embed
            self.time_embed = nn.Parameter(torch.zeros(1, self.T, D // 2))
            self.decoder_time_embed = nn.Parameter(torch.zeros(1, self.T, Dd // 2))
            nn.init.normal_(self.time_embed, std=0.02); nn.init.normal_(self.decoder_time_embed, std=0.02)
        if ppe_type == 'fourier' and self.pos_mode != 'index':
            self.ppe = FourierPosEmbed(D // 2, 64)
            self.decoder_ppe = FourierPosEmbed(Dd // 2, 32)
            self.ppe.apply(self._init_weights); self.decoder_ppe.apply(self._init_weights)

    # ---- tokenisation ---------------------------------------------------------
    def _tok_series(self, xs):
        """xs: (B, n, L) series of the selected electrodes -> (B, n, T, D) tokens (no position code)."""
        B, n, L = xs.shape
        t = self.patch_embed.proj(xs.unsqueeze(1)).permute(0, 2, 3, 1)           # (B, n, T, D)
        if self.tokenizer == 'tnt':
            t = t + self.tok_local(xs.reshape(B * n * self.T, self.S)).view(B, n, self.T, -1)
        elif self.tokenizer == 'conv':
            t = t + self.tok_local(xs.reshape(B * n, L)).view(B, n, self.T, -1)
        return self.patch_embed.norm(t)

    def _tok_patches(self, p):
        """p: (B, m, S) individual patches -> (B, m, D) tokens (patch-wise tokenizers only)."""
        w = self.patch_embed.proj.weight.view(-1, self.S)
        t = F.linear(p, w, self.patch_embed.proj.bias)
        if self.tokenizer == 'tnt':
            t = t + self.tok_local(p.reshape(-1, self.S)).view(p.shape[0], p.shape[1], -1)
        return self.patch_embed.norm(t)

    @staticmethod
    def _pos(ch, te, T):
        """ch: (B, n, d) channel code, te: (1, T, d) time code -> (B, n, T, 2d)"""
        B, n, _ = ch.shape
        return torch.cat([ch.unsqueeze(2).expand(B, n, T, ch.shape[-1]),
                          te.unsqueeze(1).expand(B, n, T, te.shape[-1])], dim=-1)

    # ---- encoder / decoder ------------------------------------------------------
    def _encode(self, tok):
        cls = (self.cls_token + self.cls_pos_embed).expand(tok.shape[0], -1, -1)
        h = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def _decode(self, enc, mem_pos, q_pos, need=None):
        """enc: (B, 1+n_tok, D); mem_pos: (B, n_tok, Dd); q_pos: (B, n_q, T, Dd) -> (B, n_q, T, S).
        need: optional (B, n_q) bool -- electrodes whose output is used by the loss; the sub-token head
        (independent per electrode) is evaluated only for those during training."""
        mem = self.decoder_embed(enc)
        mem = torch.cat([mem[:, :1] + self.decoder_cls_pos_embed, mem[:, 1:] + mem_pos], dim=1)
        B, nq, T, Dd = q_pos.shape
        q = (self.mask_token + q_pos.reshape(B, nq * T, Dd))
        for blk in self.decoder_blocks:
            q = blk(q, mem, T)
        z = self.decoder_norm(q)
        pred = self.decoder_pred(z)                                              # (B, nq*T, S)
        if self.head == 'subattn':
            # sub-token head in chunks of electrodes; activations are recomputed in the backward pass
            # (gradient checkpointing), so its memory does not scale with the number of queries
            zz = z.view(B * nq, T, Dd)
            sel = None
            if need is not None and not bool(need.all()):
                sel = need.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                zz = zz[sel]
            outs = []
            for i in range(0, zz.shape[0], self.sub_chunk):
                zc = zz[i:i + self.sub_chunk]
                if self.training and zc.requires_grad:
                    outs.append(checkpoint(self.sub_head, zc, use_reentrant=False))
                else:
                    outs.append(self.sub_head(zc))
            res = torch.cat(outs, dim=0) if outs else zz.new_zeros(0, T, self.S)
            if sel is not None:
                full = res.new_zeros(B * nq, T, self.S)
                full[sel] = res
                res = full
            pred = pred + res.view(B, nq * T, self.S).to(pred.dtype)
        elif self.head == 'convref':
            y = pred.reshape(B * nq, T * self.S)
            pred = (y + self.ref_head(y)).view(B, nq * T, self.S)
        return pred.view(B, nq, T, self.S)

    def _time_mask(self, B, n, T, device):
        """(B, n, T) bool keep-mask over the visible electrodes' tokens (same count for every sample)."""
        if self.tmask_mode == 'span':
            s = max(1, int(round(self.tmask_ratio * T)))
            start = torch.randint(0, T - s + 1, (B, n, 1), device=device)
            ar = torch.arange(T, device=device).view(1, 1, T)
            return ~((ar >= start) & (ar < start + s))
        k = max(1, int(round((1.0 - self.tmask_ratio) * n * T)))
        ids = torch.rand(B, n * T, device=device).argsort(dim=1)[:, :k]
        keep = torch.zeros(B, n * T, dtype=torch.bool, device=device)
        keep.scatter_(1, ids, True)
        return keep.view(B, n, T)

    # ---- forward (v1 signature) --------------------------------------------------
    def forward_fixedn(self, x, active_mask, coords, visible_mask,
                       w_spatial=0.0, w_temporal=0.0,
                       reg_knn=6, reg_time_stride=10, reg_scope='recon',
                       return_components=False):
        if x.dim() == 4:
            x = x.squeeze(1)
        B, C, L = x.shape
        T, S = self.T, self.S
        dev = x.device
        train = self.training
        mixed_step = train and self.tmask_mode != 'none' and (self._calls % 2 == 1)
        token_mode = mixed_step and self.tmask_mode == 'alternate'
        if train:
            self._calls += 1

        ch_e, ch_d = self.ppe(coords), self.decoder_ppe(coords)                  # (B, C, D/2), (B, C, Dd/2)
        te_e, te_d = self.time_embed, self.decoder_time_embed                    # (1, T, .)
        n_vis = int(visible_mask.sum(dim=1).min().item())
        tok_vis = torch.zeros(B, C, T, dtype=torch.bool, device=dev)             # tokens the encoder sees

        if not token_mode:
            vis_idx = torch.argsort(visible_mask.to(torch.uint8), dim=1, descending=True, stable=True)[:, :n_vis]
            tok = self._tok_series(_gather(x, vis_idx)) + self._pos(_gather(ch_e, vis_idx), te_e, T)
            tok = tok.reshape(B, n_vis * T, -1)
            mpos = self._pos(_gather(ch_d, vis_idx), te_d, T).reshape(B, n_vis * T, -1)
            keep = torch.ones(B, n_vis, T, dtype=torch.bool, device=dev)
            if mixed_step and self.tmask_ratio > 0:
                keep = self._time_mask(B, n_vis, T, dev)
                ids = torch.argsort(keep.reshape(B, -1).to(torch.uint8), dim=1, descending=True,
                                    stable=True)[:, :int(keep[0].sum().item())]
                tok, mpos = _gather(tok, ids), _gather(mpos, ids)
            tok_vis.scatter_(1, vis_idx.unsqueeze(-1).expand(B, n_vis, T), keep)
        else:
            # random token masking: N*T tokens drawn from all ACTIVE electrodes
            act_tok = active_mask.unsqueeze(-1).expand(B, C, T).reshape(B, C * T)
            ids = (torch.rand(B, C * T, device=dev) + (~act_tok).float() * 10.0).argsort(dim=1)[:, :n_vis * T]
            ids = ids.sort(dim=1).values
            if self.tokenizer == 'conv':
                tok = _gather(self._tok_series(x).reshape(B, C * T, -1), ids)
            else:
                tok = self._tok_patches(_gather(x.reshape(B, C * T, S), ids))
            tok = tok + _gather(self._pos(ch_e, te_e, T).reshape(B, C * T, -1), ids)
            mpos = _gather(self._pos(ch_d, te_d, T).reshape(B, C * T, -1), ids)
            tok_vis.view(B, C * T).scatter_(1, ids, True)

        # ---- electrodes to decode ----
        n_mask = C - n_vis
        if not train:
            q_idx = torch.argsort(visible_mask.to(torch.uint8), dim=1, stable=True)[:, :n_mask]   # all masked
        else:
            cand = torch.ones_like(visible_mask) if token_mode else ~visible_mask
            n_q = n_mask if self.query_frac >= 1.0 else max(1, int(round(self.query_frac * n_mask)))
            score = (torch.rand(B, C, device=dev) + 2.0 * (cand & active_mask).float()
                     - 10.0 * (~cand).float())
            q_idx = score.argsort(dim=1, descending=True)[:, :n_q].sort(dim=1).values
            if mixed_step and not token_mode:
                q_idx = torch.cat([q_idx, vis_idx], dim=1)        # also decode the visible electrodes
        nq = q_idx.shape[1]

        enc = self._encode(tok)
        need = _gather(active_mask, q_idx) if train else None       # training: only electrodes with ground truth
        pred_q = self._decode(enc, mpos, self._pos(_gather(ch_d, q_idx), te_d, T), need=need)   # (B, nq, T, S)

        # ---- L1 on tokens with ground truth that the encoder did not see ----
        target_q = _gather(x, q_idx).view(B, nq, T, S)
        w = (_gather(active_mask, q_idx).unsqueeze(-1) & ~_gather(tok_vis, q_idx)).float()   # (B, nq, T)
        loss_patch = (pred_q.float() - target_q).abs().mean(dim=-1)
        L_recon = (loss_patch * w).sum() / w.sum().clamp_min(1.0)

        pred_full = torch.zeros(B, C, T, S, dtype=torch.float32, device=dev)
        pred_full.scatter_(1, q_idx.view(B, nq, 1, 1).expand(B, nq, T, S), pred_q.float())

        if w_spatial > 0.0 or w_temporal > 0.0:
            dec_elec = torch.zeros(B, C, dtype=torch.bool, device=dev).scatter_(1, q_idx, True)
            pred_signal = pred_full.view(B, C, L)
            if token_mode:
                pen = active_mask & dec_elec
                seen = tok_vis.unsqueeze(-1).expand(B, C, T, S).reshape(B, C, L)
            else:
                pen = (~visible_mask if reg_scope == 'masked' else active_mask & ~visible_mask) & dec_elec
                seen = visible_mask.unsqueeze(-1).expand(B, C, L)
            x_hat = torch.where(seen, x, pred_signal)
            dt = x_hat[:, :, 1:] - x_hat[:, :, :-1]
            tw = pen.float().unsqueeze(-1)
            L_temporal = (dt.pow(2) * tw).sum() / (tw.sum() * max(L - 1, 1)).clamp_min(1.0)
            if w_spatial > 0.0:
                L_spatial, _ = self.smoothness_loss(pred_signal, x, visible_mask, active_mask & dec_elec,
                                                    coords, knn=reg_knn, time_stride=reg_time_stride,
                                                    reg_scope=reg_scope)
            else:
                L_spatial = L_recon * 0.0
        else:
            L_spatial = L_temporal = L_recon * 0.0

        loss = L_recon + w_spatial * L_spatial + w_temporal * L_temporal
        pred = pred_full.view(B, C * T, S)
        if return_components:
            return loss, pred, L_recon.detach(), L_spatial.detach(), L_temporal.detach()
        return loss, pred

    # ---- continuous field: decode arbitrary torso positions ----------------------
    @torch.no_grad()
    def complete_at(self, x, coords, visible_mask, query_coords):
        """Completed signals at ARBITRARY torso positions.
        x: (B, C, L) normalised signals; coords: (B, C, 3); visible_mask: (B, C) bool;
        query_coords: (B, Q, 3) normalised positions.  Returns (B, Q, L)."""
        assert self.pos_mode != 'index', 'an index-embedding model cannot be queried at arbitrary positions'
        B, C, L = x.shape
        T = self.T
        n_vis = int(visible_mask.sum(dim=1).min().item())
        vis_idx = torch.argsort(visible_mask.to(torch.uint8), dim=1, descending=True, stable=True)[:, :n_vis]
        cv = _gather(coords, vis_idx)
        tok = self._tok_series(_gather(x, vis_idx)) + self._pos(self.ppe(cv), self.time_embed, T)
        enc = self._encode(tok.reshape(B, n_vis * T, -1))
        mpos = self._pos(self.decoder_ppe(cv), self.decoder_time_embed, T).reshape(B, n_vis * T, -1)
        pred = self._decode(enc, mpos, self._pos(self.decoder_ppe(query_coords), self.decoder_time_embed, T))
        return pred.reshape(B, query_coords.shape[1], L).float()


# --------------------------------------------------------------------------- #
# constructors
# --------------------------------------------------------------------------- #
def _posq(signal_shape, patch_size, embed_dim, depth, num_heads, dec_dim_default, dec_depth, dec_heads_default,
          **kwargs):
    kwargs.setdefault('loss_masked_weight', 1.0)
    kwargs.setdefault('loss_visible_weight', 0.0)
    # optional capacity overrides (hyper-parameter checks): encoder depth, decoder width / heads
    depth = kwargs.pop('enc_depth', None) or depth
    dec_dim = kwargs.pop('dec_dim', None) or dec_dim_default
    dec_heads = kwargs.pop('dec_heads', None) or dec_heads_default
    return PosQMAE(signal_shape=signal_shape, patch_size=tuple(patch_size),
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   decoder_embed_dim=dec_dim, decoder_depth=dec_depth, decoder_num_heads=dec_heads,
                   mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


# same widths / depths as the v1 factories of the same suffix (models_fixedn)
def posq_ecgi_base(signal_shape=(252, 5000), patch_size=(1, 50), dec_depth=3, **kw):
    return _posq(signal_shape, patch_size, 128, 6, 8, 64, dec_depth, 8, **kw)


def posq_ecgi_wide(signal_shape=(252, 5000), patch_size=(1, 50), dec_depth=3, **kw):
    return _posq(signal_shape, patch_size, 256, 6, 16, 128, dec_depth, 16, **kw)


def posq_ecgi_deep(signal_shape=(252, 5000), patch_size=(1, 50), dec_depth=5, **kw):
    return _posq(signal_shape, patch_size, 128, 9, 8, 64, dec_depth, 8, **kw)


def posq_ecgi_wide_deep(signal_shape=(252, 5000), patch_size=(1, 50), dec_depth=5, **kw):
    return _posq(signal_shape, patch_size, 256, 9, 16, 128, dec_depth, 16, **kw)


def posq_ecgi_wide384(signal_shape=(252, 5000), patch_size=(1, 50), dec_depth=3, **kw):
    return _posq(signal_shape, patch_size, 384, 6, 16, 192, dec_depth, 8, **kw)


POSQ_MODELS = ('posq_ecgi_base', 'posq_ecgi_wide', 'posq_ecgi_deep', 'posq_ecgi_wide_deep', 'posq_ecgi_wide384')

# v2 options stored in args.json (name -> default); defaults reproduce "v1 + query decoder"
V2_OPTS = dict(dec_tsa=False, tsa_order='post', dec_ssa=False, tokenizer='linear', head='linear', time_embed='sincos', ppe_type='mlp',
               tmask_mode='none', tmask_ratio=0.0, query_frac=1.0, dec_depth=None,
               sub_len=5, inner_dim=32, inner_depth=2, head_ctx=1,
               enc_depth=None, dec_dim=None, dec_heads=None)


def build_model(args, n_keep):
    """Model from an args dict (train.py namespace / args.json). v1 model names build the v1 classes."""
    name = args['model']
    common = dict(signal_shape=(int(n_keep), int(args['crop_len'])), patch_size=tuple(args['patch_size']),
                  pos_mode=args.get('pos_embed', 'ppe'))
    if name in POSQ_MODELS:
        kw = {k: (args.get(k) if args.get(k) is not None else d) for k, d in V2_OPTS.items()}
        for k in ('dec_depth', 'enc_depth', 'dec_dim', 'dec_heads'):
            if kw[k] is None:
                kw.pop(k)
        return globals()[name](**common, **kw)
    import models_fixedn
    return models_fixedn.__dict__[name](**common)


def v2_tag(args):
    """Run-name suffix describing the v2 options that differ from the defaults."""
    if args['model'] not in POSQ_MODELS:
        return ''
    g = lambda k: args.get(k) if args.get(k) is not None else V2_OPTS[k]
    t = ''
    if g('dec_depth') is not None:
        t += f"_dd{g('dec_depth')}"
    if g('enc_depth') is not None:
        t += f"_ed{g('enc_depth')}"
    if g('dec_dim') is not None:
        t += f"_dw{g('dec_dim')}"
    if g('dec_heads') is not None:
        t += f"_dh{g('dec_heads')}"
    if g('dec_tsa'):
        t += '_tsa' + ('-pre' if g('tsa_order') == 'pre' else '')
    if g('dec_ssa'):
        t += '_ssa'
    if g('tokenizer') != 'linear':
        t += f"_tok-{g('tokenizer')}"
    if g('head') != 'linear':
        t += f"_hd-{g('head')}"
    if (g('tokenizer') == 'tnt' or g('head') == 'subattn') and (g('sub_len'), g('inner_dim'), g('inner_depth')) != (5, 32, 2):
        t += f"_s{g('sub_len')}c{g('inner_dim')}l{g('inner_depth')}"
    if g('time_embed') != 'sincos':
        t += f"_te-{g('time_embed')}"
    if g('ppe_type') != 'mlp':
        t += f"_ppe-{g('ppe_type')}"
    if g('tmask_mode') != 'none':
        t += f"_tm-{g('tmask_mode')}" + (f"{g('tmask_ratio'):g}" if g('tmask_mode') != 'alternate' else '')
    if g('query_frac') < 1.0:
        t += f"_qf{g('query_frac'):g}"
    return t
