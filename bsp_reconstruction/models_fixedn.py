"""
FixedN-Recon: fixed-count valid-channel reconstruction for ECGi.

Reuses the validated PPE-MAE backbone from the predecessor codebase (models_mae.py,
`MaskedAutoencoderECGi`) unchanged, and adds:

  * a fixed-VC forward path (`forward_fixedn`): given a per-sample VISIBLE-channel
    mask (the VC channels chosen by channel_select.py), keep exactly those
    channels visible to the encoder (whole-channel masking, mask_mode='fixed')
    and reconstruct every other channel. The L1 reconstruction loss is applied
    on the channels that are ACTIVE (have ground truth) but were masked — i.e.
    active & ~visible.

  * spatial + temporal smoothness regularizers on the reconstructed field,
    following the physics-driven spatiotemporal regularization idea of
    Yao & Yang, "Physics-driven Spatiotemporal Regularization for High-
    dimensional Predictive Modeling" (Sci. Rep. 6:39012, 2016):
        temporal  : first-order time difference  ||x(s,t) - x(s,t-1)||^2
                    (Eq. 11 with the nearest-neighbour time window)
        spatial   : distance-weighted surface Laplacian over the electrode
                    graph  || x_i - sum_j w_ij x_j ||^2, w_ij ∝ 1/d_ij over the
                    k nearest electrodes (Eq. 5–10, adapted from the torso
                    triangle mesh to the vest electrode point cloud).
    Weights are set per run via --w_spatial / --w_temporal.

The reconstructed field used by the regularizers is ASSEMBLED: ground-truth
(normalized) signal on the visible VC channels the encoder actually saw, and
the decoder prediction elsewhere. `reg_scope` selects the channels the
penalties are evaluated over:
    'recon'  : reconstructed targets only  (active & ~visible)
    'masked' : every masked channel        (~visible, incl. detector-missing
               channels that have no ground truth but are still predicted).
"""
from functools import partial

import torch
import torch.nn as nn

from models_mae import MaskedAutoencoderECGi

REG_SCOPES = ('recon', 'masked')


class FixedNReconECGi(MaskedAutoencoderECGi):
    """PPE-MAE with a fixed-visible-channel reconstruction forward path and
    spatiotemporal smoothness regularizers. All backbone modules / weights are
    inherited unchanged from MaskedAutoencoderECGi."""

    # ---- Spatiotemporal smoothness -----------------------------------------

    def smoothness_loss(self, pred_signal, x_signal, visible_mask, active_mask,
                        coords, knn=6, time_stride=10, reg_scope='recon'):
        """Spatial + temporal smoothness of the reconstructed field.

        pred_signal  : (B, C, W) decoder reconstruction (normalized space)
        x_signal     : (B, C, W) ground-truth input      (normalized; missing=0)
        visible_mask : (B, C) bool — encoder-visible VC channels
        active_mask  : (B, C) bool — channels carrying real signal
        coords       : (B, C, 3)   — normalized electrode coordinates
        knn          : neighbours for the spatial Laplacian
        time_stride  : temporal subsampling for the (heavier) spatial term
        reg_scope    : 'recon'  -> penalize active & ~visible channels
                       'masked' -> penalize all ~visible channels

        Returns (L_spatial, L_temporal) scalar tensors, averaged over the
        penalized channels and time.
        """
        B, C, W = pred_signal.shape
        device = pred_signal.device

        # Assembled field: GT on the channels the encoder saw, prediction else.
        vis = visible_mask.unsqueeze(-1)                       # (B,C,1)
        x_hat = torch.where(vis, x_signal, pred_signal)        # (B,C,W)

        # Channels the penalties act on.
        if reg_scope == 'masked':
            pen_mask = ~visible_mask                           # (B,C)
        else:  # 'recon'
            pen_mask = active_mask & (~visible_mask)
        coord_ok = coords.abs().sum(-1) > 1e-6                 # (B,C)

        # ---- temporal smoothness: first-order time difference -----------
        dt = x_hat[:, :, 1:] - x_hat[:, :, :-1]                # (B,C,W-1)
        tw = pen_mask.float().unsqueeze(-1)                    # (B,C,1)
        denom_t = tw.sum() * max(W - 1, 1)
        L_temporal = (dt.pow(2) * tw).sum() / denom_t.clamp_min(1.0)

        # ---- spatial smoothness: distance-weighted surface Laplacian ----
        # Subsample time to bound memory/compute of the neighbour gather.
        xs = x_hat[:, :, ::max(int(time_stride), 1)].contiguous()   # (B,C,Ws)
        Ws = xs.shape[2]

        # Neighbour graph from coordinates (fp32 for stable geometry). Channels
        # with NO registered coordinates (all-zero -> coord_ok False) must not
        # (a) be penalized as a target [handled by sw below], nor (b) be used as
        # a neighbour of any other channel [masked out as columns here].
        cf = coords.float()
        D = torch.cdist(cf, cf)                                # (B,C,C)
        big = (D.max().detach() + 1.0)                         # sentinel > every real distance
        eye = torch.eye(C, dtype=torch.bool, device=device).unsqueeze(0)
        D = D.masked_fill(eye, big)                            # exclude self
        D = D.masked_fill((~coord_ok).unsqueeze(1), big)       # exclude coord-less neighbours
        k = min(int(knn), C - 1)
        knn_d, knn_i = torch.topk(D, k, dim=-1, largest=False)  # (B,C,k)
        # Real neighbour slots have a finite distance (< big); self / coord-less
        # fallbacks were filled with `big` and are dropped (weight 0). Every
        # coord-valid channel has >> k coord-valid neighbours, so this only ever
        # trims the degenerate case of too few valid neighbours.
        valid_nb = knn_d < (big - 0.5)                          # (B,C,k)
        w = torch.where(valid_nb, 1.0 / (knn_d + 1e-6),
                        torch.zeros_like(knn_d))
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)         # (B,C,k)

        # Gather neighbour signals: (B, C, k, Ws) via a single gather on dim 1.
        flat_idx = knn_i.reshape(B, C * k)                     # (B, C*k)
        neigh = torch.gather(
            xs, 1, flat_idx.unsqueeze(-1).expand(B, C * k, Ws))  # (B, C*k, Ws)
        neigh = neigh.reshape(B, C, k, Ws)
        neigh_sum = (neigh * w.to(neigh.dtype).unsqueeze(-1)).sum(dim=2)   # (B,C,Ws)
        lap = xs - neigh_sum                                   # (B,C,Ws)

        sw = (pen_mask & coord_ok).float().unsqueeze(-1)       # (B,C,1)
        denom_s = sw.sum() * Ws
        L_spatial = (lap.pow(2) * sw).sum() / denom_s.clamp_min(1.0)

        return L_spatial, L_temporal

    # ---- Fixed-VC forward --------------------------------------------------

    def forward_fixedn(self, x, active_mask, coords, visible_mask,
                       w_spatial=0.0, w_temporal=0.0,
                       reg_knn=6, reg_time_stride=10, reg_scope='recon',
                       return_components=False):
        """
        x            : (B,1,H,W) or (B,H,W) — normalized signals
        active_mask  : (B,C) bool — channels carrying real signal (have GT)
        coords       : (B,C,3)
        visible_mask : (B,C) bool — VC channels kept visible (subset of active)

        The encoder masks every non-visible channel; the decoder reconstructs
        them; the L1 loss is applied on active & masked channels.

        Returns: loss, pred (B,L,ph*pw)
                 [, L_recon, L_spatial, L_temporal detached if return_components]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        encoder_missing = ~visible_mask          # mask every non-visible channel
        loss_missing = ~active_mask              # true-missing = no ground truth

        latent, mask, ids_restore = self.forward_encoder(
            x, encoder_missing, coords, mask_mode='fixed')
        pred = self.forward_decoder(latent, ids_restore, coords)

        # Reconstruction loss on active & masked patches (loss_masked_weight=1,
        # loss_visible_weight=0 -> exactly the masked-active L1 term).
        L_recon = self.forward_loss(x, pred, mask, loss_missing)

        if w_spatial > 0.0 or w_temporal > 0.0:
            pred_signal = self.unpatchify(pred).squeeze(1)     # (B,C,W)
            x_signal = x.squeeze(1)                            # (B,C,W)
            L_spatial, L_temporal = self.smoothness_loss(
                pred_signal, x_signal, visible_mask, active_mask, coords,
                knn=reg_knn, time_stride=reg_time_stride, reg_scope=reg_scope)
        else:
            z = pred.sum() * 0.0
            L_spatial, L_temporal = z, z

        loss = L_recon + w_spatial * L_spatial + w_temporal * L_temporal
        if return_components:
            return (loss, pred, L_recon.detach(),
                    L_spatial.detach(), L_temporal.detach())
        return loss, pred


# ---------------------------------------------------------------------------
# Model constructors (same backbone sizes as the predecessor codebase)
# ---------------------------------------------------------------------------
# signal_shape[0] = number of INPUT CHANNELS; a placeholder default here, always
# overridden by the caller with (n_keep, crop_len). loss weights fixed to the
# pure masked-reconstruction setting (fixed-VC never uses ratio masking).

def _fixedn(signal_shape, patch_size, embed_dim, depth, num_heads,
            dec_dim, dec_depth, dec_heads, **kwargs):
    kwargs.setdefault('loss_masked_weight', 1.0)
    kwargs.setdefault('loss_visible_weight', 0.0)
    return FixedNReconECGi(
        signal_shape=signal_shape, patch_size=tuple(patch_size),
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        decoder_embed_dim=dec_dim, decoder_depth=dec_depth,
        decoder_num_heads=dec_heads, mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def fixedn_ecgi_small(signal_shape=(252, 5000), patch_size=(1, 100), **kwargs):
    return _fixedn(signal_shape, patch_size, 64, 4, 4, 32, 2, 4, **kwargs)


def fixedn_ecgi_base(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    return _fixedn(signal_shape, patch_size, 128, 6, 8, 64, 3, 8, **kwargs)


def fixedn_ecgi_large(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    return _fixedn(signal_shape, patch_size, 128, 8, 8, 64, 8, 8, **kwargs)


def fixedn_ecgi_huge(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    return _fixedn(signal_shape, patch_size, 1024, 24, 16, 512, 8, 16, **kwargs)


# ---------------------------------------------------------------------------
# Controlled single-axis architecture variants around `base`
#   base = (embed 128, enc_depth 6, heads 8, dec_dim 64, dec_depth 3, dec_heads 8)
# Each variant changes ONE axis (width or depth) so the effect is interpretable;
# encoder head_dim=16 and decoder head_dim=8 are held constant throughout.
# ---------------------------------------------------------------------------
# WIDTH axis (enc_depth 6 / dec_depth 3 fixed):
def fixedn_ecgi_narrow(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):   # half width
    return _fixedn(signal_shape, patch_size, 64, 6, 4, 32, 3, 4, **kwargs)


def fixedn_ecgi_wide(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):     # double width
    return _fixedn(signal_shape, patch_size, 256, 6, 16, 128, 3, 16, **kwargs)


def fixedn_ecgi_wide384(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):  # width x3 (embed 384)
    # 16 enc heads (head_dim 24), 8 dec heads -- common head counts
    return _fixedn(signal_shape, patch_size, 384, 6, 16, 192, 3, 8, **kwargs)


def fixedn_ecgi_wide_deep(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):  # wide(256)+deep(9/5)
    # wide width (embed 256, 16/16 heads) combined with deep depth (enc 9, dec 5)
    return _fixedn(signal_shape, patch_size, 256, 9, 16, 128, 5, 16, **kwargs)


# DEPTH axis (embed 128 / heads 8 / dec_dim 64 fixed):
def fixedn_ecgi_shallow(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):  # enc4/dec2
    return _fixedn(signal_shape, patch_size, 128, 4, 8, 64, 2, 8, **kwargs)


def fixedn_ecgi_deep(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):     # enc9/dec5
    return _fixedn(signal_shape, patch_size, 128, 9, 8, 64, 5, 8, **kwargs)


def mcma_unet(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    """MCMA-style conv U-Net baseline (models_mcma). Deferred import avoids a
    circular import (models_mcma subclasses FixedNReconECGi)."""
    from models_mcma import mcma_unet as _factory
    return _factory(signal_shape=signal_shape, patch_size=patch_size, **kwargs)
