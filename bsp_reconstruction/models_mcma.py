"""
MCMA-style convolutional baseline for FixdElectrodsN_Recon.

Re-implements the architecture family of MCMA (Chen et al., "Multi-channel
masked autoencoder and comprehensive evaluations for reconstructing 12-lead
ECG from arbitrary single-lead ECG", npj Cardiovascular Health 2024) as a
published-MAE-family baseline for the fixed-VC reconstruction task:

  * U-Net-inspired 1-D convolutional autoencoder over the TIME axis with the
    electrode channels as the feature dimension — input (B, 252, 5000).
  * Channel identity is POSITIONAL (channel index = row), exactly MCMA's
    zero-padding strategy: the input frame keeps all 252 rows and every
    NON-VISIBLE channel is ZERO-FILLED in place. No token masking, no position
    embedding. The `coords` argument is accepted for interface parity but
    deliberately IGNORED by the representation — this baseline is
    position-agnostic by design (coords still feed the spatial smoothness
    regularizer, which is architecture-independent loss-side machinery).
  * Blocks follow MCMA's verified recipe: Conv1d kernel 5 / stride 2
    downsampling ("multi-conv": each stage adds a stride-1 refinement conv)
    and matching ConvTranspose1d upsampling with skip connections, GELU
    activations, layer + instance normalisation per conv.

Training protocol: the ORIGINAL MCMA trains with plain MSE over all leads.
Here we instead inherit FixedNReconECGi's loss / regularisation code UNCHANGED
(L1 on active & masked channels + warm-started spatial/temporal smoothness),
so the comparison against the transformer models isolates the ARCHITECTURE
FAMILY, not the objective.

Implementation: subclasses FixedNReconECGi but bypasses the transformer
constructor entirely (nn.Module.__init__ + a parameter-free patch-grid shim),
so the checkpoint holds U-Net weights only. Inherited and reused unchanged:
patchify / unpatchify, forward_loss, smoothness_loss,
_decide_n_target_channels. The token-level helpers (forward_encoder /
forward_decoder / _channel_masking / ...) are NOT usable on this class.
"""
import torch
import torch.nn as nn

from models_fixedn import FixedNReconECGi


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _PatchGrid:
    """Parameter-free stand-in for models_mae.PatchEmbed. The inherited loss /
    patch helpers only read the grid bookkeeping (grid_size / num_patches /
    patch_size); the conv baseline has no patch projection."""

    def __init__(self, signal_shape, patch_size):
        H, W = signal_shape
        ph, pw = patch_size
        assert H % ph == 0 and W % pw == 0, \
            "signal_shape must be divisible by patch_size"
        self.patch_size = patch_size
        self.grid_size = (H // ph, W // pw)   # (n_ch_patches, n_time_patches)
        self.num_patches = self.grid_size[0] * self.grid_size[1]


def _conv_block(c_in, c_out, kernel_size=5, stride=1, transpose=False):
    """MCMA conv unit: Conv1d (or ConvTranspose1d for x2 upsampling) ->
    layer normalisation (GroupNorm-1 over channels+time) -> instance
    normalisation (per-channel over time) -> GELU.

    Lengths are exact powers-of-two round trips: stride-2 Conv1d halves an
    even length (pad = k//2); ConvTranspose1d(stride=2, output_padding=1)
    doubles it back.
    """
    pad = kernel_size // 2
    if transpose:
        conv = nn.ConvTranspose1d(c_in, c_out, kernel_size, stride=2,
                                  padding=pad, output_padding=1)
    else:
        conv = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad)
    return nn.Sequential(conv,
                         nn.GroupNorm(1, c_out),
                         nn.InstanceNorm1d(c_out, affine=True),
                         nn.GELU())


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MCMAUNetECGi(FixedNReconECGi):
    """MCMA-style 1-D conv U-Net with FixedNReconECGi's public contract.

    Drop-in replacement for the transformer models: `forward_fixedn`,
    `unpatchify`, `patchify`, the `(loss, pred[, components])` return
    structure and the state_dict round trip all behave identically, so
    engine.py / train.py / inverse_evaluation/build_sources.py run unchanged.

    Layout (widths per resolution level, default (144, 192, 288, 384)):
        stem    252 -> w0            @ W      (5000)
        down1   w0  -> w1 (s2) + refine  @ W/2   (2500)
        down2   w1  -> w2 (s2) + refine  @ W/4   (1250)
        down3   w2  -> w3 (s2) + refine  @ W/8   ( 625)
        bottleneck  w3 -> w3             @ W/8
        up + concat-skip + fuse back to w0 @ W, linear head w0 -> 252
    """

    def __init__(
        self,
        signal_shape=(252, 5000),
        patch_size=(1, 50),
        widths=(144, 192, 288, 384),
        kernel_size=5,
        n_bottleneck=1,
        norm_pix_loss=False,
        # ---- loss weights (protocol parity with the _fixedn constructors) ----
        loss_masked_weight=1.0,
        loss_visible_weight=0.0,
        # ---- ratio-masking knobs, used only by forward(mask_mode='channel') ----
        mask_ratio=0.75,
        mask_strategy='total_ratio',
        mask_active_ratio=0.5,
        adaptive_threshold=0.5,
        adaptive_active_ratio=0.5,
        min_visible_channels=0,
        # Transformer-specific kwargs (embed_dim, depth, num_heads, pos_mode,
        # decoder_*, mlp_ratio, norm_layer, ...) are accepted and ignored so the
        # train.py / build_sources.py construction sites work unchanged.
        **unused_transformer_kwargs,
    ):
        # Bypass the transformer constructor on purpose: no patch-embed conv,
        # no ViT blocks, no PPE — the checkpoint holds U-Net weights only.
        nn.Module.__init__(self)

        H, W = signal_shape
        assert patch_size[0] == 1, \
            "conv baseline assumes 1-channel-high patches (channel == patch row)"
        assert W % 8 == 0, \
            "signal length must be divisible by 2^3 for the 3 stride-2 stages"
        self.signal_shape = signal_shape
        self.patch_size = tuple(patch_size)
        self.patch_embed = _PatchGrid(signal_shape, self.patch_size)
        self.norm_pix_loss = norm_pix_loss

        # Attributes read by the inherited forward_loss / _decide_n_target_channels.
        self.loss_masked_weight   = float(loss_masked_weight)
        self.loss_visible_weight  = float(loss_visible_weight)
        self.mask_ratio           = mask_ratio
        self.mask_strategy        = mask_strategy
        self.mask_active_ratio    = float(mask_active_ratio)
        self.adaptive_threshold   = float(adaptive_threshold)
        self.adaptive_active_ratio = float(adaptive_active_ratio)
        self.min_visible_channels = int(min_visible_channels)

        # ---- Encoder: stem @W, then 3x [stride-2 conv + refine conv] --------
        w0, w1, w2, w3 = widths
        k = kernel_size
        self.stem = _conv_block(H, w0, k)                        # (B, w0, W)
        self.down = nn.ModuleList([
            nn.Sequential(_conv_block(w0, w1, k, stride=2),      # (B, w1, W/2)
                          _conv_block(w1, w1, k)),
            nn.Sequential(_conv_block(w1, w2, k, stride=2),      # (B, w2, W/4)
                          _conv_block(w2, w2, k)),
            nn.Sequential(_conv_block(w2, w3, k, stride=2),      # (B, w3, W/8)
                          _conv_block(w3, w3, k)),
        ])
        self.bottleneck = nn.Sequential(
            *[_conv_block(w3, w3, k) for _ in range(n_bottleneck)])

        # ---- Decoder: ConvTranspose up, concat encoder skip, fuse conv ------
        self.up = nn.ModuleList([
            _conv_block(w3, w2, k, transpose=True),              # W/8 -> W/4
            _conv_block(w2, w1, k, transpose=True),              # W/4 -> W/2
            _conv_block(w1, w0, k, transpose=True),              # W/2 -> W
        ])
        self.fuse = nn.ModuleList([
            _conv_block(2 * w2, w2, k),
            _conv_block(2 * w1, w1, k),
            _conv_block(2 * w0, w0, k),
        ])
        self.head = nn.Conv1d(w0, H, k, padding=k // 2)          # linear output

        self._initialize_conv_weights()

    # ---- Initialisation ------------------------------------------------

    def _initialize_conv_weights(self):
        # Match the repo's Linear init style (xavier + zero bias).
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ---- Representation pipeline ---------------------------------------

    def _unet(self, x_in):
        """x_in: (B, C, W) zero-filled frame -> (B, C, W) reconstruction."""
        s0 = self.stem(x_in)                                   # (B, w0, W)
        s1 = self.down[0](s0)                                  # (B, w1, W/2)
        s2 = self.down[1](s1)                                  # (B, w2, W/4)
        z  = self.bottleneck(self.down[2](s2))                 # (B, w3, W/8)
        z = self.fuse[0](torch.cat([self.up[0](z), s2], dim=1))  # (B, w2, W/4)
        z = self.fuse[1](torch.cat([self.up[1](z), s1], dim=1))  # (B, w1, W/2)
        z = self.fuse[2](torch.cat([self.up[2](z), s0], dim=1))  # (B, w0, W)
        return self.head(z)                                    # (B, C, W)

    def _reconstruct(self, x, visible_mask):
        """Zero-fill the non-visible channels in place (MCMA's channel-position-
        preserving strategy), run the U-Net, and patchify the predicted field
        into the (B, L, ph*pw) layout the inherited loss / engine unpatchify
        expect.

        x            : (B, 1, H, W)
        visible_mask : (B, C) bool — channels the model may look at
        Returns pred (B, L, ph*pw), mask (B, L) float with 1 = masked patch.
        """
        B = x.shape[0]
        gh, gw = self.patch_embed.grid_size
        x_in = x.squeeze(1) * visible_mask[:, :gh].to(x.dtype).unsqueeze(-1)
        pred_signal = self._unet(x_in)                          # (B, C, W)
        pred = self.patchify(pred_signal.unsqueeze(1))          # (B, L, ph*pw)
        mask = (~visible_mask)[:, :gh].float().unsqueeze(2).expand(
            B, gh, gw).reshape(B, gh * gw).contiguous()
        return pred, mask

    # ---- Fixed-VC forward (same contract as FixedNReconECGi) -----------

    def forward_fixedn(self, x, active_mask, coords, visible_mask,
                       w_spatial=0.0, w_temporal=0.0,
                       reg_knn=6, reg_time_stride=10, reg_scope='recon',
                       return_components=False):
        """Drop-in counterpart of FixedNReconECGi.forward_fixedn. `coords` is
        ignored by the representation (position-agnostic baseline) but still
        feeds the inherited spatial smoothness regularizer.

        Returns: loss, pred (B,L,ph*pw)
                 [, L_recon, L_spatial, L_temporal detached if return_components]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        pred, mask = self._reconstruct(x, visible_mask)
        loss_missing = ~active_mask              # true-missing = no ground truth

        # Reconstruction loss on active & masked patches — inherited unchanged,
        # so the training protocol is IDENTICAL to the transformer models
        # (the original MCMA used plain MSE over all leads instead).
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

    # ---- Generic MAE forward (whole-channel modes only) -----------------

    def forward(self, x, missing_mask, coords, mask_mode='patch',
                loss_missing_mask=None, return_components=False):
        """Drop-in counterpart of MaskedAutoencoderECGi.forward. Only
        whole-channel masking is meaningful for a zero-filled conv input:

          'fixed'   — mask exactly missing_mask (as the parent)
          'channel' — additionally mask random active channels up to the
                      _decide_n_target_channels target (uniform sampling only;
                      the empirical channel distribution is not supported here)

        'patch' (the parent's default) raises: patch-level masking has no
        equivalent when absent channels are represented by zero-filled rows.

        Returns: loss, pred (B,L,ph*pw), mask (B,L)
                 [, L_masked, L_visible if return_components]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        gh = self.patch_embed.grid_size[0]

        ch_mask = missing_mask[:, :gh].clone()                 # bool, True = masked
        if mask_mode == 'channel':
            for b in range(ch_mask.shape[0]):
                n_already = int(ch_mask[b].sum().item())
                n_extra = self._decide_n_target_channels(n_already, gh) - n_already
                if n_extra > 0:
                    act_idx = (~ch_mask[b]).nonzero(as_tuple=False).squeeze(1)
                    perm = torch.randperm(len(act_idx), device=x.device)[:n_extra]
                    ch_mask[b, act_idx[perm]] = True
        elif mask_mode != 'fixed':
            raise NotImplementedError(
                f"mask_mode='{mask_mode}' is undefined for the conv baseline "
                "(whole-channel zero-filling only); use 'fixed' or 'channel'")

        pred, mask = self._reconstruct(x, ~ch_mask)
        loss_out = self.forward_loss(
            x, pred, mask,
            loss_missing_mask if loss_missing_mask is not None else missing_mask,
            return_components=return_components)
        if return_components:
            loss, L_masked, L_visible = loss_out
            return loss, pred, mask, L_masked, L_visible
        return loss_out, pred, mask


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------
# signal_shape[0] = number of INPUT CHANNELS; a placeholder default, always
# overridden by the caller with (n_keep, crop_len). patch_size only defines the
# (B, L, ph*pw) prediction layout shared with the transformer models.

def mcma_unet(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    """MCMA-style conv U-Net baseline: widths (144, 192, 288, 384), 3 stride-2
    stages (5000 -> 625), single-conv bottleneck; ~5.8M params at (252, 5000) —
    comparable to the ~5.4M fixedn_ecgi_wide transformer."""
    kwargs.setdefault('loss_masked_weight', 1.0)
    kwargs.setdefault('loss_visible_weight', 0.0)
    return MCMAUNetECGi(signal_shape=signal_shape,
                        patch_size=tuple(patch_size), **kwargs)
