"""
PPE-MAE: Masked Autoencoder for ECGi with Physical Position Embedding.

Extends MAEours by replacing the 2D sinusoidal positional embedding with a
hybrid scheme that separates the two axes:

  Channel axis (height):
    Physical 3D electrode coordinates (X, Y, Z) from the patient's .torso file
    are projected into embed_dim//2 via a small MLP (PhysicalPosEmbed).
    This gives each channel a position embedding grounded in actual anatomy.
    Channels with no registered torso position receive a zero coordinate vector;
    the MLP still produces a valid embedding (learned "unknown position" token).

  Time axis (width):
    Standard 1D sinusoidal embedding over the gw patch positions,
    projected to embed_dim//2.

The two halves are concatenated → embed_dim and added to patch tokens, exactly
as the original 2D sincos embed was added.

The decoder uses the same hybrid scheme for its positional embeddings.

All masking logic (channel, patch, fixed) is inherited unchanged from MAEours.
"""

from functools import partial  # noqa: F401 used in model constructors below
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: (h, w) tuple
    Returns pos_embed of shape (h*w [+1], embed_dim).
    """
    h, w = grid_size
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w then h
    grid = np.stack(grid, axis=0)       # (2, h, w)
    grid = grid.reshape([2, 1, h, w])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def get_1d_sincos_time_embed(embed_dim, n_positions):
    """
    1D sinusoidal embedding for the time (patch-width) axis.
    Returns (n_positions, embed_dim) float32.
    """
    pos = np.arange(n_positions, dtype=np.float32).reshape(-1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


# ---------------------------------------------------------------------------
# Physical Position Embedding MLP
# ---------------------------------------------------------------------------

class IndexPosEmbed(nn.Module):
    """ST-MEM-style learned per-channel-index embedding (ablation baseline).

    Replaces PhysicalPosEmbed: each channel INDEX gets a free learned vector
    (as ST-MEM's per-lead `lead_embeddings`), ignoring the electrode's 3-D
    position entirely. Ties the model to one fixed vest layout and provides
    no geometric inductive bias -- the exact contrast to the PPE.
    """
    def __init__(self, out_dim, n_ch=256):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(n_ch, out_dim))
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, coords):                     # coords only supplies (B, C)
        B, C = coords.shape[0], coords.shape[1]
        return self.emb[:C].unsqueeze(0).expand(B, -1, -1)


class PhysicalPosEmbed(nn.Module):
    """
    Projects per-channel 3D physical coordinates (X, Y, Z) into
    a channel position embedding of size `out_dim`.

    Input:  coords  (B, n_channels, 3)  — normalized to [-1, 1]
    Output: emb     (B, n_channels, out_dim)

    Channels with no registered torso position have coords = (0,0,0);
    the MLP learns a dedicated embedding for the "unknown position" case.

    Architecture: 3 -> hidden -> hidden -> out_dim with GELU activations,
    keeping it lightweight so it does not dominate the model parameter count.
    """
    def __init__(self, out_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords):
        # coords: (B, n_channels, 3)
        return self.mlp(coords)   # (B, n_channels, out_dim)


# ---------------------------------------------------------------------------
# Patch Embedding for (B, 1, H, W) signals
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits (B, 1, n_channels, signal_len) into patches via 2-D Conv.
    patch_size: (ph, pw)  e.g. (1, 50)
    Output: (B, num_patches, embed_dim)
    """
    def __init__(self, signal_shape=(256, 5000), patch_size=(1, 50),
                 in_chans=1, embed_dim=128):
        super().__init__()
        H, W = signal_shape
        ph, pw = patch_size
        assert H % ph == 0 and W % pw == 0, \
            "signal_shape must be divisible by patch_size"
        self.patch_size = patch_size
        self.grid_size = (H // ph, W // pw)   # (n_ch_patches, n_time_patches)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)          # (B, embed_dim, gh, gw)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Main MAE model
# ---------------------------------------------------------------------------

class MaskedAutoencoderECGi(nn.Module):
    """
    MAE for ECGi body-surface potential maps.

    signal_shape: (n_channels, signal_len) = (256, 5000)
    patch_size:   (ph, pw) = (1, 50)
      -> grid = (256, 100), num_patches = 25600

    mask_mode options (passed to forward):
      'channel' — whole-channel masking
      'patch'   — random patch masking
    """

    def __init__(
        self,
        signal_shape=(256, 5000),
        patch_size=(1, 50),
        in_chans=1,
        embed_dim=128,
        depth=6,
        num_heads=8,
        decoder_embed_dim=64,
        decoder_depth=3,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        mask_ratio=0.75,
        norm_pix_loss=False,
        # ---- New (Full): masking strategy + visible-channel loss ----
        mask_strategy='total_ratio',     # total_ratio | active_ratio | adaptive
        mask_active_ratio=0.5,           # fraction of active channels masked when strategy=active_ratio
        adaptive_threshold=0.5,          # natural-missing rate above which adaptive switches strategies
        adaptive_active_ratio=0.5,       # active-ratio used in adaptive mode when above threshold
        min_visible_channels=0,          # floor: ≥N active channels remain visible after masking (0 disables)
        loss_masked_weight=1.0,
        loss_visible_weight=0.0,
        # ---- New (Full): channel selection distribution ----
        channel_mask_distribution='uniform',  # uniform | empirical
        empirical_channel_weights=None,       # (256,) array; required when distribution=empirical
        pos_mode='ppe',
    ):
        super().__init__()
        self.pos_mode = pos_mode
        self.signal_shape = signal_shape   # (H, W) = (256, 5000)
        self.patch_size = patch_size       # (ph, pw)
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Masking-strategy / loss-weight knobs (all backward-compatible defaults).
        self.mask_strategy        = mask_strategy
        self.mask_active_ratio    = float(mask_active_ratio)
        self.adaptive_threshold   = float(adaptive_threshold)
        self.adaptive_active_ratio = float(adaptive_active_ratio)
        self.min_visible_channels = int(min_visible_channels)
        self.loss_masked_weight   = float(loss_masked_weight)
        self.loss_visible_weight  = float(loss_visible_weight)
        self.channel_mask_distribution = channel_mask_distribution

        # Per-channel empirical missing rates, used when
        # channel_mask_distribution=='empirical'. Stored as a buffer so it
        # tracks the model's device. Defaults to ones (uniform) when not provided.
        # n_ch = number of input channels = signal_shape[0]. This is the REDUCED
        # electrode count after dropping geometry-less / always-missing channels
        # (e.g. 252), not necessarily 256 — derive it from signal_shape so the
        # model is channel-count agnostic.
        n_ch = int(signal_shape[0])
        if empirical_channel_weights is None:
            ecw = torch.ones(n_ch, dtype=torch.float32)
        else:
            ecw = torch.as_tensor(empirical_channel_weights, dtype=torch.float32).clone()
            assert ecw.shape == (n_ch,), f'empirical_channel_weights must be ({n_ch},), got {tuple(ecw.shape)}'
        self.register_buffer('empirical_channel_weights', ecw)

        assert embed_dim % 2 == 0, "embed_dim must be even for PPE hybrid embedding"
        assert decoder_embed_dim % 2 == 0

        # ---- Encoder --------------------------------------------------------
        self.patch_embed = PatchEmbed(signal_shape, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # noqa: F841 used below
        _, gw = self.patch_embed.grid_size  # gh=256 channels, gw=time patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sinusoidal time embedding: (1, gw, embed_dim//2)
        # Stored as buffer; does not include cls token.
        time_emb = get_1d_sincos_time_embed(embed_dim // 2, gw)   # (gw, D//2)
        self.register_buffer('time_embed',
                              torch.from_numpy(time_emb).float().unsqueeze(0))

        # Fixed sinusoidal cls-token position (embed_dim): zeros, overwritten below
        self.register_buffer('cls_pos_embed',
                              torch.zeros(1, 1, embed_dim))

        # Physical position embedding MLP: projects (X,Y,Z) -> embed_dim//2
        if getattr(self, 'pos_mode', 'ppe') == 'index':
            self.ppe = IndexPosEmbed(out_dim=embed_dim // 2)
            self.decoder_ppe = IndexPosEmbed(out_dim=decoder_embed_dim // 2)
        else:
            self.ppe = PhysicalPosEmbed(out_dim=embed_dim // 2, hidden_dim=64)
            # Decoder counterpart MLP
            self.decoder_ppe = PhysicalPosEmbed(out_dim=decoder_embed_dim // 2, hidden_dim=32)
        # Fixed time embedding for decoder: (1, gw, decoder_embed_dim//2)
        dec_time_emb = get_1d_sincos_time_embed(decoder_embed_dim // 2, gw)
        self.register_buffer('decoder_time_embed',
                              torch.from_numpy(dec_time_emb).float().unsqueeze(0))
        self.register_buffer('decoder_cls_pos_embed',
                              torch.zeros(1, 1, decoder_embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ---- Decoder --------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)

        self._initialize_weights()

    # ---- Initialisation ------------------------------------------------

    def _initialize_weights(self):
        # cls_pos_embed and decoder_cls_pos_embed stay zero (no physical position
        # for the cls token — it attends to all patches regardless of location).
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---- Patch utilities -----------------------------------------------

    # ---- Hybrid PPE + time positional embedding ----------------------------

    def build_pos_embed(self, coords, ppe_mlp, time_emb_buf, cls_pos_buf):
        """
        Build per-sample positional embeddings combining physical channel
        position and sinusoidal time position.

        coords       : (B, 256, 3)  physical coordinates (normalized)
        ppe_mlp      : PhysicalPosEmbed   maps (B,256,3) -> (B,256,D//2)
        time_emb_buf : (1, gw, D//2)     fixed sinusoidal time embedding
        cls_pos_buf  : (1, 1,  D)        cls token position (zeros)

        Returns pos_embed : (B, 1 + num_patches, D)
          [0]        = cls token position
          [1:]       = channel-PPE ⊗ time-sincos for each of the gh*gw patches
        """
        B = coords.shape[0]
        gh, gw = self.patch_embed.grid_size

        # Channel PPE: (B, gh, D//2)
        ch_emb = ppe_mlp(coords[:, :gh, :])       # (B, gh, D//2)

        # Time sincos: (1, gw, D//2) -> (B, gw, D//2)
        t_emb = time_emb_buf.expand(B, -1, -1)    # (B, gw, D//2)

        # Tile: each channel repeated gw times, each time repeated gh times
        # patch order: row-major (channel fast-outer, time fast-inner)
        #   patch p = channel (p//gw), time (p%gw)
        ch_tiled = ch_emb.unsqueeze(2).expand(B, gh, gw, ch_emb.shape[-1])
        t_tiled  = t_emb.unsqueeze(1).expand(B, gh, gw, t_emb.shape[-1])
        # (B, gh, gw, D//2) each -> flatten -> (B, gh*gw, D//2)
        ch_flat = ch_tiled.reshape(B, gh * gw, ch_emb.shape[-1])
        t_flat  = t_tiled.reshape(B, gh * gw, t_emb.shape[-1])

        patch_pos = torch.cat([ch_flat, t_flat], dim=-1)   # (B, gh*gw, D)

        cls_pos = cls_pos_buf.expand(B, -1, -1)            # (B, 1, D)
        return torch.cat([cls_pos, patch_pos], dim=1)      # (B, 1+gh*gw, D)

    def patchify(self, x):
        """
        x: (B, 1, H, W)
        Returns: (B, num_patches, ph*pw)
        """
        ph, pw = self.patch_size
        H, W = self.signal_shape
        gh = H // ph
        gw = W // pw
        x = x.reshape(x.shape[0], 1, gh, ph, gw, pw)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(x.shape[0], gh * gw, ph * pw)

    def unpatchify(self, x):
        """
        x: (B, num_patches, ph*pw)
        Returns: (B, 1, H, W)
        """
        ph, pw = self.patch_size
        H, W = self.signal_shape
        gh = H // ph
        gw = W // pw
        x = x.reshape(x.shape[0], gh, gw, ph, pw, 1)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 1, H, W)

    # ---- Masking strategies (run at patch-token level) -----------------

    def _decide_n_target_channels(self, n_missing_in_sample, gh):
        """
        Per-sample target #channels-to-mask under the configured strategy.
        Returned value is the TOTAL channels masked (already-missing included),
        so callers can compute n_extra = n_target - n_missing.

        Strategies (set on the model in __init__):
          total_ratio  : mask up to int(mask_ratio * gh) channels total
                         (already-missing count toward the target — same as
                         the original training code).
          active_ratio : mask int(mask_active_ratio * n_active) of the
                         currently-active channels, regardless of missing
                         count. Visible-active count is independent of the
                         per-recording natural-missing variation.
          adaptive     : if n_missing/gh ≥ adaptive_threshold, behave like
                         active_ratio at adaptive_active_ratio; otherwise
                         behave like total_ratio.

        `min_visible_channels` (>0) clamps the result so at least N active
        channels remain visible.
        """
        n_active = max(0, gh - n_missing_in_sample)
        s = self.mask_strategy

        if s == 'active_ratio':
            n_extra = int(n_active * self.mask_active_ratio)
            n_target = n_missing_in_sample + n_extra
        elif s == 'adaptive':
            if (n_missing_in_sample / max(gh, 1)) >= self.adaptive_threshold:
                n_extra = int(n_active * self.adaptive_active_ratio)
                n_target = n_missing_in_sample + n_extra
            else:
                n_target = max(int(gh * self.mask_ratio), n_missing_in_sample)
        else:  # total_ratio (default)
            n_target = max(int(gh * self.mask_ratio), n_missing_in_sample)

        # Apply the visibility floor.
        if self.min_visible_channels > 0:
            max_extra = max(0, n_active - self.min_visible_channels)
            n_extra = max(0, n_target - n_missing_in_sample)
            n_extra = min(n_extra, max_extra)
            n_target = n_missing_in_sample + n_extra

        # Never demand more than gh channels.
        return min(n_target, gh)

    def _channel_masking(self, x, missing_mask, mask_ratio):
        """
        Whole-channel masking: select channels (not patches) to mask.

        Per sample, the total channels masked is decided by
        `_decide_n_target_channels(n_missing, gh)`. Each masked channel has
        ALL its gw patches masked.

        x:            (B, L, D)  L = num_patches = gh*gw
        missing_mask: (B, 256)   bool, True = missing
        Returns: x_vis, mask, ids_restore
          mask: (B, L) float, 1 = masked patch
        """
        B, L, D = x.shape
        gh, gw = self.patch_embed.grid_size   # e.g. (256, 100)

        # Channel-level mask: (B, gh), 1 = masked channel
        ch_mask = missing_mask[:, :gh].float()   # (B, gh)

        use_empirical = (self.channel_mask_distribution == 'empirical')

        for b in range(B):
            n_already = int(ch_mask[b].sum().item())
            n_target = self._decide_n_target_channels(n_already, gh)
            n_extra = max(0, n_target - n_already)
            if n_extra > 0:
                active_ch_idx = (ch_mask[b] == 0).nonzero(as_tuple=False).squeeze(1)
                n_extra = min(n_extra, len(active_ch_idx))
                if n_extra > 0:
                    if use_empirical:
                        # Sample without replacement, weighted by per-channel
                        # empirical missing rate. Add a small floor so channels
                        # with zero empirical rate remain sampleable (else
                        # multinomial errors when n_extra > #non-zero-weight).
                        w = self.empirical_channel_weights[active_ch_idx] + 1e-6
                        # multinomial expects non-negative; weights need not sum to 1.
                        sel = torch.multinomial(w, n_extra, replacement=False)
                        ch_mask[b, active_ch_idx[sel]] = 1.0
                    else:
                        perm = torch.randperm(len(active_ch_idx), device=x.device)[:n_extra]
                        ch_mask[b, active_ch_idx[perm]] = 1.0

        # Expand channel mask to all gw patches per channel -> (B, L)
        mask = ch_mask.unsqueeze(2).expand(B, gh, gw).reshape(B, L).contiguous()

        # Sort with noise; masked patches (mask==1) are pushed to the back,
        # noise shuffles within each group.
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(mask * 10.0 + noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only the visible patches (mask==0); count from sample 0
        # (all samples have the same n_vis after the channel-level selection above
        # targets the same n_target_ch, but clamp to the minimum across the batch
        # in case some samples had more missing channels than n_target_ch).
        n_vis_per_sample = (mask == 0).sum(dim=1)   # (B,)
        n_vis = int(n_vis_per_sample.min().item())
        ids_keep = ids_shuffle[:, :n_vis]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_vis, mask, ids_restore

    def _patch_masking(self, x, missing_mask, mask_ratio):
        """
        Random patch-level masking. Patches of missing channels are
        force-masked first. The number of *additional* patches we mask is
        the per-sample channel-level n_extra translated to patches:

          n_extra_channels       = _decide_n_target_channels(n_missing) - n_missing
          n_extra_patches_per_b  = n_extra_channels * gw

        n_keep is then min over the batch. This keeps patch and channel
        masking on the same budget under all three strategies.

        Returns x_vis, mask, ids_restore.
        """
        B, L, D = x.shape
        gh, gw = self.patch_embed.grid_size

        # Force-mask patches of missing channels: (B, L) float
        forced = missing_mask[:, :gh].unsqueeze(2).expand(
            B, gh, gw).reshape(B, L).float()

        # Per-sample n_keep budget under the chosen strategy.
        n_missing_b = missing_mask[:, :gh].sum(dim=1)               # (B,)
        n_keep_per_b = []
        for b in range(B):
            n_target = self._decide_n_target_channels(int(n_missing_b[b].item()), gh)
            # Total masked patches: structurally-missing patches + extra.
            n_mask_total = n_target * gw
            n_keep_b = max(L - n_mask_total, 0)
            n_keep_per_b.append(n_keep_b)
        n_keep = int(min(n_keep_per_b)) if n_keep_per_b else 0

        # Random noise; forced-masked patches get large noise -> sorted to back
        noise = torch.rand(B, L, device=x.device)
        noise = noise + forced * 10.0

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, L, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_vis, mask, ids_restore

    def _fixed_masking(self, x, missing_mask):
        """
        Use missing_mask exactly as-is — no extra random channels masked.
        Intended for test inference where combined_mask = missing | eval_channel_mask
        and we want to reconstruct exactly those channels, nothing more.

        missing_mask: (B, 256) bool — True = mask this channel
        """
        B, L, D = x.shape
        gh, gw = self.patch_embed.grid_size

        ch_mask = missing_mask[:, :gh].float()                          # (B, gh)
        mask = ch_mask.unsqueeze(2).expand(B, gh, gw).reshape(B, L).contiguous()

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(mask * 10.0 + noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        n_vis = int((mask == 0).sum(dim=1).min().item())
        ids_keep = ids_shuffle[:, :n_vis]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_vis, mask, ids_restore

    # ---- Encoder / Decoder ---------------------------------------------

    def forward_encoder(self, x, missing_mask, coords, mask_mode='patch'):
        """
        x:            (B, 1, H, W)
        missing_mask: (B, 256) bool
        coords:       (B, 256, 3) physical electrode coordinates (normalized)
        mask_mode:    'channel' | 'patch' | 'fixed'
        """
        # Patch embedding
        tokens = self.patch_embed(x)                     # (B, L, D)

        # Build hybrid PPE: physical channel pos + sinusoidal time pos
        pos_emb = self.build_pos_embed(
            coords, self.ppe, self.time_embed, self.cls_pos_embed)  # (B, 1+L, D)
        tokens = tokens + pos_emb[:, 1:, :]              # add patch pos (no cls)

        # Apply masking
        if mask_mode == 'channel':
            x_vis, mask, ids_restore = self._channel_masking(
                tokens, missing_mask, self.mask_ratio)
        elif mask_mode == 'fixed':
            x_vis, mask, ids_restore = self._fixed_masking(tokens, missing_mask)
        else:
            x_vis, mask, ids_restore = self._patch_masking(
                tokens, missing_mask, self.mask_ratio)

        # Prepend cls token
        cls = (self.cls_token + pos_emb[:, :1, :]).expand(x_vis.shape[0], -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)

        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)

        return x_vis, mask, ids_restore

    def forward_decoder(self, x_enc, ids_restore, coords):
        """
        x_enc:       (B, n_vis+1, encoder_dim)  (includes cls)
        ids_restore: (B, L)
        coords:      (B, 256, 3) physical coordinates
        Returns pred: (B, L, ph*pw)
        """
        x = self.decoder_embed(x_enc)                    # (B, n_vis+1, dec_dim)

        # Build decoder PPE
        dec_pos_emb = self.build_pos_embed(
            coords, self.decoder_ppe,
            self.decoder_time_embed, self.decoder_cls_pos_embed)  # (B, 1+L, dec_dim)

        # Fill in mask tokens
        n_masked = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0], n_masked, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove cls, append masks
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)            # re-prepend cls

        x = x + dec_pos_emb                               # add decoder PPE

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token -> (B, L, ph*pw)
        return x

    # ---- Loss ----------------------------------------------------------

    def forward_loss(self, x, pred, mask, missing_mask, return_components=False):
        """
        L1 reconstruction loss on patches, decomposed into:
          - L_masked  : masked + active patches (the original MAE loss)
          - L_visible : visible (un-masked) + active patches (Full-mode add-on)

        Total loss = loss_masked_weight * L_masked + loss_visible_weight * L_visible.

        Defaults (1.0, 0.0) reproduce the baseline behavior. Set
        `loss_visible_weight > 0` to also penalize reconstruction error on
        the channels the encoder DID see — useful when the goal is full
        reconstruction, not just masked-patch in-painting.

        x:            (B, 1, H, W)
        pred:         (B, L, ph*pw)
        mask:         (B, L) float, 1 = masked patch
        missing_mask: (B, 256) bool, True = missing channel

        Returns the scalar loss; if `return_components`, additionally returns
        (L_masked, L_visible) for logging.
        """
        target = self.patchify(x)   # (B, L, ph*pw)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # Per-patch active mask: 1 = active (non-missing) channel
        gh, gw = self.patch_embed.grid_size
        active_ch = ~missing_mask[:, :gh]                          # (B, gh)
        active_patch = active_ch.unsqueeze(2).expand(-1, -1, gw).reshape(
            x.shape[0], -1).float()                                # (B, L)

        loss_patch = (pred - target).abs().mean(dim=-1)            # (B, L)

        zero = loss_patch.mean() * 0.0

        # Masked + active
        w_m = mask * active_patch
        d_m = w_m.sum()
        L_masked = (loss_patch * w_m).sum() / d_m if d_m >= 1 else zero

        # Visible + active
        w_v = (1.0 - mask) * active_patch
        d_v = w_v.sum()
        L_visible = (loss_patch * w_v).sum() / d_v if d_v >= 1 else zero

        loss = (self.loss_masked_weight * L_masked
                + self.loss_visible_weight * L_visible)
        if return_components:
            return loss, L_masked.detach(), L_visible.detach()
        return loss

    # ---- Forward -------------------------------------------------------

    def forward(self, x, missing_mask, coords, mask_mode='patch',
                loss_missing_mask=None, return_components=False):
        """
        x:                 (B, 1, H, W) or (B, H, W)
        missing_mask:      (B, 256) bool — controls which channels the encoder masks
        coords:            (B, 256, 3) float — normalized physical electrode coords
        mask_mode:         'patch' | 'channel' | 'fixed'
        loss_missing_mask: (B, 256) bool — defines truly missing channels for loss.
                           If None, same as missing_mask.
        return_components: also return (L_masked, L_visible) detached for logging.

        Returns: loss, pred (B, L, ph*pw), mask (B, L)
                 [, L_masked, L_visible if return_components]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        latent, mask, ids_restore = self.forward_encoder(
            x, missing_mask, coords, mask_mode)
        pred = self.forward_decoder(latent, ids_restore, coords)
        loss_out = self.forward_loss(x, pred, mask,
                                      loss_missing_mask if loss_missing_mask is not None
                                      else missing_mask,
                                      return_components=return_components)
        if return_components:
            loss, L_masked, L_visible = loss_out
            return loss, pred, mask, L_masked, L_visible
        return loss_out, pred, mask


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

# NOTE on signal_shape: the first element is the number of INPUT CHANNELS. It is a
# placeholder default here and is ALWAYS overridden by the caller (train_v2 passes
# signal_shape=(n_keep, crop_len), n_keep=252 = the geometry-registered vest
# electrodes after dropping the 4 padding slots). Kept at 252 to reflect real usage.

def mae_ecgi_small(signal_shape=(252, 5000), patch_size=(1, 100), **kwargs):
    """Smaller variant for quick experiments. embed_dim 64, depth 4."""
    model = MaskedAutoencoderECGi(
        signal_shape=signal_shape,
        patch_size=tuple(patch_size),
        embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_ecgi_base(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    """Default model: embed_dim 128, depth 6, decoder 64/3."""
    model = MaskedAutoencoderECGi(
        signal_shape=signal_shape,
        patch_size=tuple(patch_size),
        embed_dim=128, depth=6, num_heads=8,
        decoder_embed_dim=64, decoder_depth=3, decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_ecgi_large(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    """Large variant (ViT-Base-scale encoder + standard MAE decoder):
    embed_dim 768, depth 12, heads 12; decoder 512/8/16."""
    model = MaskedAutoencoderECGi(
        signal_shape=signal_shape,
        patch_size=tuple(patch_size),
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_ecgi_huge(signal_shape=(252, 5000), patch_size=(1, 50), **kwargs):
    """Huge variant (ViT-Large-scale encoder + standard MAE decoder):
    embed_dim 1024, depth 24, heads 16; decoder 512/8/16. Very heavy on the
    ~25k-patch sequence — expect very long training."""
    model = MaskedAutoencoderECGi(
        signal_shape=signal_shape,
        patch_size=tuple(patch_size),
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


