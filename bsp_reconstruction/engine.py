"""
Training / evaluation engine for FixdElectrodsN_Recon.

Differences vs the the predecessor codebase engine:
  * Every step selects a FIXED number (VC) of VISIBLE valid channels via
    channel_select.select_visible_batch under args.select_mode; the encoder
    keeps only those channels and reconstructs the rest.
  * Per-channel VISIBLE selection counts are accumulated across training
    (usage_counts) so the spatial coverage of the selection can be visualized.
  * Loss = L1 reconstruction (on active & masked channels) + w_spatial * spatial
    smoothness + w_temporal * temporal smoothness.
  * Validation uses the SAME select mode / cluster_method as training; testing
    evaluates ALL selection configurations: cluster-greedy, cluster-kmean,
    patch and random.
  * args.select_region ('all' | 'front' | 'back') restricts the candidate
    electrodes to one torso half for training AND val/test selection alike
    (channel_select.region_mask); 'all' (default) is a strict no-op.
Metrics (MAE / RMSE / CC) are computed on the reconstructed channels
(active & ~visible) in normalized space.
"""
import math
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from channel_select import select_visible_batch


# Test-time selection configurations (mode, cluster_method). cluster_method is
# ignored for patch / random.
TEST_CONFIGS = [('cluster', 'greedy'), ('cluster', 'kmean'),
                ('patch', 'greedy'), ('random', 'greedy'),
                ('template', 'greedy')]      # v2: fixed predetermined deployment layout


def cfg_label(mode, cluster_method):
    return f"{mode}-{cluster_method}" if mode == 'cluster' else mode


def global_norm(ecg, visible_mask, eps=1e-8):
    """Global per-recording normalization for norm_mode='global'.

    Divide EVERY channel by a single scalar g = median of the per-channel std
    over the VISIBLE (VC) channels only. g depends solely on the observed
    channels, so it is available at test with no masked-channel information, and
    dividing all channels by one scalar preserves the inter-channel amplitude
    ratio the ECGi inverse needs. Returns (ecg_norm, g) with g of shape (B,)."""
    B = ecg.shape[0]
    g = torch.ones(B, device=ecg.device, dtype=ecg.dtype)
    for b in range(B):
        idx = visible_mask[b].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            g[b] = ecg[b, idx, :].std(dim=1).median().clamp_min(eps)
    return ecg / g.view(B, 1, 1), g


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    """Cosine decay with linear warmup."""
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, args, select_rng,
                    usage_counts, loss_scaler=None, log_writer=None,
                    train_logger=None):
    """Returns (avg_metrics, n_samples_processed). `usage_counts` (n_keep,) int64
    is incremented in place with the per-channel visible-selection counts."""
    model.train()
    metric_logger = MetricLogger()
    accum_iter = getattr(args, 'accum_iter', 1)
    n_samples = 0

    # Regularizer-weight warmup: linearly ramp w_spatial/w_temporal from 0 to
    # their target over the first `reg_warmup_epochs` epochs. This lets the model
    # first learn a reconstruction before the smoothness penalty reaches full
    # strength -- important for large weights, where the raw penalty (esp.
    # L_temporal) is huge at init and would otherwise dominate early training.
    # reg_warmup_epochs=0 (default) -> full weight from epoch 0 (original behavior).
    rw = getattr(args, 'reg_warmup_epochs', 0)
    reg_scale = min(1.0, (epoch + 1) / rw) if rw and rw > 0 else 1.0
    ws_eff = args.w_spatial * reg_scale
    wt_eff = args.w_temporal * reg_scale
    if train_logger is not None and rw and rw > 0:
        train_logger.write(f"# epoch {epoch}: reg_scale {reg_scale:.3f} "
                           f"-> w_spatial {ws_eff:.4g} w_temporal {wt_eff:.4g}\n")
        train_logger.flush()

    region = getattr(args, 'select_region', 'all')
    n_region_fallback = 0
    if train_logger is not None and epoch == 0:
        train_logger.write(
            f"# iters_per_epoch {len(loader)}  accum_iter {accum_iter}  "
            f"update_steps_per_epoch {len(loader) // max(accum_iter,1)}  "
            f"total_epochs {args.epochs}  vc {args.vc}  mode {args.select_mode}"
            + (f"  region {region}" if region != 'all' else "") + "\n")
        train_logger.flush()

    for step, batch in enumerate(loader):
        ecg, missing_mask, coords = batch
        ecg          = ecg.to(device, non_blocking=True)           # (B, C, L)
        missing_mask = missing_mask.to(device, non_blocking=True)  # (B, C) bool
        coords       = coords.to(device, non_blocking=True)        # (B, C, 3)
        active = ~missing_mask                                     # (B, C) bool

        # Fixed-VC visible-channel selection (varies per epoch via select_rng).
        visible_np, n_fb = select_visible_batch(
            coords.detach().cpu().numpy(), active.detach().cpu().numpy(),
            args.vc, args.select_mode, args.cluster_method, select_rng,
            region=region, return_info=True)
        n_region_fallback += n_fb
        usage_counts += visible_np.sum(axis=0).astype(np.int64)
        n_samples += visible_np.shape[0]
        visible = torch.from_numpy(visible_np).to(device)

        # Global mode: normalize by a single scale from the VC channels (physical
        # signal comes from the dataset in this mode).
        if getattr(args, 'norm_mode', 'per_channel') == 'global':
            ecg, _ = global_norm(ecg, visible)

        is_update_step = (step + 1) % accum_iter == 0
        if step % accum_iter == 0:
            optimizer.zero_grad()

        if loss_scaler is not None:
            with torch.amp.autocast('cuda'):
                loss, _, L_rec, L_sp, L_tp = model.forward_fixedn(
                    ecg, active, coords, visible,
                    w_spatial=ws_eff, w_temporal=wt_eff,
                    reg_knn=args.reg_knn, reg_time_stride=args.reg_time_stride,
                    reg_scope=args.reg_scope, return_components=True)
            loss = loss / accum_iter
            loss_scaler.scale(loss).backward()
            if is_update_step:
                loss_scaler.step(optimizer)
                loss_scaler.update()
        else:
            loss, _, L_rec, L_sp, L_tp = model.forward_fixedn(
                ecg, active, coords, visible,
                w_spatial=ws_eff, w_temporal=wt_eff,
                reg_knn=args.reg_knn, reg_time_stride=args.reg_time_stride,
                reg_scope=args.reg_scope, return_components=True)
            loss = loss / accum_iter
            loss.backward()
            if is_update_step:
                optimizer.step()

        loss_val = loss.item() * accum_iter
        if not math.isfinite(loss_val):
            raise ValueError(f"Non-finite loss: {loss_val}")

        metric_logger.update(loss=loss_val, L_recon=float(L_rec.item()),
                             L_spatial=float(L_sp.item()), L_temporal=float(L_tp.item()))

        global_step = epoch * (len(loader) // max(accum_iter, 1)) + step // max(accum_iter, 1)
        if log_writer is not None and is_update_step:
            log_writer.add_scalar('train/loss', loss_val, global_step)
            log_writer.add_scalar('train/L_recon', float(L_rec.item()), global_step)
            log_writer.add_scalar('train/L_spatial', float(L_sp.item()), global_step)
            log_writer.add_scalar('train/L_temporal', float(L_tp.item()), global_step)

        log_interval = min(10, max(1, len(loader) // max(accum_iter, 1)))
        if (train_logger is not None and is_update_step
                and (step // max(accum_iter, 1) + 1) % log_interval == 0):
            cur_lr = optimizer.param_groups[0]['lr']
            train_logger.write(
                f"epoch {epoch:04d}  iter {step + 1:05d}  loss {loss_val:.6f}  "
                f"L_recon {float(L_rec.item()):.6f}  L_spatial {float(L_sp.item()):.6f}  "
                f"L_temporal {float(L_tp.item()):.6f}  lr {cur_lr:.2e}\n")
            train_logger.flush()

    avg = {k: v.avg for k, v in metric_logger.meters.items()}
    if train_logger is not None:
        cur_lr = optimizer.param_groups[0]['lr']
        train_logger.write(f"epoch {epoch:04d}  [epoch_avg]  "
                           f"loss {avg.get('loss', 0.0):.6f}  lr {cur_lr:.2e}"
                           + (f"  region_fallback {n_region_fallback}"
                              if region != 'all' else "") + "\n")
        train_logger.flush()
    print(f"[Train epoch {epoch}] " + "  ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
    return avg, n_samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(pred_b, gt_b):
    """pred_b, gt_b: (n_eval, L) numpy (normalized). Returns mae, rmse, mean CC."""
    mae  = float(np.abs(pred_b - gt_b).mean())
    rmse = float(np.sqrt(((pred_b - gt_b) ** 2).mean()))
    cc_list = []
    for i in range(pred_b.shape[0]):
        p, g = pred_b[i], gt_b[i]
        if g.std() < 1e-8 or p.std() < 1e-8:
            continue
        cc = np.corrcoef(p, g)[0, 1]
        if np.isfinite(cc):
            cc_list.append(float(cc))
    cc = float(np.mean(cc_list)) if cc_list else 0.0
    return mae, rmse, cc


# ---------------------------------------------------------------------------
# Shared evaluation (val + test)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval(model, loader, device, vc, mode, cluster_method, seed,
             want_vis=False, save_dir=None, max_vis_samples=12, max_vis_channels=8,
             norm_mode='per_channel', region='all'):
    """Evaluate reconstruction of the (active & ~visible) channels under one
    selection (mode, cluster_method[, region]). Selection is deterministic (rng
    seeded by `seed`) so the metric is reproducible across epochs / modes."""
    model.eval()
    is_cuda = (device.type == 'cuda')
    rng = np.random.default_rng(seed)
    all_mae, all_rmse, all_cc, all_sids, all_neval, all_nvis = [], [], [], [], [], []
    total_loss, n_batches, n_vis = 0.0, 0, 0
    compute_s, sel_s, n_forward = 0.0, 0.0, 0     # inference timing accumulators
    wall0 = time.time()

    if want_vis:
        fig_dir = os.path.join(save_dir, 'figures'); npy_dir = os.path.join(save_dir, 'signals')
        os.makedirs(fig_dir, exist_ok=True); os.makedirs(npy_dir, exist_ok=True)

    for batch in loader:
        ecg, missing_mask, coords, mu, sigma, sids = batch
        ecg          = ecg.to(device)
        missing_mask = missing_mask.to(device)
        coords       = coords.to(device)
        active = ~missing_mask

        _ts = time.time()
        visible_np = select_visible_batch(
            coords.cpu().numpy(), active.cpu().numpy(), vc, mode, cluster_method, rng,
            region=region)
        visible = torch.from_numpy(visible_np).to(device)
        sel_s += time.time() - _ts

        if norm_mode == 'global':
            ecg, _ = global_norm(ecg, visible)

        # Metrics do not depend on the smoothness terms -> skip them (weights 0).
        if is_cuda: torch.cuda.synchronize()
        _t0 = time.time()
        loss, pred = model.forward_fixedn(ecg, active, coords, visible,
                                          w_spatial=0.0, w_temporal=0.0)
        pred_signal = model.unpatchify(pred).squeeze(1)            # (B, C, W)
        if is_cuda: torch.cuda.synchronize()
        compute_s += time.time() - _t0
        n_forward += ecg.shape[0]
        total_loss += float(loss.item()); n_batches += 1

        recon_mask = active & (~visible)
        mu_np = mu.numpy(); sigma_np = sigma.numpy()

        for b in range(ecg.shape[0]):
            ch_idx = recon_mask[b].nonzero(as_tuple=False).squeeze(1)
            if len(ch_idx) == 0:
                continue
            pred_b = pred_signal[b, ch_idx, :].cpu().float().numpy()
            gt_b   = ecg[b, ch_idx, :].cpu().float().numpy()
            mae, rmse, cc = _compute_metrics(pred_b, gt_b)
            all_mae.append(mae); all_rmse.append(rmse); all_cc.append(cc)
            sid = sids[b] if isinstance(sids, (list, tuple)) else sids
            all_sids.append(sid); all_neval.append(int(len(ch_idx)))
            all_nvis.append(int(visible_np[b].sum()))

            if want_vis and n_vis < max_vis_samples:
                ci = ch_idx.cpu().numpy()
                sel = (np.linspace(0, len(ci) - 1, max_vis_channels).astype(int)
                       if len(ci) > max_vis_channels else np.arange(len(ci)))
                mb = mu_np[b, ci]; sb = sigma_np[b, ci]
                pred_dn = pred_b[sel] * sb[sel, None] + mb[sel, None]
                gt_dn   = gt_b[sel]   * sb[sel, None] + mb[sel, None]
                sid_safe = str(sid).replace('/', '_').replace(' ', '_')
                np.save(os.path.join(npy_dir, f"{sid_safe}__gt.npy"), gt_dn)
                np.save(os.path.join(npy_dir, f"{sid_safe}__pred.npy"), pred_dn)
                _save_comparison_figure(gt_dn, pred_dn, ci[sel], sid_safe,
                                        mae, rmse, cc, fig_dir)
                n_vis += 1

    results = {
        'loss': total_loss / max(n_batches, 1),
        'mae':  float(np.mean(all_mae))  if all_mae  else 0.0,
        'rmse': float(np.mean(all_rmse)) if all_rmse else 0.0,
        'cc':   float(np.mean(all_cc))   if all_cc   else 0.0,
        'cc_std':  float(np.std(all_cc))  if all_cc  else 0.0,
        'mae_std': float(np.std(all_mae)) if all_mae else 0.0,
        'n_samples': len(all_mae),
        'mean_n_recon': float(np.mean(all_neval)) if all_neval else 0.0,
        'mean_n_visible': float(np.mean(all_nvis)) if all_nvis else 0.0,
        # inference timing (compute = forward + unpatchify, GPU-synced)
        'infer_ms_per_sample': (compute_s / max(n_forward, 1)) * 1000.0,
        'select_ms_per_sample': (sel_s / max(n_forward, 1)) * 1000.0,
        'infer_wall_s': time.time() - wall0,
        'mae_per_sample': all_mae, 'rmse_per_sample': all_rmse,
        'cc_per_sample': all_cc, 'sample_ids': all_sids,
    }
    return results


def evaluate(model, loader, device, args):
    """Validation. Uses --val_select_mode/--val_cluster_method when set, so the
    model can be validated (and its best checkpoint chosen) under a DEPLOYMENT
    selection that differs from training -- e.g. train greedy, validate kmean.
    Falls back to the training selection otherwise."""
    vmode = getattr(args, 'val_select_mode', None) or args.select_mode
    vcm = getattr(args, 'val_cluster_method', None) or args.cluster_method
    res = run_eval(model, loader, device, args.vc, vmode,
                   vcm, seed=args.seed, want_vis=False,
                   norm_mode=getattr(args, 'norm_mode', 'per_channel'),
                   region=getattr(args, 'select_region', 'all'))
    print("  ".join(f"{k}: {res[k]:.4f}" for k in ('loss', 'mae', 'rmse', 'cc')))
    return res


def test_all_modes(model, loader, device, args, save_dir):
    """Testing: evaluate every selection config (cluster-greedy, cluster-kmean,
    patch, random), each with its own figures, all under args.select_region.
    Returns {label: metrics}."""
    out = {}
    region = getattr(args, 'select_region', 'all')
    for mode, cm in TEST_CONFIGS:
        if mode == 'template' and region != 'all':      # the fixed full-torso template has no in-region variant here
            print(f"  [test template        ] skipped (select_region={region}; half-torso templates: training_sweeps/eval_region_runs.sh)")
            continue
        label = cfg_label(mode, cm)
        vis_dir = os.path.join(save_dir, f'test_{label}')
        res = run_eval(model, loader, device, args.vc, mode, cm,
                       seed=args.seed, want_vis=True, save_dir=vis_dir,
                       norm_mode=getattr(args, 'norm_mode', 'per_channel'),
                       region=getattr(args, 'select_region', 'all'))
        out[label] = res
        print(f"  [test {label:16s}] mae {res['mae']:.4f}  rmse {res['rmse']:.4f}  "
              f"cc {res['cc']:.4f}  (n={res['n_samples']}, "
              f"n_recon~{res['mean_n_recon']:.0f}, n_vis~{res['mean_n_visible']:.0f})")
    return out


# ---------------------------------------------------------------------------
# Visualization helper (per-channel GT vs reconstruction)
# ---------------------------------------------------------------------------
def _save_comparison_figure(gt, pred, ch_indices, sid, mae, rmse, cc, fig_dir):
    n_eval, L = gt.shape
    t = np.arange(L)
    fig, axes = plt.subplots(n_eval, 1, figsize=(14, 2.2 * n_eval),
                             sharex=True, constrained_layout=True)
    if n_eval == 1:
        axes = [axes]
    fig.suptitle(f"Sample: {sid}\nMAE={mae:.4f}  RMSE={rmse:.4f}  CC={cc:.4f}",
                 fontsize=10, y=1.01)
    for i, ax in enumerate(axes):
        ax.plot(t, gt[i],   color='steelblue', linewidth=0.8, label='Original')
        ax.plot(t, pred[i], color='tomato', linewidth=0.8, linestyle='--',
                label='Reconstructed')
        ax.set_ylabel(f"Ch {ch_indices[i]}", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel("Time (samples)", fontsize=8)
    fig.savefig(os.path.join(fig_dir, f"{sid}.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Simple metric logger
# ---------------------------------------------------------------------------
class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.total += val * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


class MetricLogger:
    def __init__(self):
        self.meters = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)
