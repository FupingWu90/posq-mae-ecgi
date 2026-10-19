"""
Train FixdElectrodsN_Recon: reconstruct a full 252-electrode ECGi recording from
a FIXED number (VC) of visible valid channels + physical electrode coordinates.

Key differences from the predecessor codebase (mask-ratio MAE):
  * fixed VC visible channels (whole-channel masking only), selected per mode:
      cluster (greedy FPS | kmean medoids) | patch | random
  * loss = L1 reconstruction + w_spatial * spatial-smoothness
                              + w_temporal * temporal-smoothness  (STRE-style)
  * validation uses the SAME select mode as training; testing evaluates ALL
    configs: cluster-greedy, cluster-kmean, patch, random.
  * per-electrode VC-selection counts are accumulated over training and rendered
    as an unrolled-torso spatial map (viz_usage).

Output dir encodes the tunable args:
  runs/data-<tag>/fixedn_<model>_vc<VC>_<mode>[-<cmethod>]_ws<..>-wt<..>_sc-<scope>
                  _k<knn>s<stride>_ebs<eff_batch>_crop<..>_ep<..>/
"""
import argparse, json, os, sys, time, datetime
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models_fixedn
import models_v2
from channel_select import MODES, CLUSTER_METHODS, REGIONS
from dataset import build_experiment_datasets, precompute_preprocessed
from engine import (adjust_learning_rate, train_one_epoch, evaluate,
                    test_all_modes, cfg_label)
from viz_usage import plot_usage


def get_args():
    p = argparse.ArgumentParser('FixdElectrodsN_Recon training')
    # data / split
    p.add_argument('--crop_len', type=int, default=5000)
    p.add_argument('--no_preprocess', action='store_true', default=False)
    p.add_argument('--no_prep_cache', action='store_true', default=False,
                   help='disable the full-recording preprocessing cache (preprocess '
                        'every access instead; slower but no ~11GB disk cache)')
    p.add_argument('--dataset_tag', type=str, default='cohortA+cohortB_vs_external')
    p.add_argument('--pp_norm', type=str, default='centroid')
    p.add_argument('--norm_mode', type=str, default='global',
                   choices=['global', 'per_channel'],
                   help="signal normalization: global (default; x = prep_mV / g, one "
                        "scalar g = median std of the VC channels -> preserves "
                        "inter-channel amplitude ratio, inverse-ready) | per_channel "
                        "(z-score each channel; NOT deployable for the inverse)")
    # fixed-VC selection
    p.add_argument('--vc', type=int, default=32,
                   help='number of visible valid channels kept as encoder input')
    p.add_argument('--select_mode', type=str, default='cluster', choices=list(MODES),
                   help='train/val visible-channel selection mode')
    p.add_argument('--cluster_method', type=str, default='greedy', choices=list(CLUSTER_METHODS),
                   help="cluster sub-method: greedy (farthest-point) | kmean (medoids)")
    p.add_argument('--val_select_mode', type=str, default=None, choices=list(MODES),
                   help='validation/best-ckpt selection mode; default = --select_mode. '
                        'Set to decouple val from training (e.g. train greedy, validate kmean).')
    p.add_argument('--val_cluster_method', type=str, default=None, choices=list(CLUSTER_METHODS),
                   help='validation cluster sub-method; default = --cluster_method')
    p.add_argument('--select_region', type=str, default='all', choices=list(REGIONS),
                   help="restrict the visible-electrode candidates to one torso half "
                        "(front = anterior, back = posterior; channel_select.region_mask) "
                        "for the TRAINING selection and the val/test/deploy selection "
                        "alike. Default 'all' = whole torso (unchanged behaviour).")
    p.add_argument('--min_recon', type=int, default=1,
                   help='drop recordings with fewer than vc+min_recon active channels, so '
                        'every recording can supply VC visible channels and leave >=min_recon '
                        'to reconstruct (keeps the per-batch visible count uniform)')
    # smoothness regularizers (STRE-style)
    p.add_argument('--w_spatial', type=float, default=0.0,
                   help='weight of the spatial (surface-Laplacian) smoothness term')
    p.add_argument('--w_temporal', type=float, default=0.0,
                   help='weight of the temporal (first-difference) smoothness term')
    p.add_argument('--reg_knn', type=int, default=6,
                   help='neighbours for the spatial Laplacian')
    p.add_argument('--reg_time_stride', type=int, default=10,
                   help='temporal subsampling for the spatial term (compute/memory)')
    p.add_argument('--reg_scope', type=str, default='recon', choices=['recon', 'masked'],
                   help="channels the smoothness acts on: recon (active&~visible) | "
                        "masked (all ~visible)")
    p.add_argument('--reg_warmup_epochs', type=int, default=0,
                   help='linearly ramp w_spatial/w_temporal from 0 to target over the '
                        'first N epochs (0 = full weight from epoch 0). Stabilises large '
                        'weights whose raw penalty is huge at init (esp. temporal).')
    # model
    p.add_argument('--model', type=str, default='fixedn_ecgi_base',
                   choices=['fixedn_ecgi_small', 'fixedn_ecgi_base',
                            'fixedn_ecgi_large', 'fixedn_ecgi_huge',
                            # controlled single-axis variants around base:
                            'fixedn_ecgi_narrow', 'fixedn_ecgi_wide',
                            'mcma_unet',
                            'fixedn_ecgi_wide384', 'fixedn_ecgi_wide_deep',
                            'fixedn_ecgi_shallow', 'fixedn_ecgi_deep']
                           + list(models_v2.POSQ_MODELS))
    # ---- v2 (PosQ-MAE) options; only read by the posq_* models (models_v2.py) ----
    p.add_argument('--dec_depth', type=int, default=None,
                   help='depth of the position-queried decoder (default: the factory value)')
    p.add_argument('--enc_depth', type=int, default=None, help='encoder depth override (capacity checks)')
    p.add_argument('--dec_dim', type=int, default=None, help='decoder width override (capacity checks)')
    p.add_argument('--dec_heads', type=int, default=None, help='decoder attention heads override')
    p.add_argument('--dec_tsa', action='store_true', default=False,
                   help='temporal self-attention among the queries of the same electrode in every decoder block')
    p.add_argument('--tsa_order', type=str, default='post', choices=['post', 'pre'],
                   help='decoder block order with --dec_tsa: post = cross-attn -> temporal self-attn -> MLP '
                        '(default); pre = temporal self-attn -> cross-attn -> MLP')
    p.add_argument('--dec_ssa', action='store_true', default=False,
                   help='v2: spatial query interaction -- the queries of the same time patch (across electrodes) attend to one another in every decoder block')
    p.add_argument('--tokenizer', type=str, default='linear', choices=['linear', 'tnt', 'conv'],
                   help='input tokenizer: linear patch embedding (v1) | + intra-patch self-attention | + conv stem')
    p.add_argument('--head', type=str, default='linear', choices=['linear', 'subattn', 'convref'],
                   help='output head: linear (v1) | + sub-token local attention across patch boundaries | + conv refinement')
    p.add_argument('--sub_len', type=int, default=5, help='sub-token length (samples) of tnt / subattn')
    p.add_argument('--inner_dim', type=int, default=32, help='sub-token width of tnt / subattn')
    p.add_argument('--inner_depth', type=int, default=2, help='number of sub-token blocks of tnt / subattn')
    p.add_argument('--head_ctx', type=int, default=1, help='subattn: neighbouring patches visible on each side')
    p.add_argument('--time_embed', type=str, default='sincos', choices=['sincos', 'learned'])
    p.add_argument('--ppe_type', type=str, default='mlp', choices=['mlp', 'fourier'])
    p.add_argument('--tmask_mode', type=str, default='none', choices=['none', 'dropout', 'span', 'alternate'],
                   help='time-patch masking mixed with whole-electrode masking (every second training step)')
    p.add_argument('--tmask_ratio', type=float, default=0.0)
    p.add_argument('--query_frac', type=float, default=1.0,
                   help='fraction of the masked electrodes decoded per TRAINING step (eval decodes all)')
    p.add_argument('--tag', type=str, default='', help='prefix of the run directory name (e.g. the pilot id)')
    p.add_argument('--downstream_splits', type=str, nargs='*', default=['val'],
                   help="splits of the post-training downstream AT/RT report. v2 pilots use the VALIDATION "
                        "patients only ('val' -> at_rt_results_val.json); pass test_internal test_external "
                        "for a frozen configuration (-> at_rt_results.json)")
    p.add_argument('--patch_size', type=int, nargs=2, default=[1, 50])
    # training
    p.add_argument('--epochs', type=int, default=800)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--accum_iter', type=int, default=4)
    p.add_argument('--aug_times', type=int, default=8)
    # data-scaling / composition experiments (patient-level training subsets;
    # val/test and the 252-channel input are unchanged)
    p.add_argument('--train_frac', type=float, default=1.0,
                   help='fraction of training PATIENTS kept (stratified per cohort)')
    p.add_argument('--train_sources', type=str, default='both',
                   choices=['both', 'cohortA', 'cohortB', 'sim', 'sim+both'],
                   help='which training cohorts may contribute; sim = Route A '
                        'simulated cohort only, sim+both = simulated + clinical')
    # simulation experiments (simulation/make_dataset.py)
    p.add_argument('--real_repeat', type=int, default=1,
                   help='sim+both: list every clinical recording this many times '
                        'per epoch (balanced real:sim sampling)')
    p.add_argument('--val_source', type=str, default='auto', choices=['auto', 'real', 'sim'],
                   help='validation split for checkpoint selection (auto: sim for '
                        'train_sources=sim, clinical otherwise)')
    p.add_argument('--init_from', type=str, default='',
                   help='checkpoint whose model weights initialise training '
                        '(sim-pretrain -> clinical fine-tune); optimizer restarts')
    p.add_argument('--init_tag', type=str, default='',
                   help='short label of --init_from for the run name (default: "sim")')
    p.add_argument('--test_limit', type=int, default=0,
                   help='smoke tests: evaluate only the first N test recordings')
    p.add_argument('--subsample_seed', type=int, default=0,
                   help='seed of the patient subsampling draw')
    p.add_argument('--rng_seed', type=int, default=None,
                   help='training-only RNG (init, shuffling, augmentation, greedy '
                        'restarts). Default: --seed. The patient SPLIT and the '
                        'val/test deployment selection ALWAYS use --seed, so '
                        'multi-seed runs stay comparable on identical splits.')
    p.add_argument('--pos_embed', type=str, default='ppe', choices=['ppe', 'index'],
                   help="channel position code: 'ppe' (physical 3-D MLP) or "
                        "'index' (ST-MEM-style learned per-channel-index embedding)")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--blr', type=float, default=1e-3)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--min_lr', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=80)
    p.add_argument('--use_amp', action='store_true', default=False)
    # misc
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str,
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs'))
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--save_every', type=int, default=40)
    p.add_argument('--eval_every', type=int, default=20)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--test_only', action='store_true', default=False)
    p.add_argument('--no_downstream_eval', dest='downstream_eval', action='store_false',
                   default=True,
                   help='skip the post-training downstream inverse->AT/RT report '
                        '(source 1 vs source 4 vs source 2 + commercial). The report '
                        'uses inverse_evaluation/cache_hsp references if present (build with '
                        'inverse_evaluation/cache_references.py) else computes them on the fly.')
    return p


def build_run_name(args, eff):
    mode = args.select_mode + (f"-{args.cluster_method}" if args.select_mode == 'cluster' else '')
    # val/best-ckpt selection tag, only when it differs from the training selection
    vmode = (getattr(args, 'val_select_mode', None) or args.select_mode)
    vcm = (getattr(args, 'val_cluster_method', None) or args.cluster_method)
    vlabel = vmode + (f"-{vcm}" if vmode == 'cluster' else '')
    val_tag = f"_val-{vlabel}" if vlabel != mode else ""
    nm = 'gl' if args.norm_mode == 'global' else 'pc'
    rw = getattr(args, 'reg_warmup_epochs', 0)
    rw_tag = f"-rw{rw}" if rw and rw > 0 else ""
    tf = getattr(args, 'train_frac', 1.0)
    ts = getattr(args, 'train_sources', 'both')
    ss = getattr(args, 'subsample_seed', 0)
    data_tag = ((f"_tf{tf:g}" if tf < 1.0 else "")
                + (f"_src-{ts}" if ts != 'both' else "")
                + (f"_ss{ss}" if (tf < 1.0 or ts != 'both') and ss != 0 else ""))
    pe = getattr(args, 'pos_embed', 'ppe')
    rs = getattr(args, 'rng_seed', None)
    data_tag = (f"_rs{rs}" if rs is not None and rs != args.seed else "") + data_tag
    data_tag = (f"_pe-{pe}" if pe != 'ppe' else "") + data_tag
    rr_ = getattr(args, 'real_repeat', 1)
    data_tag = data_tag + (f"_rr{rr_}" if ts == 'sim+both' and rr_ > 1 else "")
    vs_ = getattr(args, 'val_source', 'auto')
    data_tag = data_tag + (f"_val-{vs_}" if vs_ not in ('auto',) and not (vs_ == 'sim' and ts == 'sim') and not (vs_ == 'real' and ts != 'sim') else "")
    if getattr(args, 'init_from', ''):
        data_tag = data_tag + f"_init-{getattr(args, 'init_tag', '') or 'sim'}"
    reg = getattr(args, 'select_region', 'all')
    data_tag = data_tag + (f"_reg-{reg}" if reg != 'all' else "")   # last in the chain
    pre = (f"{args.tag}__" if getattr(args, 'tag', '') else '')
    return (f"{pre}{args.model}{models_v2.v2_tag(vars(args))}_vc{args.vc}_{mode}{val_tag}_nm-{nm}"
            f"_ws{args.w_spatial:g}-wt{args.w_temporal:g}{rw_tag}_sc-{args.reg_scope}"
            f"_k{args.reg_knn}s{args.reg_time_stride}"
            f"_ebs{eff}_crop{args.crop_len}_ep{args.epochs}{data_tag}")


def save_ckpt(model, opt, scaler, epoch, best, usage_counts, usage_samples, ckpt_dir, fname):
    d = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
         'epoch': epoch, 'best_val_mae': best,
         'usage_counts': np.asarray(usage_counts), 'usage_samples': int(usage_samples)}
    if scaler is not None:
        d['scaler'] = scaler.state_dict()
    torch.save(d, os.path.join(ckpt_dir, fname))


def _write_stats(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=2, default=str)


def _gpu_gb(kind, device):
    if device.type != 'cuda':
        return None
    fn = torch.cuda.max_memory_reserved if kind == 'reserved' else torch.cuda.max_memory_allocated
    return round(fn() / 1e9, 2)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    rng_seed = args.rng_seed if args.rng_seed is not None else args.seed
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    if device.type == 'cuda':
        cudnn.benchmark = True
    print(f"Device: {device}")

    eff = args.batch_size * args.accum_iter
    run = args.run_name or build_run_name(args, eff)
    save_dir = os.path.join(args.output_dir, f"data-{args.dataset_tag}", run)
    ckpt_dir = os.path.join(save_dir, 'checkpoints'); tb_dir = os.path.join(save_dir, 'tensorboard')
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True); Path(tb_dir).mkdir(parents=True, exist_ok=True)

    # ---- datasets ----
    min_active = args.vc + args.min_recon
    prep_cache = (not args.no_prep_cache) and (not args.no_preprocess)
    train_ds, val_ds, ti_ds, te_ds, info = build_experiment_datasets(
        crop_len=args.crop_len, preprocess=not args.no_preprocess, seed=args.seed,
        aug_times=args.aug_times, pp_norm=args.pp_norm, tag='exp',
        min_active=min_active, prep_cache=prep_cache, norm_mode=args.norm_mode,
        verbose=True, train_frac=args.train_frac, train_sources=args.train_sources,
        subsample_seed=args.subsample_seed, train_rng_seed=rng_seed,
        real_repeat=args.real_repeat, val_source=args.val_source)
    n_keep = info['n_keep']
    if args.test_limit > 0:
        ti_ds.samples = ti_ds.samples[:args.test_limit]; te_ds.samples = te_ds.samples[:args.test_limit]
        print(f"[smoke] test sets truncated to {args.test_limit} recordings each")

    # Build the full-recording preprocessing cache upfront (parallel) so the heavy
    # FIR/Savitzky-Golay/notch filtering is paid once, not every epoch. The random
    # crop is still applied per-access after loading, so augmentation is preserved.
    if prep_cache:
        t_pc = time.time()
        all_s = train_ds.samples + val_ds.samples + ti_ds.samples + te_ds.samples
        precompute_preprocessed(all_s, info['keep_idx'], train_ds.masks,
                                workers=max(args.num_workers, 1), verbose=True)
        print(f"Preprocessing cache ready in {time.time()-t_pc:.0f}s")

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    np.save(os.path.join(save_dir, 'keep_idx.npy'), info['keep_idx'])
    with open(os.path.join(save_dir, 'split_info.json'), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in info.items() if k != 'train_rate'}, f, indent=2, default=str)

    # Re-seed each worker's random-crop RNG per epoch so augmentation crops
    # differ across workers (fork copies the dataset's rng at an identical state).
    def _winit(worker_id):
        import numpy as _np
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        if wi is not None:
            wi.dataset.rng = _np.random.default_rng(torch.initial_seed() % (2**32))
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True,
                                   drop_last=sh, worker_init_fn=_winit)
    train_loader = dl(train_ds, True); val_loader = dl(val_ds, False)
    ti_loader = dl(ti_ds, False); te_loader = dl(te_ds, False)

    # ---- model ----
    model = models_v2.build_model(vars(args), n_keep)      # v1 names -> v1 classes, posq_* -> PosQ-MAE
    model.to(device)
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ck0['model'], strict=True), None
        print(f"[init] weights initialised from {args.init_from} (epoch {ck0.get('epoch', '?')}); "
              f"optimizer and schedule restart")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.model} signal_shape=({n_keep},{args.crop_len}) params={nparams:,}")
    print(f"VC={args.vc}  select_mode={args.select_mode}"
          + (f"/{args.cluster_method}" if args.select_mode == 'cluster' else '')
          + (f"  select_region={args.select_region}" if args.select_region != 'all' else '')
          + f"  w_spatial={args.w_spatial}  w_temporal={args.w_temporal}"
          + f"  reg_scope={args.reg_scope}")
    print(f"Save dir: {save_dir}")

    if args.lr is None:
        args.lr = args.blr
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda') if (args.use_amp and device.type == 'cuda') else None

    writer = SummaryWriter(log_dir=tb_dir)
    train_log = open(os.path.join(save_dir, 'train.log'), 'a', buffering=1)
    et_log = open(os.path.join(save_dir, 'eval_test.log'), 'a', buffering=1)
    def log_et(tag, ep, st):
        scal = {k: v for k, v in st.items() if isinstance(v, (int, float))}
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {tag} ep{ep:04d}  " + \
               "  ".join(f"{k} {v:.6f}" for k, v in scal.items())
        print(line); et_log.write(line + '\n')

    # selection rng: separate stream so per-epoch selection varies but is seeded
    select_rng = np.random.default_rng(rng_seed + 1)
    usage_counts = np.zeros(n_keep, dtype=np.int64)
    usage_samples = 0

    # ---- training-statistics recording ----
    stats_path = os.path.join(save_dir, 'training_stats.json')
    stats = dict(model=args.model, params=int(nparams), n_keep=n_keep, vc=args.vc,
                 select_mode=cfg_label(args.select_mode, args.cluster_method),
                 select_region=args.select_region,
                 amp=bool(scaler is not None), batch_size=args.batch_size,
                 accum_iter=args.accum_iter, eff_batch=eff, aug_times=args.aug_times,
                 epochs_total=args.epochs, iters_per_epoch=len(train_loader),
                 crop_len=args.crop_len, min_active=min_active, prep_cache=prep_cache,
                 norm_mode=args.norm_mode,
                 w_spatial=args.w_spatial, w_temporal=args.w_temporal, reg_scope=args.reg_scope,
                 n_train=len(train_ds), n_val=len(val_ds),
                 n_test_internal=len(ti_ds), n_test_external=len(te_ds),
                 n_sim_train=info.get('n_sim_train', 0), train_sources=args.train_sources,
                 real_repeat=args.real_repeat, val_source=info.get('val_source', 'real'),
                 init_from=args.init_from,
                 device=str(device),
                 gpu_name=(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'))
    epoch_times = []

    start_epoch, best = 0, float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start_epoch = ck.get('epoch', 0) + 1; best = ck.get('best_val_mae', float('inf'))
        if 'usage_counts' in ck:
            usage_counts = np.asarray(ck['usage_counts']).astype(np.int64)
            usage_samples = int(ck.get('usage_samples', 0))
        if scaler and 'scaler' in ck: scaler.load_state_dict(ck['scaler'])
        print(f"Resumed from epoch {start_epoch-1}")

    if not args.test_only:
        print(f"Training {args.epochs} epochs (lr={args.lr:.2e}, eff_batch={eff})")
        t0 = time.time()
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(opt, epoch, args)
            ep_t0 = time.time()
            ts, n_s = train_one_epoch(model, train_loader, opt, device, epoch, args,
                                      select_rng, usage_counts, loss_scaler=scaler,
                                      log_writer=writer, train_logger=train_log)
            ep_dt = time.time() - ep_t0
            epoch_times.append(ep_dt)
            usage_samples += n_s
            for k, v in ts.items(): writer.add_scalar(f'train/{k}', v, epoch)
            writer.add_scalar('time/epoch_sec', ep_dt, epoch)
            if epoch % args.eval_every == 0 or epoch + 1 == args.epochs:
                vs = evaluate(model, val_loader, device, args)
                log_et('VAL ', epoch, vs)
                for k, v in vs.items():
                    if isinstance(v, (int, float)): writer.add_scalar(f'val/{k}', v, epoch)
                if vs['mae'] < best:
                    best = vs['mae']
                    save_ckpt(model, opt, scaler, epoch, best, usage_counts, usage_samples,
                              ckpt_dir, 'checkpoint_best.pth')
            if epoch % args.save_every == 0 or epoch + 1 == args.epochs:
                save_ckpt(model, opt, scaler, epoch, best, usage_counts, usage_samples,
                          ckpt_dir, 'checkpoint_last.pth')

            # crash-safe rolling training statistics
            recent = float(np.mean(epoch_times[-20:]))
            stats.update(epochs_done=epoch + 1,
                         last_epoch_sec=round(ep_dt, 1),
                         mean_epoch_sec=round(recent, 1),
                         ms_per_iter=round(recent / max(len(train_loader), 1) * 1000, 1),
                         samples_per_sec=round(len(train_ds) / recent, 1) if recent > 0 else None,
                         gpu_peak_alloc_gb=_gpu_gb('alloc', device),
                         gpu_peak_reserved_gb=_gpu_gb('reserved', device),
                         elapsed_train_hours=round((time.time() - t0) / 3600, 3),
                         est_total_train_hours=round(recent * args.epochs / 3600, 2),
                         val_best_mae=round(float(best), 6))
            _write_stats(stats_path, stats)
            writer.flush()
        print(f"Training time: {datetime.timedelta(seconds=int(time.time()-t0))}")

        # ---- electrode-usage spatial visualization ----
        mode_label = cfg_label(args.select_mode, args.cluster_method)
        try:
            plot_usage(usage_counts, usage_samples, info['keep_idx'],
                       os.path.join(save_dir, 'usage'), args.vc, mode_label)
        except Exception as e:
            print(f"[warn] usage visualization failed: {e}")

    # ---- final test: internal + external, ALL selection configs ----
    bp = os.path.join(ckpt_dir, 'checkpoint_best.pth')
    if os.path.exists(bp):
        ck = torch.load(bp, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        best_ep = ck.get('epoch', '?'); print(f"[Test] loaded best ckpt (epoch {best_ep})")
    else:
        best_ep = args.epochs - 1
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()   # measure inference peak separately
    test_t0 = time.time()
    compare = {}
    for name, loader, ds in [('test_internal', ti_loader, ti_ds),
                             ('test_external', te_loader, te_ds)]:
        res = test_all_modes(model, loader, device, args,
                             save_dir=os.path.join(save_dir, name))
        scal = {lab: {k: v for k, v in m.items() if isinstance(v, (int, float))}
                for lab, m in res.items()}
        with open(os.path.join(save_dir, f'{name}_results.json'), 'w') as f:
            json.dump({'metrics': scal, 'n': len(ds), 'best_epoch': str(best_ep)}, f, indent=2)
        compare[name] = scal
    with open(os.path.join(save_dir, 'test_compare.json'), 'w') as f:
        json.dump({'best_epoch': str(best_ep), 'n_keep': n_keep, 'vc': args.vc,
                   'train_select_mode': cfg_label(args.select_mode, args.cluster_method),
                   'select_region': args.select_region,
                   'results': compare,
                   'n_test_internal': info['n_test_internal'],
                   'n_test_external': info['n_test_external']}, f, indent=2)

    # ---- finalize inference statistics (compute-only, averaged over configs) ----
    infer = [m['infer_ms_per_sample'] for cfgs in compare.values() for m in cfgs.values()
             if 'infer_ms_per_sample' in m]
    sel = [m['select_ms_per_sample'] for cfgs in compare.values() for m in cfgs.values()
           if 'select_ms_per_sample' in m]
    stats.update(infer_ms_per_sample=round(float(np.mean(infer)), 2) if infer else None,
                 select_ms_per_sample=round(float(np.mean(sel)), 2) if sel else None,
                 infer_gpu_peak_gb=_gpu_gb('alloc', device),
                 test_wall_sec=round(time.time() - test_t0, 1),
                 best_epoch=str(best_ep))
    _write_stats(stats_path, stats)
    print(f"Done. Results in {save_dir}")
    print(f"Stats: {stats_path}")
    train_log.close(); et_log.close(); writer.close()

    # ---- post-training downstream inverse -> AT/RT report ----
    # Reconstruct BSP -> ECGi inverse -> heart-surface AT/RT, scoring
    # source 1 (VC-only) vs source 4 (model recon) against source 2 (all-valid)
    # and the commercial .potential. Robust: never breaks a completed training.
    if getattr(args, 'downstream_eval', False):
        try:
            dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse_evaluation')
            if dd not in sys.path:
                sys.path.insert(0, dd)
            from report_at_rt import report
            ds_splits = list(getattr(args, 'downstream_splits', None) or [])
            if 'val' in ds_splits:
                report(save_dir, splits=('val',), out_json=os.path.join(save_dir, 'at_rt_results_val.json'))
            tests = tuple(s_ for s_ in ds_splits if s_ != 'val')
            if tests:
                report(save_dir, splits=tests)
        except Exception as e:
            print(f"[warn] downstream AT/RT report skipped: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main(get_args().parse_args())
