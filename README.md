# PPE-MAE: few-electrode electrocardiographic imaging with a physical-position masked autoencoder

Experiment code for *"Generalisable ECGi Body-Surface Potential Reconstruction in Real-World
Clinical Recordings with a Physical-Position Masked Autoencoder"*.

A masked autoencoder whose tokens carry a **physical position embedding** (PPE, an MLP of each
electrode's normalised 3-D coordinates) reconstructs the complete 252-channel body-surface
potential (BSP) field of an ECGi vest from only *N* visible electrodes selected on the torso
geometry. A fixed, physics-based Method-of-Fundamental-Solutions (MFS) inverse then recovers the
epicardial potentials, from which activation time (AT), repolarisation time (RT) and
activation–recovery interval (ARI) maps are extracted and compared against a clinical reference.
With *N* = 16 electrodes the reconstruction matches the inverse computed from all valid vest
electrodes in activation accuracy and surpasses it in repolarisation accuracy on noisy recordings.

> **Data are not included.** The recordings are clinical data that cannot be redistributed. All
> cohort identifiers in this repository are anonymised (`cohortA`, `cohortB` = the two internal
> sub-cohorts of one centre; `external` = the held-out external centre).

## Pipeline and repository layout

The repository follows the three stages of the method, one directory each:

```
N visible electrodes  --(1) bsp_reconstruction/-->  full 252-channel BSP field
                      --(2) inverse_evaluation/--->  epicardial potentials -> AT / RT / ARI vs reference
                      --(3) baselines_analyses/--->  classical & linear baselines, robustness analyses
```

| Directory | Stage | Contents |
|---|---|---|
| `bsp_reconstruction/` | 1 — reconstruct the body-surface field | `train.py` (single training/evaluation entry point), `models_mae.py` (PPE-MAE backbone: `PhysicalPosEmbed`, `IndexPosEmbed` ablation), `models_fixedn.py` (fixed-*N* reconstruction model, size factories, losses), `models_mcma.py` (convolutional baseline), `channel_select.py` (greedy / k-means / patch / random selection, optional front/back torso restriction), `dataset.py`, `preprocess.py`, `detect_missing.py`, `torso_loader.py`, `engine.py`, `viz_usage.py` |
| `inverse_evaluation/` | 2 — inverse problem and downstream metrics | `mfs.py` + `inverse.py` (MFS, Tikhonov, floored L-curve; `calibration.json` = deployed hyper-parameters), `at_rt.py` (Wyatt AT/RT/ARI with vertex gating), `build_sources.py` (the compared BSP fields), `eval_downstream.py` (per-recording metrics), `report_at_rt.py` (cohort aggregates), `cache_references.py`, `calibrate_inverse.py`, `heart_coords.py` (patient-independent epicardial coordinates), `spatial_maps.py`, deployment-matrix and diagnostic scripts, `cache_hsp/*.sh` cache drivers |
| `baselines_analyses/` | 3 — baselines and analyses | `lmmse_baseline.py` (population linear-MMSE), `interp_baseline.py` (Laplacian / thin-plate-spline completion), `lam_recal.py` (per-*N* regularisation re-calibration of the raw inverse), `fidelity_baselines.py` (BSP/HSP fidelity of the classical completions), `layout_shift_test.py` (electrode-layout-shift stress test, PPE vs index embedding), `dense_field_test.py` (continuous-field decoding at arbitrary torso points), `seed_robustness.py` (deployment-selection seed robustness), shard drivers `run_*.sh`, `common.py` |
| `training_sweeps/` | training configurations | Serial single-GPU sweep drivers (selection mode, regularisation, width/depth, electrode count, data scaling, learned baselines, training seeds, front/back-restricted selection); idempotent, resume from the last checkpoint |
| `run.sh` | | Minimal training launcher |

Generated artefacts (`runs/`, `cache/`, `inverse_evaluation/cache_hsp/*/`,
`inverse_evaluation/spatial_out/`, `baselines_analyses/generated/`, `figures/`) are git-ignored.

## Installation

Python 3.11, PyTorch ≥ 2.x with CUDA (training and inference used one 32-GB GPU).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # exact versions used for the experiments
```

Shell drivers activate `${VENV:-<repo>/venv}`; set `VENV` if your environment lives elsewhere.

## Data layout

The pipeline reads the vendor's export format of a 252-electrode ECGi vest: per recording an
`*.ecgdata` file (256 channels, 1 kHz) plus, per patient, the `.torso` (electrode positions),
`.heart` (ventricular epicardial mesh), `.valve` (valve-plane surfaces capping the mesh openings)
and, optionally, `.potential` (the vendor's epicardial reconstruction used as clinical reference).
Patients are directories; cohort roots are configured with environment variables:

```bash
export ECGI_COHORT_A_ROOT=/path/to/cohortA    # internal sub-cohort A  (training only)
export ECGI_COHORT_B_ROOT=/path/to/cohortB    # internal sub-cohort B  (train / val / internal test)
export ECGI_EXTERNAL_ROOT=/path/to/external   # external centre       (test only)
export ECGI_CACHE_DIR=/path/to/cache          # optional: preprocessing + missing-mask cache
export ECGI_FIG_DIR=/path/to/figures          # optional: where diagnostic plots are written
```

`bsp_reconstruction/dataset.py::collect_split` builds the patient-level split (seed 42): cohort
B is split into train / val / internal-test patients, cohort A is training-only, the external
cohort is test-only. Missing/bad channels are detected once per recording by three
gain-invariant criteria (`detect_missing.py`) and cached; the 4 geometry-less padding channels
are dropped globally (256 → 252). Patient directory names become part of the recording
identifiers, so keep them pseudonymous.

## Stage 1 — training the reconstruction model

The final configuration (greedy farthest-point selection during training, k-means medoid
selection at validation/deployment, temporal-smoothness weight 3 with a 100-epoch warm-up,
"wide" backbone, effective batch 64, 800 epochs):

```bash
cd bsp_reconstruction
python train.py --model fixedn_ecgi_wide --batch_size 32 --accum_iter 2 --patch_size 1 50 \
  --vc 16 --norm_mode global --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy --val_select_mode cluster --val_cluster_method kmean \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20
```

Useful flags: `--vc N` (number of visible electrodes), `--pos_embed {ppe,index}` (physical vs
ST-MEM-style index embedding), `--model mcma_unet` (convolutional baseline), `--select_region
{all,front,back}` (restrict the visible-electrode candidates to one torso half, training and
deployment alike), `--train_frac` / `--train_sources` (data-scaling experiments), `--rng_seed`
(training-only randomness; `--seed` fixes the patient split and the deployment selection and
should stay at 42). Run directories are named from the configuration
(`runs/data-cohortA+cohortB_vs_external/<model>_vc<N>_...`) and contain `args.json`,
`checkpoints/`, `test_compare.json` (BSP metrics under all selection modes) and, after the
automatic stage-2 evaluation, `at_rt_results.json`.

## Stage 2 — inverse problem and AT/RT evaluation

```bash
cd inverse_evaluation
# (once) model-independent references: all-valid inverse + commercial map, per split
python cache_references.py --run_dir <run_dir> --split test_internal
python cache_references.py --run_dir <run_dir> --split test_external
# cohort aggregates (mean/median AT, RT, ARI, HSP metrics; deploy selection = k-means)
python report_at_rt.py --run_dir <run_dir>
# per-recording CSV for paired tests (--region front|back restricts the deployment selection)
python eval_downstream.py --run_dir <run_dir> --n 1000000 --vc 16 --mode cluster --cluster_method kmean \
    --split test_internal --out ../baselines_analyses/generated/vc_percase/vc16_test_internal.csv
# re-calibrate the inverse (alpha_in, alpha_out, lambda floor) on the validation split
python calibrate_inverse.py --run_dir <run_dir>
```

Every method (reconstruction, raw *N*-electrode inverse, Laplacian / thin-plate-spline
interpolation, LMMSE, all-valid inverse) passes through the identical MFS operator; AT is scored
on QRS-usable and RT/ARI on T-usable vertices defined from the reference field, so all methods
are compared on identical vertex sets (`at_rt.py`, `eval_downstream.py`). Epicardial quantities
live on the vendor's ventricular mesh; `heart_coords.py` maps every heart onto a common
(circumferential angle, apex–base) grid using the valve surfaces to orient the long axis.

## Stage 3 — baselines and analyses

```bash
cd baselines_analyses
./run_vc_percase.sh                                   # per-recording metrics for every N
./run_interp_shards.sh                                # Laplacian / TPS completion
./run_lmmse.sh                                        # linear-MMSE completion (fit, calibrate, evaluate)
./run_lam_recal.sh                                    # per-N lambda floor for the raw inverse
python fidelity_baselines.py --split test_internal --shard 0 --nshards 4   # ... then --summary
python layout_shift_test.py                           # layout-shift stress test (PPE vs index)
python dense_field_test.py --sanity; python dense_field_test.py --calibrate; python dense_field_test.py --eval --split test_internal; python dense_field_test.py --summary
python seed_robustness.py --n 80 --seeds 42 7 123     # deployment-selection seed robustness
```

The drivers in `training_sweeps/` document every training configuration used (selection modes,
regularisation, architecture width/depth, electrode counts 4–64, training-set fractions,
learned baselines, training seeds, front/back-restricted training).

## Notes

* Amplitude normalisation is a single ratio-preserving per-recording scale computed from the
  visible electrodes only (`bsp_reconstruction/dataset.py`, `engine.py`), so that reconstructed
  fields keep physically meaningful inter-channel amplitude ratios for the inverse problem.
* Channels without a registered position are never selected as visible inputs and never enter
  the inverse; they are still reconstructed through a learned "unknown-position" embedding.
* The vendor file parsers (`bsp_reconstruction/torso_loader.py`, `inverse_evaluation/mesh_io.py`)
  read the XML geometry exports of the acquisition system; adapt them for other formats.

## Citation

If you use this code, please cite the paper (reference to be added upon publication).

## License

To be decided by the authors before public release.
