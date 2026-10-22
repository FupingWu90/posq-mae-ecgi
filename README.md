# PosQ-MAE: few-electrode electrocardiographic imaging with a position-queried masked autoencoder

Experiment code and released weights for *"Sixteen electrodes approximate full-vest
electrocardiographic imaging in real-world clinical recordings"* (PosQ-MAE, a position-queried masked
autoencoder).

PosQ-MAE completes the full 252-channel body-surface potential (BSP) field of an ECGi vest from
only *N* measured electrodes. An encoder reads the measured electrodes, each token carrying a
**physical position embedding** (an MLP of the electrode's normalised 3-D coordinates); a decoder
reconstructs the signal at any queried torso position by cross-attention to the encoded
electrodes, and a local sub-token attention head restores the within-patch temporal detail. A
fixed, physics-based Method-of-Fundamental-Solutions (MFS) inverse then recovers the epicardial
potentials, from which activation time (AT), repolarisation time (RT) and activation–recovery
interval (ARI) maps are extracted and compared against a clinical reference. With *N* = 16
electrodes (a sixteen-fold reduction) the downstream activation-time error stays within 2 ms of the
inverse computed from all valid vest electrodes, on an internal and on a held-out external cohort.

> **Data are not included.** The recordings are clinical data that cannot be redistributed. All
> cohort identifiers in this repository are anonymised (`cohortA`, `cohortB` = the two internal
> sub-cohorts of one centre; `external` = the held-out external centre). The trained weights of the
> final model (*N* = 16, training seed 42) are included in `results/final_model/` (see *Released
> weights* below).

This release supersedes the PPE-MAE code of the first manuscript version (kept as the `v1`
predecessor, see *Previous release* below): the model is new (`bsp_reconstruction/models_v2.py`),
every comparison is made under **two deployment layouts** (a fixed electrode template and
per-recording k-means placement), and the learned baselines of the paper are the models of the
previous release re-evaluated under the new protocol.

## Pipeline and repository layout

```
N measured electrodes --(1) bsp_reconstruction/-->  full 252-channel BSP field
                      --(2) inverse_evaluation/--->  epicardial potentials -> AT / RT / ARI vs reference
                      --(3) baselines_analyses/--->  classical & learned baselines, robustness and clinical analyses
                          simulation/           ->  surface-source simulated recordings (ground-truth experiments)
                          training_sweeps/      ->  registry and drivers of every training run of the paper
```

| Directory | Stage | Contents |
|---|---|---|
| `bsp_reconstruction/` | 1 — complete the body-surface field | `train.py` (single training/evaluation entry point; v1 and v2 models), `models_v2.py` (**PosQ-MAE**: position-queried cross-attention decoder, sub-token output head, tokenizer / masking / interaction variants, `complete_at()` for arbitrary query positions, `build_model()`), `models_mae.py` + `models_fixedn.py` (v1 PPE-MAE backbone and fixed-*N* model, kept as the learned baseline and as the encoder of PosQ-MAE), `models_mcma.py` (convolutional baseline), `channel_select.py` (greedy / k-means / patch / random selection and the fixed **template** layout with nearest-available substitution), `templates/` (the fixed deployment layouts for *N* = 4 … 64, built by `build_templates.py` from the mean training-set electrode geometry), `dataset.py`, `preprocess.py`, `detect_missing.py`, `torso_loader.py`, `engine.py`, `paths.py`, `viz_usage.py` |
| `inverse_evaluation/` | 2 — inverse problem and downstream metrics | `mfs.py` + `inverse.py` (MFS, Tikhonov, floored L-curve; `calibration.json` = deployed hyper-parameters), `at_rt.py` (Wyatt AT/RT/ARI with vertex gating), `build_sources.py` (the compared BSP fields), `eval_downstream.py` (per-recording metrics under a chosen layout: `--mode template` or `--mode cluster --cluster_method kmean`), `at_maps.py` (per-vertex AT/RT maps), `report_at_rt.py` (cohort aggregates), `cache_references.py`, `calibrate_inverse.py`, `eval_sim_gt.py` (ground-truth evaluation on simulated recordings), `heart_coords.py`, `spatial_maps.py`, deployment-matrix and diagnostic scripts |
| `simulation/` | surface-source simulation | `simcore.py` (equivalent-double-layer epicardial source, eikonal activation scenarios, solid-angle forward model with a bounded-torso BEM correction), `noise.py` (beat tiling, gain calibration, missing-channel fingerprints, noise), `make_dataset.py` (writes vendor-format recordings plus ground truth) |
| `baselines_analyses/` | 3 — baselines and analyses | `classical_baselines_template.py` (Laplacian / thin-plate-spline / population linear-MMSE completion under a chosen layout), `lam_recal_v2.py` (per-*N* regularisation re-calibration of the raw inverse), `dense_field_v2.py` (continuous-field decoding at arbitrary torso points), `layout_shift_v2.py` (layout-shift stress test, physical positions vs index embedding), `index_jitter_v2.py`, `deploy_robustness.py` and `region_deploy_v2.py` (deployment-layout and half-torso robustness), `patient_level_stats.py` and `patient_ids.py` (patient-level paired tests), `activation_site_error.py` (earliest/latest-activation localisation endpoint), `plausibility.py` (reference-free conduction-velocity plausibility), `gate_sens.py`, `compare_pilots.py` / `compare_pilots_val.py` (paired per-recording tests between runs), `val_engine_metrics.py`, `test_engine_metrics.py`, `pilot_metrics.py`, `baselines_two_layouts.py`, `compare_classical_layouts.py`, `results_v2.py` (the main comparison tables: cohort means and paired Wilcoxon tests of every method under both layouts), `cpu_timing.py`, `infer_time_idle.py`, `bench_variants.py`, cross-attention and query-interaction diagnostics, adoption-gate and parity checks, unit tests (`test_models_v2.py`, `test_attention_modules.py`, `test_pilot_logic.py`) |
| `training_sweeps/` | training configurations | `pilots.py` (registry of the architecture pilots and the rules of the unattended queue) and `phase3.py` (registry of the final-configuration runs: electrode counts, seeds, index embedding, data scaling, selection modes, half-torso training, simulation); `run_queue.sh` / `run_phase3.sh` (serial single-GPU queues, idempotent, resume from the last checkpoint); evaluation drivers for both deployment layouts (`eval_test_runs.sh`, `val_percase.sh`, `follow_*.sh`), for the baselines (`eval_baselines_template.sh`, `eval_classical_template.sh`, `build_template_vc_cache.sh`) and for the analyses (`run_*.sh`, `extract_atmaps.sh`, `eval_region_runs.sh`, `eval_sim_gt_runs.sh`); the recorded automatic decisions of the queue (`*_decision.json`, `final_epochs.json`) |
| `results/v1_reference/` | | Aggregate reference numbers of the previous release's final model under the new protocol (validation completion error under both layouts, convergence, template-substitution statistics), used by the adoption rules of the pilot queue and by the fairness checks; regenerated by `baselines_analyses/v1_reference.py` |
| `run.sh` | | Launcher of the final configuration |

Generated artefacts (`runs/`, `cache/`, `data/`, `results/`, `figures/`, `inverse_evaluation/cache_hsp/*/`,
`baselines_analyses/generated/`, `training_sweeps/logs/`) are git-ignored.

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
export ECGI_SIM_ROOT=/path/to/sim             # optional: simulated cohort written by simulation/make_dataset.py
export ECGI_CACHE_DIR=/path/to/cache          # optional: preprocessing + missing-mask cache
export ECGI_FIG_DIR=/path/to/figures          # optional: where diagnostic plots are written
export ECGI_V1_ROOT=/path/to/previous-release # optional: checkout of the v1 release (learned baselines, see below)
```

`bsp_reconstruction/dataset.py::collect_split` builds the patient-level split (seed 42): cohort
B is split into train / val / internal-test patients, cohort A is training-only, the external
cohort is test-only. Missing/bad channels are detected once per recording by three
gain-invariant criteria (`detect_missing.py`) and cached; the 4 geometry-less padding channels
are dropped globally (256 → 252). Patient directory names become part of the recording
identifiers, so keep them pseudonymous.

## Stage 1 — training the completion model

The final configuration (PosQ-MAE "wide": 6 encoder blocks, 3 cross-attention decoder blocks of
width 128 with 16 heads, sub-token attention head with 5-sample sub-tokens of width 32, greedy
farthest-point electrode selection during training, the fixed template layout for checkpoint
selection, temporal-smoothness weight 3 with a 100-epoch warm-up, effective batch 64, 800 epochs):

```bash
cd bsp_reconstruction
python train.py --tag FINAL --model posq_ecgi_wide --head subattn --inner_depth 1 --sub_len 5 --inner_dim 32 \
  --batch_size 32 --accum_iter 2 --patch_size 1 50 --vc 16 --norm_mode global \
  --w_spatial 0 --w_temporal 3 --reg_warmup_epochs 100 --reg_scope recon \
  --select_mode cluster --cluster_method greedy --val_select_mode template \
  --aug_times 1 --epochs 800 --use_amp --num_workers 8 --save_every 40 --eval_every 20 --downstream_splits val
```

(`run.sh` runs exactly this.) Useful flags: `--vc N` (number of measured electrodes),
`--pos_embed {ppe,index}` (physical position embedding vs an ST-MEM-style channel-index
embedding on the same backbone), `--head {linear,subattn,convref}`, `--tokenizer {linear,tnt,conv}`,
`--dec_tsa` / `--dec_ssa` (query interaction along time / across electrodes), `--dec_depth`,
`--dec_dim`, `--dec_heads`, `--enc_depth` (capacity), `--tmask_mode` (mixed masking),
`--query_frac` (fraction of the hidden electrodes decoded per step), `--time_embed`,
`--select_mode {cluster,patch,random,template}` with `--cluster_method {greedy,kmean}` (training-time
selection), `--val_select_mode` (layout used for checkpoint selection), `--select_region
{all,front,back}` (restrict the candidates to one torso half), `--train_frac` / `--train_sources`
(data scaling; `sim`, `sim+both` for the simulation experiments), `--init_from` (initialise from a
simulation-pretrained checkpoint), `--rng_seed` (training-only randomness; `--seed` fixes the
patient split and the deployment selection and should stay at 42), `--model fixedn_ecgi_wide`
(the v1 PPE-MAE model) and `--model mcma_unet` (convolutional baseline). Run directories are named
from the configuration (`runs/data-cohortA+cohortB_vs_external/<tag>__<model>_vc<N>_...`) and
contain `args.json`, `checkpoints/`, `test_compare.json` (held-out BSP metrics under every
selection mode, including the fixed template) and the downstream results of the requested splits.

**Released weights.** `results/final_model/` holds the checkpoint of the final model reported in the
paper (*N* = 16, training seed 42, selected on the validation split under the fixed template layout):
`args.json` (the complete configuration), `keep_idx.npy` (the 252 retained channels) and
`checkpoints/checkpoint_best.pth` (model weights only, 5.46 M parameters, about 22 MB). Install it as
the run directory `runs/data-cohortA+cohortB_vs_external/FINAL__.../` that the evaluation scripts
expect, then evaluate it:

```bash
python bsp_reconstruction/install_final_model.py        # --link symlinks instead of copying
bash training_sweeps/eval_test_runs.sh FINAL              # both test cohorts x both deployment layouts
```

The weights were trained on the clinical recordings and contain no patient data. The two further
training seeds (P3S43, P3S44) and the ablation runs are not distributed.

**Deployment layouts.** Every model is evaluated under (i) the fixed template layout
(`bsp_reconstruction/templates/template_vc<N>.npz`: k-means medoids of the mean normalised
electrode positions of the training geometries, with nearest-available substitution of missing
template electrodes) and (ii) per-recording k-means placement on the valid electrodes of each
recording. `python bsp_reconstruction/build_templates.py` rebuilds the templates from the training
split.

**Experiment registry.** `training_sweeps/pilots.py` lists the architecture pilots (all trained on
top of the final training recipe; decided on the validation split under both layouts): the base
query decoder (DEC1), decoder heads (DECH), temporal self-attention among the queries (DEC2),
tokenizers (LOC1 conv, LOC2 sub-token self-attention), output heads (LOC3 sub-token attention =
final, LOC3B convolutional refinement, LOC3D two sub-token blocks), smoothness-loss ablations
(LOC4 w_t = 0, LOC5 w_t = 1), mixed masking (MSK1, MSK3), learned time embedding (EMB1), query
fraction (DEC4), schedule (EPL3) and capacity (CAPE, CAPD, CAPW). `training_sweeps/phase3.py`
lists the runs of the final configuration: electrode counts 4–64 (P3N4 … P3N64), query
interaction (P3INT1–3), training seeds (P3S43, P3S44), index embedding × 3 seeds (P3IDX42–44),
training-set fractions 1/2, 1/4, 1/8 (P3TF05, P3TF025, P3TF0125), training-time selection modes
(P3SELKM, P3SELPT, P3SELRD), half-torso training (P3REGF, P3REGB) and the simulation group
(P3SIM, P3SIM2R, P3SIMR, P3FT05/025/0125). `python training_sweeps/phase3.py next` prints the next
command of the queue; `run_phase3.sh` runs it unattended.

## Stage 2 — inverse problem and AT/RT evaluation

```bash
# (once) model-independent references: all-valid inverse + commercial map, per split
python inverse_evaluation/cache_references.py --run_dir <run_dir> --split test_internal
python inverse_evaluation/cache_references.py --run_dir <run_dir> --split test_external
# raw N-electrode inverse under the fixed template layout, every N (CPU)
bash training_sweeps/build_template_vc_cache.sh
# per-recording downstream metrics of a run on both test cohorts under both layouts
bash training_sweeps/eval_test_runs.sh FINAL          # -> baselines_analyses/generated/percase_candidate/<ID>_vc<N>_<split>_<layout>.csv
# the same for the validation recordings, and the held-out completion error under both layouts
LAYOUT=template bash training_sweeps/val_percase.sh FINAL; LAYOUT=kmeans bash training_sweeps/val_percase.sh FINAL
python baselines_analyses/val_engine_metrics.py --run_dir <run_dir>
# paired per-recording tests between runs (commercial and all-valid references)
python baselines_analyses/compare_pilots.py --split test_internal --layout template FINAL P3IDX42
# cohort aggregates / re-calibration of the inverse (alpha_in, alpha_out, lambda floor) on the validation split
python inverse_evaluation/report_at_rt.py --run_dir <run_dir>
python inverse_evaluation/calibrate_inverse.py --run_dir <run_dir>
```

Every method (completion, raw *N*-electrode inverse, Laplacian / thin-plate-spline interpolation,
LMMSE, all-valid inverse, learned baselines) passes through the identical MFS operator; AT is scored
on QRS-usable and RT/ARI on T-usable vertices defined from the reference field, so all methods
are compared on identical vertex sets (`at_rt.py`, `eval_downstream.py`). Epicardial quantities
live on the vendor's ventricular mesh; `heart_coords.py` maps every heart onto a common
(circumferential angle, apex–base) grid using the valve surfaces to orient the long axis.

## Stage 3 — baselines and analyses

```bash
bash training_sweeps/eval_classical_template.sh         # Laplacian / TPS / LMMSE completion under the template layout
bash training_sweeps/eval_baselines_template.sh         # learned baselines of the previous release under the template layout
bash training_sweeps/run_lam_recal.sh                   # per-N lambda floor for the raw inverse
bash training_sweeps/run_dense_field.sh                 # continuous-field decoding at arbitrary torso points
bash training_sweeps/run_deploy_robustness.sh           # deployment-layout robustness + half-torso deployment
bash training_sweeps/eval_region_runs.sh P3REGF P3REGB  # half-torso-trained models under both protocols
python baselines_analyses/layout_shift_v2.py            # layout-shift stress test (physical positions vs index embedding)
python baselines_analyses/index_jitter_v2.py            # index-slot jitter vs the per-recording advantage of the position code
bash training_sweeps/extract_atmaps.sh FINAL; python baselines_analyses/plausibility.py --id FINAL   # per-vertex maps, conduction-velocity plausibility
python baselines_analyses/patient_level_stats.py        # patient-level paired tests
python baselines_analyses/activation_site_error.py --id FINAL   # earliest / latest-activation localisation endpoint
python baselines_analyses/gate_sens.py                  # vertex-gating sensitivity
python baselines_analyses/cpu_timing.py --run_dir <run_dir> --threads 1   # single-core CPU completion time
```

Simulation (ground-truth experiments): `python simulation/make_dataset.py` writes the simulated
cohort under `ECGI_SIM_ROOT`; the P3SIM* entries of `training_sweeps/phase3.py` train on it and
`bash training_sweeps/eval_sim_gt_runs.sh` scores every model against the simulated ground truth
under both layouts (`inverse_evaluation/eval_sim_gt.py`).

The cohort clinical-characteristics table and the stratified analyses by acquisition state and QRS
duration were derived from the study case-report tables, which are patient data; the scripts that
read them are not distributed. They operate on the per-recording CSVs produced above.

### Channel-loss experiments and patient hold-out (2026-10-20)

```bash
# observed channel loss: strata of the number of valid electrodes, threshold sweep, binned curves (both layouts)
python baselines_analyses/heavy_loss_strata.py
# emulated heavy loss: an extra 25 / 50 / 75 % of the active channels of every test recording flagged missing
# before preprocessing and selection; the reduced vest's own inverse becomes the extra source 'src2_reduced'
bash training_sweeps/run_extra_loss.sh            # calls inverse_evaluation/eval_downstream.py --extra_loss 0.25 --extra_loss_seed 1 ...
python baselines_analyses/extra_loss_summary.py   # tables / figure of the emulated-loss experiment
# training with a patient hold-out (one data-folder name per line); the held-out recordings form the role 'holdout'
python bsp_reconstruction/train.py ... --holdout_file my_holdout.txt
python inverse_evaluation/eval_downstream.py --run_dir <run> --split holdout --n 1000000 --vc 16 --mode template --out holdout.csv
bash training_sweeps/eval_holdout_runs.sh <RUN_ID>          # the same for both layouts -> results/percase_holdout/
python inverse_evaluation/cache_references.py --run_dir <run> --split holdout   # reference cache (all-valid inverse, commercial maps) of the held-out recordings
python baselines_analyses/holdout_summary.py --id <RUN_ID>  # aggregates of the held-out patients next to the internal test set (both layouts)
bash training_sweeps/eval_classical_holdout.sh               # Laplacian / TPS / LMMSE on the hold-out split, LMMSE covariance refitted without the held-out patients
```

## Previous release

The learned baselines of the paper (the v1 PPE-MAE model, the ST-MEM-style index-embedding model and
the convolutional MCMA model) and the k-means-layout results of the classical completions are the
frozen artefacts of the first release of this repository, re-evaluated under the new protocol.
`bsp_reconstruction/paths.py` reads them from `ECGI_V1_ROOT` (default `<repo>/v1`): a checkout of
that release with its `runs/`, `cache/`, `inverse_evaluation/cache_hsp/` and
`baselines_analyses/generated/`. All v1 models can also be retrained from this repository
(`--model fixedn_ecgi_wide`, `--pos_embed index`, `--model mcma_unet`; the drivers of the first
release list their configurations).

## Notes

* Amplitude normalisation is a single ratio-preserving per-recording scale computed from the
  measured electrodes only (`bsp_reconstruction/dataset.py`, `engine.py`), so that completed
  fields keep physically meaningful inter-channel amplitude ratios for the inverse problem.
* Channels without a registered position are never selected as inputs and never enter the inverse;
  they are still completed through a learned "unknown-position" embedding.
* The vendor file parsers (`bsp_reconstruction/torso_loader.py`, `inverse_evaluation/mesh_io.py`)
  read the XML geometry exports of the acquisition system; adapt them for other formats.
* The template layouts in `bsp_reconstruction/templates/` are aggregates (mean electrode geometry of
  the training vest fits, k-means medoid indices); they contain no patient-specific information.

## Citation

If you use this code or the released weights, please cite the paper:

> F. Wu et al., *Sixteen electrodes approximate full-vest electrocardiographic imaging in real-world
> clinical recordings*, 2026 (submitted). Code and weights: https://github.com/FupingWu90/posq-mae-ecgi
> (archived release: Zenodo DOI: [![DOI](https://zenodo.org/badge/1358517805.svg)](https://doi.org/10.5281/zenodo.23573242)
).

## License

MIT License; see `LICENSE`.
