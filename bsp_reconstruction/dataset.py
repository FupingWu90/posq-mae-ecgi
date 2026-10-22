"""
ECGi dataset for FixdElectrodsN_Recon (fixed valid-channel reconstruction).

Data collection, cross-dataset patient split, recursive torso lookup, channel
reduction (256 -> n_keep = 252 geometry-registered electrodes) and the disk-
cached high-pass missing detector are all IDENTICAL to the predecessor codebase — the same
`cache/missing_masks_exp_keep252_*.pkl` is reused.

Channel count: the 4 always-geometry-less padding slots {252,253,254,255} are
dropped globally -> every sample is 252-wide. NOTE that within those 252 kept
channels a given patient's torso may still leave a few UNREGISTERED (222–252
registered per recording, mean ~251; ~80% of recordings register all 252).
Unregistered channels get normalized coords (0,0,0) via _coords_for (nan->0).
The engine's VC selection and the model's spatial smoothness both treat all-zero
coords as invalid, so such channels are never chosen as a visible VC and are
excluded from the distance-based spatial term; they are still reconstructed
(active ones carry ground truth) via the backbone's learned "unknown-position"
embedding, exactly as in the predecessor codebase.

What changes vs the predecessor codebase: there is no eval-time channel masking baked into
the dataset. The fixed-VC visible-channel selection (cluster / patch / random)
is performed in the engine so that (a) it can vary per epoch and (b) the test
set can be evaluated under all three modes from a single dataset. Return
signatures:
  train    : (ecg, missing_mask, coords)
  val/test : (ecg, missing_mask, coords, mu, sigma, sample_id)
`missing_mask` marks detector-flagged channels (~active); the engine derives
active = ~missing_mask and chooses the VC visible subset from the active set.
"""
import os, glob, random, pickle, hashlib, uuid
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_ecgi
from torso_loader import load_torso_coords, normalize_coords
from detect_missing import detect_missing

CROP_LEN = 5000
MIN_LEN  = 5000
NCH_RAW  = 256

# Cohort roots: one sub-directory per patient holding the vendor exports
# (*.ecgdata, .torso, .heart, .valve, .potential).  Override with environment variables.
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT_A_ROOT = os.environ.get("ECGI_COHORT_A_ROOT", os.path.join(_PROJ, "data", "cohortA"))   # internal sub-cohort A (train only)
COHORT_B_ROOT = os.environ.get("ECGI_COHORT_B_ROOT", os.path.join(_PROJ, "data", "cohortB"))   # internal sub-cohort B (train / val / test_internal)
EXTERNAL_ROOT = os.environ.get("ECGI_EXTERNAL_ROOT", os.path.join(_PROJ, "data", "external"))  # external test cohort
# Simulated cohort (simulation/make_dataset.py): <SIM_ROOT>/{train,val,test}/<cohort>-<patient>/CI/*.ecgdata
SIM_ROOT = os.environ.get("ECGI_SIM_ROOT", os.path.join(_PROJ, "data", "sim"))
COHORT_B_SPLIT = (19, 5, 13)   # train / val / test patients (sums to 37)

# v2: caches are READ from the frozen v1 tree when present and WRITTEN only under the
# v2 workspace (paths.py). Keys are unchanged, so every v1 entry is reused as is.
from paths import V1_CACHE_DIR, V2_CACHE_DIR, assert_v2
CACHE_DIR = V2_CACHE_DIR
PREP_CACHE_DIR = os.path.join(CACHE_DIR, "prep")   # cached FULL preprocessed recordings (new entries)
V1_PREP_CACHE_DIR = os.path.join(V1_CACHE_DIR, "prep")   # read-only


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _read_ecgdata(path):
    try:
        with open(path, 'rb') as f:
            L = int(np.fromfile(f, np.int32, 1)[0])
            C = int(np.fromfile(f, np.int32, 1)[0])
            d = np.fromfile(f, np.float32, -1)
        if C != NCH_RAW or d.size != L * C:
            return None
        return d.reshape(L, C).T
    except Exception:
        return None


def _header_ok(path):
    try:
        with open(path, 'rb') as f:
            L = int(np.fromfile(f, np.int32, 1)[0]); C = int(np.fromfile(f, np.int32, 1)[0])
        return (L >= MIN_LEN) and (C == NCH_RAW)
    except Exception:
        return False


def find_torso_near(patient_dir, ecg_path):
    """Co-located torso wins (cohort B: CI/*.torso); else patient root; else recursive."""
    d = os.path.dirname(ecg_path)
    for cand in (glob.glob(os.path.join(d, '*.torso')),
                 glob.glob(os.path.join(patient_dir, '*.torso')),
                 glob.glob(os.path.join(patient_dir, '**', '*.torso'), recursive=True)):
        if cand:
            return cand[0]
    return None


def _center_crop(ecg, n=CROP_LEN):
    L = ecg.shape[1]
    if L <= n:
        return ecg
    s = (L - n) // 2
    return ecg[:, s:s + n]


# --------------------------------------------------------------------------- #
# Sample collection + cross-dataset split
# --------------------------------------------------------------------------- #
def _patient_dirs(root):
    return sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p)))


def _ecgfiles(pdir, scope):
    if scope == '**':
        return sorted(glob.glob(os.path.join(pdir, '**', '*.ecgdata'), recursive=True))
    return sorted(glob.glob(os.path.join(pdir, scope, '*.ecgdata')))


def collect_split(seed=42, holdout=None):
    """role -> list of (sample_id, filepath, torso_path); roles:
    'train','val','test_internal','test_external'. Returns (buckets, cohortB_assign).
    holdout: optional set of patient folder names (any cohort); their recordings are moved from 'train'
    (and 'val') into the extra role 'holdout' AFTER the seed-based split, so every other role is unchanged
    (2026-10-20: invasive-mapping hold-out retrain; the list lives in a private file, never printed)."""
    buckets = {'train': [], 'val': [], 'test_internal': [], 'test_external': [],
               'sim_train': [], 'sim_val': [], 'sim_test': [], 'holdout': []}
    seen = set()

    def add(role, root_tag, pdir, scope):
        for ef in _ecgfiles(pdir, scope):
            rp = os.path.realpath(ef)
            if rp in seen or not _header_ok(ef):
                continue
            seen.add(rp)
            stem = os.path.splitext(os.path.basename(ef))[0]
            sid = f"{root_tag}__{os.path.basename(pdir)}__{stem}"
            buckets[role].append((sid, ef, find_torso_near(pdir, ef)))

    for p in _patient_dirs(COHORT_A_ROOT):
        add('train', 'cohortA', os.path.join(COHORT_A_ROOT, p), '**')

    cohortB_pat = _patient_dirs(COHORT_B_ROOT)
    rng = random.Random(seed)
    order = list(range(len(cohortB_pat))); rng.shuffle(order)
    n_tr, n_va, _ = COHORT_B_SPLIT
    cohortB_train = sorted(cohortB_pat[i] for i in order[:n_tr])
    cohortB_val   = sorted(cohortB_pat[i] for i in order[n_tr:n_tr + n_va])
    cohortB_test  = sorted(cohortB_pat[i] for i in order[n_tr + n_va:])
    for p in cohortB_train: add('train',         'cohortB', os.path.join(COHORT_B_ROOT, p), 'CI')
    for p in cohortB_val:   add('val',           'cohortB', os.path.join(COHORT_B_ROOT, p), 'CI')
    for p in cohortB_test:  add('test_internal', 'cohortB', os.path.join(COHORT_B_ROOT, p), 'CI')

    for p in _patient_dirs(EXTERNAL_ROOT):
        add('test_external', 'external', os.path.join(EXTERNAL_ROOT, p), 'CI')

    # simulated cohort (optional): sample ids 'sim__<cohort>-<patient>__<stem>'
    for split in ('train', 'val', 'test'):
        root = os.path.join(SIM_ROOT, split)
        if os.path.isdir(root):
            for p in _patient_dirs(root):
                add(f'sim_{split}', 'sim', os.path.join(root, p), 'CI')

    if holdout:
        hold = set(holdout)
        for role in ('train', 'val'):
            keep, moved = [], []
            for rec in buckets[role]:
                (moved if rec[0].split('__', 2)[1] in hold else keep).append(rec)
            buckets[role] = keep; buckets['holdout'] += moved
        buckets['holdout'].sort()
    return buckets, dict(cohortB_train=cohortB_train, cohortB_val=cohortB_val, cohortB_test=cohortB_test)


# --------------------------------------------------------------------------- #
# Geometry (keep-set) + coordinate caches
# --------------------------------------------------------------------------- #
_COORD_CACHE = {}


def _coords_for(torso_path, pp_norm='centroid'):
    if torso_path is None:
        return np.zeros((NCH_RAW, 3), np.float32)
    key = (torso_path, pp_norm)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    try:
        c = normalize_coords(load_torso_coords(torso_path), pp_norm=pp_norm)
        c = np.nan_to_num(c, nan=0.0).astype(np.float32)
    except Exception:
        c = np.zeros((NCH_RAW, 3), np.float32)
    _COORD_CACHE[key] = c
    return c


def _raw_coords_for(torso_path):
    """Raw coords (256,3), NaN for unregistered — used by spatial detector. Cached."""
    key = ('raw', torso_path)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    if torso_path is None:
        c = np.full((NCH_RAW, 3), np.nan, np.float32)
    else:
        try:
            c = load_torso_coords(torso_path).astype(np.float32)
        except Exception:
            c = np.full((NCH_RAW, 3), np.nan, np.float32)
    _COORD_CACHE[key] = c
    return c


def geometry_union(samples):
    reg = np.zeros(NCH_RAW, bool); seen_t = set()
    for _, _, tp in samples:
        if tp is None or tp in seen_t:
            continue
        seen_t.add(tp)
        reg |= ~np.isnan(_raw_coords_for(tp)[:, 0])
    return reg


# --------------------------------------------------------------------------- #
# Per-file missing-mask precompute (disk cached; identical to the predecessor codebase)
# --------------------------------------------------------------------------- #
def _content_key(fpath):
    """Cache key of a recording: the real path, plus mtime+size for files under
    SIM_ROOT (simulated recordings are regenerated in place, so a path-only key
    would serve stale masks / preprocessed signals)."""
    rp = os.path.realpath(fpath)
    if rp.startswith(os.path.realpath(SIM_ROOT)):
        st = os.stat(rp)
        return f"{rp}|{st.st_mtime_ns}|{st.st_size}"
    return rp


def _cache_path(keep_idx, tag, root=None):
    h = hashlib.md5(keep_idx.tobytes()).hexdigest()[:8]
    return os.path.join(root or CACHE_DIR, f"missing_masks_{tag}_keep{len(keep_idx)}_{h}.pkl")


def precompute_masks(all_samples, keep_idx, tag='exp', verbose=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = _cache_path(keep_idx, tag)                       # v2 (written when new masks appear)
    rpath = cpath if os.path.exists(cpath) else _cache_path(keep_idx, tag, V1_CACHE_DIR)   # else read v1
    cache = {}
    if os.path.exists(rpath):
        try:
            with open(rpath, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    masks = {}; n_new = 0
    for i, (sid, fp, tp) in enumerate(all_samples):
        rp = _content_key(fp)
        if rp in cache:
            masks[fp] = cache[rp]; continue
        ecg = _read_ecgdata(fp)
        if ecg is None:
            masks[fp] = np.ones(len(keep_idx), bool); cache[rp] = masks[fp]; continue
        ecg = _center_crop(ecg)[keep_idx]
        coords = _raw_coords_for(tp)[keep_idx]
        m = detect_missing(ecg, coords=coords)
        masks[fp] = m; cache[rp] = m; n_new += 1
        if verbose and (i % 200 == 0):
            print(f"  [precompute_masks] {i}/{len(all_samples)} (new={n_new})", flush=True)
    if n_new:
        try:
            with open(assert_v2(cpath), 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"  [precompute_masks] cache save failed: {e}")
    if verbose:
        print(f"  [precompute_masks] {len(masks)} files, {n_new} newly computed, cache={rpath if not n_new else cpath}")
    return masks


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _normalize(ecg, missing_mask):
    n = ecg.shape[0]
    out = np.zeros_like(ecg)
    mus = np.zeros(n, np.float32); sigmas = np.ones(n, np.float32)
    for i in range(n):
        if missing_mask[i]:
            continue
        ch = ecg[i]; mu = ch.mean(); sd = ch.std()
        if sd < 1e-8:
            sd = 1.0
        mus[i] = mu; sigmas[i] = sd
        out[i] = (ch - mu) / sd
    return out, mus, sigmas


# --------------------------------------------------------------------------- #
# Full-recording preprocessing cache (filter once, crop per-epoch after)
# --------------------------------------------------------------------------- #
def _prep_cache_name(fpath, keep_idx, missing_mask):
    key = (_content_key(fpath).encode()
           + np.asarray(keep_idx).tobytes()
           + np.asarray(missing_mask, dtype=bool).tobytes())
    return hashlib.md5(key).hexdigest() + '.npy'


def _prep_cache_path(fpath, keep_idx, missing_mask):
    """Path a NEW cache entry is written to (v2 workspace)."""
    return os.path.join(PREP_CACHE_DIR, _prep_cache_name(fpath, keep_idx, missing_mask))


def _prep_existing(fpath, keep_idx, missing_mask):
    """Existing cache entry (v2 first, then the read-only v1 cache) or None."""
    name = _prep_cache_name(fpath, keep_idx, missing_mask)
    for root in (PREP_CACHE_DIR, V1_PREP_CACHE_DIR):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return None


def _build_one_prep(task):
    """Compute + atomically cache the FULL preprocessed recording for one file.
    Returns 1 if newly written, 0 if already present. Module-level so it is
    picklable for multiprocessing."""
    fpath, keep_idx, mask = task
    if _prep_existing(fpath, keep_idx, mask) is not None:
        return 0
    path = assert_v2(_prep_cache_path(fpath, keep_idx, mask))
    raw = _read_ecgdata(fpath)
    if raw is None:
        prep = np.zeros((len(keep_idx), MIN_LEN), np.float32)
    else:
        prep = preprocess_ecgi(raw[keep_idx].astype(np.float32), mask).astype(np.float32)
    tmp = path[:-4] + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
    try:
        np.save(tmp, prep)
        os.replace(tmp + '.npy', path)
    except Exception:
        try:
            os.remove(tmp + '.npy')
        except Exception:
            pass
    return 1


def precompute_preprocessed(all_samples, keep_idx, masks, workers=8, verbose=True):
    """Build the full-recording preprocessing cache upfront (parallel), so the
    heavy filtering is paid once instead of during the first epoch."""
    os.makedirs(PREP_CACHE_DIR, exist_ok=True)
    tasks = [(fp, keep_idx, masks[fp]) for _, fp, _ in all_samples]
    n_new = 0
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_build_one_prep, tasks, chunksize=8)):
                n_new += r
                if verbose and i % 200 == 0:
                    print(f"  [precompute_preprocessed] {i}/{len(tasks)} (new={n_new})", flush=True)
    else:
        for t in tasks:
            n_new += _build_one_prep(t)
    if verbose:
        print(f"  [precompute_preprocessed] {len(tasks)} files, {n_new} newly cached "
              f"in {PREP_CACHE_DIR}")
    return n_new


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ECGiDataset(Dataset):
    def __init__(self, samples, masks, keep_idx, mode='train',
                 crop_len=CROP_LEN, preprocess=True,
                 seed=42, aug_times=1, pp_norm='centroid', prep_cache=True,
                 norm_mode='per_channel'):
        self.samples = samples
        self.masks = masks
        self.keep_idx = keep_idx
        self.n_keep = len(keep_idx)
        self.mode = mode
        self.crop_len = crop_len
        self.preprocess = preprocess
        self.aug_times = aug_times if mode == 'train' else 1
        self.pp_norm = pp_norm
        self.rng = np.random.default_rng(seed)
        # 'per_channel' -> z-score each channel here (VC-independent, so done in
        # the dataset). 'global' -> single per-recording scale g = median std of
        # the N VISIBLE (VC) channels; VC is chosen in the engine, so global-mode
        # normalization is DEFERRED to the engine and the dataset returns the
        # physical (preprocessed) signal.
        self.norm_mode = norm_mode
        # Cache the FULL preprocessed recording per file so the expensive
        # FIR/Savitzky-Golay/notch filtering runs ONCE (not per epoch). The
        # random crop is applied AFTER loading the cached signal, so per-epoch
        # crop augmentation is fully preserved (preprocessing is shift-local, so
        # preprocess-then-crop matches crop-then-preprocess except at window
        # edges, where filtering the full signal is actually cleaner).
        self.prep_cache = bool(prep_cache and preprocess)
        self.prep_dir = PREP_CACHE_DIR
        if self.prep_cache:
            os.makedirs(self.prep_dir, exist_ok=True)

    def __len__(self):
        return len(self.samples) * self.aug_times

    def _crop(self, ecg):
        L = ecg.shape[1]
        if L == self.crop_len:
            return ecg
        if self.mode == 'train':
            s = int(self.rng.integers(0, L - self.crop_len + 1))
        else:
            s = (L - self.crop_len) // 2
        return ecg[:, s:s + self.crop_len]

    def _prep_path(self, fpath, missing_mask):
        return _prep_cache_path(fpath, self.keep_idx, missing_mask)

    def _load_preprocessed_full(self, fpath, missing_mask):
        """Return the FULL preprocessed recording (n_keep, L), from disk cache if
        present, else compute it, cache atomically, and return it. Cropping and
        normalization happen in __getitem__ AFTER this."""
        hit = _prep_existing(fpath, self.keep_idx, missing_mask)
        if hit is not None:
            try:
                return np.load(hit, mmap_mode='r')
            except Exception:
                pass
        path = assert_v2(self._prep_path(fpath, missing_mask))
        raw = _read_ecgdata(fpath)
        if raw is None:
            prep = np.zeros((self.n_keep, MIN_LEN), np.float32)
        else:
            prep = preprocess_ecgi(raw[self.keep_idx].astype(np.float32),
                                   missing_mask).astype(np.float32)
        tmp = path[:-4] + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
        try:
            np.save(tmp, prep)              # writes tmp + '.npy'
            os.replace(tmp + '.npy', path)  # atomic publish
        except Exception:
            try:
                os.remove(tmp + '.npy')
            except Exception:
                pass
        return prep

    def __getitem__(self, idx):
        sid, fpath, tpath = self.samples[idx % len(self.samples)]
        missing_mask = self.masks[fpath].copy()

        if self.prep_cache:
            full = self._load_preprocessed_full(fpath, missing_mask)  # (n_keep, L)
            L = full.shape[1]
            if L < self.crop_len:                                     # degenerate/short
                ecg = np.zeros((self.n_keep, self.crop_len), np.float32)
                ecg[:, :L] = np.asarray(full)
            else:
                ecg = np.array(self._crop(full), dtype=np.float32)    # random crop AFTER preprocess
        else:
            ecg = _read_ecgdata(fpath)
            if ecg is None or ecg.shape[1] < self.crop_len:
                ecg = np.zeros((NCH_RAW, self.crop_len), np.float32)
            ecg = ecg[self.keep_idx]
            ecg = self._crop(ecg)
            if self.preprocess:
                ecg = preprocess_ecgi(ecg, missing_mask)

        if self.norm_mode == 'global':
            # Normalization deferred to the engine (needs the VC selection).
            # Return the physical preprocessed signal; missing channels are zero.
            ecg = ecg.astype(np.float32).copy()
            ecg[missing_mask] = 0.0
            mus = np.zeros(self.n_keep, np.float32)
            sigmas = np.ones(self.n_keep, np.float32)
        else:
            ecg, mus, sigmas = _normalize(ecg, missing_mask)
            ecg = ecg.astype(np.float32)

        coords = _coords_for(tpath, self.pp_norm)[self.keep_idx].astype(np.float32)

        ecg_t = torch.from_numpy(ecg)
        miss_t = torch.from_numpy(missing_mask)
        coords_t = torch.from_numpy(coords)

        if self.mode == 'train':
            return ecg_t, miss_t, coords_t

        # val / test: VC visible-channel selection happens in the engine.
        return (ecg_t, miss_t, coords_t,
                torch.from_numpy(mus), torch.from_numpy(sigmas), sid)


# --------------------------------------------------------------------------- #
# Build everything
# --------------------------------------------------------------------------- #
def _filter_min_active(samples, masks, n_keep, min_active):
    """Drop recordings with fewer than `min_active` active (non-missing)
    channels. Needed so every recording can supply VC visible channels AND
    leave channels to reconstruct, which keeps the per-batch visible count
    uniform (no encoder truncation). Returns (kept_samples, n_dropped)."""
    if min_active <= 0:
        return samples, 0
    kept = [s for s in samples if (n_keep - int(masks[s[1]].sum())) >= min_active]
    return kept, len(samples) - len(kept)


def subsample_train(train_s, train_frac=1.0, train_sources='both',
                    subsample_seed=0, verbose=True):
    """Data-scaling / composition control for the training set (patient level).

    train_sources : 'both' | 'cohortA' | 'cohortB'  -- which cohorts may contribute
    train_frac    : fraction of training PATIENTS kept, stratified per cohort
                    (recordings of one patient are highly correlated, so the
                    unit of subsampling is the patient, not the recording)
    subsample_seed: RNG seed of the patient draw (fixed -> nested subsets are
                    NOT guaranteed across fractions; same seed -> same draw)
    sample ids are '<cohort>__<patient>__<stem>'. Returns the kept list.
    """
    if train_sources == 'both' and train_frac >= 1.0:
        return train_s
    by = {}
    for smp in train_s:
        coh, pat = smp[0].split('__', 2)[:2]
        by.setdefault(coh, {}).setdefault(pat, []).append(smp)
    if train_sources != 'both':
        by = {k: v for k, v in by.items() if k == train_sources}
    rng = random.Random(subsample_seed)
    kept = []
    for coh in sorted(by):
        pats = sorted(by[coh])
        rng.shuffle(pats)
        n_keep_p = max(1, int(round(train_frac * len(pats)))) if train_frac < 1.0 else len(pats)
        sel = sorted(pats[:n_keep_p])
        if verbose:
            print(f"[subsample] {coh}: {n_keep_p}/{len(pats)} patients "
                  f"-> {sum(len(by[coh][q]) for q in sel)} recordings "
                  f"(frac={train_frac:g}, sources={train_sources}, seed={subsample_seed})")
        for q in sel:
            kept += by[coh][q]
    return kept


def build_experiment_datasets(crop_len=CROP_LEN, preprocess=True,
                              seed=42, aug_times=1, pp_norm='centroid',
                              tag='exp', min_active=0, prep_cache=True,
                              norm_mode='per_channel', verbose=True,
                              train_frac=1.0, train_sources='both',
                              subsample_seed=0, train_rng_seed=None,
                              real_repeat=1, val_source='auto', holdout_patients=None):
    """train_sources: 'both' | 'cohortA' | 'cohortB'  (clinical cohorts)
                      'sim'                       (simulated cohort only)
                      'sim+both'                  (simulated + clinical, mixed)
    real_repeat : when mixing, every clinical recording is listed this many
                  times so that real:sim sampling is balanced although the
                  simulated set is larger (per-epoch weight, not new data)
    val_source  : 'auto' (sim val for 'sim', clinical val otherwise) | 'real' | 'sim'
    keep_idx (the 252-channel input) always comes from the CLINICAL training
    geometries so that every model shares the identical input width."""
    buckets, cohortB_assign = collect_split(seed=seed, holdout=holdout_patients)
    train_s = buckets['train']; val_s = buckets['val']
    if holdout_patients:
        n_hold_pat = len({r[0].split('__', 2)[1] for r in buckets['holdout']})
        if verbose:
            print(f"[holdout] {len(buckets['holdout'])} recordings of {n_hold_pat} patients removed from train/val "
                  f"(train now {len(train_s)}, val {len(val_s)})")
    ti_s = buckets['test_internal']; te_s = buckets['test_external']
    use_sim = train_sources in ('sim', 'sim+both')
    sim_train = buckets.get('sim_train', []); sim_val = buckets.get('sim_val', [])
    if use_sim and not sim_train:
        raise RuntimeError(f'train_sources={train_sources} but no simulated recordings under {SIM_ROOT}/train')
    if val_source == 'auto':
        val_source = 'sim' if train_sources == 'sim' else 'real'
    if val_source == 'sim':
        if not sim_val:
            raise RuntimeError(f'val_source=sim but no simulated recordings under {SIM_ROOT}/val')
        val_s = sim_val
    if verbose:
        print(f"Samples — train {len(train_s)} (cohortA+cohortB_train), val {len(val_s)} (cohortB_val), "
              f"test_internal {len(ti_s)} (cohortB_test), test_external {len(te_s)} (external)")
        print(f"cohort B train {cohortB_assign['cohortB_train']}")
        print(f"cohort B val   {cohortB_assign['cohortB_val']}")
        print(f"cohort B test  {cohortB_assign['cohortB_test']}")

    reg = geometry_union(train_s)
    keep_idx = np.where(reg)[0].astype(np.int64)
    if verbose:
        dropped = sorted(set(range(NCH_RAW)) - set(keep_idx.tolist()))
        print(f"Geometry-registered (train): {len(keep_idx)}; dropped (no geometry): {dropped}")

    all_samples = train_s + val_s + ti_s + te_s + (sim_train if use_sim else [])
    masks = precompute_masks(all_samples, keep_idx, tag=tag, verbose=verbose)

    train_rate = np.mean([masks[fp] for _, fp, _ in train_s], axis=0)
    always = np.where(train_rate >= 1.0)[0]
    if len(always):
        if verbose:
            print(f"Dropping {len(always)} always-missing kept channels: "
                  f"{[int(keep_idx[i]) for i in always]}")
        good = np.ones(len(keep_idx), bool); good[always] = False
        keep_idx = keep_idx[good]
        masks = {fp: m[good] for fp, m in masks.items()}
        train_rate = train_rate[good]
    n_keep = len(keep_idx)

    # Drop recordings with too few active channels so every recording can supply
    # VC visible channels (+ >=1 to reconstruct) -> uniform per-batch visible count.
    if min_active > 0:
        drops = {}
        train_s, drops['train']         = _filter_min_active(train_s, masks, n_keep, min_active)
        val_s,   drops['val']           = _filter_min_active(val_s,   masks, n_keep, min_active)
        ti_s,    drops['test_internal'] = _filter_min_active(ti_s,    masks, n_keep, min_active)
        te_s,    drops['test_external'] = _filter_min_active(te_s,    masks, n_keep, min_active)
        if verbose:
            print(f"min_active={min_active}: dropped recordings with < {min_active} "
                  f"active channels -> {drops}")

    if verbose:
        print(f"FINAL model input width n_keep = {n_keep}")
        print(f"Train per-channel missing rate: mean {train_rate.mean():.3f} "
              f"min {train_rate.min():.3f} max {train_rate.max():.3f}")
        for name, s in [('train', train_s), ('val', val_s),
                        ('test_internal', ti_s), ('test_external', te_s)]:
            r = np.mean([masks[fp].mean() for _, fp, _ in s]) if s else float('nan')
            print(f"  per-file missing ratio [{name:13s}] mean={r:.3f}  n={len(s)}")

    # Data-scaling / composition control: applied AFTER keep_idx (geometry union
    # of the FULL training set) and the always-missing drop, so every run keeps
    # the identical 252-channel input and the identical val/test sets; only the
    # set of training patients changes.
    n_train_full = len(train_s)
    if use_sim:
        # clinical part: subsampled as usual ('sim+both' -> both cohorts), then
        # repeated real_repeat x for balanced sampling; 'sim' -> no clinical data
        real_part = [] if train_sources == 'sim' else subsample_train(
            train_s, train_frac=train_frac, train_sources='both',
            subsample_seed=subsample_seed, verbose=verbose)
        if min_active > 0:
            sim_train, n_drop_sim = _filter_min_active(sim_train, masks, n_keep, min_active)
            if verbose:
                print(f"min_active={min_active}: dropped {n_drop_sim} simulated recordings")
        train_s = sim_train + real_part * max(1, int(real_repeat))
        if verbose:
            print(f"[sim] training set: {len(sim_train)} simulated + {len(real_part)} clinical "
                  f"x{max(1, int(real_repeat))} = {len(train_s)} entries/epoch (sources={train_sources})")
    else:
        train_s = subsample_train(train_s, train_frac=train_frac,
                                  train_sources=train_sources,
                                  subsample_seed=subsample_seed, verbose=verbose)
    n_train_pat = len({smp[0].split('__', 2)[1] for smp in train_s})
    if verbose and (train_frac < 1.0 or train_sources != 'both'):
        print(f"[subsample] training set: {len(train_s)}/{n_train_full} recordings, "
              f"{n_train_pat} patients (x aug_times={aug_times} -> "
              f"{len(train_s) * aug_times} samples/epoch)")

    common = dict(masks=masks, keep_idx=keep_idx, crop_len=crop_len,
                  preprocess=preprocess, seed=seed, pp_norm=pp_norm,
                  prep_cache=prep_cache, norm_mode=norm_mode)
    train_common = dict(common, seed=(train_rng_seed if train_rng_seed is not None
                                       else seed))
    train_ds = ECGiDataset(train_s, mode='train', aug_times=aug_times,
                           **train_common)
    val_ds   = ECGiDataset(val_s,   mode='val',   **common)
    ti_ds    = ECGiDataset(ti_s,    mode='test',  **common)
    te_ds    = ECGiDataset(te_s,    mode='test',  **common)

    info = dict(keep_idx=keep_idx, n_keep=n_keep, train_rate=train_rate,
                cohortB_assign=cohortB_assign, n_train=len(train_s), n_val=len(val_s),
                n_train_full=n_train_full, n_train_patients=n_train_pat,
                train_frac=train_frac, train_sources=train_sources,
                subsample_seed=subsample_seed, real_repeat=real_repeat,
                val_source=val_source, n_sim_train=(len(sim_train) if use_sim else 0),
                n_test_internal=len(ti_s), n_test_external=len(te_s),
                n_holdout=len(buckets['holdout']),
                n_holdout_patients=len({r[0].split('__', 2)[1] for r in buckets['holdout']}))
    return train_ds, val_ds, ti_ds, te_ds, info
