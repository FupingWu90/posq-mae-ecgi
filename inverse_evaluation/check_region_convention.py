"""
Assertion-style check of the torso 'front'/'back' electrode-region convention
used by bsp_reconstruction/channel_select.py (region_mask / select_visible_channels(region=..)).

For a few real patients it verifies that
  1. channel_select.{AP_AXIS, ANT_SIGN} == heart_coords.{AP_AXIS, ANT_SIGN}
     (bsp_reconstruction/ re-declares the cohort convention because it must not import
     inverse_evaluation/; the two copies must never drift apart);
  2. the heart-mesh centroid lies on the 'front' side of the torso-electrode
     centroid, i.e. ANT_SIGN * (heart_centroid - torso_centroid)[AP_AXIS] > 0
     (.heart and .torso share one coordinate frame);
  3. region_mask on the NORMALISED coordinates (what the selector receives)
     agrees with the raw-coordinate sign test, front/back partition the
     registered electrodes, and both halves are non-empty;
and prints the number of front / back electrodes per patient.

  python check_region_convention.py [--n 3] [--split test_internal]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bsp_reconstruction')
sys.path.insert(0, _CODE)

from mesh_io import find_patient_files, load_surface          # noqa: E402
from torso_loader import load_torso_coords, normalize_coords  # noqa: E402
from dataset import collect_split                             # noqa: E402
import heart_coords                                           # noqa: E402
import channel_select                                         # noqa: E402
from channel_select import region_mask, AP_AXIS, ANT_SIGN     # noqa: E402


def check_patient(torso_path, verbose=True):
    files = find_patient_files(os.path.dirname(torso_path))
    assert files['heart'] is not None, f"no .heart next to {torso_path}"
    heart_V, _, _ = load_surface(files['heart'])
    raw = load_torso_coords(torso_path).astype(np.float64)      # (256,3), NaN = unregistered
    reg = ~np.isnan(raw[:, 0])
    torso_c = raw[reg].mean(0)
    heart_c = heart_V.astype(np.float64).mean(0)
    d_ap = ANT_SIGN * (heart_c - torso_c)[AP_AXIS]
    assert d_ap > 0, (f"heart centroid is NOT on the 'front' side: "
                      f"ANT_SIGN*(heart-torso)[AP_AXIS] = {d_ap:.2f}")

    # sign test on the raw frame vs region_mask on the normalised coords
    norm = np.nan_to_num(normalize_coords(raw.astype(np.float32), pp_norm='centroid'), nan=0.0)
    front, back = region_mask(norm, 'front'), region_mask(norm, 'back')
    raw_front = reg & (ANT_SIGN * (raw[:, AP_AXIS] - torso_c[AP_AXIS]) > 0)
    assert np.array_equal(front, raw_front), "normalised-frame front != raw-frame front"
    assert not np.any(front & back), "front and back overlap"
    assert np.array_equal(front | back, reg), "front|back != registered electrodes"
    assert region_mask(norm, 'all').all()
    assert front.sum() > 0 and back.sum() > 0, "empty half"
    # informational: fraction of heart vertices anterior of the electrode centroid
    h_front = float((ANT_SIGN * (heart_V[:, AP_AXIS] - torso_c[AP_AXIS]) > 0).mean())
    if verbose:
        pdir = os.path.dirname(torso_path)
        label = os.path.basename(pdir)
        if label in ('CI', 'ci'):                       # torso sits in <patient>/CI/
            label = os.path.basename(os.path.dirname(pdir)) + '/' + label
        print(f"  {label:20s} registered={reg.sum():3d}  front={front.sum():3d}  "
              f"back={back.sum():3d}  heart-torso AP offset (anterior>0) = {d_ap:6.1f}  "
              f"heart verts anterior = {100*h_front:5.1f}%  [{heart_V.shape[0]} verts]")
    return int(front.sum()), int(back.sum()), float(d_ap)


def main(n=3, split='test_internal'):
    assert (channel_select.AP_AXIS, channel_select.ANT_SIGN) == \
        (heart_coords.AP_AXIS, heart_coords.ANT_SIGN), \
        "channel_select.{AP_AXIS,ANT_SIGN} drifted from heart_coords"
    print(f"convention: AP_AXIS={AP_AXIS} ANT_SIGN={ANT_SIGN:+.0f} "
          f"(channel_select == heart_coords: OK)")
    buckets, _ = collect_split(seed=42)
    seen, done = set(), 0
    for sid, ecg, torso in buckets[split]:
        if torso is None or torso in seen:
            continue
        seen.add(torso)
        check_patient(torso)
        done += 1
        if done >= n:
            break
    assert done == n, f"only {done} patients checked"
    print(f"all {done} patients pass (heart centroid on the 'front' side)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--split', default='test_internal',
                   choices=['train', 'val', 'test_internal', 'test_external'])
    a = p.parse_args()
    main(a.n, a.split)
