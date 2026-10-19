"""Patient identity of a test recording (sid = 'cohort__folder__file'): internal test = the patient folder (11 test
patients, one folder each); external = md5 of the torso geometry file (208 folders -> 202 unique patients: six
patients were exported twice under a second folder name with byte-identical geometry).  Returns {sid: key};
keys are opaque and must never be printed."""
import os, sys, hashlib
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V2, 'bsp_reconstruction'))
from dataset import collect_split
_CACHE = {}


def patient_map(split):
    if split not in _CACHE:
        b, _ = collect_split(seed=42)
        out, tmd5 = {}, {}
        for sid, _ecg, torso in b[split]:
            if not torso:
                continue
            folder = tuple(sid.split('__', 2)[:2])
            if split == 'test_external':
                if folder not in tmd5:
                    tmd5[folder] = hashlib.md5(open(torso, 'rb').read()).hexdigest()
                out[sid] = tmd5[folder]
            else:
                out[sid] = folder[1]
        _CACHE[split] = out
    return _CACHE[split]
