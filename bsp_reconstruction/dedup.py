"""Content de-duplication rule of the internal test set (2026-10-21).  The paper counts unique recordings: the raw internal
test list of dataset.collect_split holds 276 files, one of which is a byte-identical copy of another recording of the same
patient (identity map of the frozen v1 baselines_analyses/common._identity: md5 per file; validation and external test sets
hold no duplicates, the training list is left exactly as trained).  Every evaluator and loader keeps one recording per
content key (the first in sorted sample-id order, as baselines_analyses/holdout_summary.dedup) by calling ok(sid) and skipping the
rest.  The dropped sample ids are stored only as md5 digests of the id string (dedup_exclude.json next to this file,
written by baselines_analyses/dedup_rule.py --build), so nothing identifying is written or printed."""
import os, json, hashlib
_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dedup_exclude.json')
_KEYS = None


def excluded():
    """set of md5 digests of the excluded sample ids (both the raw id and its '/'-safe form); empty if no file."""
    global _KEYS
    if _KEYS is None:
        _KEYS = set()
        if os.path.exists(_FILE):
            for v in json.load(open(_FILE)).get('exclude', {}).values():
                _KEYS.update(v)
    return _KEYS


def key(sid):
    return hashlib.md5(str(sid).encode()).hexdigest()


def ok(sid):
    """True unless `sid` (a sample id, its '/'-safe form, or the basename of a per-recording cache file) is excluded."""
    k = excluded()
    if not k:
        return True
    s = str(sid)
    if s.endswith('.npz') or s.endswith('.csv') or s.endswith('.json'):
        s = s.rsplit('.', 1)[0]
    return key(s) not in k and key(s.replace('/', '_')) not in k


def filter_recs(recs):
    """(sid, ecg, torso) tuples minus the excluded ones."""
    return [r for r in recs if ok(r[0])]
