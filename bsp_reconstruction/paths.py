"""Central path definitions.

V2_ROOT is this repository.  V1_ROOT points to a checkout of the previous release (the frozen v1 code base whose
trained runs and baseline artefacts the analyses read: the v1 PPE-MAE model, the ST-MEM-style index-embedding model,
MCMA and the classical completions under the per-recording k-means layout).  Everything this code base writes stays
under V2_ROOT; `assert_v2` guards the writers.
"""
import os

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # this repository
V1_ROOT = os.environ.get('ECGI_V1_ROOT', os.path.join(V2_ROOT, 'v1'))         # previous release (read-only)

V1_CACHE_DIR = os.path.join(V1_ROOT, 'cache')            # read-only: prep/ + missing-mask pickle
V2_CACHE_DIR = os.environ.get('ECGI_CACHE_DIR', os.path.join(V2_ROOT, 'cache'))   # new cache entries go here
V1_HSP_CACHE = os.path.join(V1_ROOT, 'inverse_evaluation', 'cache_hsp')   # read-only reference / VC-only caches
V2_HSP_CACHE = os.path.join(V2_ROOT, 'inverse_evaluation', 'cache_hsp')
V1_RUNS = os.path.join(V1_ROOT, 'runs', 'data-cohortA+cohortB_vs_external')
V2_RUNS = os.path.join(V2_ROOT, 'runs')
V1_GENERATED = os.path.join(V1_ROOT, 'baselines_analyses', 'generated')
V2_GENERATED = os.path.join(V2_ROOT, 'baselines_analyses', 'generated')
FIG_DIR_V2 = os.environ.get('ECGI_FIG_DIR', os.path.join(V2_ROOT, 'figures'))
TEMPLATE_DIR = os.path.join(V2_ROOT, 'bsp_reconstruction', 'templates')


def assert_v2(path):
    """Raise if `path` lies inside the (read-only) previous-release tree; returns the path otherwise."""
    rp = os.path.realpath(path)
    if rp == V1_ROOT or rp.startswith(V1_ROOT + os.sep):
        raise PermissionError(f'this code base must not write into the previous-release tree: {rp}')
    return path


def first_existing(*paths):
    """First existing path among the arguments, else None."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None
