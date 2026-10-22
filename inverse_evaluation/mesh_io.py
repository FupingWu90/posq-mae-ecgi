"""
Mesh I/O for CardioInsight `.heart` and `.torso` XML files.

Both file types share the namespace `http://www.cardioinsight.com/Entities`.
Each contains:
  Vertices       — list of <Vector3D><X/><Y/><Z/></Vector3D>
  Faces          — list of <Face><index1/><index2/><index3/>...</Face>
  HoleVertices   — vertices around mesh holes (e.g. valve openings); IGNORED
  HoleFaces      — faces around mesh holes; IGNORED

Geometry units appear to be centimetres (heart bbox ~10 cm extent).

The `.torso` file additionally contains a `ChannelIndices` block mapping each
torso vertex to a 0-based index in the 256-channel ECG array. Channels not
listed in `ChannelIndices` have no electrode position in this recording.
"""

import os
import glob
import xml.etree.ElementTree as etree
import numpy as np

NS = '{http://www.cardioinsight.com/Entities}'


def load_surface(path):
    """
    Read a CardioInsight `.heart` or `.torso` XML file.

    Returns
    -------
    V : (n_vertices, 3) float32 — vertex coordinates
    F : (n_faces, 3) int32      — triangle vertex indices
    channel_indices : (n_vertices,) int32 or None
        Only populated for `.torso` files. None for `.heart`.
    """
    root = etree.parse(path).getroot()
    verts_el = root.find(NS + 'Vertices')
    faces_el = root.find(NS + 'Faces')
    if verts_el is None or faces_el is None:
        raise ValueError(f'{path}: missing Vertices/Faces')

    n_v = len(verts_el)
    V = np.empty((n_v, 3), dtype=np.float32)
    for i, child in enumerate(verts_el):
        V[i, 0] = float(child[0].text)
        V[i, 1] = float(child[1].text)
        V[i, 2] = float(child[2].text)

    F = np.empty((len(faces_el), 3), dtype=np.int32)
    for i, child in enumerate(faces_el):
        F[i, 0] = int(child[0].text)
        F[i, 1] = int(child[1].text)
        F[i, 2] = int(child[2].text)

    ch_el = root.find(NS + 'ChannelIndices')
    if ch_el is not None and len(ch_el) > 0:
        channel_indices = np.array([int(c.text) for c in ch_el], dtype=np.int32)
        if len(channel_indices) != n_v:
            channel_indices = None
    else:
        channel_indices = None

    return V, F, channel_indices


def vertex_normals(V, F):
    """
    Outward unit normals at each vertex.

    Per-face normals via (v1-v0) × (v2-v0); per-vertex normal averaged across
    incident faces (area-weighted naturally because the cross product magnitude
    equals 2 × triangle area). Final normal is unit-normalised.

    The CardioInsight convention is consistent CCW orientation viewed from
    outside the surface, so the cross-product convention used here gives
    outward normals. We do **not** verify orientation against the centroid —
    a globally flipped mesh would invert all normals and the Neumann residual
    would simply absorb the sign during Tikhonov solve. If sign matters for a
    downstream interpretation, post-check `(N · (V - V.mean())) > 0`.

    Parameters
    ----------
    V : (n_v, 3) float
    F : (n_f, 3) int

    Returns
    -------
    N : (n_v, 3) float32, unit-norm rows
    """
    n_v = V.shape[0]
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # (n_f, 3); magnitude = 2 * area
    N = np.zeros((n_v, 3), dtype=np.float64)
    np.add.at(N, F[:, 0], fn)
    np.add.at(N, F[:, 1], fn)
    np.add.at(N, F[:, 2], fn)
    norm = np.linalg.norm(N, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (N / norm).astype(np.float32)


def find_patient_files(patient_dir):
    """
    Locate the `.heart`, `.torso`, `.landmark`, `.valve` files in a patient root
    folder. Returns a dict with keys 'heart', 'torso', 'landmark', 'valve'.
    Missing keys map to None.
    """
    out = {'heart': None, 'torso': None, 'landmark': None, 'valve': None}
    for ext in out:
        matches = glob.glob(os.path.join(patient_dir, f'*.{ext}'))
        if matches:
            out[ext] = matches[0]
    return out


def import_pot(pot_f):
    """Read a `.potential` file. Returns (pot, len, electrodes); pot shape: (electrodes, len)."""
    with open(pot_f, 'rb') as f:
        L = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        E = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        d = np.fromfile(f, dtype=np.float32, count=-1)
    return d.reshape(L, E).T, L, E


def import_ecgdata(ecg_f):
    """Read an `.ecgdata` file. Returns (ecg, len, electrodes); shape: (electrodes, len)."""
    with open(ecg_f, 'rb') as f:
        L = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        E = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        d = np.fromfile(f, dtype=np.float32, count=-1)
    return d.reshape(L, E).T, L, E
