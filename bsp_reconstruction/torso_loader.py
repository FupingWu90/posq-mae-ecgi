"""
Load physical 3D electrode coordinates from .torso XML files.

The .torso file (CardioInsight XML) contains:
  - Vertices: list of (X, Y, Z) 3D coordinates — one per electrode
  - ChannelIndices: list of channel indices (0-based) mapping vertices to
                    the 256-channel ECG array

Usage:
  coords = load_torso_coords(torso_path)
  # coords: (256, 3) float32, NaN for channels with no physical position

The returned array can be passed to the model as per-sample physical
position embeddings for the channel dimension.
"""

import numpy as np
try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree


def load_torso_coords(torso_path):
    """
    Parse a .torso XML file and return a (256, 3) float32 array of (X, Y, Z)
    coordinates. Channels not present in the torso file are filled with NaN.

    Parameters
    ----------
    torso_path : str

    Returns
    -------
    coords : np.ndarray, shape (256, 3), dtype float32
    """
    tree = etree.parse(torso_path)
    root = tree.getroot()

    tags = [str(child.tag).lower() for child in root]

    # Find ChannelIndices element
    ch_idx_pos = next(i for i, t in enumerate(tags) if 'channelindices' in t)
    ch_indices_el = root[ch_idx_pos]
    channel_indices = [int(el.text) for el in ch_indices_el]

    # Find Vertices element
    vert_idx_pos = next(i for i, t in enumerate(tags) if 'vertices' in t and 'hole' not in t)
    vertices_el = root[vert_idx_pos]

    n_verts = len(vertices_el)
    assert n_verts == len(channel_indices), (
        f"Vertex count ({n_verts}) != ChannelIndices count ({len(channel_indices)})"
    )

    coords = np.full((256, 3), np.nan, dtype=np.float32)
    for i, vert in enumerate(vertices_el):
        ch = channel_indices[i]
        if 0 <= ch < 256:
            coords[ch, 0] = float(vert[0].text)  # X
            coords[ch, 1] = float(vert[1].text)  # Y
            coords[ch, 2] = float(vert[2].text)  # Z

    return coords


def normalize_coords(coords, pp_norm='centroid'):
    """
    Normalize electrode coordinates.

    pp_norm='minmax':
        Normalize each axis to [-1, 1] using the bounding box of valid electrodes.
        Simple but origin is anatomically arbitrary and scale varies with body size.

    pp_norm='centroid':
        Subtract the centroid of all valid electrode positions (approximates the
        heart center, since electrodes surround it) then divide by mean
        electrode-to-centroid distance. Result: heart-relative unit-distance coords.
        More physically meaningful for ECGi — two electrodes at the same anatomical
        position on patients of different sizes get similar embeddings.

    coords  : (256, 3) float32, NaN for channels with no registered position
    Returns : (256, 3) float32, NaN channels set to 0
    """
    out = np.zeros_like(coords)
    valid = ~np.isnan(coords[:, 0])
    if valid.sum() == 0:
        return out
    valid_coords = coords[valid]

    if pp_norm == 'centroid':
        centroid = valid_coords.mean(axis=0)
        centered = valid_coords - centroid
        mean_dist = np.sqrt((centered ** 2).sum(axis=1)).mean()
        scale = mean_dist if mean_dist > 1e-8 else 1.0
        out[valid] = (centered / scale).astype(np.float32)
    else:  # 'minmax'
        cmin = valid_coords.min(axis=0)
        cmax = valid_coords.max(axis=0)
        crange = np.where(cmax - cmin > 1e-8, cmax - cmin, 1.0)
        out[valid] = (2.0 * (valid_coords - cmin) / crange - 1.0).astype(np.float32)

    return out


def find_torso_file(patient_dir):
    """
    Find the .torso file in a patient directory (non-recursive — torso files
    sit directly in the patient root, not in subdirectories).

    Returns path string or None if not found.
    """
    import glob
    import os
    matches = glob.glob(os.path.join(patient_dir, '*.torso'))
    return matches[0] if matches else None
