import numpy as np
import scipy as sp

def new_ch_idx(old_idx, good_channels):
    return (np.array(good_channels) == old_idx).argmax()


def forward_bl(X, bl_type, block_labels):
    blocks = set(block_labels)
    means = np.full((X.shape[0], len(blocks), X.shape[2], 1), np.nan)
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] /= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'data_mean':
            means[:, [ii]] = X[:, idxs].mean(axis=(1, 3), keepdims=True)
            X[:, idxs] /= means[:, [ii]]
        else:
            raise ValueError
    return X, means


def invert_bl(X, bl_type, means, block_labels):
    blocks = set(block_labels)
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] *= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'data_mean':
            X[:, idxs] *= means[:, [ii]]
        else:
            raise ValueError
    return X


def get_pcs(d, center_pca):
    mean = d.mean(axis=0)
    if center_pca:
        d = (d - mean) / np.sqrt(d.shape[0])
    return sp.linalg.eigh(d.T.dot(d), eigvals=(39-2, 39)), mean


def flip(pcs):
    fl = 2 * (abs(pcs.max(axis=-1, keepdims=True)) >= abs(pcs.min(axis=-1, keepdims=True))) -1
    return fl * pcs
