import numpy as np
import scipy as sp


def baseline_mean_std(block_labels, good_channels, baselines):
    blocks = sorted(set(block_labels))
    bl_mean = np.zeros((len(blocks), 40, len(good_channels)))
    bl_std = np.zeros((len(blocks), 40, len(good_channels)))
    for ii, block in enumerate(blocks):
        for band in range(40):
            data = baselines[(block, band)][good_channels]
            bl_mean[ii, band] = data.mean(axis=-1)
            bl_std[ii, band] = data.std(axis=-1)
    return bl_mean, bl_std

def new_ch_idx(old_idx, good_channels):
    return (np.array(good_channels) == old_idx).argmax()


def forward_bl(X, bl_type, bl_mean, bl_std, block_labels):
    blocks = set(block_labels)
    means = np.full((X.shape[0], len(blocks), X.shape[2], 1), np.nan)
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] /= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore':
            X[:, idxs] -= bl_mean[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] /= bl_std[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'data_mean':
            means[:, [ii]] = X[:, idxs].mean(axis=(1, 3), keepdims=True)
            X[:, idxs] /= means[:, [ii]]
        else:
            raise ValueError
    return X, means


def invert_bl(X, bl_type, means, bl_mean, bl_std, block_labels):
    blocks = set(block_labels)
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] *= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore':
            X[:, idxs] *= bl_std[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] += bl_mean[ii, :, np.newaxis, :, np.newaxis]
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

def remove_pc1(X, pcs, baselines):
    pass
