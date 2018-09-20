import numpy as np

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
