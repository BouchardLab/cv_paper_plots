import h5py
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import cm
from sklearn.linear_model import (RANSACRegressor, TheilSenRegressor,
                                  HuberRegressor)

from ecog.utils import bands
from .xfreq_analysis import good_examples_and_channels, get_vsmc_electrodes



def load_data(fname):
    baselines = dict()
    with h5py.File(fname) as f:
        block_labels = f['block'].value
        for key, value in f.items():
            if ('block' in key) and ('band' in key):
                items = key.split('_')
                block, band = int(items[2]), int(items[4])
                baselines[(block, band)] = np.squeeze(value.value).astype('float32')
        good_examples, good_channels = good_examples_and_channels(f['X0'].value)
        vsmc_electrodes = get_vsmc_electrodes(f)
        tokens = f['tokens'].value.astype('str').tolist()

        good_examples = sorted(np.nonzero(good_examples)[0].tolist())

        good_channels = sorted(np.nonzero(vsmc_electrodes * good_channels)[0].tolist())
        n_trials, n_channels, n_time = f['X0'].shape
        print(len(good_examples), len(good_channels))
        X = np.zeros((40, len(good_examples), len(good_channels), n_time), dtype='float32')
        block_labels = block_labels[good_examples]
        for ii in range(40):
            X[ii] = f['X{}'.format(ii)][good_examples][:, good_channels]
        labels = f['y'][good_examples]
    return X, baselines, good_examples, good_channels, tokens, block_labels, labels


def baseline_mean_std(block_labels, good_channels, baselines, mb=None, cfs=None):
    blocks = sorted(set(block_labels))
    bl_mean = np.zeros((len(blocks), 40, len(good_channels)))
    bl_std = np.zeros((len(blocks), 40, len(good_channels)))
    for ii, block in enumerate(blocks):
        if mb is not None:
            try:
                bb_mb = mb[block][0]
            except KeyError:
                bb_mb = mb[str(block)][0]
            bb = calc_bb_estimate(cfs, bb_mb[...,0], bb_mb[...,1])
        for band in range(40):
            data = baselines[(block, band)][good_channels]
            if (mb is not None) and (band < 29):
                bb_i = bb[band]
                data = data - bb_i
            bl_mean[ii, band] = data.mean(axis=-1)
            bl_std[ii, band] = data.std(axis=-1)
    return bl_mean, bl_std

def new_ch_idx(old_idx, good_channels):
    return (np.array(good_channels) == old_idx).argmax()


def forward_bl(X, bl_type, bl_mean, bl_std, block_labels, mb=None, cfs=None):
    blocks = sorted(set(block_labels))
    means = np.full((X.shape[0], len(blocks), X.shape[2], 1), np.nan)
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] /= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore':
            X[:, idxs] -= bl_mean[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] /= bl_std[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore_bb':
            bb = calc_bb_estimate(cfs, mb[idxs][..., 0], mb[idxs][..., 1])
            bb[bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1]] = 0.
            X[:, idxs] -= bb + bl_mean[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] /= bl_std[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'data_mean':
            means[:, [ii]] = X[:, idxs].mean(axis=(1, 3), keepdims=True)
            X[:, idxs] /= means[:, [ii]]
        else:
            raise ValueError
    return X, means


def invert_bl(X, bl_type, means, bl_mean, bl_std, block_labels):
    blocks = sorted(set(block_labels))
    for ii, block in enumerate(blocks):
        idxs = block_labels == block
        if bl_type == 'bl_mean':
            X[:, idxs] *= bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore':
            X[:, idxs] *= bl_std[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] += bl_mean[ii, :, np.newaxis, :, np.newaxis]
        elif bl_type == 'bl_zscore_bb':
            bb = calc_bb_estimate(cfs, mb[idxs][..., 0], mb[idxs][..., 1])
            bb[bands.chang_lab['cfs'] >= bands.neuro.min_freqs[-1]] = 0.
            X[:, idxs] *= bl_std[ii, :, np.newaxis, :, np.newaxis]
            X[:, idxs] += bb + bl_mean[ii, :, np.newaxis, :, np.newaxis]
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


def log_log_robust_regression(cfs, y, kind=0):
    assert y.shape[0] == 40
    y = y.reshape(40, -1)
    x = np.tile(cfs[:, np.newaxis], (1, y.shape[1]))
    y = np.log(y).ravel()
    x = np.log(x).ravel()[:, np.newaxis]
    if kind == 0:
        model = RANSACRegressor()
    elif kind == 1:
        model = TheilSenRegressor(n_jobs=-1)
    elif kind == 2:
        model = HuberRegressor()
    else:
        raise ValueError
    model.fit(x, y)
    yp = model.predict(x)
    u = np.square(y - yp)
    v = np.square(y - y.mean())
    R2 = 1. - u/v
    if kind == 0:
        return model.estimator_.coef_, model.estimator_.intercept_, np.median(R2)
    elif kind in [1, 2]:
        return model.coef_, model.intercept_, np.median(R2)
    else:
        raise ValueError


def calc_bb_fit(cfs, X, labels=None, kind=2, comm=None):
    if labels is not None:
        cvs = sorted(set(labels))
        bls = np.full((len(cvs), X.shape[2], X.shape[3], 2), np.nan)
        medR2 = np.full((len(cvs), X.shape[2], X.shape[3]), np.nan)
        for ii, cv in enumerate(cvs):
            print(ii, cv)
            idxs = np.where(labels == cv)[0]
            Xp = X[:, idxs].mean(axis=1)
            for ch in range(X.shape[2]):
                for tt in range(X.shape[3]):
                    rval = log_log_robust_regression(cfs, Xp[:, ch, tt], kind)
                    bls[ii, ch, tt] = rval[:2]
                    medR2[ii, ch, tt] = rval[2]
        return bls, medR2
    else:
        shape = X.shape[1:]
        bls = np.full((np.prod(shape), 2,), np.nan)
        medR2 = np.full(np.prod(shape), np.nan)
        for ii, Xp in enumerate(X.reshape(X.shape[0], -1).T):
            rval = log_log_robust_regression(cfs, Xp, kind)
            bls[ii] = rval[:2]
            medR2[ii] = rval[2]
    return bls.reshape(shape + (2,)), medR2.reshape(shape)


def calc_bb_estimate(cfs, m, b):
    for ii in range(m.ndim):
        cfs = cfs[..., np.newaxis]
    cfs = np.log(cfs)
    m = m[np.newaxis]
    b = b[np.newaxis]
    return np.exp(m * cfs + b)

def correlate_ts(x, y):
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    num = (xm * ym).mean(axis=-1)
    den = xm.std(axis=-1) * ym.std(axis=-1)
    return (num / den)


def auto_corr(X, lags=51):
    df = pd.DataFrame(X.reshape(-1, 258).T)
    n = df.shape[1]
    ac = np.zeros(lags)
    for ii in range(n):
        for jj in range(lags):
            ac[jj] += df[ii].autocorr(jj - lags //2) / n
    return ac


def plot_PC1s(pcs, evs, mean, faxes=None, title=False, ylabel=None):
    if pcs.ndim > 3:
        pcs = pcs.reshape(-1, 3, 40)
        mean = mean.reshape(-1, 40)
        evs = evs.reshape(-1, 3)
    freqs = bands.chang_lab['cfs']
    if faxes is None:
        faxes = plt.subplots(1, 2, figsize=(2.5, 12))
    f, (ax0, ax1) = faxes
    ratios = evs[:, 1:]/evs[:, [0]]
    beta_weights = np.zeros(pcs.shape[0])
    beta_weights = (pcs[:, 0, 14:20]).sum(axis=-1)
    beta_weights -= beta_weights.min()
    beta_weights /= beta_weights.max()
    for ii, pc in enumerate(pcs):
        ax0.plot(freqs, pcs[ii, 0],c=cm.Greys(beta_weights[ii]), alpha=1.)
    ax0.plot(freqs, np.median(pcs[:, 0], axis=0), c='r')
    ax0.axhline(0, 0, 1, c='blue', ls='--')
    ax1.errorbar([0, 1], ratios.mean(axis=0), yerr=ratios.std(axis=0),
                 c='k', ls='none', marker='.')
    ax1.axhline(1, 0, 1, linestyle='--', c='gray')
    pc0s = pcs[:, 0] / np.linalg.norm(pcs[:, 0], axis=1, keepdims=True)
    if mean is not None:
        mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)
        ips = abs(np.sum(pc0s * mean, axis=1))
        ax1.errorbar(2, ips.mean(), yerr=ips.std(), c='k', marker='.')
    #ax0.set_xscale('log')
    if mean is None:
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels([r'$\frac{e_2}{e_1}$', r'$\frac{e_3}{e_1}$'])
        ax1.set_xlim(-1, 2)
    else:
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels([r'$\frac{e_2}{e_1}$', r'$\frac{e_3}{e_1}$', r'$|\mu\cdot PC1|$'])
        ax1.set_xlim(-1, 3)
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0, 1])
    if ylabel is not None:
        ax0.set_ylabel(ylabel)
    return
