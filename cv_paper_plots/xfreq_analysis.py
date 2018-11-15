import h5py, os

import numpy as np
from scipy import stats
from sklearn.linear_model import (LinearRegression, HuberRegressor,
                                  RANSACRegressor, TheilSenRegressor)

from ecog.utils import bands
from .style import (subject_labels, subject_colors,
                    axes_label_fontstyle, ticklabel_fontstyle,
                    tickparams_fontstyle)


plot_time = np.array([0, .5+.6])
plot_idx = (plot_time * 200.).astype(int)
s = slice(*plot_idx.tolist())

hg_power_time = np.array([.5-.07, .5 + .14])
hg_power_idx = (hg_power_time * 200.).astype(int)
hg_power_s = slice(*plot_idx.tolist())


def get_vsmc_electrodes(f):
    vsmc = np.concatenate([f['anatomy']['preCG'].value, f['anatomy']['postCG'].value])
    vsmc_electrodes = np.zeros(256)
    # Electrodes at 1-indexed
    vsmc_electrodes[vsmc - 1] = 1
    return vsmc_electrodes


def good_examples_and_channels(data):
    """Find good examples and channels.

    First removes all examples and channels that are completely NaN.
    The removes remaining examples and channels that are partially NaN.

    Parameters
    ----------
    data : ndarray (examples, channels, time)
        Data array.
    Returns
    -------
    good_examples : list
        Binary mask of examples that are not NaN.
        Same length as first dimension of data.
    good_channels : list
        Binary mask of channels that are not NaN.
        Same length as second dimension of data.
    """
    data = data.sum(axis=2)
    nan_time = np.isnan(data)

    # First exlude examples and channels that are all NaN
    bad_examples = np.all(nan_time, axis=1)
    bad_channels = np.all(nan_time, axis=0)
    data_good = data.copy()
    data_good[bad_examples] = 0.
    data_good[:, bad_channels] = 0.

    # Then exclude examples and channels with any NaNs from remaining group
    # Often a channel is bad only for a subset of blocks, so mask them first
    partial_bad_channels = np.isnan(data_good.sum(axis=0))
    data_good[:, partial_bad_channels] = 0.
    partial_bad_examples = np.isnan(data_good.sum(axis=1))
    good_examples = np.logical_not(partial_bad_examples) * np.logical_not(bad_examples)
    good_channels = np.logical_not(partial_bad_channels) * np.logical_not(bad_channels)

    return good_examples, good_channels


def get_cv_idxs(y, good_examples):
    y_counts = np.zeros(57)
    cv_idxs = []
    for ii in range(57):
        y_counts[ii] = (y == ii).sum()
        if (y_counts[ii] >= 10):
            cv_idxs.append(np.nonzero(y == ii)[0].tolist())
        else:
            cv_idxs.append([])

    keep_cvs = y_counts >= 10
    n_cv = keep_cvs.sum()

    cv_idxs = [sorted(list(set(idxs).intersection(good_examples))) for idxs in cv_idxs]
    cv_idxs = [idxs for idxs in cv_idxs if len(idxs) > 0]
    return cv_idxs, n_cv


def save_power(f, channel, cv, subject, bb=False, bb2=False):
    """Save the power spectrum matrix.

    Parameters
    ----------
    f : h5py file handle
    channel : int
        ECoG array channel.
    cv : str
        CV to select.
    subject : str
        Subject name for file name.
    """
    if bb2:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        n_time = X.shape[-1]
        cv_idx = tokens.index(cv)
        cv_idx = tokens.index(cv)
        batch_idxs = np.where(np.equal(y, cv_idx))[0]
        power_data = X[:, batch_idxs][:, :, channel].mean(axis=1)
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power_bb2.npz'.format(subject, cv, channel)), **{'power_data': power_data})
    elif bb:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        pcs = f['pcs']
        pc0s = pcs[:, 0]
        n_time = X.shape[-1]
        cv_idx = tokens.index(cv)
        batch_idxs = np.where(np.equal(y, cv_idx))[0]
        power_data = X[:, batch_idxs][:, :, channel].mean(axis=1)
        print(X.shape, pc0s.shape, power_data.shape)
        power_proj = pc0s[channel].dot(power_data)
        print(power_proj.shape)
        power_data[:29] -= pc0s[channel][:29, np.newaxis] * power_proj[np.newaxis]
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power_bb.npz'.format(subject, cv, channel)), **{'power_data': power_data})
    else:
        y = f['y'].value
        good_examples, good_channels = good_examples_and_channels(f['X0'].value)
        good_channels = np.nonzero(good_channels)[0].tolist()
        assert channel in good_channels
        n_time = f['X0'].shape[-1]
        cv_idx = f['tokens'].value.astype('str').tolist().index(cv)
        batch_idxs = np.nonzero(np.equal(y, cv_idx) * good_examples)[0].tolist()
        power_data = np.zeros((40, 258))
        for ii in range(40):
            power_data[ii] = np.nanmean(f['X{}'.format(ii)][batch_idxs][:, channel], axis=0)
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power.npz'.format(subject, cv, channel)), **{'power_data': power_data})


def save_correlations(f, subject, channel=None, bb=False, bb2=False):
    if bb2:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        n_time = X.shape[-1]
        assert plot_idx[-1] <= n_time
        n_time = plot_idx[-1]

        n_ch = X.shape[2]
        n_ex = X.shape[1]

        cv_idxs, n_cv = get_cv_idxs(y, np.arange(n_ex))

        def normalize(a):
            a -= np.mean(a, axis=-1, keepdims=True)
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            return a

        xcorr_freq = np.zeros((40, n_cv, n_ch))
        hg_ts = np.zeros((n_cv, n_ch, n_time))
        b_ts, hg_ts = extract_b_hg(X, labels=y)
        hg_ts = hg_ts.mean(axis=0)[..., s]
        b_ts = b_ts.mean(axis=0)[..., s]
        hg_ts = normalize(hg_ts)
        b_ts = normalize(b_ts)

        for jj, idxs in enumerate(cv_idxs):
            other_ts = X[:, idxs].mean(axis=1)[..., s]
            other_ts = normalize(other_ts)
            xcorr_freq[:, jj] = np.sum(hg_ts[jj][np.newaxis] * other_ts, axis=-1)

        ones = np.ones_like(hg_ts[0, 0])
        n_overlap = np.correlate(ones, ones, mode='full')
        xcorr_time = np.zeros((n_overlap.size, n_cv, n_ch))
        acorr_time = np.zeros((2, n_overlap.size, n_cv, n_ch))
        for ii in range(n_cv):
            for jj in range(n_ch):
                hg = hg_ts[ii, jj]
                b = b_ts[ii, jj]
                xcorr_time[:, ii, jj] = np.correlate(hg, b, mode='full')
                #acorr_time[0, :, ii, jj] = np.correlate(hg, hg, mode='full')
                #acorr_time[1, :, ii, jj] = np.correlate(b, b, mode='full')

        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_correlations_bb2.npz'.format(subject)), **{'xcorr_freq': xcorr_freq,
                                                                         'xcorr_time': xcorr_time,
                                                                         'acorr_time': acorr_time})
    elif bb:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        pcs = f['pcs']
        pc0s = pcs[:, 0]
        n_time = X.shape[-1]
        assert plot_idx[-1] <= n_time
        n_time = plot_idx[-1]

        n_ch = X.shape[2]
        n_ex = X.shape[1]

        cv_idxs, n_cv = get_cv_idxs(y, np.arange(n_ex))

        def normalize(a):
            a -= np.mean(a, axis=-1, keepdims=True)
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            return a

        xcorr_freq = np.zeros((40, n_cv, n_ch))
        hg_ts = np.zeros((n_cv, n_ch, n_time))
        X_hg = extract_hg(X)
        for jj, idxs in enumerate(cv_idxs):
                hg_ts[jj] = X_hg[:, idxs].mean(axis=(0, 1))[..., s]
        hg_ts = normalize(hg_ts)

        b_ts = np.zeros((n_cv, n_ch, n_time))
        for jj, idxs in enumerate(cv_idxs):
            other_ts = X[:, idxs].mean(axis=1)[..., s]
            power_proj = (pc0s.T[..., np.newaxis] * other_ts).sum(axis=0)
            other_ts[:29] -= pc0s.T[..., np.newaxis][:29] * power_proj[np.newaxis]
            b_ts[jj] = extract_b[other_ts].mean(axis=0)
            other_ts = normalize(other_ts)
            xcorr_freq[:, jj] = np.sum(hg_ts[jj][np.newaxis] * other_ts, axis=-1)
        b_ts = normalize(b_ts)

        ones = np.ones_like(hg_ts[0, 0])
        n_overlap = np.correlate(ones, ones, mode='full')
        xcorr_time = np.zeros((n_overlap.size, n_cv, n_ch))
        acorr_time = np.zeros((2, n_overlap.size, n_cv, n_ch))
        for ii in range(n_cv):
            for jj in range(n_ch):
                hg = hg_ts[ii, jj]
                b = b_ts[ii, jj]
                xcorr_time[:, ii, jj] = np.correlate(hg, b, mode='full')
                #acorr_time[0, :, ii, jj] = np.correlate(hg, hg, mode='full')
                #acorr_time[1, :, ii, jj] = np.correlate(b, b, mode='full')

        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_correlations_bb.npz'.format(subject)), **{'xcorr_freq': xcorr_freq,
                                                                         'xcorr_time': xcorr_time,
                                                                         'acorr_time': acorr_time})
    else:
        good_examples, good_channels = good_examples_and_channels(f['X0'].value)
        n_time = f['X0'].shape[-1]
        assert plot_idx[-1] <= n_time
        n_time = plot_idx[-1]


        vsmc_electrodes = get_vsmc_electrodes(f)


        good_examples = np.nonzero(good_examples)[0].tolist()

        good_channels = np.nonzero(vsmc_electrodes * good_channels)[0].tolist()
        if channel is not None:
            assert channel in good_channels
            good_channels = [channel]

        cv_idxs, n_cv = get_cv_idxs(f['y'].value, good_examples)

        n_ch = len(good_channels)
        n_ex = len(good_examples)

        def normalize(a):
            a -= np.mean(a, axis=-1, keepdims=True)
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            return a

        hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                                  bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
        b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                                 bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])
        hb_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][3],
                                  bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][3])
        b_bands = np.logical_or(b_bands, hb_bands)
        b_bands = range(10, 21)

        xcorr_freq = np.zeros((40, n_cv, n_ch))
        hg_ts = np.zeros((hg_bands.sum(), n_cv, n_ch, n_time))
        for ii, c in enumerate(np.nonzero(hg_bands)[0]):
            for jj, idxs in enumerate(cv_idxs):
                hg_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
        hg_ts = np.mean(hg_ts, axis=0)
        hg_ts = normalize(hg_ts)

        for ii in range(40):
            for jj, idxs in enumerate(cv_idxs):
                other_ts = normalize(f['X{}'.format(ii)][idxs][:, good_channels].mean(axis=0)[..., s])
                xcorr_freq[ii, jj] = np.sum(hg_ts[jj] * other_ts, axis=-1)

        b_ts = np.zeros((len(b_bands), n_cv, n_ch, n_time))
        for ii, c in enumerate(b_bands):
            for jj, idxs in enumerate(cv_idxs):
                b_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
        b_ts = np.mean(b_ts, axis=0)
        b_ts = normalize(b_ts)
        ones = np.ones_like(hg_ts[0, 0])
        n_overlap = np.correlate(ones, ones, mode='full')
        xcorr_time = np.zeros((n_overlap.size, n_cv, n_ch))
        acorr_time = np.zeros((2, n_overlap.size, n_cv, n_ch))
        for ii in range(n_cv):
            for jj in range(n_ch):
                hg = hg_ts[ii, jj]
                b = b_ts[ii, jj]
                xcorr_time[:, ii, jj] = np.correlate(hg, b, mode='full')
                #acorr_time[0, :, ii, jj] = np.correlate(hg, hg, mode='full')
                #acorr_time[1, :, ii, jj] = np.correlate(b, b, mode='full')

        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_correlations.npz'.format(subject)), **{'xcorr_freq': xcorr_freq,
                                                                               'xcorr_time': xcorr_time,
                                                                        'acorr_time': acorr_time})
def save_time_correlations(f, subject, channel=None):
    good_examples, good_channels = good_examples_and_channels(f['X0'].value)
    n_time = f['X0'].shape[-1]
    assert plot_idx[-1] <= n_time
    n_time = plot_idx[-1]

    vsmc_electrodes = get_vsmc_electrodes(f)

    good_examples = np.nonzero(good_examples)[0].tolist()

    good_channels = np.nonzero(vsmc_electrodes * good_channels)[0].tolist()
    if channel is not None:
        assert channel in good_channels
        good_channels = [channel]

    cv_idxs, n_cv = get_cv_idxs(f['y'].value, good_examples)

    n_ch = len(good_channels)
    n_ex = len(good_examples)

    def normalize(a, axis=-1):
        a -= np.mean(a, axis=-1, keepdims=True)
        a /= np.linalg.norm(a, axis=-1, keepdims=True)
        return a

    hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
    b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                             bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])
    hb_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][3],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][3])
    b_bands = np.logical_or(b_bands, hb_bands)
    b_bands = range(10, 21)

    hg_ts = np.zeros((hg_bands.sum(), n_cv, n_ch, n_time))
    for ii, c in enumerate(np.nonzero(hg_bands)[0]):
        for jj, idxs in enumerate(cv_idxs):
            hg_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
    hg_ts = np.mean(hg_ts, axis=0)
    hg_ts = normalize(hg_ts, axis=0)

    b_ts = np.zeros((len(b_bands), n_cv, n_ch, n_time))
    for ii, c in enumerate(b_bands):
        for jj, idxs in enumerate(cv_idxs):
            b_ts[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)[..., s]
    b_ts = np.mean(b_ts, axis=0)
    b_ts = normalize(b_ts, axis=0)

    xcorr_time = np.sum(hg_ts * b_ts, axis=0)


    np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_time_correlations.npz'.format(subject)), **{'xcorr_time': xcorr_time})


def save_hg_power(f, subject, bb=False, bb2=False):
    if bb2:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        n_time = X.shape[-1]
        assert plot_idx[-1] <= n_time
        n_time = plot_idx[-1]

        n_ch = X.shape[2]
        n_ex = X.shape[1]

        cv_idxs, n_cv = get_cv_idxs(y, np.arange(n_ex))

        power_data = extract_hg(X, labels=y).mean(axis=0)[..., s]
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_bb2.npz'.format(subject)), **{'power_data': power_data})
    elif bb:
        y = f['y']
        X = f['X']
        tokens = f['tokens']
        pcs = f['pcs']
        pc0s = pcs[:, 0]
        n_time = X.shape[-1]
        assert plot_idx[-1] <= n_time
        n_time = plot_idx[-1]

        n_ch = X.shape[2]
        n_ex = X.shape[1]

        cv_idxs, n_cv = get_cv_idxs(y, np.arange(n_ex))

        power_data = np.zeros((n_cv, n_ch, n_time))
        for jj, idxs in enumerate(cv_idxs):
            power_data[jj] = extract_hg(X)[:, idxs].mean(axis=(0, 1))[..., s]
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_bb.npz'.format(subject)), **{'power_data': power_data})
    else:
        vsmc_electrodes = get_vsmc_electrodes(f)

        good_examples, good_channels = good_examples_and_channels(f['X0'].value)
        good_channels = np.nonzero(vsmc_electrodes * good_channels)[0].tolist()
        n_time = f['X0'].shape[-1]

        good_examples = np.nonzero(good_examples)[0].tolist()

        cv_idxs, n_cv = get_cv_idxs(f['y'].value, good_examples)

        hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                                  bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
        hg_bands = np.nonzero(hg_bands)[0].tolist()

        power_data = np.zeros((len(hg_bands), n_cv, len(good_channels), n_time))
        for ii, c in enumerate(hg_bands):
            for jj, idxs in enumerate(cv_idxs):
                power_data[ii, jj] = f['X{}'.format(c)][idxs][:, good_channels].mean(axis=0)
        power_data = power_data.mean(axis=0)
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power.npz'.format(subject)), **{'power_data': power_data})


def plot_power(subject, channel, cv, axes, vmin=None, vmax=None, bb=False, bb2=True):
    """Plot the power spectrum matrix.

    Parameters
    ----------
    subject : str
        Subject name for file name.
    channel : int
        ECoG array channel.
    cv : str
        CV to select.
    """
    ax0, ax1 = axes
    axes = [ax for ax in axes if (ax is not None)]
    if bb2:
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power_bb2.npz'.format(subject, cv, channel)))['power_data']
    elif bb:
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power_bb.npz'.format(subject, cv, channel)))['power_data']
    else:
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_{}_{}_power.npz'.format(subject, cv, channel)))['power_data']

    if ax0 is not None:
        print(power_data[::-1, s].min(), power_data[::-1, s].max())
        print(power_data.shape)
        im = ax0.imshow(power_data[::-1, s], interpolation='nearest', cmap='afmhot',
                        aspect='auto', vmin=vmin, vmax=vmax)
        yticklabels = [5, 25, 75]
        yticks = [40-np.searchsorted(bands.chang_lab['cfs'], y, side='right') for y in
                  yticklabels]
        yticklabels.append(200)
        yticks.append(0)
        ax0.set_yticks(yticks)
        ax0.set_yticklabels(yticklabels)
        ax0.set_title('Electrode: {}'.format(channel), **axes_label_fontstyle)
        ax0.set_ylabel('Freq. (Hz)', **axes_label_fontstyle)
        ax0.axvline(100, 0, 1, linestyle='--', c='white', lw=1.)
        #ax0.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)

    if ax1 is not None:
        hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                                  bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
        b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                                 bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])

        hb_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][3],
                                  bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][3])
        b_bands = np.logical_or(b_bands, hb_bands)
        b_bands = range(10, 21)
        hg = power_data[hg_bands].mean(axis=0)
        b = power_data[b_bands].mean(axis=0)

        b = b[s]
        hg = hg[s]

        hg -= hg.min()
        hg /= hg.max()
        hg = 2. * hg - 1
        b -= b.min()
        b /= b.max()
        b = 2. * b - 1

        ax1.plot(hg, c='r', lw=2)
        ax1.plot(b, c='k', lw=2)
        ax1.set_ylabel('Normalized\nAmplitude', **axes_label_fontstyle)
        ax1.set_xlabel('Time (ms)', **axes_label_fontstyle)
        ax1.set_xlim([0, plot_idx[-1]])
        ax1.axvline(100, 0, 1, linestyle='--', lw=1., c='gray')
    for ax in axes:
        ax.set_xticks([0, 100, plot_idx[-1]])
        ax.set_xticklabels([-500, 0, int(1000 * plot_time[-1])-500])
        ax.tick_params(**tickparams_fontstyle)
    return im


def plot_correlations(subjects, ax, kind='freq', bb=False, bb2=False):
    if not isinstance(subjects, list):
        subjects = [subjects]
    for subject in subjects:
        c = subject_colors[subject]
        if bb2:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb2.npz'.format(subject)))
        elif bb:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb.npz'.format(subject)))
        else:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations.npz'.format(subject)))
            xcorr_time = d['xcorr_time']
        xcorr_freq = d['xcorr_freq']

        if kind == 'freq':
            mean = xcorr_freq.mean(axis=(1, 2))
            sem = xcorr_freq.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_freq.shape[1:]))
            idxs = bands.chang_lab['cfs'] <= 60
            x = bands.chang_lab['cfs'][idxs]
            mean = mean[idxs]
            sem = sem[idxs]
            ax.plot(x, mean, c,
                    label=subject_labels[subject].replace('ect', '.'),
                    lw=1)
            ax.fill_between(x, mean-sem, mean+sem, alpha=.5,
                    facecolor=c)
            ax.set_xlim(0, 60)
            ax.set_xlabel('Freq. (Hz)', **axes_label_fontstyle)
            ax.set_ylabel(r'H$\gamma$ Correlation', **axes_label_fontstyle)
            ax.set_ylim(-.22, .5)
            ax.axhline(-.22, 15./60, 29./60, linestyle='-', c='black', lw=3.)
        elif kind == 'time':
            mean = xcorr_time.mean(axis=(1, 2))
            sem = xcorr_time.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_time.shape[1:]))
            ax.plot(mean)
            ax.fill_between(np.arange(mean.size), mean-sem, mean+sem, alpha=.5)
            n_time = xcorr_time.shape[0]
            ax.set_xticks([0, n_time // 2, n_time])
            ax.set_xticklabels(int(1000 * (n_time // 2) * (1/200.)) * np.array([-1, 0, 1]))
            ax.set_xlabel('Lag (ms)', **axes_label_fontstyle)
            ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation', **axes_label_fontstyle)
        else:
            raise NotImplementedError
    ax.legend(loc='upper left', ncol=2,
              prop={'size': ticklabel_fontstyle['fontsize']},
              labelspacing=.2, columnspacing=.6,
              handlelength=.4, handletextpad=.4)
    ax.axhline(0, linestyle='--', c='steelblue', lw=1.)
    ax.tick_params(**tickparams_fontstyle)


def plot_time_correlations(subjects, ax):
    if not isinstance(subjects, list):
        subjects = [subjects]
    for subject in subjects:
        c = subject_colors[subject]
        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_time_correlations.npz'.format(subject)))
        xcorr_time = d['xcorr_time']

        mean = xcorr_time.mean(axis=0)
        sem = xcorr_time.std(axis=0) / np.sqrt(xcorr_time.shape[0])
        x = np.arange(mean.size)
        mean = mean
        sem = sem
        ax.plot(x, mean, c)
        ax.fill_between(x, mean-sem, mean+sem, alpha=.5,
                facecolor=c)
        #ax.set_xlim(0, 40)
        #ax.set_ylim(-.2, None)
        ax.set_xlabel('Time (ms)', **axes_label_fontstyle)
        ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation', **axes_label_fontstyle)
    ax.axhline(0, linestyle='--', c='gray', lw=1.)
    ax.set_xticks([0, 100, plot_idx[-1]])
    ax.set_xticklabels([-500, 0, int(1000 * plot_time[-1])-500])
    ax.tick_params(**tickparams_fontstyle)


def plot_correlation_histogram(subjects, ax, cs=None, bb=False, bb2=False):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]

    for subject, c in zip(subjects, cs):
        if bb2:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb2.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb2.npz'.format(subject)))['power_data']
        elif bb:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb.npz'.format(subject)))['power_data']
        else:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power.npz'.format(subject)))['power_data']
        xcorr_time = d['xcorr_time']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel()

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        x, y = power_data[pos], corr_data[pos]
        slope, intercept, _, _, _ = stats.linregress(x, y)
        yp = 0
        xp = -intercept / slope
        cutoff = xp

        ax.hist(corr_data, bins=50, histtype='step', fill=False, color='k', lw=1)
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 200])
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylabel('Counts', **axes_label_fontstyle)
        ax.set_xlabel(r'H$\gamma$-$\beta$ Correlation', **axes_label_fontstyle)

        if bb2:
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff_bb2.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})
        elif bb:
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff_bb.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})
        else:
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})
    ax.tick_params(**tickparams_fontstyle)


def plot_power_histogram(subjects, ax, cs=None, bb=False, bb2=True):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]

    for subject, c in zip(subjects, cs):
        if bb2:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb2.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb2.npz'.format(subject)))['power_data']
        elif bb:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb.npz'.format(subject)))['power_data']
        else:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power.npz'.format(subject)))['power_data']
        xcorr_time = d['xcorr_time']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel()

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[pos], corr_data[pos])
        yp = 0
        xp = -intercept / slope
        cutoff = xp

        ax.hist(power_data, bins=50, histtype='step', fill=False, color='k', lw=1)
        ax.set_ylabel('Counts', **axes_label_fontstyle)
        ax.set_xlabel(r'Average H$\gamma$ Power (zscore)', **axes_label_fontstyle)
        ax.set_ylim(0, 500)
        ax.set_yticks([0, 500])
        ax.set_xlim(-.5, None)
        ax.set_xticks([-.5, 0, 1])

        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                            'cv_channels': power_data_not_flat >= cutoff})
    ax.tick_params(**tickparams_fontstyle)


def plot_power_correlations(subjects, ax, pos_only=True, cutoff_pct=None, bb=False, bb2=False):
    if not isinstance(subjects, list):
        subjects = [subjects]

    xmin = 0
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    x_cutoff = -np.inf
    cutoffs = []
    x_cutoffs = []
    return_cutoff_pct = []

    for subject in subjects:
        c = subject_colors[subject]
        if bb2:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb2.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb2.npz'.format(subject)))['power_data']
        elif bb:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_bb.npz'.format(subject)))['power_data']
        else:
            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations.npz'.format(subject)))
            power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power.npz'.format(subject)))['power_data']
        xcorr_time = d['xcorr_time']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel()

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        x, y = power_data[pos], corr_data[pos]
        """
        print(x.shape, y.shape)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        print(slope, intercept)
        m = TheilSenRegressor().fit(x[:,np.newaxis], y)
        slope = np.asscalar(m.coef_)
        intercept = np.asscalar(m.intercept_)
        print(slope, intercept)
        m = RANSACRegressor().fit(x[:,np.newaxis], y).estimator_
        slope = np.asscalar(m.coef_)
        intercept = np.asscalar(m.intercept_)
        print(slope, intercept)
        m = HuberRegressor(epsilon=1.0).fit(x[:,np.newaxis], y)
        slope = np.asscalar(m.coef_)
        intercept = np.asscalar(m.intercept_)
        print(slope, intercept)
        """
        m = LinearRegression().fit(x[:,np.newaxis], y)
        slope = np.asscalar(m.coef_)
        intercept = np.asscalar(m.intercept_)
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
        print(('{}, \tm: {:0.2e}, \tb: {:0.2e}, \tR^2: {:0.2e}, ' +
               '\tp: {:0.2e}, \tstderr: {:0.2e}').format(subject, slope,
                                                         intercept,
                                                         rvalue**2,
                                                         pvalue, stderr))
        yp = 0
        xp = -intercept / slope
        cutoff = xp
        if cutoff_pct is not None:
            cutoff = np.percentile(pos, cutoff_pct)
        cutoffs.append(cutoff)
        x_cutoff = max(cutoff, x_cutoff)
        x_cutoffs.append(x_cutoff)

        if pos_only:
            if c is None:
                c = 'k'
            n_groups = 9
            pts_x = power_data[pos]
            pts_y = corr_data[pos]
            idxs = np.argsort(pts_x)
            pts_x = pts_x[idxs]
            pts_y = pts_y[idxs]
            n_pts = pts_x.size
            n_per_group = n_pts // n_groups
            if n_per_group * n_groups != n_pts:
                n_per_group += 1
            start_idx = 0
            xs = []
            ys = []
            x_stds = []
            y_stds = []
            for ii in range(n_groups):
                if ii == n_groups-1:
                    x = pts_x[start_idx:]
                    y = pts_y[start_idx:]
                else:
                    x = pts_x[start_idx:start_idx + n_per_group]
                    y = pts_y[start_idx:start_idx + n_per_group]
                xs.append(x.mean())
                ys.append(y.mean())
                x_stds.append(x.std() / np.sqrt(x.size))
                y_stds.append(y.std() / np.sqrt(y.size))
                if ii == 0:
                    label = subject_labels[subject]
                else:
                    label = None
                ax.errorbar(xs[-1], ys[-1], xerr=x_stds[-1], yerr=y_stds[-1], c=c,
                        alpha=1., label=label)
                start_idx += n_per_group

            delta_x = max(.02, 1.1 * np.max(x_stds))
            delta_y = max(.05, 1.1 * np.max(y_stds))
            #xmin = min(xmin, min(xs) - delta_x)
            xmax = max(xmax, max(xs) + delta_x)
            ymin = min(ymin, min(ys) - delta_y)
            ymax = max(ymax, max(ys) + delta_y)

            x_width = xmax - xmin
            x_center = (xmax + xmin) / 2.
            y_width = ymax - ymin
            y_center = (ymax + ymin) / 2.
            width = max(x_width, y_width)

            ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation', **axes_label_fontstyle)
            ax.set_xlabel(r'Average H$\gamma$ Power (zscore)', **axes_label_fontstyle)
            x = np.linspace(min(xs), max(xs), 2)
            y = slope * x + intercept
            if cutoff_pct is None:
                ax.plot(x, y, '-', c=c, alpha=.5)
            if cutoff_pct is None:
                return_cutoff_pct.append((pos < cutoff).sum() / pos.size)
        else:
            raise NotImplementedError
            neg = power_data < 0
            ax.plot(x, y, 'r--')
            ax.scatter(power_data, corr_data, marker='.', c='k', alpha=.1)
            ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation', **axes_label_fontstyle)
            ax.set_xlabel(r'Average H$\gamma$ Power (zscore)', **axes_label_fontstyle)
            slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[neg], corr_data[neg])
            x = np.linspace(power_data.min(), 0, 1000)
            y = slope * x + intercept
            ax.plot(x, y, 'r--')
            ax.set_xlim(power_data.min(), -power_data.min())

        if bb2:
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff_bb2.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})
        elif bb:
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff_bb.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})
        else:
            print('hi')
            np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                                'cv_channels': power_data_not_flat >= cutoff})

    x_width = xmax - xmin
    x_center = (xmax + xmin) / 2.
    y_width = ymax - ymin
    y_center = (ymax + ymin) / 2.
    width = max(x_width, y_width)
    for cutoff, x_cutoff in zip(cutoffs, x_cutoffs):
        if cutoff > 0:
            ax.axvline(cutoff, 0,  abs(y_center - width / 2.)/ width,
                       linestyle='--', c='gray', lw=1.)
            ax.axhline(0, 0, (x_cutoff - x_center + width / 2.) / width,
                       linestyle='--', c='gray', lw=1.)
    ax.set_xlim(0, width)
    ax.set_ylim(y_center - width / 2., y_center + width / 2.)
    ax.set_xticks([0, .75])
    ax.tick_params(**tickparams_fontstyle)
    ax.legend(loc='lower right', ncol=1,
              prop={'size': ticklabel_fontstyle['fontsize']})
              #labelspacing=.2, columnspacing=.6,
              #handlelength=.4, handletextpad=.4)
    return return_cutoff_pct


def plot_resolved_power_correlations(subjects, ax, hline_c='gray', bb=False, bb2=False):
    if not isinstance(subjects, list):
        subjects = [subjects]

    ymin = np.inf
    ymax = -np.inf

    for subject in subjects:
        c = subject_colors[subject]
        if bb2:
            cv_channels = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff.npz'.format(subject)))['cv_channels']

            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb2.npz'.format(subject)))
        elif bb:
            cv_channels = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff_bb.npz'.format(subject)))['cv_channels']

            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations_bb.npz'.format(subject)))
        else:
            cv_channels = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                                  '{}_hg_power_cutoff.npz'.format(subject)))['cv_channels']

            d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                        '{}_correlations.npz'.format(subject)))
        print(cv_channels.sum())
        xcorr_freq = d['xcorr_freq']

        xcorr_freq_high = xcorr_freq[:, cv_channels]

        idxs = bands.chang_lab['cfs'] <= 60
        x = bands.chang_lab['cfs'][idxs]
        mean = xcorr_freq_high.mean(axis=(1))
        sem = xcorr_freq_high.std(axis=(1)) / np.sqrt(np.prod(xcorr_freq_high.shape[1]))
        mean = mean[idxs]
        sem = sem[idxs]
        ymin = min(mean.min(), ymin)
        ymax = max(mean.max(), ymax)
        c = subject_colors[subject]
        ax.plot(x, mean, '-', color='white', lw=.5)
        ax.fill_between(x, mean-sem, mean+sem, edgecolor=c,
                        facecolor=c, alpha=.6)
        ax.set_xlabel('Freq. (Hz)', **axes_label_fontstyle)
        ax.set_ylabel(r'H$\gamma$ Correlation.', **axes_label_fontstyle)

        xcorr_freq_low = xcorr_freq[:, ~cv_channels]
        mean = xcorr_freq_low.mean(axis=(1))
        sem = xcorr_freq_low.std(axis=(1)) / np.sqrt(np.prod(xcorr_freq_low.shape[1]))
        mean = mean[idxs]
        sem = sem[idxs]
        ymin = min(mean.min(), ymin)
        ymax = max(mean.max(), ymax)
        ax.plot(x, mean, '-', color=c, lw=.6)
        ax.fill_between(x, mean-sem, mean+sem, edgecolor=c,
                        facecolor=c, alpha=1.)
    ax.set_xlabel('Freq. (Hz)', **axes_label_fontstyle)
    ax.set_ylabel(r'H$\gamma$ Correlation.', **axes_label_fontstyle)
    ax.set_xlim(0, 60)
    ax.axhline(0, linestyle='--', c=hline_c, lw=.5)
    ax.tick_params(**tickparams_fontstyle)
    ymin = np.floor(ymin * 10) / 10
    ymax = np.ceil(ymax * 10) / 10
    ax.set_ylim(ymin, ymax)

    ax.axhline(ymin, 15./60, 29./60, linestyle='-', c='black', lw=3.)


def extract_b_hg(X, labels=None):
    return extract_b(X, labels), extract_hg(X, labels)


def extract_b(X, labels=None):
    b_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][2],
                             bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][2])
    X_b = X[b_bands]
    if labels is None:
        return X_b
    else:
        cvs = sorted(set(labels))
        shape = X_b.shape
        shapep = (shape[0], len(cvs)) + shape[2:]
        Xp = np.full(shapep, np.nan)
        for ii, cv in enumerate(cvs):
            idxs = np.where(labels == cv)[0]
            Xp[:, ii] = X_b[:, idxs].mean(axis=1)
        return Xp

def extract_hg(X, labels=None):
    hg_bands = np.logical_and(bands.chang_lab['cfs'] >= bands.neuro['min_freqs'][-1],
                              bands.chang_lab['cfs'] <= bands.neuro['max_freqs'][-1])
    X_hg = X[hg_bands]
    if labels is None:
        return X_hg
    else:
        cvs = sorted(set(labels))
        shape = X_hg.shape
        shapep = (shape[0], len(cvs)) + shape[2:]
        Xp = np.full(shapep, np.nan)
        for ii, cv in enumerate(cvs):
            idxs = np.where(labels == cv)[0]
            Xp[:, ii] = X_hg[:, idxs].mean(axis=1)
        return Xp
