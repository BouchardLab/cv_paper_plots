import h5py, os

import numpy as np
from scipy import stats

from ecog.utils import bands


plot_time = np.array([0, .5+.6])
plot_idx = (plot_time * 200.).astype(int)
s = slice(*plot_idx.tolist())

hg_power_time = np.array([.5-.07, .5 + .14])
hg_power_idx = (hg_power_time * 200.).astype(int)
hg_power_s = slice(*plot_idx.tolist())


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


def save_power(f, channel, cv, subject):
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


def save_correlations(f, subject, channel=None):
    good_examples, good_channels = good_examples_and_channels(f['X0'].value)
    n_time = f['X0'].shape[-1]
    assert plot_idx[-1] <= n_time
    n_time = plot_idx[-1]

    vsmc = np.concatenate([f['anatomy']['preCG'].value, f['anatomy']['postCG'].value])
    vsmc_electrodes = np.zeros(256)
    vsmc_electrodes[vsmc] = 1

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
            acorr_time[0, :, ii, jj] = np.correlate(hg, hg, mode='full')
            acorr_time[1, :, ii, jj] = np.correlate(b, b, mode='full')
    
    np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_correlations.npz'.format(subject)), **{'xcorr_freq': xcorr_freq,
                                                                           'xcorr_time': xcorr_time,
                                                                    'acorr_time': acorr_time})

def save_hg_power(f, subject):
    vsmc = np.concatenate([f['anatomy']['preCG'].value, f['anatomy']['postCG'].value])
    vsmc_electrodes = np.zeros(256)
    vsmc_electrodes[vsmc] = 1
    
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


def plot_power(subject, channel, cv, axes, vmin=None, vmax=None):
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
    power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                          '{}_{}_{}_power.npz'.format(subject, cv, channel)))['power_data']
    
    im = ax0.imshow(power_data[::-1, s], interpolation='nearest', cmap='afmhot', aspect='auto', vmin=vmin, vmax=vmax)
    ax0.set_yticks(np.arange(0, 40, 5))
    ax0.set_yticklabels(bands.chang_lab['cfs'][::-5].astype(int))
    ax0.set_title('{}_{}_{}'.format(subject, channel, cv))
    ax0.set_ylabel('Freq. (Hz)')
    ax0.set_xlabel('Time (ms)')
    
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
    for ax in [ax0, ax1]:
        ax.set_xticks([0, 100, plot_idx[-1]])
        ax.set_xticklabels([-500, 0, int(1000 * plot_time[-1])-500])
    ax1.set_ylabel('Normalized\nAmplitude')
    ax1.set_xlabel('Time (ms)')
    ax1.set_xlim([0, plot_idx[-1]])


def plot_correlations(subject, ax, kind='freq'):
    d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                '{}_correlations.npz'.format(subject)))
    xcorr_freq = d['xcorr_freq']
    xcorr_time = d['xcorr_time']

    if kind == 'freq':
        mean = xcorr_freq.mean(axis=(1, 2))
        sem = xcorr_freq.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_freq.shape[1:]))
        idxs = bands.chang_lab['cfs'] <= 40
        x = bands.chang_lab['cfs'][idxs]
        mean = mean[idxs]
        sem = sem[idxs]
        ax.plot(x, mean, 'k')
        ax.fill_between(x, mean-sem, mean+sem, alpha=.5,
                facecolor='gray')
        ax.set_xlim(0, 40)
        ax.set_ylim(-.2, None)
        ax.set_xlabel('Freq. (Hz)')
        ax.set_ylabel(r'H$\gamma$ Corr. Coef.')
    elif kind == 'time':
        mean = xcorr_time.mean(axis=(1, 2))
        sem = xcorr_time.std(axis=(1, 2)) / np.sqrt(np.prod(xcorr_time.shape[1:]))
        ax.plot(mean)
        ax.fill_between(np.arange(mean.size), mean-sem, mean+sem, alpha=.5)
        n_time = xcorr_time.shape[0]
        ax.set_xticks([0, n_time // 2, n_time])
        ax.set_xticklabels(int(1000 * (n_time // 2) * (1/200.)) * np.array([-1, 0, 1]))
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel(r'H$\gamma$-$\beta$ Corr. Coef.')
    else:
        raise NotImplementedError
    ax.axhline(0, linestyle='--', c='black')


def plot_correlation_histogram(subjects, ax, cs=None):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]

    for subject, c in zip(subjects, cs):
        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_correlations.npz'.format(subject)))
        xcorr_time = d['xcorr_time']
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power.npz'.format(subject)))['power_data']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel() 

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[pos], corr_data[pos])
        yp = 0
        xp = -intercept / slope
        cutoff = xp

        ax.hist(corr_data, bins=50, histtype='step', fill=False, color='k', lw=2)
        ax.set_ylabel('Counts')
        ax.set_xlabel(r'H$\gamma$-$\beta$ Correlation (R)')

        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                            'cv_channels': power_data_not_flat >= cutoff})


def plot_power_histogram(subjects, ax, cs=None):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]

    for subject, c in zip(subjects, cs):
        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_correlations.npz'.format(subject)))
        xcorr_time = d['xcorr_time']
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power.npz'.format(subject)))['power_data']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel() 

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[pos], corr_data[pos])
        yp = 0
        xp = -intercept / slope
        cutoff = xp

        ax.hist(power_data, bins=50, histtype='step', fill=False, color='k', lw=2)
        ax.set_ylabel('Counts')
        ax.set_xlabel(r'Average H$\gamma$ Power (z-score)')
        ax.axvline(xp, ls='--', color='black')
     
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                            'cv_channels': power_data_not_flat >= cutoff})
    
    
def plot_power_correlations(subjects, ax, num, cs=None, cutoff_pct=None):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    x_cutoff = -np.inf
    return_cutoff_pct = []

    for subject, c in zip(subjects, cs):
        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_correlations.npz'.format(subject)))
        xcorr_time = d['xcorr_time']
        power_data = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power.npz'.format(subject)))['power_data']
        power_data_not_flat = np.mean(power_data[..., hg_power_s], axis=-1)
        power_data = power_data_not_flat.ravel() 

        n_time = xcorr_time.shape[0]
        corr_data = np.ravel(xcorr_time[n_time // 2])

        pos = power_data >= 0
        slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[pos], corr_data[pos])
        yp = 0
        xp = -intercept / slope
        cutoff = xp
        if cutoff_pct is not None:
            cutoff = np.percentile(pos, cutoff_pct)
        x_cutoff = max(cutoff, x_cutoff)

        if num == 0:
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
            start_idx = 0
            xs = []
            ys = []
            x_stds = []
            y_stds = []
            for ii in range(n_groups):
                xs.append(pts_x[start_idx:start_idx + n_per_group].mean())
                ys.append(pts_y[start_idx:start_idx + n_per_group].mean())
                x_stds.append(pts_x[start_idx:start_idx + n_per_group].std())
                y_stds.append(pts_y[start_idx:start_idx + n_per_group].std())
                ax.errorbar(xs[-1], ys[-1], xerr=x_stds[-1], yerr=y_stds[-1], c=c,
                        alpha=.8)
                start_idx += n_per_group

            delta_x = max(.1, 1.1 * np.max(x_stds))
            delta_y = max(.5, 1.1 * np.max(y_stds))
            if xs[0] - delta_x < xmin:
                xmin = xs[0] - delta_x
            if xs[-1] + delta_x > xmax:
                xmax = xs[-1] + delta_x
            if ys[0] - delta_y < ymin:
                ymin = ys[0] - delta_y
            if ys[-1] + delta_y > ymax:
                ymax = ys[-1] + delta_y
            x_width = xmax - xmin
            x_center = (xmax + xmin) / 2.
            y_width = ymax - ymin
            y_center = (ymax + ymin) / 2.
            width = max(x_width, y_width)

            ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation (R)')
            ax.set_xlabel(r'Average H$\gamma$ Power (z-score)')
            x = np.linspace(xmin, xmax, 2)
            y = slope * x + intercept
            ax.plot(x, y, '-', c=c)
            if cutoff > 0:
                ax.axvline(cutoff, 0,  abs(y_center - width / 2.)/ width, linestyle='--', c='black')
                ax.axhline(0, 0, (x_cutoff - x_center + width / 2.) / width, linestyle='--', c='black')
            ax.set_xlim(x_center - width / 2., x_center + width / 2.)
            ax.set_ylim(y_center - width / 2., y_center + width / 2.)
            if cutoff_pct is None:
                return_cutoff_pct.append((pos < cutoff).sum() / pos.size)
        elif num == 1:
            neg = power_data < 0
            ax.plot(x, y, 'r--')
            ax.scatter(power_data, corr_data, marker='.', c='k', alpha=.1)
            ax.set_ylabel(r'H$\gamma$-$\beta$ Correlation (R)')
            ax.set_xlabel(r'Average H$\gamma$ Power (z-score)')
            slope, intercept, r_value, p_value, std_err = stats.linregress(power_data[neg], corr_data[neg])
            x = np.linspace(power_data.min(), 0, 1000)
            y = slope * x + intercept
            ax.plot(x, y, 'r--')
            ax.set_xlim(power_data.min(), -power_data.min())
        else:
            raise NotImplementedError
     
        np.savez(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(subject)), **{'cutoff': xp,
                                                                            'cv_channels': power_data_not_flat >= cutoff})
    return return_cutoff_pct


def plot_resolved_power_correlations(subjects, ax, cs=None):
    if not isinstance(subjects, list):
        subjects = [subjects]
        cs = len(subjects) * [cs]
    if cs is None:
        cs = len(subjects) * [cs]

    for subject, c in zip(subjects, cs):
        cv_channels = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(subject)))['cv_channels']

        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_correlations.npz'.format(subject)))
        xcorr_freq = d['xcorr_freq']

        xcorr_freq_high = xcorr_freq[:, cv_channels]

        idxs = bands.chang_lab['cfs'] <= 40
        x = bands.chang_lab['cfs'][idxs]
        mean = xcorr_freq_high.mean(axis=(1))
        sem = xcorr_freq_high.std(axis=(1)) / np.sqrt(np.prod(xcorr_freq_high.shape[1]))
        mean = mean[idxs]
        sem = sem[idxs]
        #ax.plot(x, mean, '--', color=c)
        ax.fill_between(x, mean-sem, mean+sem, alpha=1., edgecolor=c, hatch='//',
                        facecolor='none')
        ax.set_xlabel('Freq. (Hz)')
        ax.set_ylabel(r'H$\gamma$ Corr. Coef.')
        
        xcorr_freq_low = xcorr_freq[:, ~cv_channels]
        mean = xcorr_freq_low.mean(axis=(1))
        sem = xcorr_freq_low.std(axis=(1)) / np.sqrt(np.prod(xcorr_freq_low.shape[1]))
        mean = mean[idxs]
        sem = sem[idxs]
        #ax.plot(x, mean, color=c)
        ax.fill_between(x, mean-sem, mean+sem, alpha=1., edgecolor=c,
                        facecolor='none')
    ax.set_xlabel('Freq. (Hz)')
    ax.set_ylabel(r'H$\gamma$ Corr. Coef.')
    ax.set_xlim(0, 40)
