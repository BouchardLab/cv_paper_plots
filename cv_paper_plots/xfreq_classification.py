import os
import numpy as np

from ecog.utils.bands import neuro, chang_lab

from .style import subject_colors, ticklabel_fontsize, axes_label_fontsize

def plot_xfreq_classification(subjects, band_abbreviations, bands,
                              single_accuracy, multi_accuracy, chance, axes):
    ax0, ax1 = axes
    for ii, ba in enumerate(band_abbreviations):
        for jj, (s, ch) in enumerate(zip(subjects, chance)):
            col = subject_colors[s]
            x = ii + .1 *(jj-1.5)
            y = single_accuracy[ba][jj, 2] / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            ax0.errorbar(x, ym, yerr=ysem, fmt='.', c=col)
            y = (multi_accuracy[ba][jj, 2] -
                 single_accuracy['hg'][jj, 2]) / single_accuracy['hg'][jj, 2]
            y = (multi_accuracy[ba][jj, 2] -
                 single_accuracy['hg'][jj, 2]) / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            ax1.errorbar(x, ym, yerr=ysem, fmt='.', c=col)
        ax0.plot([ii - .2, ii + .2],
                 2 * [np.mean(single_accuracy[ba][:, 2]/chance)], 'b')
        ax1.plot([ii - .2, ii + .2],
                 2 * [np.mean((multi_accuracy[ba][:, 2] -
                      single_accuracy['hg'][:, 2]) / chance)], 'b')
        """
        ax1.plot([ii - .2, ii + .2],
                 2 * [np.mean((multi_accuracy[ba][:, 2] -
                      single_accuracy['hg'][:, 2]) / single_accuracy['hg'][:, 2])], 'b')
        """
    for ax in [ax0, ax1]:
        ax.set_xticks(np.arange(len(band_abbreviations)))
        ax.set_xticklabels(bands)
        ax.set_xlim(-.5, 4.5)
        ax.tick_params(labelsize=ticklabel_fontsize)
    ax0.set_ylim(.5, None)
    ax0.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ % Accuracy', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/HG', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance', fontsize=axes_label_fontsize)
    ax0.plot([-10, 10], [1, 1], '--', c='blue', lw=.5)
    ax1.plot([-10, 10], [0, 0], '--', c='blue', lw=.5)

def plot_correlation_vs_accuracy(subjects, band_abbreviations, bands,
                                 single_accuracy, multi_accuracy, chance, axes):
    if not isinstance(subjects, list):
        subjects = [subjects]

    ax0, ax1 = axes

    cfs = chang_lab['cfs']

    for ii, (s, ch) in enumerate(zip(subjects, chance)):
        c = subject_colors[s]
        cv_channels = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                              '{}_hg_power_cutoff.npz'.format(s)))['cv_channels']

        d = np.load(os.path.join(os.environ['HOME'], 'plots/xfreq/data',
                    '{}_correlations.npz'.format(s)))
        xcorr_freq = d['xcorr_freq']

        xcorr_freq_high = xcorr_freq[:, cv_channels]
        print(xcorr_freq_high.shape)

        for jj, ba in enumerate(band_abbreviations):

            idxs = np.logical_and(cfs > neuro['min_freqs'][jj],
                                  cfs < neuro['max_freqs'][jj])
            x = xcorr_freq_high[idxs]
            print(x.shape)
            xm = x.mean()
            xsem = x.std() / np.sqrt(x.size)

            y = single_accuracy[ba][ii, 2] / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            ax0.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='.', c=c)

            y = (multi_accuracy[ba][ii, 2] - single_accuracy['hg'][ii, 2]) * 100 #/ ch
            y = (multi_accuracy[ba][ii, 2] -
                 single_accuracy['hg'][ii, 2]) / single_accuracy['hg'][ii, 2]
            y = (multi_accuracy[ba][ii, 2] - single_accuracy['hg'][ii, 2]) / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            ax1.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='.', c=c)

    ax0.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ % Accuracy', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/HG', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance', fontsize=axes_label_fontsize)
    ax0.axhline(1, linestyle='--', c='blue', lw=.5)
    ax1.axhline(0, linestyle='--', c='blue', lw=.5)
    for ax in axes:
        ax.set_xlabel(r'H$\gamma$ Corr. Coef.', fontsize=axes_label_fontsize)
        ax.tick_params(labelsize=ticklabel_fontsize)
