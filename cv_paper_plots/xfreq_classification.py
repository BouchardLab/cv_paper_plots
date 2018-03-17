import os
import numpy as np
from scipy.stats import wilcoxon

from ecog.utils.bands import neuro, chang_lab

from .style import (subject_colors, subject_labels,
                    ticklabel_fontstyle, axes_label_fontstyle,
                    tickparams_fontstyle)

def plot_xfreq_classification(subjects, band_abbreviations, bands,
                              single_accuracy, multi_accuracy, chance, axes):
    ax0, ax1 = axes
    for ii, ba in enumerate(band_abbreviations):
        y = single_accuracy[ba][:, 2]
        ch = chance
        pA = wilcoxon(y.ravel(), ch.ravel())[1] * 5
        print('{}, pA: {:0.2e}'.format(ba, pA))
        y = multi_accuracy[ba][:, 2] / chance
        y_hg = single_accuracy['hg'][:, 2] / chance
        pB = wilcoxon(y.ravel(), y_hg.ravel())[1] * 5
        print('{}, pB: {:0.2e}\n'.format(ba, pB))
        if pA < 1e-3:
            ax0.text(ii, .75, '⁎⁎', fontsize=ticklabel_fontstyle['fontsize'], horizontalalignment='center')
        elif pA < 1e-2:
            ax0.text(ii, .75, '⁎', fontsize=ticklabel_fontstyle['fontsize'], horizontalalignment='center')
        else:
            ax0.text(ii, .75, 'n.s.', fontsize=ticklabel_fontstyle['fontsize'], horizontalalignment='center')
        for jj, (s, ch) in enumerate(zip(subjects, chance)):
            col = subject_colors[s]
            x = ii + .125 *(jj-1.5)
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
            if ii == 0:
                label = subject_labels[s]
            else:
                label = None
            ax1.errorbar(x, ym, yerr=ysem, fmt='.', c=col, label=label)
        ax0.plot([ii - .2, ii + .2],
                 2 * [np.mean(single_accuracy[ba][:, 2]/chance)], 'b', alpha=.7)
        ax1.plot([ii - .2, ii + .2],
                 2 * [np.mean((multi_accuracy[ba][:, 2] -
                      single_accuracy['hg'][:, 2]) / chance)], 'b', alpha=.7)
    for jj, (s, ch) in enumerate(zip(subjects, chance)):
        col = subject_colors[s]
        y = single_accuracy['hg'][jj, 2] / ch
        std = y.std()
        ax1.plot([5. + .1 * (jj-1.5), 5. + .1 * (jj-1.5)], [-std, std], c=col)
        """
        ax1.plot([ii - .2, ii + .2],
                 2 * [np.mean((multi_accuracy[ba][:, 2] -
                      single_accuracy['hg'][:, 2]) / single_accuracy['hg'][:, 2])], 'b')
        """
    ax1.text(2.5, 2, 'all n.s.', fontsize=ticklabel_fontstyle['fontsize'], horizontalalignment='center')
    for ax in [ax0, ax1]:
        ax.tick_params(**tickparams_fontstyle)
    ax0.set_xticks(np.arange(len(band_abbreviations)))
    ax0.set_xticklabels(bands)
    ax0.set_xlim(-.5, 4.5)
    ax1.set_xticks(np.arange(len(band_abbreviations) + 1))
    ax1.set_xticklabels(bands + [r'H$\gamma$ std.'])
    ax1.set_xlim(-.5, 5.5)
    ax0.set_ylim(.5, None)
    ax0.set_yticks([1, 2, 3])
    ax0.set_ylabel('Accuracy/chance', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ % Accuracy', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ Accuracy/HG', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance', **axes_label_fontstyle)
    ax0.plot([-10, 10], [1, 1], '--', c='steelblue', lw=.5)
    ax1.plot([-10, 10], [0, 0], '--', c='steelblue', lw=.5)
    ax1.legend(loc='lower left', fontsize=ticklabel_fontstyle['fontsize'], ncol=2)

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

        for jj, ba in enumerate(band_abbreviations):

            idxs = np.logical_and(cfs > neuro['min_freqs'][jj],
                                  cfs < neuro['max_freqs'][jj])
            x = xcorr_freq_high[idxs]
            xm = x.mean()
            xsem = x.std() / np.sqrt(x.size)

            y = single_accuracy[ba][ii, 2] / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            if jj == 0:
                label = subject_labels[s]
            else:
                label = None
            ax0.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='.', c=c,
                         label=label)

            y = (multi_accuracy[ba][ii, 2] - single_accuracy['hg'][ii, 2]) * 100 #/ ch
            y = (multi_accuracy[ba][ii, 2] -
                 single_accuracy['hg'][ii, 2]) / single_accuracy['hg'][ii, 2]
            y = (multi_accuracy[ba][ii, 2] - single_accuracy['hg'][ii, 2]) / ch
            ym = y.mean()
            ysem = y.std() / np.sqrt(10)
            ax1.errorbar(xm, ym, xerr=xsem, yerr=ysem, fmt='.', c=c)

    ax0.set_ylabel('Accuracy/chance', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ % Accuracy', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ Accuracy/HG', **axes_label_fontstyle)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance', **axes_label_fontstyle)
    ax0.set_yticks([1, 2, 3])
    ax0.axhline(1, linestyle='--', c='steelblue', lw=.5)
    ax1.axhline(0, linestyle='--', c='steelblue', lw=.5)
    for ax in axes:
        ax.set_xlabel(r'H$\gamma$ Corr. Coef.', **axes_label_fontstyle)
        ax.tick_params(**tickparams_fontstyle)
    ax0.legend(loc='best', ncol=1,
              prop={'size': ticklabel_fontstyle['fontsize']})
