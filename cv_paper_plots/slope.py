import numpy as np
from scipy.stats import linregress

from .style import (subject_colors as colors,
                    subject_labels as labels,
                    ticklabel_fontsize,
                    axes_label_fontsize)

def plot_cv_slope(subjects, deep, linear, random, training_size, keys, axes,
                  legend=False):
    ax0, ax1 = axes
    lw = 2
    n_subjects, _, n_iter = deep[keys[0]].shape
    n_fracs = len(keys)
    accuracies = np.zeros((3, n_subjects, n_fracs, n_iter))
    slopes = np.zeros((2, n_subjects, n_iter))
    for ii, key in enumerate(keys):
        deepi = deep[key]
        lineari = linear[key]
        randomi = random[:, ii]
        for jj in range(n_subjects):
            accuracies[0, jj, ii] = lineari[jj, 2]
            accuracies[1, jj, ii] = deepi[jj, 2]
            accuracies[2, jj, ii] = randomi[jj].mean(axis=(0, 1))

    for jj, s in enumerate(subjects):
        x = np.array(keys) + .004 * (jj - 1.5)
        y = accuracies[1, jj] / accuracies[2, jj]
        y = accuracies[1, jj] / accuracies[2, jj, -1]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                    c=colors[s], label=labels[s], lw=lw)

        x = np.array(keys) + .004 * (jj - 1.5) + .002
        y = accuracies[0, jj] / accuracies[2, jj]
        y = accuracies[0, jj] / accuracies[2, jj, -1]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                     fmt=':', c=colors[s], lw=lw)

        for kk in range(n_iter):
            x = training_size[jj, :, kk]
            y = accuracies[0, jj, :, kk] / accuracies[2, jj, :, kk]
            y = accuracies[0, jj, :, kk] / accuracies[2, jj, -1, kk]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            slopes[0, jj, kk] = slope * 1000.

            y = accuracies[1, jj, :, kk] / accuracies[2, jj, :, kk]
            y = accuracies[1, jj, :, kk] / accuracies[2, jj, -1, kk]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            slopes[1, jj, kk] = slope * 1000.

    ax0.set_xlim(.45, 1.05)
    if legend:
        ax0.legend(loc='best')
    ax0.axhline(1, c='gray', linestyle='--', lw=1)
    ax0.set_xlabel('Training dataset fraction', fontsize=axes_label_fontsize)
    ax0.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)

    for ii, s in enumerate(subjects):
        x = np.array([0, 1]) + .05 * (ii - 1.5)
        y = slopes[:, ii]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax1.errorbar(x, ym, yerr=yerr,
                     c=colors[s], lw=lw)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Linear', 'Deep'])
    ax1.set_xlim(-.5, 1.5)
    ax1.axhline(0, c='gray', linestyle='--', lw=1)
    ax1.set_xlabel('CV task', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance per 1k training examples',
                   fontsize=axes_label_fontsize)
    print(slopes.shape)
    print(('Deep networks scale better with dataset size than logistic regresion ' +
           'with an improvement of {} $\pm$ {}  and {} $\pm$ {} ' +
           'over chance per 1000 training samples respectively. This improvement ' +
           'is summarized across subjects in Fig \\ref{}fig:slope{}B. ' +
           'For the subject with highest accuracy (Subject 1), the ' +
           'change in accuracy over chance per 1000 training examples ' +
           'for deep networks and logistic regression are ' +
           '{} $\pm$ {} and {} $\pm$ {} respectively. ' +
           'For the subject with highest slope (Subject 4), the ' +
           'change in accuracy over chance per 1000 training examples ' +
           'for deep networks and logistic regression are ' +
           '{} $\pm$ {} and {} $\pm$ {} respectively.').format(np.round(slopes[1].mean(), 1),
                                       np.round(slopes[1].std(), 1),
                                       np.round(slopes[0].mean(), 1),
                                       np.round(slopes[0].std(), 1),
                                       '{',
                                       '}',
                                       np.round(slopes[1, 0].mean(), 1),
                                       np.round(slopes[1, 0].std(), 1),
                                       np.round(slopes[0, 0].mean(), 1),
                                       np.round(slopes[0, 0].std(), 1),
                                       np.round(slopes[1, 3].mean(), 1),
                                       np.round(slopes[1, 3].std(), 1),
                                       np.round(slopes[0, 3].mean(), 1),
                                       np.round(slopes[0, 3].std(), 1)))
    for ax in axes:
        ax.tick_params(labelsize=ticklabel_fontsize)
