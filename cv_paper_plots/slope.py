import numpy as np
from scipy.stats import linregress
from scipy.stats import wilcoxon
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

from .style import (subject_colors as colors,
                    subject_labels as labels,
                    ticklabel_fontsize,
                    axes_label_fontsize)
from .stats import (permute_paired_diffs,
                    permute_paired_regression,
                    parametric_slopes_test)

def plot_cv_slope(subjects, deep, linear, random, training_size, keys, axes,
                  legend=False):
    ax0, ax1 = axes
    lw = 2
    n_subjects, _, n_iter = deep[keys[0]].shape
    n_fracs = len(keys)
    accuracies = np.zeros((3, n_subjects, n_fracs, n_iter))
    slopes = np.zeros((2, n_subjects, 3))
    random = random.mean(axis=-1)
    for ii, key in enumerate(keys):
        deepi = deep[key]
        lineari = linear[key]
        randomi = random[:, ii]
        for jj in range(n_subjects):
            accuracies[0, jj, ii] = lineari[jj, 2]
            accuracies[1, jj, ii] = deepi[jj, 2]
            accuracies[2, jj, ii] = randomi[jj]

    for jj, s in enumerate(subjects):
        x = np.array(keys) + .004 * (jj - 1.5)
        yl = accuracies[1, jj] / accuracies[2, jj]
        ym = yl.mean(axis=-1)
        yerr = yl.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                    c=colors[s], label=labels[s], lw=lw)

        x = np.array(keys) + .004 * (jj - 1.5) + .002
        yd = accuracies[0, jj] / accuracies[2, jj]
        ym = yd.mean(axis=-1)
        yerr = yd.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                     fmt=':', c=colors[s], lw=lw)

        """
        x = training_size[jj]
        print(x.shape, yl.shape)
        xs = np.tile(x.T.ravel(), 2)
        ys = np.concatenate((yl.T.ravel(), yd.T.ravel()))
        plt.figure()
        plt.scatter(xs, ys)
        deep = np.concatenate((np.zeros(len(keys) * n_iter),
                               np.ones(len(keys) * n_iter))).astype(bool)
        df = pd.DataFrame.from_dict({'x': xs, 'y':ys, 'deep': deep})
        formula = 'y ~ x : C(deep)'
        lm = ols(formula, df)
        fit = lm.fit()
        print(fit.summary())
        """
        x = training_size[jj] / 1000.
        y = accuracies[0, jj] / accuracies[2, jj]
        slope0, intercept0, r_value0, p_value0, std_err0 = linregress(x.ravel(), y.ravel())
        print(slope0, r_value0, p_value0, std_err0)
        y = accuracies[1, jj] / accuracies[2, jj]
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x.ravel(), y.ravel())
        print(slope1, r_value1, p_value1, std_err1)
        p = parametric_slopes_test(slope0, std_err0, x.size, slope1, std_err1, x.size) * 4
        slopes[0, jj] = slope0, std_err0, p
        slopes[1, jj] = slope1, std_err1, p
        print()

    ax0.set_xlim(.45, 1.05)
    if legend:
        ax0.legend(loc='best', ncol=2, fontsize=ticklabel_fontsize)
    ax0.axhline(1, c='steelblue', linestyle='--', lw=1)
    ax0.set_xlabel('Training dataset fraction', fontsize=axes_label_fontsize)
    ax0.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)
    ax0.set_yticks([1, 5, 10, 15, 20])

    for ii, s in enumerate(subjects):
        x = np.array([0, 1]) + .05 * (ii - 1.5)
        ym = slopes[:, ii, 0]
        yerr = slopes[:, ii, 1]
        p = slopes[1, ii, 2]
        ax1.errorbar(x, ym, yerr=yerr,
                     c=colors[s], lw=lw)
        print(p)
        if p < .001:
            ax1.text(x[1] + .05, ym[1], '⁎⁎', color=colors[s], fontsize=axes_label_fontsize)
            """
        elif p < .01:
            ax1.text(x[1] + .05, ym[1], '⁎⁎', color=colors[s], fontsize=axes_label_fontsize)
            """
        elif p < .05:
            ax1.text(x[1] + .05, ym[1], '⁎', color=colors[s], fontsize=axes_label_fontsize)
        else:
            ax1.text(x[1] + .05, ym[1], 'n.s.', color=colors[s], fontsize=axes_label_fontsize)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Logistic', 'Deep'])
    ax1.set_xlim(-.5, 1.5)
    ax1.axhline(0, c='steelblue', linestyle='--', lw=1)
    ax1.set_title('Consonant\nVowel', fontsize=axes_label_fontsize)
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance per 1k training examples',
                   fontsize=axes_label_fontsize)
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
