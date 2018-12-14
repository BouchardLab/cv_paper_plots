import numpy as np
from scipy.stats import linregress
from scipy.stats import wilcoxon
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

from .style import (subject_colors as colors,
                    subject_labels as labels,
                    ticklabel_fontstyle,
                    axes_label_fontstyle,
                    tickparams_fontstyle)
from .stats import (permute_paired_diffs,
                    permute_paired_regression,
                    parametric_slopes_test)

def plot_cv_slope(subjects, deep, linear, random, training_size, keys, axes,
                  legend=False, show_significance=False, normalize_chance=True):
    ax0, ax1 = axes
    lw = 2
    n_subjects, _, n_iter = deep[keys[0]].shape
    n_fracs = len(keys)
    accuracies = np.zeros((3, n_subjects, n_fracs, n_iter))
    slopes = np.zeros((2, n_subjects, 3))
    slopes_crossval = np.zeros((2, n_subjects, 10))
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
        if normalize_chance:
            yl = accuracies[1, jj] / accuracies[2, jj]
        else:
            yl = accuracies[1, jj]
        ym = yl.mean(axis=-1)
        yerr = yl.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                    c=colors[s], label=labels[s], lw=lw)

        x = np.array(keys) + .004 * (jj - 1.5) + .002
        if normalize_chance:
            yd = accuracies[0, jj] / accuracies[2, jj]
        else:
            yd = accuracies[0, jj]
        ym = yd.mean(axis=-1)
        yerr = yd.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                     fmt=':', c=colors[s], lw=lw)

        x = training_size[jj] / 1000.
        if normalize_chance:
            y = accuracies[0, jj] / accuracies[2, jj]
        else:
            y = accuracies[0, jj]
        for ii in range(y.shape[1]):
            slopes_crossval[0, jj, ii] = linregress(x[:, ii], y[:, ii])[0]
        slope0, intercept0, r_value0, p_value0, std_err0 = linregress(x.ravel(), y.ravel())
        #print(slope0, r_value0, p_value0, std_err0)
        if normalize_chance:
            y = accuracies[1, jj] / accuracies[2, jj]
        else:
            y = accuracies[1, jj]
        for ii in range(y.shape[1]):
            slopes_crossval[1, jj, ii] = linregress(x[:, ii], y[:, ii])[0]
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x.ravel(), y.ravel())
        print(s, slope1, std_err1)
        #print(slope1, r_value1, p_value1, std_err1)
        p = parametric_slopes_test(slope0, std_err0, x.size, slope1, std_err1, x.size) * 4
        slopes[0, jj] = slope0, std_err0, p
        slopes[1, jj] = slope1, std_err1, p
        #print()
    if show_significance:
        anova_model = []
        anova_subject = []
        anova_slope = []
        for model in [0, 1]:
            for subject in range(4):
                sl = slopes_crossval[model, subject]
                anova_slope.append(sl)
                anova_model.append(model * np.ones_like(sl))
                anova_subject.append(subject * np.ones_like(sl))
        df = pd.DataFrame.from_dict({'slope': np.array(anova_slope).ravel(),
                                     'model': np.array(anova_model).ravel(),
                                     'subject': np.array(anova_subject).ravel()})
        formula = "slope ~ C(model, Treatment(0)) + C(subject, Treatment(0))"
        lm = ols(formula, df)
        fit = lm.fit()
        print(fit.summary())
        for table in fit.summary().tables:
            print(table.as_latex_tabular())


    ax0.set_xlim(.45, 1.05)
    if legend:
        ax0.legend(loc='best', ncol=2, fontsize=ticklabel_fontstyle['fontsize'])
    if normalize_chance:
        ax0.axhline(1, c='steelblue', linestyle='--', lw=1)
        ax0.set_ylabel('Accuracy/chance', **axes_label_fontstyle)
        ax0.set_yticks([1, 10, 20, 30])
    else:
        ax0.set_ylabel('Accuracy', **axes_label_fontstyle)
        ax0.set_yticks([0, .25, .5])
        for jj, s in enumerate(subjects):
            x = np.array(keys) + .004 * (jj - 1.5)
            y = accuracies[2, jj].mean(axis=-1)
            ax0.plot(x, y, c=colors[s], linestyle='--', lw=1)
    ax0.set_xlabel('Training dataset fraction', **axes_label_fontstyle)

    for ii, s in enumerate(subjects):
        x = np.array([0, 1]) + .05 * (ii - 1.5)
        ym = slopes[:, ii, 0]
        yerr = slopes[:, ii, 1]
        p = slopes[1, ii, 2]
        ax1.errorbar(x, ym, yerr=yerr,
                     c=colors[s], lw=lw)
        if show_significance:
            print(p)
            if p < .001:
                ax1.text(x[1] + .05, ym[1], '⁎⁎', color=colors[s], fontsize=ticklabel_fontstyle['fontsize'])
                """
            elif p < .01:
                ax1.text(x[1] + .05, ym[1], '⁎⁎', color=colors[s], fontsize=ticklabel_fontstyle['fontsize'])
                """
            elif p < .05:
                ax1.text(x[1] + .05, ym[1], '⁎', color=colors[s], fontsize=ticklabel_fontstyle['fontsize'])
            else:
                ax1.text(x[1] + .05, ym[1], 'n.s.', color=colors[s], fontsize=ticklabel_fontstyle['fontsize'])

    ax1.set_xticks([0, 1])
    ax1.set_xlim(-.5, 1.5)
    ax1.axhline(0, c='steelblue', linestyle='--', lw=1)
    if normalize_chance:
        ax1.set_ylabel(r'$\Delta$ Accuracy/chance per 1k training examples',
                       **axes_label_fontstyle)
    else:
        ax1.set_ylabel(r'$\Delta$ Accuracy per 1k training examples',
                       **axes_label_fontstyle)
    ax1.set_title('Consonant\nVowel', **axes_label_fontstyle)
    if show_significance:
        print(('Deep networks scale better with dataset size than logistic regression ' +
               'with an improvement of {}x $\pm$ {}  and {}x $\pm$ {} ' +
               'over chance per 1000 training samples respectively. This improvement ' +
               'is summarized across subjects in Fig \\ref{{fig:slope}}B. ' +
               'For the subject with highest accuracy (Subject 1), the ' +
               'change in accuracy over chance per 1000 training examples ' +
               'for deep networks and logistic regression are ' +
               '{}x $\pm$ {} and {}x $\pm$ {} respectively. ' +
               'For the subject with highest slope (Subject 4), the ' +
               'change in accuracy over chance per 1000 training examples ' +
               'for deep networks and logistic regression are ' +
               '{}x $\pm$ {} and {}x $\pm$ {} respectively.').format(np.round(slopes[1, :, 0].mean(), 1),
                                           np.round(slopes[1, :, 0].std(), 1),
                                           np.round(slopes[0, :, 0].mean(), 1),
                                           np.round(slopes[0, :, 0].std(), 1),
                                           np.round(slopes[1, 0, 0], 1),
                                           np.round(slopes[1, 0, 1], 1),
                                           np.round(slopes[0, 0, 0], 1),
                                           np.round(slopes[0, 0, 1], 1),
                                           np.round(slopes[1, 3, 0], 1),
                                           np.round(slopes[1, 3, 1], 1),
                                           np.round(slopes[0, 3, 0], 1),
                                           np.round(slopes[0, 3, 1], 1)))
    for ax in axes:
        ax.tick_params(**tickparams_fontstyle)
    ax1.set_xticklabels(['Logistic', 'Deep'], **axes_label_fontstyle)
