import numpy as np
from scipy.stats import wilcoxon
from .style import (subject_colors as colors,
                    subject_labels as labels,
                    ticklabel_fontsize,
                    axes_label_fontsize)
from .stats import permute_paired_diffs


def plot_cv_accuracy(subjects, deep, linear, random, ax, task='Consonant\nVowel',
        legend=True, ymax=None):
    lw = 2
    n_subjects, _, n_iter = deep.shape
    data = np.zeros((3, n_subjects, n_iter))
    random = random.mean(axis=(1, 2))
    for ii in range(n_subjects):
        data[0, ii] = linear[ii, 2]
        data[1, ii] = deep[ii, 2]
        data[2, ii] = random[ii]

    for ii, s in enumerate(subjects):
        x = np.array([0, 1]) + .05 * (ii-1.5)
        y = data[:2, ii] / data[2, ii][np.newaxis,...]
        p = wilcoxon(y[0], y[1])[1] * 4
        p0 = wilcoxon(data[0, ii], data[2, ii])[1] * 4
        p1 = wilcoxon(data[1, ii], data[2, ii])[1] * 4
        print('{}: p={}, {}, {}'.format(labels[s], p, p0, p1))
        p = permute_paired_diffs(y[0], y[1])[2] * 4
        print('{}: p={}'.format(s, p))
        ym = np.mean(y, axis=-1)
        yerr = np.std(y, axis=-1) / np.sqrt(n_iter)
        ax.errorbar(x, ym, yerr=yerr,
                    c=colors[s], label=labels[s].replace('ect', '.'), lw=lw)
        if p < .001:
            ax.text(x[1] + .1, ym[1], '⁎⁎⁎', color=colors[s])
        elif p < .01:
            ax.text(x[1] + .1, ym[1], '⁎⁎', color=colors[s])
        elif p < .05:
            ax.text(x[1] + .1, ym[1], '⁎', color=colors[s])
    p = wilcoxon(data[0].ravel(), data[1].ravel())[1] * 5
    print('all subject: p={}'.format(p))

    diff = data[1] - data[0]
    task_name = task.split(' ')[0].replace('\n', ' ').lower()
    print(('The highest deep network accuracy for a single subject ' +
          'on the {} task is for Subject 1 which is {}$\pm$ {}\% ' +
          '({} times chance, {}\%) and {}$\pm$ {}\% ({} times chance, ' +
          '{}\%) for logistic regression and deep networks ' +
          'respectively, which is a {}\% ' +
          'improvement. ' + 
          'Mean {} classification accuracy across subjects (XX way) ' +
          'with deep networks is {} $\pm$ {}\%. For logistic regression, ' +
          'it is {} $\pm$ {}\%.').format(
                                task_name,
                                np.round(100 * data[1,0].mean(), 1),
                                np.round(100 * data[1,0].std(), 1),
                                np.round((data[1,0] / data[2,0]).mean(), 1),
                                np.round(100 * data[2,0].mean(), 1),
                                np.round(100 * data[0,0].mean(), 1),
                                np.round(100 * data[0,0].std(), 1),
                                np.round((data[0,0] / data[2,0]).mean(), 1),
                                np.round(100 * data[2,0].mean(), 1),
                                np.round(100 * ((data[1, 0] / data[0,0]).mean()
                                    - 1), 1),
                                task_name,
                                np.round(100 * data[1].mean(), 1),
                                np.round(100 * data[1].std(), 1),
                                np.round(100 * data[0].mean(), 1),
                                np.round(100 * data[0].std(), 1)))
    if legend:
        ax.legend(loc='upper left', prop={'size': ticklabel_fontsize})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Logistic', 'Deep'])
    ax.set_xlim(-.5, 1.5)
    ax.axhline(1, c='steelblue', linestyle='--', lw=1)
    ax.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)
    ax.set_title(task, fontsize=axes_label_fontsize)
    ax.set_ylim([None, ymax])
    if ymax is not None and ymax < 3:
        ax.set_yticks([1, 2])
    elif ymax is not None and ymax < 10:
        ax.set_yticks([1, 5, 9])
    else:
        ax.set_yticks([1, 5, 10, 15, 20])
    ax.tick_params(labelsize=ticklabel_fontsize)
