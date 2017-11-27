import numpy as np
from .style import subject_colors as colors, subject_labels as labels


colors = subject_colors

def plot_cv_accuracy(subjects, deep, linear, random, ax, task='CV task',
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
        ym = np.mean(y, axis=-1)
        yerr = np.std(y, axis=-1) / np.sqrt(n_iter)
        ax.errorbar(x, ym, yerr=yerr,
                    c=colors[s], label=labels[s], lw=lw)
    diff = data[1] - data[0]
    print(('{} classification accuracy (XX way) improves from {} $\pm$ ' +
          '{}\%  to {} $\pm$ {}\% across subjects for logistic regression ' +
          'compared to deep networks respectively. The highest single subject ' +
          '{} accuracies are for Subject 1 which are {}$\pm$ {}\% ' +
          '({} times chance, {}\%) and {}$\pm$ {}\% ({} times chance, ' +
          '{}\%) for logistic regression and deep networks ' +
          'respectively, which is a {}\% ' +
          'improvement.').format(task.split(' ')[0],
                                np.round(100 * data[0].mean(), 1),
                                np.round(100 * data[0].std(), 1),
                                np.round(100 * data[1].mean(), 1),
                                np.round(100 * data[1].std(), 1),
                                task.split(' ')[0].lower(),
                                np.round(100 * data[0,0].mean(), 1),
                                np.round(100 * data[0,0].std(), 1),
                                int(np.round((data[0,0] / data[2,0]).mean(), 0)),
                                np.round(100 * data[2,0].mean(), 1),
                                np.round(100 * data[1,0].mean(), 1),
                                np.round(100 * data[1,0].std(), 1),
                                int(np.round((data[1,0] / data[2,0]).mean(), 0)),
                                np.round(100 * data[2,0].mean(), 1),
                                np.round(100 * ((data[1, 0] / data[0,0]).mean() - 1), 1)))
    if legend:
        ax.legend(loc='best')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear', 'Deep'])
    ax.set_xlim(-.5, 1.5)
    ax.axhline(1, c='gray', linestyle='--', lw=1)
    ax.set_ylabel('Accuracy/chance')
    ax.set_xlabel(task)
    ax.set_ylim([None, ymax])
    if ymax is not None and ymax < 3:
        ax.set_yticks([1, 2])
