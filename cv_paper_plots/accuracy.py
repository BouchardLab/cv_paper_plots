import numpy as np

def plot_cv_accuracy(deep, linear, random, labels, key, colors, ax, task='cv'):
    lw = 2
    assert task in ['cv', 'c', 'v', 'p', 'm']
    n_subjects, _, n_iter = deep[key].shape
    data = np.zeros((3, n_subjects, n_iter))
    deepi = deep[key]
    lineari = linear[key]
    randomi = random[:, -1].mean(axis=(1, 2))
    for ii, label in enumerate(labels):
        data[0, ii] = lineari[ii, 2]
        data[1, ii] = deepi[ii, 2]
        data[2, ii] = randomi[ii]

    for ii, (label, c) in enumerate(zip(labels, colors)):
        x = np.array([0, 1]) + .05 * (ii-1.5)
        y = data[:2, ii] / data[2, ii][np.newaxis,...]
        ym = np.mean(y, axis=-1)
        yerr = np.std(y, axis=-1) / np.sqrt(n_iter)
        ax.errorbar(x, ym, yerr=yerr,
                    c=c, label=label, lw=lw)
    ax.legend(loc='best')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear', 'Deep'])
    ax.set_xlim(-.5, 1.5)
    ax.axhline(1, c='gray', linestyle='--')
    ax.set_ylabel('Accuracy/chance')
    ax.set_xlabel('CV task')
