import numpy as np

def plot_cv_accuracy(deep, linear, chance, labels, key, colors, ax, task='cv'):
    lw = 2
    assert task in ['cv', 'c', 'v', 'p', 'm']
    n_subjects, _, n_iter = deep[key].shape
    data = np.zeros((2, n_subjects, n_iter))
    deepi = deep[key]
    lineari = linear[key]
    for ii, label in enumerate(labels):
        data[0, ii] = lineari[ii, 2]
        data[1, ii] = deepi[ii, 2]

    for ii, (label, c) in enumerate(zip(labels, colors)):
        ax.errorbar([0, 1], data[:, ii].mean(axis=1)/chance[ii],
                    yerr=data[:,ii].std(axis=1)/np.sqrt(n_iter)/chance[ii],
                    c=c, label=label, lw=lw)
    ax.legend(loc='best')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear', 'Deep'])
    ax.set_xlim(-.5, 1.5)
    ax.axhline(1, c='gray', linestyle='--')
