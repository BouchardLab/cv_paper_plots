import numpy as np
from scipy.stats import linregress

def plot_cv_slope(deep, linear, random, training_size, labels, keys, colors, axes):
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
        for jj, label in enumerate(labels):
            accuracies[0, jj, ii] = lineari[jj, 2]
            accuracies[1, jj, ii] = deepi[jj, 2]
            accuracies[2, jj, ii] = randomi[jj].mean(axis=(0, 1))

    for jj, (label, c) in enumerate(zip(labels, colors)):
        x = np.array(keys) + .004 * (jj - 1.5)
        y = accuracies[1, jj] / accuracies[2, jj]
        y = accuracies[1, jj] / accuracies[2, jj, -1]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                    c=c, label=label, lw=lw)

        x = np.array(keys) + .004 * (jj - 1.5) + .002
        y = accuracies[0, jj] / accuracies[2, jj]
        y = accuracies[0, jj] / accuracies[2, jj, -1]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax0.errorbar(x, ym, yerr=yerr,
                     fmt=':', c=c, lw=lw)

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
    ax0.legend(loc='best')
    ax0.axhline(1, c='gray', linestyle='--')
    ax0.set_xlabel('Training dataset fraction')
    ax0.set_ylabel('Accuracy/chance')

    for ii, (label, c) in enumerate(zip(labels, colors)):
        x = np.array([0, 1]) + .05 * (ii - 1.5)
        y = slopes[:, ii]
        ym = y.mean(axis=-1)
        yerr = y.std(axis=-1) / np.sqrt(n_iter)
        ax1.errorbar(x, ym, yerr=yerr,
                     c=c, label=label, lw=lw)
    ax1.legend(loc='best')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Linear', 'Deep'])
    ax1.set_xlim(-.5, 1.5)
    ax1.axhline(0, c='gray', linestyle='--')
    ax1.set_xlabel('CV task')
    ax1.set_ylabel(r'$\Delta$ Accuracy/chance per 1k training examples')
    print('turn frac units into examples units')
