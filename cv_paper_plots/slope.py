import numpy as np
from scipy.stats import linregress

def plot_cv_slope(deep, linear, chance, labels, keys, colors, axes):
    ax0, ax1 = axes
    lw = 2
    n_subjects, _, n_iter = deep[keys[0]].shape
    n_fracs = len(keys)
    accuracies = np.zeros((2, n_subjects, n_fracs, n_iter))
    slopes = np.zeros((2, n_subjects, n_iter))
    for ii, key in enumerate(keys):
        deepi = deep[key]
        lineari = linear[key]
        for jj, label in enumerate(labels):
            accuracies[0, jj, ii] = lineari[jj, 2]
            accuracies[1, jj, ii] = deepi[jj, 2]

    for jj, (label, c) in enumerate(zip(labels, colors)):
        label = labels[0]
        ax0.errorbar(keys, accuracies[1, jj].mean(axis=-1)/chance[jj],
                    yerr=accuracies[1, jj].std(axis=-1)/np.sqrt(n_iter)/chance[jj],
                    c=c, label=label, lw=lw)
        ax0.errorbar(keys, accuracies[0, jj].mean(axis=-1)/chance[jj],
                    yerr=accuracies[0, jj].std(axis=-1)/np.sqrt(n_iter)/chance[jj],
                    fmt=':', c=c, label=label, lw=lw)
        for kk in range(n_iter):
            slope, intercept, r_value, p_value, std_err = linregress(keys,
                    accuracies[0, jj, :, kk])
            slopes[0, jj, kk] = slope
            slope, intercept, r_value, p_value, std_err = linregress(keys, accuracies[1, jj, :, kk])
            slopes[1, jj, kk] = slope

    ax0.set_xlim(.5, 1)
    ax0.legend(loc='best')
    ax0.axhline(1, c='gray', linestyle='--')

    for ii, (label, c) in enumerate(zip(labels, colors)):
        ax1.errorbar([0, 1], slopes[:, ii].mean(axis=1)/chance[ii],
                     yerr=slopes[:,ii].std(axis=1)/np.sqrt(n_iter)/chance[ii],
                     c=c, label=label, lw=lw)
    ax1.legend(loc='best')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Linear', 'Deep'])
    ax1.set_xlim(-.5, 1.5)
    ax1.axhline(1, c='gray', linestyle='--')
    print('turn frac units into examples units')
