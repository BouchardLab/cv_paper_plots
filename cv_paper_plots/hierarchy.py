import os, pickle
import numpy as np
import functools

from scipy import cluster
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ranksums, ttest_1samp

from .style import (subjects, subject_labels, subject_colors,
                    axes_label_fontstyle, ticklabel_fontstyle,
                    tickparams_fontstyle)


def create_dendrogram(features, labels, color_threshold=None,
                      title=None, save_path=None, ax=None,
                      deep=True):
    """
    Create dendrogram from data X. Averages over labels y.
    """
    def color(z, thresh, groups, k):
        dist = z[k-57, 2]
        child = z[k-57, 0].astype('int')
        while child > 56:
            child = z[child-57, 0].astype(int)
        if dist > thresh:
            set_c = 'gray'
        else:
            for c, idxs in groups.items():
                if child in idxs:
                    set_c = c
        return set_c

    z = cluster.hierarchy.ward(features)
    r = cluster.hierarchy.dendrogram(z, labels=labels,
                                     no_plot=True)
    old_idx = []
    for cv in r['ivl']:
        old_idx.append(labels.index(cv))
    if deep:
        groups = {'green': old_idx[0:13],
                  'red': old_idx[13:25],
                  'blue': old_idx[25:36],
                  'black': old_idx[36:57]}
    else:
        groups = {'red': old_idx[0:12],
                  'green': old_idx[12:25],
                  'blue': old_idx[25:34],
                  'black': old_idx[34:57]}

    if color_threshold is not None:
        r = cluster.hierarchy.dendrogram(z, labels=labels,
                                         link_color_func=functools.partial(color,
                                             z, color_threshold, groups),
                                         ax=ax)
    return z, r


def plot_dendrogram(yhs, threshold, cvs, max_d, ax, deep=False):
    ax.axhline(threshold, 0, 1, linestyle='--', c='gray', lw=1)

    z, r = create_dendrogram(yhs, cvs, threshold, ax=ax, deep=deep)
    ax.set_xticks([])
    ax.set_ylabel('Distance', labelpad=0, **axes_label_fontstyle)
    ax.set_ylim(None, max_d)
    ax.set_yticks([0, max_d])
    ax.set_yticklabels([0, max_d])
    ax.tick_params(*tickparams_fontstyle)
    return z, r


def plot_distance_vs_clusters(z, threshold, max_d, ax):
    ax.axhline(threshold, 0, 1, linestyle='--', c='gray', lw=1)
    ds = z[:, 2]
    bins = np.linspace(0, ds.max(), 1000)
    h, b = np.histogram(ds, bins, density=False)
    cs = np.cumsum(h[::-1])[::-1]
    ax.set_ylim(None, max_d)
    ax.plot(cs, b[1:], c='k')
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_yticks([0, max_d])
    ax.set_yticklabels([0, max_d])
    ax.set_ylim(None, max_d)
    ax.tick_params(axis='both', which='major', pad=1)
    ax.tick_params(**tickparams_fontstyle)
    ax.set_xlabel('# Clusters', labelpad=-2, **axes_label_fontstyle)
    ax.set_ylabel('Distance', labelpad=-15, **axes_label_fontstyle)
    return cs, b[1:]


def plot_cv_accuracy(cv_accuracy, ax):
    folds, n_cvs = cv_accuracy.shape
    ax.barh(range(n_cvs), np.nanmean(cv_accuracy, axis=0)[::-1], height=.7,
            edgecolor='k', color='none')
    ax.set_ylim(np.array([0, 57])-.5)
    ax.set_yticks([])
    ax.set_xticks([0, .5])
    ax.set_xlim(0, .75)
    ax.set_xticklabels([0, .5])
    ax.tick_params(**tickparams_fontstyle)
    ax.set_xlabel('Accuracy', labelpad=0, **axes_label_fontstyle)


def plot_soft_confusion(yhs, r, f, ax, cax, deep=True):
    im = ax.imshow(yhs, cmap='gray_r', interpolation='nearest',
            vmin=0, vmax=yhs.max())
    ax.set_xticks(np.linspace(0, 56, 57))
    ax.set_xticklabels(r['ivl'], **ticklabel_fontstyle)
    ax.set_yticks(np.linspace(0, 56, 57))
    ax.set_yticklabels(r['ivl'], **ticklabel_fontstyle)
    ax.set_ylabel('Target CV', **axes_label_fontstyle)
    ax.set_xlabel('Predicted CV', **axes_label_fontstyle)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    tick_offset = .02
    tick_scale = -.05
    for ii, label in enumerate(ax.yaxis.get_majorticklabels()):
        label.set_position([tick_scale*(((ii+1)%2)-.5)-tick_offset, 1])

    tick_offset = -.025
    tick_scale = -.025
    for ii, label in enumerate(ax.xaxis.get_majorticklabels()):
        label.set_position([0, 1+tick_scale*((ii%3)-.5)-tick_offset])
    #ax.tick_params(**tickparams_fontstyle)

    c = f.colorbar(im, cax=cax, orientation='horizontal')
    if deep:
        c.set_ticks([0, .1])
    else:
        c.set_ticks([0, .4])
    c.ax.tick_params(**tickparams_fontstyle)


def load_predictions(folder, files):
    consonants = ['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r',
                  's', r'$\int$', 't', r'$\theta$', 'v', 'w', 'j', 'z']
    vowels = ['a', 'i', 'u']

    cvs = []
    for c in consonants:
        for v in vowels:
            cvs.append(c+v)

    def load(path):
        with open(path, 'rb') as f:
            dicts, dicts2, y_dims, has_data = pickle.load(f, encoding='latin1')
        indices_dicts, y_hat_dicts, logits_dicts = dicts2
        return indices_dicts, y_hat_dicts

    indices = []
    y_hats = []
    for f in files:
        indices_dicts, y_hat_dicts = load(os.path.join(folder, f))
        indices_dicts, y_hat_dicts = indices_dicts[0], y_hat_dicts[0]
        for key in sorted(y_hat_dicts.keys()):
            y_hats.append(y_hat_dicts[key][0])
        for key in sorted(indices_dicts.keys()):
            indices.append(indices_dicts[key][0])

    yhs = np.zeros((57, 57))
    correct = np.zeros((len(indices), 57))
    total = np.zeros_like(correct)
    for ii, (idxs, pys) in enumerate(zip(indices, y_hats)):
        for (y, yh), py in zip(idxs, pys):
            yhs[y] += py
            if y == yh:
                correct[ii, y] += 1
            total[ii, y] += 1
    print(total)
    print(correct)
    cv_accuracy = correct / total
    yhs /= yhs.sum(axis=1, keepdims=True)


    z, r = create_dendrogram(yhs, cvs)

    old_idx = []
    for cv in r['ivl']:
        old_idx.append(cvs.index(cv))
    yhs = yhs[old_idx]
    yhs = yhs[:, old_idx]
    cv_accuracy = cv_accuracy[:, old_idx]
    return yhs, cv_accuracy, r['ivl']

def load_correlations(folder, files):
    def load(path):
        data = np.load(path)
        try:
            ccmjar = data['ccmjar']
        except KeyError:
            ccmjar = None
        return data['ccp'], data['ccm'], data['ccv'], ccmjar

    dp = []
    dm = []
    dv = []
    dmjar = []

    for fname in files:
        path = os.path.join(folder, fname)
        data = load(path)
        dp.append(data[0])
        dm.append(data[1])
        dv.append(data[2])
        dmjar.append(data[3])
    return dp, dm, dv, dmjar

def plot_correlations(dp, dm, dv, dmjar, ax, deep=True):

    box_params = {'notch': False,
                  'sym': '',
                  'vert': False,
                  'whis': 0,
                  'labels': ('Vowel',
                             'Constriction\nDegree',
                             'Constriction\nLocation',
                             'Major\nArticulator'),
                  'positions': [0, 1, 2, 3],
                  'medianprops': {'color': 'black', 'linewidth': 1},
                  'boxprops': {'color': 'black', 'linewidth': 1}}

    data = [np.concatenate(x) for x in [dv, dm, dp, dmjar]]

    def draw_sig(ax, x, y0, y1, n_stars, right=False):
        fraction = .2 / (y1 - y0)
        if right:
            fraction = -fraction
            offset = .03
        else:
            offset = -.03
        ax.annotate("", xy=(x, y0), xycoords='data',
        xytext=(x, y1), textcoords='data',
        arrowprops=dict(arrowstyle="-", ec='k',
        connectionstyle="bar,fraction={}".format(fraction)))
        ax.text(x+offset, .5 * (y0 + y1), n_stars*'⁎', fontsize=ticklabel_fontstyle['fontsize'],
                verticalalignment='center')

    if not (ttest_1samp(np.concatenate(dmjar), 0)[1] * 4 < .05):
        raise ValueError
    if not (ttest_1samp(np.concatenate(dp), 0)[1] * 4 < .05):
        raise ValueError
    if not (ttest_1samp(np.concatenate(dm), 0)[1] * 4 < .05):
        raise ValueError
    if not (ttest_1samp(np.concatenate(dv), 0)[1] * 4 < .05):
        raise ValueError
    if deep:
        if wilcoxon(np.concatenate(dmjar), np.concatenate(dp))[1] * 4 < 1e-10:
            draw_sig(ax, .15, 2, 3, 2)
            if not (ttest_1samp(np.concatenate(dmjar), 0)[1] * 4 < .05):
                raise ValueError
        if wilcoxon(np.concatenate(dp), np.concatenate(dm))[1] * 4 < 1e-10:
            draw_sig(ax, -.03, 1, 2, 2)
            if not (ttest_1samp(np.concatenate(dp), 0)[1] * 4 < .05):
                raise ValueError
        if wilcoxon(np.concatenate(dmjar), np.concatenate(dm))[1] * 4 < 1e-10:
            draw_sig(ax, -.06, 1, 3, 2)
            if not (ttest_1samp(np.concatenate(dm), 0)[1] * 4 < .05):
                raise ValueError
        if ttest_1samp(np.concatenate(dv), 0)[1] * 4 < 1e-4:
            ax.text(-.07, 0, '⁎', fontsize=ticklabel_fontstyle['fontsize'], verticalalignment='center')
    else:
        if wilcoxon(np.concatenate(dmjar), np.concatenate(dp))[1] * 4 < 1e-10:
            draw_sig(ax, .095, 2, 3, 2)
        if wilcoxon(np.concatenate(dp), np.concatenate(dm))[1] * 4 < 1e-10:
            draw_sig(ax, .33, 1, 2, 2, right=True)
        if wilcoxon(np.concatenate(dmjar), np.concatenate(dm))[1] * 4 < 1e-10:
            draw_sig(ax, -.06, 1, 3, 2)
        if ttest_1samp(np.concatenate(dv), 0)[1] * 4 < 1e-4:
            ax.text(-.07, 0, '⁎', fontsize=ticklabel_fontstyle['fontsize'], verticalalignment='center')
    bp = ax.boxplot(data, **box_params)
    ax.set_xlim([-.1, .65])
    ax.set_xlabel('Correlation Coefficient', **axes_label_fontstyle)
    ax.tick_params(**tickparams_fontstyle)
    for ii, x in enumerate([dv, dm, dp, dmjar]):
        for s, xs in zip(subjects, x):
            if ii == 0:
                label = subject_labels[s]
            else:
                label = None
            plt.plot(np.median(xs), ii, 'o',
                     markersize=4, c=subject_colors[s], label=label)
    ax.legend(loc='lower right', ncol=2, prop={'size': ticklabel_fontstyle['fontsize']})
