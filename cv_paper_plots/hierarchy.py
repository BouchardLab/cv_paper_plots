import os, pickle
import numpy as np
import functools

from scipy import cluster
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_1samp

from .style import (subjects, subject_labels, subject_colors,
                    axes_label_fontsize, ticklabel_fontsize)


def create_dendrogram(features, labels, color_threshold=None,
                      title=None, save_path=None, ax=None):
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
    groups = {'green': old_idx[0:13],
              'red': old_idx[13:25],
              'blue': old_idx[25:36],
              'black': old_idx[36:57]}

    if color_threshold is not None:
        r = cluster.hierarchy.dendrogram(z, labels=labels,
                                         link_color_func=functools.partial(color,
                                             z, color_threshold, groups),
                                         ax=ax)
    return z, r


def plot_dendrogram(yhs, threshold, cvs, max_d, ax):
    ax.axhline(threshold, 0, 1, linestyle='--', c='k', lw=1)

    z, r = create_dendrogram(yhs, cvs, threshold, ax=ax)
    ax.set_xticks([])
    ax.set_ylabel('Distance', fontsize=axes_label_fontsize, labelpad=0)
    ax.set_ylim(None, max_d)
    ax.set_yticks([0, max_d])
    ax.set_yticklabels([0, max_d])
    ax.tick_params(labelsize=ticklabel_fontsize)
    return z, r


def plot_distance_vs_clusters(z, threshold, max_d, ax):
    ax.axhline(threshold, 0, 1, linestyle='--', c='k', lw=1)
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
    ax.tick_params(labelsize=ticklabel_fontsize)
    ax.set_xlabel('# Clusters', fontsize=axes_label_fontsize, labelpad=0)
    ax.set_ylabel('Distance', fontsize=axes_label_fontsize, labelpad=-10)


def plot_cv_accuracy(cv_accuracy, ax):
    folds, n_cvs = cv_accuracy.shape
    ax.barh(range(n_cvs), cv_accuracy.mean(axis=0)[::-1], height=.7,
            edgecolor='k', color='none')
    ax.set_ylim(np.array([0, 57])-.5)
    ax.set_yticks([])
    ax.set_xticks([0, .5])
    ax.set_xticklabels([0, .5])
    ax.tick_params(labelsize=ticklabel_fontsize)
    ax.set_xlabel('Accuracy', fontsize=axes_label_fontsize, labelpad=0)


def plot_soft_confusion(yhs, r, f, ax, cax):
    im = ax.imshow(yhs, cmap='gray_r', interpolation='nearest',
            vmin=0, vmax=yhs.max())
    ax.set_xticks(np.linspace(0, 56, 57))
    ax.set_xticklabels(r['ivl'])
    ax.set_yticks(np.linspace(0, 56, 57))
    ax.set_yticklabels(r['ivl'])
    ax.set_ylabel('Target CV', fontsize=axes_label_fontsize)
    ax.set_xlabel('Predicted CV', fontsize=axes_label_fontsize)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    tick_offset = .02
    tick_scale = -.05
    pos = 0
    for label in ax.yaxis.get_majorticklabels():
        label.set_position([tick_scale*(((pos+1)%2)-.5)-tick_offset, 1])
        pos += 1

    tick_offset = -.01
    tick_scale = -.03
    pos = 0
    for label in ax.xaxis.get_majorticklabels():
        label.set_position([0, 1+tick_scale*(((pos+1)%2)-.5)-tick_offset])
        pos += 1
    ax.tick_params(labelsize=ticklabel_fontsize-2)

    c = f.colorbar(im, cax=cax, orientation='horizontal')
    c.set_ticks([0, .1])
    c.ax.tick_params(labelsize=ticklabel_fontsize)


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

def plot_correlations(dp, dm, dv, dmjar, ax):

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
    print(ranksums(np.concatenate(dmjar), np.concatenate(dp)))
    print(ranksums(np.concatenate(dp), np.concatenate(dm)))
    print(ranksums(np.concatenate(dmjar), np.concatenate(dm)))
    print(ttest_1samp(np.concatenate(dv), 0))
    bp = ax.boxplot(data, **box_params)
    ax.set_xlim([-.06, .65])
    ax.set_xlabel('Correlation Coefficient', fontsize=axes_label_fontsize)
    ax.tick_params(labelsize=ticklabel_fontsize)
    for ii, x in enumerate([dv, dm, dp, dmjar]):
        for s, xs in zip(subjects, x):
            if ii == 0:
                label = subject_labels[s]
            else:
                label = None
            plt.plot(np.median(xs), ii, 'o',
                     markersize=4, c=subject_colors[s], label=label)
    ax.legend(loc='lower right', ncol=2, prop={'size': ticklabel_fontsize})
