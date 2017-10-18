import os, pickle
import numpy as np
import functools

from scipy import cluster
import matplotlib.pyplot as plt


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
            set_c = 'k'
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
    groups = {'#1f77b4': old_idx[0:13],
              '#ff7f0e': old_idx[13:25],
              '#2ca02c': old_idx[25:42],
              '#9467bd': old_idx[42:57]}

    if color_threshold is not None:
        r = cluster.hierarchy.dendrogram(z, labels=labels,
                                         link_color_func=functools.partial(color,
                                             z, color_threshold, groups),
                                         ax=ax)
    return z, r


def corr_box_plot(p, m, v):
    place_25 = np.sort(p)[np.round(int(p.size*.25))]
    place_med = np.median(p)
    place_75 = np.sort(p)[np.round(int(p.size*.75))]
    manner_25 = np.sort(m)[np.round(int(m.size*.25))]
    manner_med = np.median(m)
    manner_75 = np.sort(m)[np.round(int(m.size*.75))]
    vowel_25 = np.sort(v)[np.round(int(v.size*.25))]
    vowel_med = np.median(v)
    vowel_75 = np.sort(v)[np.round(int(v.size*.75))]
    box_params = {'notch': False,
                  'sym': '',
                  'vert': False,
                  'whis': 0,
                  'labels': ('Vowel configuration', 'Constriction degree', 'Constriction location'),
                  'medianprops': {'color': 'black', 'linewidth': 2},
                  'boxprops': {'color': 'gray', 'linewidth': 2}}
    data = [v, m, p]
    f = plt.figure()
    plt.boxplot(data, **box_params)
    plt.xlabel('Correlation Coefficient')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    return f


def plot_dendrogram(yhs, threshold, cvs, max_d, ax):
    ax.axhline(threshold, 0, 1, linestyle='--', c='k')

    z, r = create_dendrogram(yhs, cvs, threshold, ax=ax)
    ax.set_xticks([])
    ax.set_ylabel('Distance')
    ax.set_ylim(None, max_d)
    ax.set_yticks([0, max_d])
    ax.set_yticklabels([0, max_d], fontsize=8)
    return z, r


def plot_distance_vs_clusters(z, threshold, max_d, ax):
    ax.axhline(threshold, 0, 1, linestyle='--', c='k')
    ds = z[:, 2]
    bins = np.linspace(0, ds.max(), 1000)
    h, b = np.histogram(ds, bins, density=False)
    cs = np.cumsum(h[::-1])[::-1]
    ax.set_ylim(None, max_d)
    ax.plot(cs, b[1:], c='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(None, max_d)
    ax.set_xlabel('# Clusters')


def plot_cv_accuracy(cv_accuracy, ax):
    folds, n_cvs = cv_accuracy.shape
    ax.barh(range(n_cvs), cv_accuracy.mean(axis=0)[::-1], height=.7,
            edgecolor='k', color='none')
    ax.set_ylim(np.array([0, 57])-.5)
    ax.set_yticks([])
    ax.set_xticks([0, .5])
    ax.set_xticklabels([0, .5])
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Accuracy')


def plot_soft_confusion(yhs, r, f, ax, cax):
    im = ax.imshow(yhs, cmap='gray_r', interpolation='nearest',
            vmin=0, vmax=yhs.max())
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(np.linspace(0, 56, 57))
    ax.set_xticklabels(r['ivl'], rotation='vertical', fontsize=6)
    ax.set_yticks(np.linspace(0, 56, 57))
    ax.set_yticklabels(r['ivl'], fontsize=6)
    ax.set_ylabel('Target CV')
    ax.set_xlabel('Predicted CV')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()

    tick_offset = -.01
    tick_scale = -.05
    pos = 0
    for label in ax.yaxis.get_majorticklabels():
        label.set_position([1+tick_scale*(((pos+1)%2)-.5)-tick_offset, 1])
        pos += 1

    pos = 0
    for label in ax.xaxis.get_majorticklabels():
        label.set_position([0, 1+tick_scale*(((pos+1)%2)-.5)-tick_offset])
        pos += 1

    c = f.colorbar(im, cax=cax)
    c.set_ticks([0, .1])
    c.ax.tick_params(labelsize=8)
    cax.set_ylabel('Probability')


def load_predictions(folder, files):
    consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r',
                         's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
    vowels = sorted(['aa', 'ee', 'oo'])

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
        dp.extend(data[0])
        dm.extend(data[1])
        dv.extend(data[2])
        dmjar.extend(data[3])
    return dp, dm, dv, dmjar

def plot_correlations(dp, dm, dv, dmjar, ax):

    box_params = {'notch': False,
                  'sym': '', 
                  'vert': False,
                  'whis': 0,
                  'labels': ('Vowel',
                             'Manner',
                             'Place',
                             'Maj. Art.'),
                  'positions': [2-.375, 3-.375, 4-.375, 5-.375],
                  'medianprops': {'color': 'black', 'linewidth': 1}, 
                  'boxprops': {'color': 'black', 'linewidth': 1}} 

    data = [dv, dm, dp, dmjar]
    bp = ax.boxplot(data, **box_params)
    c = 'k' 
    for ii in range(len(bp['boxes'])):
        plt.setp(bp['boxes'][ii], color=c)
        plt.setp(bp['caps'][2*ii], color=c)
        plt.setp(bp['caps'][2*ii+1], color=c)
        plt.setp(bp['whiskers'][2*ii], color=c)
        plt.setp(bp['whiskers'][2*ii+1], color=c)
        plt.setp(bp['medians'][ii], color=c)
    ax.plot(0,0, '-', c='red', label='Deep')
    ax.plot(0,0, '-', c='black', label='Linear')
    ax.set_xlim([-.06, .65])
    ax.set_xlabel('Correlation Coefficient')
