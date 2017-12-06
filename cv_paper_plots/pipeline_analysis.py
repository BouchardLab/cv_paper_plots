import h5py, os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import ecog
from ecog.utils.electrodes import load_electrode_labels
from ecog.signal_processing import hilbert_transform, gaussian
from ecog.tokenize.transcripts import parse, make_df
from ecog.utils import bands
from ecog.signal_processing import zscore

from .style import axes_label_fontsize, ticklabel_fontsize

def make_slice(idx, transcript, rate):
    align = transcript['align'].loc[idx]
    start_idx = int(rate * (align-.5))
    stop_idx = int(rate * (align + .8))
    return slice(start_idx, stop_idx)

def make_colors(n):
    colors = []
    for ii in range(n):
        if ii % 2 == 0:
            colors.append('k')
        else:
            colors.append('gray')
    return colors


def plot_electrodes_by_time(data, ax):
    n_elec, n_time = data.shape
    cmap = matplotlib.cm.get_cmap('Reds')
    cs = make_colors(n_elec)
    x = np.linspace(-500, 800, n_time)
    for ii in range(n_elec):
        y = data[ii] / 3
        ax.plot(x, y + ii, c=cs[ii], lw=.5)
    ax.set_xticks([-500, 0, 800])
    ax.set_yticks([])
    ax.set_yticklabels([None])
    if n_elec == 1:
        ax.set_ylabel('Electrode Voltage')
    else:
        ax.set_ylabel('Electrodes')
    ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axvline(0, 0, 1, linestyle='--', c='black', lw=1)
    ax.tick_params(labelsize=ticklabel_fontsize)


def plot_40bands_by_time(data, band_idxs, sl, rate, block_path, ax):
    n_idxs = len(band_idxs)
    cs = make_colors(n_idxs)

    cfs = bands.chang_lab['cfs']
    stds = bands.chang_lab['sds']
    filters = [gaussian(data, rate, cfs[idx], stds[idx]) for idx in band_idxs]
    ht = np.squeeze(hilbert_transform(data, rate, filters).real)
    ht, means, stds = zscore(ht, mode='file', sampling_freq=400., block_path=block_path)
    ht = ht[:, sl]
    x = np.linspace(-500, 800, ht.shape[1])
    for ii in range(n_idxs):
        y = ht[ii] / 2
        ax.plot(x, y + ii, c=cs[ii], lw=1.5)
    ax.set_xticks([-500, 0, 800])
    ax.set_yticks(np.arange(n_idxs)[::3])
    ax.set_yticklabels(cfs.astype(int)[band_idxs][::3])
    ax.set_ylabel('Frequency', fontsize=axes_label_fontsize)
    ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axvline(0, 0, 1, linestyle='--', c='black', lw=1)
    ax.tick_params(labelsize=ticklabel_fontsize)


def plot_40bandsAA_by_time(data, band_idxs, sl, rate, block_path, ax):
    n_idxs = len(band_idxs)
    cs = make_colors(n_idxs)

    cfs = bands.chang_lab['cfs']
    stds = bands.chang_lab['sds']
    filters = [gaussian(data, rate, cfs[idx], stds[idx]) for idx in band_idxs]
    ht = np.squeeze(abs(hilbert_transform(data, rate, filters)))
    ht, means, stds = zscore(ht, mode='file', sampling_freq=400., block_path=block_path)
    ht = ht[:, sl]
    x = np.linspace(-500, 800, ht.shape[1])
    for ii in range(n_idxs):
        y = ht[ii] / 2
        ax.plot(x, y + ii, c=cs[ii], lw=1.5)
    ax.set_xticks([-500, 0, 800])
    ax.set_yticks(np.arange(n_idxs)[::3])
    ax.set_yticklabels(cfs.astype(int)[band_idxs][::3])
    ax.set_ylabel('Frequency', fontsize=axes_label_fontsize)
    ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axvline(0, 0, 1, linestyle='--', c='black', lw=1)
    ax.tick_params(labelsize=ticklabel_fontsize)


def plot_neurobands_by_time(data, sl, rate, block_path, ax):
    n_bands = len(bands.neuro['bands'])
    cs = make_colors(n_bands)
    cfs = bands.chang_lab['cfs']
    stds = bands.chang_lab['sds']

    for ii, (minf, maxf) in enumerate(zip(bands.neuro['min_freqs'], bands.neuro['max_freqs'])):
        filters = [gaussian(data, rate, c, s) for c, s in zip(cfs, stds) if ((c >= minf) and (c <= maxf))]
        ht = np.squeeze(abs(hilbert_transform(data, rate, filters)))
        ht, m, s = zscore(ht, mode='file', sampling_freq=400., block_path=block_path)
        ht = ht[:, sl].mean(axis=0) / 2
        x = np.linspace(-500, 800, ht.size)
        ax.plot(x, ht + ii, c=cs[ii], lw=1.5)
    ax.set_xticks([-500, 0, 800])
    ax.set_yticks(np.arange(n_bands))
    ax.set_yticklabels([r'$\theta$', r'$\alpha$', r'L$\beta$', r'H$\beta$', r'$\gamma$', r'H$\gamma$'])
    ax.set_ylabel('Frequency band', fontsize=axes_label_fontsize)
    ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axvline(0, 0, 1, linestyle='--', c='black', lw=1)
    ax.tick_params(labelsize=ticklabel_fontsize)


def plot_datapoints(data, slices, rate, block_path, axes):
    n_elects, n_time = data.shape
    cs = make_colors(n_elects)
    cfs = bands.chang_lab['cfs']
    stds = bands.chang_lab['sds']

    minf = bands.neuro['min_freqs'][-1]
    maxf = bands.neuro['max_freqs'][-1]
    filters = [gaussian(data, rate, c, s) for c, s in zip(cfs, stds) if ((c >= minf) and (c <= maxf))]
    ht = abs(hilbert_transform(data, rate, filters)).mean(axis=0)
    ht, m, s = zscore(ht, mode='file', sampling_freq=400., block_path=block_path)
    for ii, (ax, sl) in enumerate(zip(axes, slices)):
        all_y = ht[:, sl] / 3
        x = np.linspace(-500, 800, all_y.shape[-1])
        for jj in range(n_elects):
            y = all_y[jj]
            ax.plot(x, y + jj, c=cs[jj], lw=1.5)
        ax.set_xticks([-500, 0, 800])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        if ii == 0:
            ax.set_ylabel(r'Electrodes H$\gamma$')
        if ii == (len(slices)-1):
            ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
            ax.set_xticks([-500, 0, 800])
        else:
            ax.set_xticks([])
        ax.axvline(0, 0, 1, linestyle='--', c='black', lw=1)
    ax.tick_params(labelsize=ticklabel_fontsize)


def plot_network(n_nodes, n_layers, ax):
    xs = np.linspace(.1, .9, n_layers)
    ys = np.linspace(.1, .9, n_nodes)
    r = min((.9 - .15) / (2 * n_nodes), (.9 - .15) / (2 * n_layers))
    alpha = .3
    for y1 in ys:
        for y2 in ys:
            if abs(y2 - y1) < .5:
                for ii, x1 in enumerate(xs[:-1]):
                    x2 = xs[ii + 1]
                    ax.plot([x1 + r, x2 - r], [y1, y2], 'k', alpha=alpha,lw=.5)
    for y in np.linspace(.1, .9, n_nodes):
        for x in xs:
            ax.add_artist(Circle((x , y), r, fill=False, lw=.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.tick_params(labelsize=ticklabel_fontsize)


def make_bracket(l_ys, r_ys, ax):
    lw = 2
    min_y = min(min(l_ys), min(r_ys))
    max_y = max(max(l_ys), max(r_ys))
    ax.plot([0, 0], [min_y, max_y], 'k', clip_on=False, lw=lw)
    for y in l_ys:
        ax.plot([-.1, 0], [y, y], 'k', clip_on=False, lw=lw)
    for y in r_ys:
        ax.plot([0, .1], [y, y], 'k', clip_on=False, lw=lw)
    ax.axis('off')
