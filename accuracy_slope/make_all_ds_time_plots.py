#!/usr/bin/env python

import copy, os, h5py, argparse
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cv_paper_plots.style import axes_label_fontsize, ticklabel_fontsize


folder = '/home/jesse/plots/ds/data'

data = ['ec2_cv_time_indep.npz',
        'ec9_cv_time_indep.npz',
        'gp31_cv_time_indep.npz',
        'gp33_cv_time_indep.npz']
r_data = ['ec2_scv_time_indep.npz',
          'ec9_scv_time_indep.npz',
          'gp31_scv_time_indep.npz',
          'gp33_scv_time_indep.npz']

c_all = []
v_all = []

x = np.arange(-100*5,158*5, 5)
fig = plt.figure(figsize=(2.5, 1.425))
l_edge = .2
r_edge = .05
b_edge = .275
t_edge = .05
width = 1. - l_edge - r_edge
height = 1. - b_edge - t_edge
ax = fig.add_axes([l_edge, b_edge, width, height])

for ii, (d, rd) in enumerate(zip(data, r_data)):

    data_fname = os.path.join(folder, d)
    with np.load(data_fname) as f:
        c, v = f['c_ita'], f['v_ita']

    data_fname = os.path.join(folder, rd)
    with np.load(data_fname) as f:
        rc, rv = f['sc_ita'], f['sv_ita']

    n_data = c.shape[0]
    c_mean = c.mean(axis=0)
    c_std = c.std(axis=0)
    v_mean = v.mean(axis=0)
    v_std = v.std(axis=0)
    assert n_data > 0

    rc_mean = rc.mean()
    rv_mean = rv.mean()

    c_all.append(c/rc_mean)
    v_all.append(v/rv_mean)

    if ii == 0:
        c_label = 'Single subj. consonant'
    else:
        c_label = None
    if ii == 0:
        v_label = 'Single subj. vowel'
    else:
        v_label = None

    """
    p = ax.fill_between(x, (c_mean-c_std/np.sqrt(n_data))/rc_mean,
                        (c_mean+c_std/np.sqrt(n_data))/rc_mean,
                     facecolor='black', edgecolor='black',
                     alpha=.4, label=c_label)
    if c_label is not None:
        legend.append(p)

    p = ax.fill_between(x, (v_mean-v_std/np.sqrt(n_data))/rv_mean,
                        (v_mean+v_std/np.sqrt(n_data))/rv_mean,
                     facecolor='red', edgecolor='red',
                     alpha=.4, label=v_label)
    if v_label is not None:
        legend.append(p)
    """


c_all = np.vstack(c_all)
v_all = np.vstack(v_all)
n_data = c_all.shape[0]

c_mean = c_all.mean(axis=0)
c_std = c_all.std(axis=0)
v_mean = v_all.mean(axis=0)
v_std = v_all.std(axis=0)

c_label = 'Consonant'
ax.fill_between(x, c_mean-c_std/np.sqrt(n_data),
                              c_mean+c_std/np.sqrt(n_data),
                 facecolor='black', edgecolor='black',
                 label=c_label)

v_label = 'Vowel'
ax.fill_between(x, v_mean-v_std/np.sqrt(n_data),
                              v_mean+v_std/np.sqrt(n_data),
                 facecolor='gray', edgecolor='gray',
                 label=v_label)
plt.plot(x, np.ones_like(x), '-', color='gray', lw=1)

ax.axvline(0, 1 ,0, linestyle='--', color='gray', lw=1)
ax.set_xlabel('Time (ms)', fontsize=axes_label_fontsize)
ax.set_ylabel('Accuracy/chance', fontsize=axes_label_fontsize)
ax.set_xlim(x.min(),x.max())
ax.set_xticks([x.min(),0, x.max()])
ax.set_xticklabels([-500, 0, 800])
ax.set_ylim(.75, 3)
ax.set_yticks(np.arange(1, 4))
ax.set_yticklabels(np.arange(1, 4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(labelsize=ticklabel_fontsize)
ax.legend(loc='best', borderaxespad=0, prop={'size': ticklabel_fontsize})
plt.savefig(os.path.join(os.environ['HOME'], 'Downloads/time_accuracy_all.pdf'))
plt.savefig(os.path.join(os.environ['HOME'], 'Downloads/time_accuracy_all.png'), dpi=300)
