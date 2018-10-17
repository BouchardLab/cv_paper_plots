import h5py, os, pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm

from ecog.utils import bands
from cv_paper_plots.style import (subject_colors, letter_fontstyle,
                                  ticklabel_fontstyle, subject_labels)

import scipy as sp
from cv_paper_plots import xfreq_analysis as xfa
from cv_paper_plots import broadband
from cv_paper_plots.xfreq_analysis import good_examples_and_channels
from cv_paper_plots.broadband import (forward_bl, invert_bl, new_ch_idx, get_pcs, flip,
                                      baseline_mean_std, plot_PC1s, load_data, log_log_robust_regression)

from mpi4py import MPI
from datetime import datetime
time_start = datetime.now()

_np2mpi = {np.dtype(np.float32): MPI.FLOAT,
           np.dtype(np.float64): MPI.DOUBLE,
           np.dtype(np.int): MPI.LONG,
           np.dtype(np.intc): MPI.INT}

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


kind_names = {0: 'RANSAC', 1: 'TheilSen', 2: 'Huber'}
kind = 2
folder = os.path.join(os.environ['HOME'],'Development/data/ecog/AA_ff/')
subjects = ['ec2', 'ec9', 'gp31', 'gp33']
files = ['EC2_blocks_1_8_9_15_76_89_105_CV_AA_ff_align_window_-0.5_to_0.79_none.h5',
         'EC9_blocks_15_39_46_49_53_60_63_CV_AA_ff_align_window_-0.5_to_0.79_none.h5',
         'GP31_blocks_1_2_4_6_9_21_63_65_67_69_71_78_82_83_CV_AA_ff_align_window_-0.5_to_0.79_none.h5',
         'GP33_blocks_1_5_30_CV_AA_ff_align_window_-0.5_to_0.79_none.h5']
freqs = bands.chang_lab['cfs']

subject_idx = 0
subject = subjects[subject_idx]

if rank == 0:
    fname = os.path.join(folder, files[subject_idx])
    X, baselines, good_examples, good_channels, tokens, block_labels, labels = load_data(fname)
    bl_mean, bl_std = baseline_mean_std(block_labels, good_channels, baselines)
    shape = X.shape
    n_idxs = shape[1]
    n_iter = int(np.ceil(n_idxs / size))
    print(n_idxs, n_iter, size)
    X_bls = np.full(shape[1:] + (2,), np.nan)
    X_medR2 = np.full(shape[1:], np.nan)
    print('load', datetime.now() - time_start)
else:
    n_iter = None
    n_idxs = None
    shape = None
    X_bls = None
    X_medR2 = None
n_iter = comm.bcast(n_iter, 0)
n_idxs = comm.bcast(n_idxs, 0)
shape = comm.bcast(shape, 0)
nf = freqs.size
sub_shape = shape[2:]
print(shape, sub_shape, n_idxs)

for ii in range(n_iter):
    start = ii * size
    end = min(start+size, n_idxs)
    # Split data
    send_x = None
    recv_x = np.empty((nf,) + sub_shape)
    if rank == 0:
        send_x = np.empty((size, nf) + sub_shape)
        axes = list(range(send_x.ndim))
        axes[0:2] = [1, 0]
        send_x[:end-start] = np.transpose(X[:, start:end], axes=axes)
    comm.Scatter(send_x, recv_x, 0)
    # Fit
    recv_bls = None
    recv_medR2 = None
    if rank == 0:
       recv_bls = np.full((size,) + sub_shape + (2,), np.nan)
       recv_medR2 = np.full((size,) + sub_shape, np.nan)
    if start + rank < end:
        print(start+rank)
        send_bls, send_medR2 = broadband.calculate_baselines(freqs, recv_x, kind=kind)
    else:
        send_bls = np.full(sub_shape + (2,), np.nan)
        send_medR2 = np.full(sub_shape, np.nan)
    comm.Gather(send_bls, recv_bls, 0)
    if rank == 0:
        print(float(ii)*size / n_idxs, datetime.now() - time_start)
        X_bls[start:end] = recv_bls[:end-start]

if rank == 0:
    np.savez('{}_baselines'.format(subject), X_bls=X_bls, X_medR2=X_medR2, kind=kind)
