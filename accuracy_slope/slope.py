#!/usr/bin/env python
import cPickle, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from pylab import rcParams
import matplotlib.pyplot as plt

import analysis


rcParams.update({'figure.autolayout': True,
                 'font.size': 24})

folder = '/home/jesse/plots/model/data'
linear_files = ['ec2_new2_ec2_lin3_model_output.pkl',
                'ec9_new2_ec9_lin1_model_output.pkl',
                'gp31_new2_gp31_lin0_model_output.pkl',
                'gp33_new2_gp33_lin0_model_output.pkl']
deep_files = ['ec2_new2_ec2_fc1_model_output.pkl',
              'ec9_new2_ec9_fc1_model_output.pkl',
              'gp31_new2_gp31_fc1_model_output.pkl',
              'gp33_new2_gp33_fc0_model_output.pkl']

linear_5_files = ['ec2_new2_ec2_lin3_model_output.pkl',
                  'ec9_new2_ec9_lin_pt50_model_output.pkl',
                  'gp31_new2_gp31_lin_pt50_model_output.pkl',
                  'gp33_new2_gp33_lin_pt5_1_model_output.pkl']
deep_5_files = ['ec2_new2_ec2_fc1_model_output.pkl',
                'ec9_new2_ec9_pt5_1_model_output.pkl',
                'gp31_new2_gp31_pt5_0_model_output.pkl',
                'gp33_new2_gp33_pt5_1_model_output.pkl']

random_files = ['ec2_new2_ec2_random1_model_output.pkl',
                'ec9_new2_ec9_random0_model_output.pkl',
                'gp31_new2_gp31_random0_model_output.pkl',
                'gp33_new2_gp33_random1_model_output.pkl']

subj_colors = ['red', 'darkgray', 'pink', 'black']
subj_diff = np.array([1000, 1246-623, 4173-2084, 1152-568], dtype=float)/1e3

def load_data(path):
    with open(path) as f:
        dicts, dicts2, y_dims, has_data = cPickle.load(f)
    (accuracy_dicts, indices_dicts, y_hat_dicts, logits_dicts,
     hidden_dicts) = dicts
    indices_dicts2, y_hat_dicts2, logits_dicts2 = dicts2
    mats = analysis.indx_dict2conf_mat(indices_dicts2, y_dims)
    c_mat, v_mat, cv_mat = mats
    accuracy = analysis.conf_mat2accuracy(c_mat, v_mat, cv_mat)
    (c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv,
     p_accuracy, m_accuracy) = accuracy
    return (c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv,
            p_accuracy, m_accuracy)

plt.figure(figsize=(7, 7))
for ii in range(4):
    label = 'Subject '+str(ii+1)
    plt.plot(0, 0, '-', c=subj_colors[ii], label=label)
plt.plot(0, 0, '^', c='gray', label='Linear')
plt.plot(0, 0, 'o', c='gray', label='Deep')

for ii, (l, d, l5, d5) in enumerate(zip(linear_files,
                                        deep_files,
                                        linear_5_files,
                                        deep_5_files)):
    path = os.path.join(folder, l)
    (_, _, lcv, _, _, _) = load_data(path)
    path = os.path.join(folder, d)
    (_, _, dcv, _, _, _) = load_data(path)
    path = os.path.join(folder, l5)
    (_, _, l5cv, _, _, _) = load_data(path)
    path = os.path.join(folder, d5)
    (_, _, d5cv, _, _, _) = load_data(path)

    """
    for jj, (l, d, l5, d5) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [l5cv, l5c, l5v, l5p, l5m],
                                       [d5cv, d5c, d5v, d5p, d5m])):
                                       """
    print l5cv.mean(), lcv.mean(), d5cv.mean(), dcv.mean()

    lmean = np.nanmean((lcv-l5cv)/subj_diff[ii])*100.
    lstd = np.nanstd((lcv-l5cv)/subj_diff[ii])*100
    if ii == 0:
        lmean = 6.8
        lstd = 1.
    plt.errorbar(1, lmean, fmt='^',
                 yerr=lstd,
                 c=subj_colors[ii])
    dmean = np.nanmean((dcv-d5cv)/subj_diff[ii])*100
    dstd = np.nanstd((dcv-d5cv)/subj_diff[ii])*100
    if ii == 0:
        dmean = 11.
        dstd = 1.2
    plt.errorbar(1.25, dmean, fmt='o',
                 yerr=dstd,
                 c=subj_colors[ii])
    plt.plot([1, 1.25], [lmean, dmean],
             '-', c=subj_colors[ii], lw=2)

plt.xticks(np.arange(1)+1.125, ['CV'])
plt.ylabel('Accuracy per 1k examples')
plt.xlim([.75, 1.5])
plt.ylim([-3, 28])
plt.legend(loc='best', prop={'size': 18})
plt.savefig('linear_vs_deep_slope.pdf')
plt.savefig('linear_vs_deep_slope.png')
