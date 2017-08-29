import numpy as np

def plot_cv_accuracy(deep, linear, chance, labels, keys, colors, ax):
    n_subjects, _, n_iter = deep[keys[0]].shape
    data = np.zeros((2, n_subjects, n_iter))
    key = keys[-1]
    deepi = deep[key]
    lineari = linear[key]
    for ii, label in enumerate(labels):
        data[0, ii] = lineari[ii, 2]
        data[1, ii] = deepi[ii, 2]

    for ii, (label, c) in enumerate(zip(labels, colors)):
        ax.errorbar([0, 1], data[:, ii].mean(axis=1)/chance[ii],
                    yerr=data[:,ii].std(axis=1)/np.sqrt(n_iter)/chance[ii],
                    c=c, label=label)
    ax.legend(loc='best')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear', 'Deep'])
    ax.set_xlim(-.5, 1.5)
    ax.axhline(1, c='gray', linestyle='--')




"""
rcParams.update({'figure.autolayout': True,
                 'font.size': 18})

fig = plt.figure(figsize=(7, 5))
for ii in range(4):
    label = 'Subject '+str(ii+1)
    plt.plot(0, 0, '-', c=subj_colors[ii], label=label)
plt.plot(0, 0, '^', c='gray', label='Linear')
plt.plot(0, 0, 'o', c='gray', label='Deep')
for ii, (linear_fname, deep_fname, random_fname) in enumerate(zip(linear_files,
                                                                  deep_files,
                                                                  random_files)):
    path = os.path.join(folder, linear_fname)
    (lc, lv, lcv, _, lp, lm) = load_data(path)
    path = os.path.join(folder, deep_fname)
    (dc, dv, dcv, _, dp, dm) = load_data(path)
    path = os.path.join(folder, random_fname)
    (rc, rv, rcv, _, rp, rm) = load_data(path)

    for jj, (l, d, r) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [rcv, rc, rv, rp, rm])):
        if jj >= 1 and jj < 3:
            chance = np.nanmean(r)
            plt.errorbar(jj+1, np.nanmean(l)/chance, fmt='^',
                         yerr=np.nanstd(l)/chance,
                         c=subj_colors[ii])
            plt.errorbar(jj+1.25, np.nanmean(d)/chance, fmt='o',
                         yerr=np.nanstd(d)/chance,
                         c=subj_colors[ii])
            plt.plot([jj+1, jj+1.25], [np.nanmean(l)/chance, np.nanmean(d)/chance],
                     '-', c=subj_colors[ii], lw=2)
            plt.plot([jj+1, jj+1.25], [1/chance, 1/chance],
                    ':', c=subj_colors[ii], lw=2)
plt.plot([0, 10], [1, 1], '--', c='gray', lw=1, label='Chance')
plt.plot([0, 0], [0, 0], ':', c='gray', lw=1, label='Max')
plt.xticks(np.arange(1, 3)+1.125, ['Cons.', 'Vowel'])
plt.xlim([1.75, 3.5])
plt.ylim([.7, 23])
plt.legend(bbox_to_anchor=(1, 1), prop={'size': 14},
           loc='upper right')
plt.savefig('linear_vs_deep_accuracy2.pdf')
plt.savefig('linear_vs_deep_accuracy2.png')
plt.close()

fig = plt.figure(figsize=(7, 3))
for ii in range(4):
    label = 'Subject '+str(ii+1)
    plt.plot(0, 0, '-', c=subj_colors[ii], label=label)
plt.plot(0, 0, '^', c='gray', label='Linear')
plt.plot(0, 0, 'o', c='gray', label='Deep')
for ii, (linear_fname, deep_fname, random_fname) in enumerate(zip(linear_files,
                                                                  deep_files,
                                                                  random_files)):
    path = os.path.join(folder, linear_fname)
    (lc, lv, lcv, _, lp, lm) = load_data(path)
    path = os.path.join(folder, deep_fname)
    (dc, dv, dcv, _, dp, dm) = load_data(path)
    path = os.path.join(folder, random_fname)
    (rc, rv, rcv, _, rp, rm) = load_data(path)

    for jj, (l, d, r) in enumerate(zip([lcv, lc, lv, lp, lm],
                                       [dcv, dc, dv, dp, dm],
                                       [rcv, rc, rv, rp, rm])):
        if jj >= 3:
            chance = np.nanmean(r)
            plt.errorbar(jj+1, np.nanmean(l)/chance, fmt='^',
                         yerr=np.nanstd(l)/chance,
                         c=subj_colors[ii])
            plt.errorbar(jj+1.25, np.nanmean(d)/chance, fmt='o',
                         yerr=np.nanstd(d)/chance,
                         c=subj_colors[ii])
            plt.plot([jj+1, jj+1.25], [np.nanmean(l)/chance, np.nanmean(d)/chance],
                     '-', c=subj_colors[ii], lw=2)
            plt.plot([jj+1, jj+1.25], [1/chance, 1/chance],
                    ':', c=subj_colors[ii], lw=2)
plt.plot([2.5, 5.75], [1, 1], '--', c='gray', lw=1, label='Chance')
plt.xticks(np.arange(3, 5)+1.125, ['Place', 'Manner'])
plt.xlim([3.5, 5.75])
plt.ylim([.7, 3.4])
plt.savefig('linear_vs_deep_accuracy3.pdf')
plt.savefig('linear_vs_deep_accuracy3.png')
plt.close()
"""
