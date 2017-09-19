import pickle, os

import numpy as np

from cv_paper_plots.analysis import consonant_equiv, vowel_equiv, place_equiv, manner_equiv

rng = np.random.RandomState(20170801)
subjects = ['ec2', 'ec9', 'gp31', 'gp33']
fracs = [.5, .6, .7, .8, .9, 1.]
folds = list(range(10))
n_trials = 10

chance_data = np.zeros((5, len(subjects), len(fracs), len(folds), n_trials))
training_size = np.zeros((len(subjects), len(fracs), len(folds)))

with open(os.path.join(os.environ['HOME'],
                       'plots/ds/data/dataset_summary.pkl'), 'rb') as f:
    train_subjects, test_subjects = pickle.load(f, encoding='latin1')

for ii, s in enumerate(subjects):
    for jj, f in enumerate(fracs):
        for fold in folds:
            for t in range(n_trials):
                train = train_subjects[s][f][fold].ravel()
                test = test_subjects[s][f][fold].ravel()
                train_pdf, _ = np.histogram(train, np.arange(0, 58), density=True)
                yh = rng.multinomial(1, train_pdf, test.size).argmax(axis=1)
                d = np.array(test == yh)
                chance_data[0, ii, jj, fold, t] = np.nanmean(d)
                d = np.array([consonant_equiv(test_y, yh_y) for test_y, yh_y in zip(test, yh)])
                chance_data[1, ii, jj, fold, t] = np.nanmean(d)
                d = np.array([vowel_equiv(test_y, yh_y) for test_y, yh_y in zip(test, yh)])
                chance_data[2, ii, jj, fold, t] = np.nanmean(d)
                d = np.array([place_equiv(test_y, yh_y) for test_y, yh_y in zip(test, yh)])
                chance_data[3, ii, jj, fold, t] = np.nanmean(d)
                d = np.array([manner_equiv(test_y, yh_y) for test_y, yh_y in zip(test, yh)])
                chance_data[4, ii, jj, fold, t] = np.nanmean(d)

            training_size[ii, jj, fold] = train.size

np.savez(os.path.join(os.environ['HOME'], 'plots/ds/data/dataset_chance.npz'),
         **{'chance_data': chance_data, 'training_size': training_size})
