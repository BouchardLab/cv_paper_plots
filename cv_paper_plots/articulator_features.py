import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_articulator_state_matrix(ax=None):
    consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
    vowels = sorted(['aa', 'ee', 'oo'])
    assert len(set(consonants)) == 19
    assert len(set(vowels)) == 3
    cvs = sorted([c+v for c, v in itertools.product(consonants, vowels)])
    assert len(set(cvs)) == 57
    labels = ['lips', 'tongue', 'larynx', 'jaw', 'back tounge', 'high tongue']
    
    features = np.zeros((57, 6), dtype=int)
    # b
    features[:3, (0, 2, 3)] = 1
    # d
    features[3:6, (1, 2, 3)] = 1
    # f
    features[6:9, (0, 3)] = 1
    # g
    features[9:12, (1, 2)] = 1
    # h
    # None
    # k
    features[15:18, 1] = 1
    # l
    features[18:21, (1, 2, 3)] = 1
    # m
    features[21:24, (0, 2, 3)] = 1
    # n
    features[24:27, (1, 2, 3)] = 1
    # p
    features[27:30, (0, 3)] = 1
    # r
    features[30:33, (1, 2, 3)] = 1
    # s
    features[33:36, (1, 3)] = 1
    # sh
    features[36:39, (1, 3)] = 1
    # t
    features[39:42, (1, 3)] = 1
    # th
    features[42:45, (1, 3)] = 1
    # v
    features[45:48, (0, 2, 3)] = 1
    # w
    features[48:51, (0, 1, 2, 3)] = 1
    # y
    features[51:54, (1, 2, 3)] = 1
    # z
    features[54:, (1, 2, 3)] = 1

    # Vowels
    # aa
    features[::3, 4] = 1
    # ee
    features[1::3, 5] = 1
    # oo
    features[2::3, (4, 5)] = 1
    if ax is not None:
        ax.imshow(features.T, cmap='gray', interpolation='nearest')
        ax.set_xticks(range(57))
        ax.set_xticklabels(19 * ['\\a\\', '\\i\\', '\\u\\'])
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels)
        for x in np.linspace(.5, 56.5, 57):
            ax.axvline(x, 0, 1, c='gray')
        for y in np.linspace(.5, 4.5, 5):
            ax.axhline(y, 0, 1, c='gray')
    return (cvs, labels, features)

def get_phonetic_feature_matrix():
    consonants = sorted(['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p',
                         'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'])
    vowels = sorted(['aa', 'ee', 'oo'])
    assert len(set(consonants)) == 19
    assert len(set(vowels)) == 3
    cvs = sorted([c+v for c, v in itertools.product(consonants, vowels)])
    assert len(set(cvs)) == 57
    labels = ['bilabial', 'secondary labial', 'labiodental', 'dental',
              'alveolar', 'post alveolar', 'velar', 'voiced', 'mandibular',
              'oral stop', 'fricative', 'approximate', 'nasal stop',
              'lateral', 'rhotic', 'back tongue', 'high tongue',
              'lip rounding', 'jaw open']
    assert len(set(labels)) == 19
    pmv = {'place': slice(0, 9), 'manner': slice(9, 15), 'vowel': slice(15, 19)}
    
    features = np.zeros((57, 19), dtype=int)
    # b
    features[:3, (0, 7, 8, 9)] = 1
    # d
    features[3:6, (4, 7, 8, 9)] = 1
    # f
    features[6:9, (2, 8, 10)] = 1
    # g
    features[9:12, (6, 7, 9)] = 1
    # h
    # None
    # k
    features[15:18, (6, 9)] = 1
    # l
    features[18:21, (4, 7, 8, 13)] = 1
    # m
    features[21:24, (0, 7, 8, 12)] = 1
    # n
    features[24:27, (4, 7, 8, 9, 12)] = 1
    # p
    features[27:30, (0, 8, 9)] = 1
    # r
    features[30:33, (1, 5, 7, 8, 11, 14)] = 1
    # s
    features[33:36, (4, 8, 10)] = 1
    # sh
    features[36:39, (1, 5, 8, 10)] = 1
    # t
    features[39:42, (4, 8, 9)] = 1
    # th
    features[42:45, (3, 8, 10)] = 1
    # v
    features[45:48, (2, 7, 8, 10)] = 1
    # w
    features[48:51, (0, 6, 7, 8, 11)] = 1
    # y
    features[51:54, (5, 7, 8, 11)] = 1
    # z
    features[54:, (4, 7, 8, 10)] = 1

    # Vowels
    # aa
    features[::3, (15, 18)] = 1
    # ee
    features[1::3, 16] = 1
    # oo
    features[2::3, (15, 16, 17)] = 1
    import matplotlib.pyplot as plt
    plt.imshow(features.T, cmap='gray', interpolation='nearest')
    return (cvs, labels, pmv, features)
