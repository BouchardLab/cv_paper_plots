import itertools
import numpy as np
import matplotlib.pyplot as plt

from .style import (subjects, subject_labels, subject_colors,
                    axes_label_fontsize, ticklabel_fontsize)


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
        plot(ax, features, labels)
    return (cvs, labels, features)

def get_phonetic_feature_matrix(ax=None):
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
    if ax is not None:
        plot(ax, features, labels)
    return (cvs, labels, pmv, features)


def plot(ax, features, labels):
    consonants = ['b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r',
                  's', r'$\int$', 't', r'$\theta$', 'v', 'w', 'j', 'z']
    n = len(labels)
    ax.imshow(features.T, cmap='gray', interpolation='nearest')
    ax.set_xticks(range(57))
    ax.set_xticklabels(19 * ['/a/', '/i/', '/u/'],
                       fontsize=ticklabel_fontsize-1)
    for ii, label in enumerate(ax.xaxis.get_major_ticks()):
        label.set_pad(5*((ii % 3) - 1) + 6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=ticklabel_fontsize-1)
    lw = 1.
    for x in np.linspace(.5, 55.5, 56):
        ax.axvline(x, 0, 1, c='gray', lw=lw)
    for y in np.linspace(.5, n-1.5, n-1):
        ax.axhline(y, 0, 1, c='gray', lw=lw)
    for ii in range(19):
        hbracket(ax, ii*3-.25, ii*3+2.25, -.5, consonants[ii])
    vbracket(ax, 56.75, -.25, 5.25, 'Major\nArticulator')
    vbracket(ax, 56.75, 5.75, 14.25, 'Constriction\nLocation')
    vbracket(ax, 56.75, 14.75, 20.25, 'Constriction\nDegree')
    vbracket(ax, 56.75, 20.75, 24.25, 'Vowel')


def hbracket(ax, x0, x1, y, text):
    fraction = .75 / (x1 - x0)
    ax.annotate("", xy=(x0, y), xycoords='data',
    xytext=(x1, y), textcoords='data',
    arrowprops=dict(arrowstyle="-", ec='k',
    connectionstyle="bar,fraction={}".format(fraction)),
    annotation_clip=False)
    ax.text(.5 * (x0 + x1), y-1.5, '/{}/'.format(text),
            fontsize=ticklabel_fontsize-1,
            verticalalignment='center',
            horizontalalignment='center')

def vbracket(ax, x, y0, y1, text):
    fraction = .75 / (y1 - y0)
    ax.annotate("", xy=(x, y0), xycoords='data',
    xytext=(x, y1), textcoords='data',
    arrowprops=dict(arrowstyle="-", ec='k',
    connectionstyle="bar,fraction={}".format(fraction)),
    annotation_clip=False)
    ax.text(x+1.5, .5 * (y0 + y1), text, fontsize=ticklabel_fontsize-1,
            verticalalignment='center')
