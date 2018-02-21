import numpy as np
import scipy.stats as stats
from scipy.stats import linregress

def permute_paired_diffs(x, y, n_perm=10000):
    x = x.ravel()
    y = y.ravel()
    assert x.size == y.size
    f = lambda x, y: abs(y.mean() - x.mean())
    value = f(x, y)
    b_values = np.zeros(n_perm)
    for ii in range(n_perm):
        switch = np.random.binomial(1, .5, size=x.size)
        xp = switch * x + (1-switch) * y
        yp = (1-switch) * x + switch * y
        b_values[ii] = f(xp, yp)
    return value, b_values, (value <= b_values).mean()

def permute_diffs(x, y, n_perm=10000):
    x = x.ravel()
    y = y.ravel()
    f = lambda x, y: abs(y.mean() - x.mean())
    value = f(x, y)
    b_values = np.zeros(n_perm)
    combine = np.catenate([x, y])
    for ii in range(n_perm):
        shuffle = np.random.permutation(combine)
        xp = shuffle[:x.size]
        yp = shuffle[x.size:]
        b_values[ii] = f(xp, yp)
    return value, b_values, (value <= b_values).mean()

def permute_paired_regression(x, y0, y1, n_perm=10000):
    assert x.shape == y0.shape
    assert y0.shape == y1.shape

    def f(x, y0, y1):
        s0 = linregress(x.ravel(), y0.ravel())[0]
        s1 = linregress(x.ravel(), y1.ravel())[0]
        return abs(s0-s1)

    value = f(x, y0, y1)
    b_values = np.zeros(n_perm)
    for ii in range(n_perm):
        switch = np.random.binomial(1, .5, size=y0.shape)
        y0p = switch * y0 + (1-switch) * y1
        y1p = switch * y1 + (1-switch) * y0
        b_values[ii] = f(x, y0p, y1p)
    return value, b_values, (value <= b_values).mean()

def parametric_slopes_test(m0, s0, n0, m1, s1, n1):
    t = (m0 - m1) / np.sqrt(s0**2 + s1**2)
    n = n0 + n1 - 4
    return stats.t.sf(abs(t), n)
