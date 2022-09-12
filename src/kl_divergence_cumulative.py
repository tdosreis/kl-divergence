import numpy as np


def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True)
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n

    def interpolate_(x_):
        yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate_


def cumulative_kl(x, y, fraction=0.5):
    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    ex = np.min(dx)
    ey = np.min(dy)
    e = np.min([ex, ey])*fraction
    n = len(x)
    P = ecdf(x)
    Q = ecdf(y)
    KL = (1.0/n)*np.sum(np.log((P(x) - P(x-e))/(Q(x) - Q(x-e))))
    return KL
