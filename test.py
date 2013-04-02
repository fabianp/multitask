import numpy as np
from scipy import linalg
from scipy.sparse import linalg as splinalg
from numpy import testing

import multitask as mt

def test_trace_1():
    B = np.ones((3, 3))
    X = np.random.randn(100, 9)
    y = np.dot(X, B.ravel('F')) + .1 * np.random.randn(100)

    alpha = 10.
    B_, _ = mt.trace(X, y, alpha, 0., (3, 3), rtol=1e-10)

    # KKT conditions
    grad = - np.dot(X.T, y - np.dot(X, B_.ravel('F')))
    M = (grad / alpha).reshape(B.shape, order='F')
    assert np.all(linalg.svdvals(M) < 1. + 1e-3)
    testing.assert_allclose(np.dot(M.ravel('F'), B_.ravel('F')),
        - linalg.svdvals(B_).sum())

def test_low_rank_1():
    # when X = identity, solution is the SVD
    X = np.eye(100)
    y = np.random.randn(100)
    u1, v1 = mt.low_rank(X, y, 0, (10, 1))
    u2, s, v2 = splinalg.svds(y.reshape((10, 10), order='F'), 1)
    c = u1[0] / u2[0]
    testing.assert_allclose(u1, c * u2, rtol=1e-3)
    u3, v3 = mt.rank_one(X, y, 0, 10, rtol=1e-12)
    c = u3[0] / u2[0]
    testing.assert_allclose(u3, c * u2, rtol=1e-3)

    y = np.random.randn(100, 2)
    u1, v1 = mt.low_rank(X, y, 0, (10, 1), rtol=1e-12)
    u2, s, v2 = splinalg.svds(y[:, 1].reshape((10, 10), order='F'), 1)
    c = u1[1][0] / u2[0]
    testing.assert_allclose(u1[1], c * u2, rtol=1e-3)
