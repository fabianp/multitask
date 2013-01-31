import numpy as np
from scipy import linalg
from numpy import testing

from multitask import trace

def test_1():
    B = np.ones((3, 3))
    X = np.random.randn(100, 9)
    y = np.dot(X, B.ravel('F')) + .1 * np.random.randn(100)

    alpha = 10.
    B_, _ = trace(X, y, alpha, 0., (3, 3), rtol=1e-10)

    # KKT conditions
    grad = - np.dot(X.T, y - np.dot(X, B_.ravel('F')))
    M = (grad / alpha).reshape(B.shape, order='F')
    assert np.all(linalg.svdvals(M) < 1. + 1e-3)
    testing.assert_allclose(np.dot(M.ravel('F'), B_.ravel('F')),
        - linalg.svdvals(B_).sum())



