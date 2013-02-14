# .. License: Simplified BSD ..
# .. Author: Fabian Pedregosa <fabian@fseoane.net> ..

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg

def ridge(X, y, alpha, beta, n_task, method='iterative', M=None, rtol=1e-5, verbose=False, warm_start=None):
    """
    Multitask ridge model (refs ?)

    MSE + alpha ||w_mean||^2 + beta ||w - w_mean||^2

    Parameters
    ----------
    X: {array, sparse matrix, LinearOperator}
    y : array
    alpha: float
    beta: float
    method: {'iterative', 'direct'}
    shape_B: tuple of size (2,)
        Contains the desired shape of the output matrix

    Returns
    -------
    B : array, shape = shape_B
    """
    m = X.shape[1] // n_task
    assert X.shape[1] == n_task * m # check that division is integer
    shape_B = (X.shape[1] / n_task, n_task)

    X = splinalg.aslinearoperator(X)

    if M is None:
        def matvec(z):
            w_mean = np.mean(z.reshape(shape_B, order='F'), 1)
            return X.rmatvec(X.matvec(z)) + (beta) * z + (alpha - beta) * np.tile(w_mean, n_task)
    else:
        M_kron = sparse.kron(sparse.eye(m, m), M)
        def matvec(z):
            return X.rmatvec(X.matvec(z)) + M_kron.dot(z)

    K = splinalg.LinearOperator((X.shape[1], X.shape[1]), matvec=matvec, rmatvec=matvec, dtype=X.dtype)
    Xy = X.rmatvec(y)
    if warm_start is not None:
        warm_start = warm_start.ravel('F')
    def f(t):
        if verbose:
            print('RTOL: %s' % linalg.norm(K.matvec(t) - Xy, 2))

    sol, info = splinalg.cg(K, Xy, tol=rtol, x0=warm_start, callback=f)
    B = sol.reshape(shape_B, order='F')
    return B
