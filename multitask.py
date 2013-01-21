
__version__ = '0.1'

from scipy import linalg
from scipy.sparse import linalg as splinalg

def multitask_ridge(X, y, alpha, beta, shape_B, rtol=1e-5, verbose=False, warm_start=None):
    """
    Multitask ridge model (refs ?)

    MSE + alpha ||w_mean||^2 + beta ||w - w_mean||^2

    Parameters
    ----------
    X: {array, sparse matrix, LinearOperator}
    y : array
    alpha: float
    beta: float
    shape_B: tuple of size (2,)
        Contains the desired shape of the output matrix
    """
    n_subj = shape_B[1]

    X = splinalg.aslinearoperator(X)

    def matvec(z):
        w_mean = np.mean(z.reshape(shape_B, order='F'), 1)
        return X.rmatvec(X.matvec(z)) + (beta) * z + (alpha - beta) * np.tile(w_mean, n_subj)

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
