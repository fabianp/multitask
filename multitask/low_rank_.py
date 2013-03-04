# encoding: utf-8
# .. License: Simplified BSD ..
# .. Author: Fabian Pedregosa <fabian@fseoane.net> ..

import numpy as np
from scipy import sparse, linalg, optimize
from scipy.sparse import linalg as splinalg


def low_rank(X, y, alpha, shape_u, Z=None, prior_u=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
    """
    Computes the model

         min_{u, v} || y - X vec(u v.T) - Z w||_2 ^2 + alpha ||u - u0||_2 ^2

    subject to ||u|| = 1.

    Parameters
    ----------
    X : {array. sparse matrix, LinearOpeator}, shape = [n_samples, n_features]
    y : array-like, shape = [n_samples] or [n_samples, n_targets]
    prior_u: array, shape=shape_u

    maxiter : number of iterations
       set to infinity to iterate until convergence

    Returns
    -------
    """
    X = splinalg.aslinearoperator(X)
    y = np.asarray(y)
    assert len(shape_u) == 2
    shape_v = (X.shape[1] / shape_u[0], shape_u[1]) # TODO: check first dimension is integer
    if u0 is None:
        u0 = np.random.randn(*shape_u)
    if v0 is None:
        v0 = np.ones(shape_v) # np.random.randn(shape_B[1])
    w0 = None
    if Z is not None:
        w0 = np.zeros(Z.shape[1])
    if prior_u is None:
        prior_u = np.ones_like(u0).ravel()
    else:
        prior_u = np.array(prior_u).ravel()

    if y.ndim == 1:
        Y = y.reshape((y.size, 1))
    else:
        Y = y
    U, V, W = [], [], []
    for yi in Y.T:
        n_iter = 0
        while n_iter <= maxiter: # this allows to set maxiter to inf
            n_iter += 1
            old_u0 = u0.copy()

            # update v
            v0 = v0.reshape(shape_v)
            size_id = X.shape[1] / shape_v[0]
            Kron_v = sparse.kron(v0, sparse.eye(size_id, shape_u[0] * shape_u[1] / shape_v[1]))
            def K_matvec(z):
                return Kron_v.T.dot(X.rmatvec(X.matvec(Kron_v.dot(z)))) + alpha * z
            K = splinalg.LinearOperator((Kron_v.shape[1], Kron_v.shape[1]), matvec=K_matvec, dtype=X.dtype)
            if Z is None:
                Ky = Kron_v.T.dot(X.rmatvec(yi)) + alpha * prior_u
            else:
                Ky = Kron_v.T.dot(X.rmatvec(yi - np.dot(Z, w0))) + alpha * prior_u
            u0, info = splinalg.cg(K, Ky, x0=u0.ravel(), tol=rtol * 1e-3)
            u0 = u0.reshape(shape_u, order='F')
            u0 = u0 / linalg.norm(u0)

            # update u
            Kron_u = sparse.kron(sparse.eye(X.shape[1] / shape_u[0], shape_v[1] * shape_v[0] / shape_u[1]), u0)
            def K2_matvec(z):
                return Kron_u.T.dot(X.rmatvec(X.matvec(Kron_u.dot(z))))
            K = splinalg.LinearOperator((Kron_u.shape[1], Kron_u.shape[1]), matvec=K2_matvec, dtype=X.dtype)
            if Z is None:
                Ky = Kron_u.T.dot(X.rmatvec(yi))
            else:
                Ky = Kron_u.T.dot(X.rmatvec(yi - np.dot(Z, w0)))
            vt0, info = splinalg.cg(K, Ky, x0=v0.T.ravel(), tol=rtol * 1e-3)
            vt0 = vt0.reshape((shape_v[1], shape_v[0]), order='F')
            v0 = vt0.T

            # update w
            if Z is not None:
                # TODO: cache SVD(Z)
                w0 = linalg.lstsq(Z, yi - X.matvec(np.dot(u0, v0.T).ravel('F')))[0]

            if verbose:
                v0 = v0.reshape(shape_v)
                if Z is None:
                    pobj = np.linalg.norm(yi - X.matvec(np.dot(u0, v0.T).ravel('F'))) ** 2 + \
                           alpha * linalg.norm(u0 - prior_u) ** 2
                else:
                    pobj = np.linalg.norm(yi - X.matvec(np.dot(u0, v0.T).ravel('F')) - np.dot(Z, w0)) ** 2 + \
                           alpha * linalg.norm(u0 - prior_u) ** 2

                print('POBJ: %s' % pobj)

            if linalg.norm(old_u0 - u0, 2) < rtol:
                break
        U.append(u0.copy())
        V.append(v0.copy())
        if Z is not None:
            W.append(w0.copy())
    if y.ndim == 1:
        U = U[0]
        V = V[0]
        if Z is not None:
            W = W[0]
    if Z is None:
        return U, V
    else:
        return U, V, W


def khatri_rao(A, B):
    """
    Compute the Khatri-rao product, where the partition is taken to be
    the vectors along axis one.

    This is a helper function for rank_one

    Parameters
    ----------
    A : array, shape (n, p)
    B : array, shape (m, p)
    AB : array, shape (nm, p), optimal
        if given, result will be stored here

    Returns
    -------
    a*b : array, shape (nm, p)
    """
    num_targets = A.shape[1]
    assert B.shape[1] == num_targets
    return (A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
            ).reshape(num_targets, len(B) * len(A)).T


def CGNR(matmat, rmatmat, b, x0, maxiter=100, rtol=1e-6):
    """
    Parameters
    ----------
    matmat : callable
        matmat(X) returns A.dot(X)
    rmatmat : callable
        rmatmat(X) returns A.T.dot(X)
    b : array of shape (n, k)

    Returns
    -------
    x : approximate solution, shape (n, k)
    r : residual for the normal equation, shape (k,)

    References
    ----------
    Yousef Saad, “Iterative Methods for Sparse Linear Systems, Second Edition”,
    SIAM, pp. 151-172, pp. 272-275, 2003 http://www-users.cs.umn.edu/~saad/books.html
    """

    r = b - matmat(x0)
    z = rmatmat(r)
    k = b.shape[1]
    p = z
    i = 0
    residuals = np.inf
    while i < maxiter and np.any(np.abs(residuals) > b.shape[1] * linalg.norm(x0, 'fro') * rtol):
        i += 1
        w = matmat(p)
        alpha = (z * z).sum(0) / (w * w).sum(0)
        x0 += alpha.reshape((-1, k)) * p
        r -= alpha * w
        z_new = rmatmat(r)
        beta = (z_new * z_new).sum(0) / (z * z).sum(0)
        z = z_new
        residuals = (z * z).sum(0)
        p = z_new + beta.reshape((-1, k)) * p
    return x0, residuals


def rank_one(X, y, alpha, size_u, prior_u=None, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
    """
    multi-target rank one model

        ||y - X vec(u v.T)||_2 ^2

    TODO: prior_u

    Parameters
    ----------
    X : sparse matrix, shape (n, p)
    Y : array-lime, shape (n, k)
    size_u : integer
        Must be divisor of p
    u0 : array
        Initial value for u
    v0 : array
        Initial value for v
    rtol : float
    maxiter : int
        maximum number of iterations
    verbose : bool
        If True, prints the value of the objective
        function at each iteration

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : XXX
    """

    X = splinalg.aslinearoperator(X)
    y = np.asarray(y)
    n_task = y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    # .. some auxiliary functions ..
    # .. used in conjugate gradient ..
    def obj(a, b):
        uv0 = khatri_rao(b, a)
        return .5 * linalg.norm(y - X.matvec(uv0), 'fro') ** 2

    def matmat(X, a, b):
        """
        X (b * a)
        """
        uv0 = khatri_rao(b, a)
        return X.matvec(uv0)

    def rmatmat1(X, a, b):
        """
        (I kron a^T) X^T b
        """
        b = X.rmatvec(b).T
        B = b.reshape((n_task, -1, a.shape[0]), order='F')
        res = np.einsum("ijk, ik -> ij", B, a.T).T
        return res

    def rmatmat2(X, a, b):
        """
        (a^T kron I) X^T b
        """
        b = X.rmatvec(b).T
        B = b.reshape((n_task, -1, a.shape[0]), order='C')
        tmp = np.einsum("ijk, ik -> ij", B, a.T).T
        return tmp

    if u0 is None:
        u0 = np.ones(size_u * n_task)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    counter = 0
    u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    rtol0 = np.inf
    while counter < maxiter and (rtol0 > rtol):
        counter += 1

        # .. update v0 ..
        v0, res_v = CGNR(
            lambda z: matmat(X, u0, z),
            lambda z: rmatmat2(X, u0, z), y, v0, maxiter=np.inf, rtol=.1)

        # .. update u0 ..
        u0, res_u = CGNR(
            lambda z: matmat(X, z, v0),
            lambda z: rmatmat1(X, v0, z), y, u0, maxiter=np.inf, rtol=.1)

        # .. need to recompute rv for new u0 ..
        res_v = rmatmat2(X, u0, matmat(X, u0, v0) - y)
        rtol0 = (np.abs(res_u).max() + np.abs(res_v).max()) / (linalg.norm(u0, 'fro') + linalg.norm(v0, 'fro'))
        if verbose:
            print 'RELATIVE TOLERANCE: %s' % rtol0

        if rtol0 < rtol:
            break

        if Z is not None:
            # TODO: cache SVD(Z)
            w0 = linalg.lstsq(Z, y - X.matvec(khatri_rao(v0, u0)))[0]

    if Z is None:
        return u0, v0
    else:
        return u0, v0, w0


if __name__ == '__main__':
    size_u, size_v = 10, 8
    X = sparse.csr_matrix(np.random.randn(1000, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .3 * np.random.randn(X.shape[0])
    y = np.array([i * y for i in range(1, 10)]).T
    u, v, w0 = rank_one(X, y, .1, size_u, Z=np.random.randn(X.shape[0], 2), verbose=True)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
