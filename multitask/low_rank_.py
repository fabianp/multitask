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
    n_task = b.shape[1]
    p = z
    i = 0
    residuals = np.inf
    while i < maxiter and np.any(np.abs(residuals) > linalg.norm(x0, 'fro') * rtol):
        i += 1
        w = matmat(p)
        tmp = (w * w).sum(0)
        tmp[np.abs(tmp) < np.finfo(np.float).eps] = np.finfo(np.float).eps
        alpha = (z * z).sum(0) / tmp
        x0 += alpha.reshape((-1, n_task)) * p
        r -= alpha * w
        z_new = rmatmat(r)
        beta = (z_new * z_new).sum(0) / (z * z).sum(0)
        z = z_new
        residuals = (z * z).sum(0)
        p = z_new + beta.reshape((-1, n_task)) * p
    return x0, residuals


def PCGNR(matmat, rmatmat, b, x0, M, maxiter=100, rtol=1e-6):
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
    r_tilde = rmatmat(r)
    z = M.matmat(r_tilde)
    k = b.shape[1]
    p = z
    i = 0
    residuals = np.inf
    while i < maxiter and np.any(np.abs(residuals) > b.shape[1] * linalg.norm(x0, 'fro') * rtol):
        i += 1
        w = matmat(p)
        tmp = (w * w).sum(0)
        tmp[np.abs(tmp) < np.finfo(np.float).eps] = np.finfo(np.float).eps
        alpha = (z * r_tilde).sum(0) / tmp
        x0 += alpha.reshape((-1, k)) * p
        r -= alpha * w
        r_tilde_new = rmatmat(r)
        z_new = M.matmat(r_tilde_new)
        beta = (z_new * r_tilde_new).sum(0) / (z * r_tilde).sum(0)
        z = z_new
        r_tilde = r_tilde_new
        residuals = (z * z).sum(0)
        p = z_new + beta.reshape((-1, k)) * p
        #print 'P', linalg.norm(residuals)
    return x0, residuals


def matmat2(X, a, b, n_task):
    """
    X (b * a)
    """
    uv0 = khatri_rao(b, a)
    return X.matvec(uv0)


def rmatmat1(X, a, b, n_task):
    """
    (I kron a^T) X^T b
    """
    b1 = X.rmatvec(b[:X.shape[0]]).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='F')
    res = np.einsum("ijk, ik -> ij", B, a.T).T
    #res += np.sqrt(alpha) * b[X.shape[0]:]
    return res


def rmatmat2(X, a, b, n_task):
    """
    (a^T kron I) X^T b
    """
    b1 = X.rmatvec(b).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='C')
    tmp = np.einsum("ijk, ik -> ij", B, a.T).T
    return tmp


def rank_one(X, Y, alpha, size_u, prior_u=None, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
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
    Mv : LinearOperator
        preconditioner for the least squares problem ||y - X(u \kron I)v||

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : XXX
    """

    X = splinalg.aslinearoperator(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    # .. some auxiliary functions ..
    # .. used in conjugate gradient ..
    def obj(a, b):
        uv0 = khatri_rao(b, a)
        return .5 * linalg.norm(Y - X.matvec(uv0), 'fro') ** 2

    def matmat1(X, a, b, n_task):
        """
        X (b * a) with regularization
        """
        uv0 = khatri_rao(b, a)
        t0 = X.matvec(uv0)
        tmp = np.vstack((t0, np.sqrt(alpha) * a))
        return tmp


    if u0 is None:
        u0 = np.ones(size_u * n_task)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    size_v = X.shape[1] / size_u
    u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    w0 = np.empty((size_u + size_v, n_task))
    w0[:size_u] = u0
    w0[size_u:] = v0
    w0[size_u:] = v0
    w0 = w0.reshape((-1,), order='F')

    def f(w):
        W = w.reshape((-1, n_task), order='F')
        u, v = W[:size_u], W[size_u:]
        return obj(u, v)

    def fprime(w):
        W = w.reshape((-1, n_task), order='F')
        u, v = W[:size_u], W[size_u:]
        tmp = Y - matmat2(X, u, v, n_task)
        grad = np.empty((size_u + size_v, n_task))  # TODO: do outside
        grad[:size_u] = rmatmat1(X, v, tmp, n_task)
        grad[size_u:] = rmatmat2(X, u, tmp, n_task)
        return - grad.reshape((-1,), order='F')


    def call(x):
        print('OBJ %s' % f(x))
    out = optimize.minimize(f, w0, jac=fprime, tol=rtol, callback=call, method='L-BFGS-B')
    W = out.x.reshape((-1, n_task), order='F')
    if Z is not None:
        return W[:size_u], W[size_u:], None
    else:
        return W[:size_u], W[size_u:]



def rank_one_proj(X, Y, alpha, size_u, prior_u=None, Z=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
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
    Mv : LinearOperator
        preconditioner for the least squares problem ||y - X(u \kron I)v||

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : XXX
    """

    #X = splinalg.aslinearoperator(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if u0 is None:
        u0 = np.ones(size_u * n_task)
    if u0.shape[0] == size_u:
        u0 = np.repeat(u0, n_task).reshape((-1, n_task), order='F')
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    size_v = X.shape[1] / size_u
    u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))

    cutoff = 1e-3
    if verbose:
        print('Precomputing the singular value decomposition ...')
    U, s, Vt = linalg.svd(X)
    if verbose:
        print('Done')
    sigma = 1. / s[s > cutoff]
    if verbose:
        print('Precomputing least squares solution ...')
    ls_sol = (Vt.T[:, :sigma.size] * sigma).dot(U.T[:sigma.size]).dot(Y)
    ls_sol = ls_sol.reshape((-1, n_task))
    Kern = Vt[np.sum(s > cutoff):].T
    KK = Kern.dot(Kern.T)
    if verbose:
        print('Done')
    x0 = khatri_rao(v0, u0)
    X_ = splinalg.aslinearoperator(X)
    obj_old = np.inf
    counter = 0

    def power2(A, q, n_iter=10):
        for _ in range(n_iter):
            z = (A * q).sum(1)
            z = z.reshape((1, z.shape[0], z.shape[1]))
            z = (A.transpose((1, 0, 2)) * z).sum(1)
            q = (z / np.sqrt((z.T * z.T).sum(1)))
            q = q.reshape((1, q.shape[0], q.shape[1]))
            s = (A * q).sum(1)
            s = s.reshape((1, s.shape[0], s.shape[1]))
            s = (A.transpose((1, 0, 2)) * s).sum(1)
            s = s.reshape((1, s.shape[0], s.shape[1]))
            s = (s * q).sum(1)
            assert s.size == n_task
        return np.sqrt(s), q.reshape((q.shape[1], q.shape[2]))

    if verbose:
        print('Starting projection iteration ...')

    while counter < maxiter:
        counter += 1
        tmp = KK.dot(x0)
        proj = tmp + ls_sol
        proj = proj.reshape((u0.shape[0], v0.shape[0], n_task), order='F')
        s, v0 = power2(proj, v0, 1)
        tmp = v0.reshape((1, v0.shape[0], v0.shape[1]))
        u0 = (proj * tmp).sum(1) / s
        v0 = v0 * s
        # for i in range(n_task):
        #     Xi = proj[:, :, i]
        #     s1, v1 = power(Xi, v0[:, i], 2)
        #     v0[:, i] = v1 * s1
        #     u0[:, i] = np.dot(Xi, v1) / s1
        x0 = khatri_rao(v0, u0)
        obj_new = linalg.norm(Y - X.dot(x0), 'fro') ** 2

        if verbose:
            #print('TOL %s' % (linalg.norm(tol, np.inf) / obj_new))
            print('OBJ %s' % obj_new)

        #print(np.abs(obj_new - obj_old) / obj_new)
        if np.abs(obj_new - obj_old) < rtol * obj_new * 0.1:
            break

        obj_old = obj_new

    if Z is not None:
        return u0, v0, None
    else:
        return u0, v0


if __name__ == '__main__':
    size_u, size_v = 9, 48
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .3 * np.random.randn(X.shape[0])
    y = np.array([i * y for i in range(1, 101)]).T
    u, v, w0 = rank_one_proj(X.A, y, .1, size_u, u0=np.random.randn(size_u),
                             Z=np.random.randn(X.shape[0], 2), verbose=True, maxiter=500)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
