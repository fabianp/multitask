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


def rank_one(X, Y, alpha, size_u, u0=None, v0=None, Z=None, rtol=1e-6, verbose=False, maxiter=1000):
    """
    multi-target rank one model

        ||y - X vec(u v.T) - Z w||^2 + alpha * ||u - u_0||^2

    Parameters
    ----------
    X : array-like, sparse matrix or LinearOperator, shape (n, p)
    Y : array-lime, shape (n, k)
    size_u : integer
        Must be divisor of p
    u0 : array
    Z : array, sparse matrix or LinearOperator, shape (n, q)
    rtol : float
    maxiter : int
        maximum number of iterations
    verbose : int
        1 : a bit verbose
        2 : very verbose

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : XXX
    """

    X = splinalg.aslinearoperator(X)
    if Z is None:
        # create identity operator
        Z_ = splinalg.LinearOperator(shape=(X.shape[0], 1),
            matvec=lambda x: np.zeros((X.shape[0], x.shape[1])),
            rmatvec=lambda x: np.zeros((1, x.shape[1])))
    else:
        Z_ = splinalg.aslinearoperator(Z)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if u0 is None:
        u0 = np.ones((size_u, n_task))
    if u0.size == size_u:
        u0 = u0.reshape((-1, 1))
        u0 = np.repeat(u0, n_task, axis=1)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    size_v = X.shape[1] / size_u
    #u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    w0 = np.zeros((size_u + size_v + Z_.shape[1], n_task))
    w0[:size_u] = u0
    w0[size_u:size_u + size_v] = v0
    w0 = w0.reshape((-1,), order='F')

    # .. some auxiliary functions ..
    # .. used in conjugate gradient ..
    def obj(X_, Y_, Z_, a, b, c, u0):
        uv0 = khatri_rao(b, a)
        cost = .5 * linalg.norm(Y_ - X_.matvec(uv0) - Z_.matvec(c), 'fro') ** 2
        reg = alpha * linalg.norm(a - u0, 'fro') ** 2
        return cost + reg

    def f(w, X_, Y_, n_task, u0):
        W = w.reshape((-1, n_task), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        return obj(X_, Y_, Z_, u, v, c, u0)

    def fprime(w, X_, Y_, n_task, u0):
        W = w.reshape((-1, n_task), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        tmp = Y_ - matmat2(X_, u, v, n_task) - Z_.matvec(c)
        grad = np.empty((size_u + size_v + Z_.shape[1], n_task))  # TODO: do outside
        grad[:size_u] = rmatmat1(X, v, tmp, n_task) - alpha * (u - u0)
        grad[size_u:size_u + size_v] = rmatmat2(X, u, tmp, n_task)
        grad[size_u + size_v:] = Z_.rmatvec(tmp)
        return - grad.reshape((-1,), order='F')

    n_split = Y.shape[1] // 20 + 1
    Y_split = np.array_split(Y, n_split, axis=1)
    U = np.zeros((size_u, n_task))
    V = np.zeros((size_v, n_task))
    C = np.zeros((Z_.shape[1], n_task))
    counter = 0
    for y_i in Y_split:
        w0_i = w0.reshape((size_u + size_v + Z_.shape[1], n_task), order='F')[:, counter:(counter + y_i.shape[1])]
        u0_i = u0[:, counter:(counter + y_i.shape[1])]
        tmp = optimize.fmin_l_bfgs_b(f, w0_i, fprime=fprime, factr=rtol / np.finfo(np.float).eps,
                    args=(X, y_i, y_i.shape[1], u0_i), maxfun=maxiter, disp=0)[0]
        W = tmp.reshape((-1, y_i.shape[1]), order='F')
        U[:, counter:counter + y_i.shape[1]] = W[:size_u]
        V[:, counter:counter + y_i.shape[1]] = W[size_u:size_u + size_v]
        C[:, counter:counter + y_i.shape[1]] = W[size_u + size_v:]
        counter += y_i.shape[1]
        if verbose:
            print('Completed %.01f%%' % ((100. * counter) / Y.shape[1]))

    if Z is None:
        return U, V
    else:
        return U, V, C



def rank_one_proj(X, Y, alpha, size_u, u0=None, v=None, rtol=1e-6, maxiter=1000, verbose=False):
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

    #X = splinalg.aslinearoperator(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]
    size_v = X.shape[1] / size_u

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    u = np.ones(size_u * n_task)
    if u0 is None:
        u0 = np.ones((size_u * size_v, n_task))
    if u.shape[0] == size_u:
        u = np.repeat(u, n_task).reshape((-1, n_task), order='F')
    if v is None:
        v = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    u = u.reshape((-1, n_task))
    v = v.reshape((-1, n_task))

    #cutoff = 1e-3
    if verbose:
        print('Precomputing the singular value decomposition ...')
    if verbose:
        print('Done')
    if verbose:
        print('Precomputing least squares solution ...')
    XY = np.dot(X.T, Y)
    if verbose:
        print('Done')
    U, sigma, Vt = linalg.svd(X, full_matrices=True)
    sigma1 = np.zeros(X.shape[1])
    sigma1[:sigma.size] = sigma
    sigma = sigma1
    obj_old = np.inf
    counter = 0

    if verbose:
        print('Starting projection iteration ...')

    sol = None
    U, svals, Vt = linalg.svd(X, full_matrices=True)
    s = np.zeros((X.shape[1], 1))
    s[:svals.size] = svals[:, None]
    while counter < maxiter:
        counter += 1
        d = np.kron(1 / v, np.ones((size_u, 1)))
        sol0 = Vt.dot(XY + alpha * d * u0)
        sol0 *= (1. / (s * s + alpha * d * d))
        sol = Vt.T.dot(sol0)
        u = (d * sol).reshape((u.shape[0], v.shape[0], n_task), order='F').sum(1) / size_v

        h = np.kron(np.ones((size_v, 1)), 1 / u)
        # import ipdb; ipdb.set_trace()
        sol0 = Vt.dot(XY + alpha * h * u0)
        sol0 *= (1. / (s * s + alpha * h * h))
        sol1 = Vt.T.dot(sol0)
        v = (h * sol).reshape((u.shape[0], v.shape[0], n_task), order='F').sum(0) / size_u

        H = np.diag(h[:, 0])
        sol = linalg.solve(X.T.dot(X) + alpha * H.T.dot(H), XY + alpha * H.T.dot(u0))
        import ipdb; ipdb.set_trace()
        v = H.dot(sol).reshape((u.shape[0], v.shape[0], n_task), order='F').sum(0) / size_u

        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        obj_new = linalg.norm(Y - X.dot(sol), 'fro') ** 2 + alpha * (linalg.norm(u - u0[:size_u]) ** 2)

        if verbose:
            print('OBJ OLD %s' % obj_old)
            print('OBJ     %s' % obj_new)
            print

        #print(np.abs(obj_new - obj_old) / obj_new)
        if np.abs(obj_new - obj_old) < rtol * obj_new * 0.1:
            break

        obj_old = obj_new

    return u, v


if __name__ == '__main__':
    size_u, size_v = 9, 48
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .1 * np.random.randn(X.shape[0])
    #y = np.array([i * y for i in range(1, 3)]).T
    u, v, w = rank_one(X.A, y, .1, size_u, Z=np.random.randn(X.shape[0], 3), verbose=True, rtol=1e-10)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
