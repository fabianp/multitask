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


def rank_one(X, Y, size_u, u0=None, v0=None, Z=None,
             rtol=1e-6, verbose=False, maxiter=1000, callback=None,
             plot=False, method='Newton-CG'):
    """
    multi-target rank one model

        ||y - X vec(u v.T) - Z w||^2 + alpha * ||u - u_0||^2

    Parameters
    ----------
    X : array-like, sparse matrix or LinearOperator, shape (n, p)
    Y_train : array-lime, shape (n, k)
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
    W : array,
        only returned if Z is specified
    """
    alpha = 0.
    X = splinalg.aslinearoperator(X)
    if Z is None:
        # create zero operator
        Z_ = splinalg.LinearOperator(shape=(X.shape[0], 1),
            matvec=lambda x: np.zeros((X.shape[0], x.shape[1])),
            rmatvec=lambda x: np.zeros((1, x.shape[1])), dtype=np.float)
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

    # .. some auxiliary functions ..
    # .. used in conjugate gradient ..
    def obj(X_, Y_, Z_, a, b, c, u0):
        uv0 = khatri_rao(b, a)
        cost = .5 * linalg.norm(Y_ - X_.matvec(uv0) - Z_.matmat(c), 'fro') ** 2
        #print('LOSS: %s' % cost)
        reg = alpha * linalg.norm(a - u0, 'fro') ** 2
        return cost + reg

    def f(w, X_, Y_, Z_, n_task, u0):
        W = w.reshape((-1, 1), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        return obj(X_, Y_, Z_, u, v, c, u0)

    def fprime(w, X_, Y_, Z_, n_task, u0):
        n_task = 1
        W = w.reshape((-1, 1), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        tmp = Y_ - matmat2(X_, u, v, 1) - Z_.matmat(c)
        grad = np.empty((size_u + size_v + Z_.shape[1], 1))  # TODO: do outside
        grad[:size_u] = rmatmat1(X, v, tmp, 1) - alpha * (u - u0)
        grad[size_u:size_u + size_v] = rmatmat2(X, u, tmp, 1)
        grad[size_u + size_v:] = Z_.rmatvec(tmp)
        return - grad.reshape((-1,), order='F')


    def hess(w, s, X_, Y_, Z_, n_task, u0):
        # TODO: regularization
        s = s.reshape((-1, 1))
        X_ = splinalg.aslinearoperator(X_)
        Z_ = splinalg.aslinearoperator(Z_)
        size_v = X_.shape[1] / size_u
        W = w.reshape((-1, 1), order='F')
        XY = X_.rmatvec(Y_)  # TODO: move out
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        s1, s2, s3 = s[:size_u], s[size_u:size_u + size_v], s[size_u + size_v:]
        W2 = X_.rmatvec(matmat2(X_, u, v, 1))
        W2 = W2.reshape((-1, s2.shape[0]), order='F')
        XY = XY.reshape((-1, s2.shape[0]), order='F')

        n_task = 1
        A_tmp = matmat2(X_, s1, v, n_task)
        As1 = rmatmat1(X_, v, A_tmp, n_task)
        tmp = matmat2(X_, u, s2, n_task)
        Ds2 = rmatmat2(X_, u, tmp, n_task)
        tmp = Z_.matvec(s3)

        Cs3 = rmatmat1(X_, v, tmp, n_task)
        tmp = matmat2(X_, s1, v, n_task).T
        Cts1 = Z_.rmatvec(tmp.T)

        tmp = matmat2(X_, u, s2, n_task)
        Bs2 = rmatmat1(X_, v, tmp, n_task) + W2.dot(s2) - XY.dot(s2)

        tmp = matmat2(X_, s1, v, n_task)
        Bts1 = rmatmat2(X_, u, tmp, n_task) + W2.T.dot(s1) - XY.T.dot(s1)

        tmp = Z_.matvec(s3)
        Es3 = rmatmat2(X_, u, tmp, n_task)

        tmp = matmat2(X_, u, s2, n_task)
        Ets2 = Z_.rmatvec(tmp)

        Fs3 = - Z_.rmatvec(Z_.matvec(s3))

        line0 = As1 + Bs2 + Cs3
        line1 = Bts1 + Ds2 + Es3
        line2 = Cts1 + Ets2 + Fs3

        return np.concatenate((line0, line1, line2)).ravel()


    if plot:
        import pylab as pl
        fig = pl.figure()
        pl.show()
        from nipy.modalities.fmri import hemodynamic_models as hdm
        canonical = hdm.glover_hrf(1., 1., size_u)
        canonical -= canonical.mean()
        pl.plot(canonical / (canonical.max() - canonical.min()), lw=4)
        pl.draw()
        pl.show()


    def do_plot(w):
        from nipy.modalities.fmri import hemodynamic_models as hdm
        canonical = hdm.glover_hrf(1., 1., size_u)
        print('PLOT')
        W = w.reshape((-1, 1), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        #u -= u.mean(0)
        #pl.clf()
        tmp = u.copy()
        sgn = np.sign(u.max(0))
        tmp *= sgn
        norm = tmp.max(0) - tmp.min(0)
        tmp = tmp / norm
        pl.plot(tmp)
        # pl.ylim((-1, 1.2))
        pl.draw()
        pl.xlim((0, size_u))

    U = np.zeros((size_u, n_task))
    V = np.zeros((size_v, n_task))
    C = np.zeros((Z_.shape[1], n_task))

    def cb(w):
        print(1)
        if plot:
            do_plot(w)
        if callback is not None:
            callback(w)

    for i in range(n_task):
        y_i = Y[:, i].reshape((-1, 1))
        w0_i = w0[:, i].ravel('F')
        u0_i = u0[:, i].reshape((-1, 1))

        args = (X, y_i, Z_, 1, u0_i)
        options = {'maxiter' : maxiter, 'xtol' : rtol}
        out = optimize.minimize(
            f, w0_i, jac=fprime, args=args, hessp=hess,
                                method=method, options=options,
                                callback=cb)

        if hasattr(out, 'nit'):
            print('Number of iterations: %s' % out.nit)
        out = out.x
        if plot:
            do_plot()
        W = out.reshape((-1, y_i.shape[1]), order='F')
        ui = W[:size_u].ravel()
        norm_ui = linalg.norm(ui)
        U[:, i] = ui / norm_ui
        V[:, i] = W[size_u:size_u + size_v].ravel() * norm_ui
        C[:, i] = W[size_u + size_v:].ravel()

    if Z is None:
        return U, V
    else:
        return U, V, C


def svd_power_method(X, Q, max_iter):
    # returns dominant singular value
    for _ in range(max_iter):
        tmp = (X * Q).sum(1)
        zk = (X.transpose((1, 0, 2)) * tmp).sum(1)
        Q = zk / np.sqrt((zk ** 2).sum(0))

    U = (X * Q).sum(1)
    return U, Q


def rank_one_gradproj(X, Y, size_u, u0=None, rtol=1e-3,
                   maxiter=50, verbose=False,
                   callback=None, v0=None, plot=False):
    """
    multi-target rank one model

        ||y - X vec(u v.T)||_2 ^2

    TODO: prior_u

    Parameters
    ----------
    X : array-like, shape (n, p)
    Y_train : array-like, shape (n, k)
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

    Reference
    ---------
    http://math.sjtu.edu.cn/faculty/zw2109/course/gprojection.pdf
    """


    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = 1
    size_v = X.shape[1] / size_u

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if plot:
        import pylab as pl

    if u0 is None:
        u0 = np.random.randn(size_u, 1)
    if u0.ndim == 1 or u0.shape[1] == 1:
        u = np.repeat(u0, n_task).reshape((-1, n_task))
    else:
        u = u0
    u = np.asfortranarray(u)
    if v0 is None:
        v = np.random.randn(size_v, n_task)
    else:
        v = v0
    w0 = khatri_rao(v, u)

    lipsch = splinalg.svds(X, 1)[1][0] ** 2
    step_size = 1. / lipsch # Landweber iteration
    print(step_size)
    obj_old = np.inf


    if plot:
        fig = pl.figure()
        pl.show()

    xk1 = w0
    XY = X.T.dot(Y)
    for n_iter in range(1, maxiter):
        print('ITER: %s' % n_iter)
        print('GRADIENT')
        Xw = X.dot(w0)
        grad = - XY + X.T.dot(Xw) #BLASify ?
        w0 -= step_size * grad
        print('PROJECTION')
        # projection step
        w_tmp = w0.reshape((size_u, size_v, n_task), order='F')
        u, v = svd_power_method(w_tmp, v, 10)
        w0 = khatri_rao(v, u)
        # Nesterov step
        if False:
            tmp = w0 + ((n_iter - 1.) / (n_iter + 2.)) * (w0 - xk1)
            xk1 = w0  # save it for next iteration
            w0 = tmp
            print('percentage of converged features: %s' % np.mean(np.abs(w0 - xk1) < rtol))
        obj_new = .5 * linalg.norm(Y - Xw, 'fro') ** 2
        print('LOSS: %s' % obj_new)
        print('TOL: %s' % (np.abs(obj_old - obj_new) / obj_new))
        if np.abs(obj_old - obj_new) / obj_new < rtol:
            print('Converged')
            break
        obj_old = obj_new
        if plot:
            print('PLOT')
            pl.clf()
            pl.ylim((-1., 1.))
            tmp = (u - u.mean(1)[:, None])
            sgn = np.sign(tmp.T.dot(u0.ravel('F')[:size_u] - u0.mean()))
            tmp *= sgn

            norm = tmp.max(0) - tmp.min(0)
            tmp = tmp / norm
            pl.plot(tmp)
            #                pl.ylim((-1, 1.2))
            pl.draw()
            pl.xlim((0, size_u))
            #pl.savefig('proj_%03d.png' % n_iter)
        if callback is not None:
            callback(w0)
    return u, v


def rank_one_frankwolfe(X, Y, size_u, u0=None, rtol=1e-3,
                   maxiter=50, verbose=False,
                   callback=None, v0=None, plot=False):
    """
    multi-target rank one model

        ||y - X vec(u v.T)||_2 ^2

    TODO: prior_u

    Parameters
    ----------
    X : array-like, shape (n, p)
    Y_train : array-like, shape (n, k)
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

    Reference
    ---------
    """


    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]
    size_v = X.shape[1] / size_u

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if plot:
        import pylab as pl

    if u0 is None:
        u0 = np.random.randn(size_u, 1)
    if u0.ndim == 1 or u0.shape[1] == 1:
        u = np.repeat(u0, n_task).reshape((-1, n_task))
    else:
        u = u0
    u = np.asfortranarray(u)
    if v0 is None:
        v = np.random.randn(size_v, n_task)
    else:
        v = v0
    w0 = khatri_rao(v, u)

    lipsch = splinalg.svds(X, 1)[1][0] ** 2
    step_size = 1. / lipsch # Landweber iteration
    obj_old = np.inf

    if plot:
        fig = pl.figure()
        pl.show()

    xk1 = w0
    XY = X.T.dot(Y)
    u2 = np.empty_like(u)
    v2 = np.empty_like(v)

    for n_iter in range(1, maxiter):
        print('ITER: %s' % n_iter)
        print('PROJECTION')
        # projection step
        Xw = X.dot(w0)
        grad = - XY + X.T.dot(Xw) #BLASify ?
        grad_tmp = grad.reshape((size_u, size_v, n_task), order='F')
        #u2, v2 = svd_power_method(grad_tmp, v2, 20)
        for j in range(n_task):
                u_svd, s, vt_svd = linalg.svd(grad_tmp[:, :, j], full_matrices=False)
                u2[:, j], v2[:, j] = u_svd[:, 0], s[0] * vt_svd[0]
        s = khatri_rao(v2, u2) - w0
        for j in range(n_task):
            yj = Y[:, j]
            sj = s[:, j]
            tmp = X.T.dot(X).dot(sj)
            alpha = (sj.dot(X.T.dot(yj)) - tmp.dot(w0[:, j])) / tmp.dot(sj)
            def f(w):
                return .5 * linalg.norm(yj - X.dot(w)) ** 2
            # def fprime(w):
            #     return - X.T.dot(yj) + X.T.dot(X.dot(w))
            # tmp =
            # alpha = optimize.line_search(f, fprime, w0[:, j], s[:, j])[0]
            w_old = w0[:, j].copy()
            w0[:, j] += alpha * s[:, j]
            #assert f(w_old) >= f(w0[:, j])
            #print((alpha, f(w_old), f(w0[:, j])))
            #import ipdb; ipdb.set_trace()


        obj_new = .5 * linalg.norm(Y - X.dot(w0), 'fro') ** 2
        print('LOSS: %s' % obj_new)
        print('TOL: %s' % (np.abs(obj_old - obj_new) / obj_new))
        if np.abs(obj_old - obj_new) / obj_new < rtol:
            print('Converged')
            #break
        obj_old = obj_new
        # w_tmp = w0.reshape((size_u, size_v, n_task), order='F')
        # u, v = svd_power_method(w_tmp, v, 10)
        # w0 = khatri_rao(v, u)
        if plot:
            w_tmp = w0.reshape((size_u, size_v, n_task), order='F')
            for j in range(n_task):
                u_svd, s, vt_svd = linalg.svd(w_tmp[:, :, j], full_matrices=False)
                u[:, j], v[:, j] = u_svd[:, 0], s[0] * vt_svd[0]
            print('PLOT')
            pl.clf()
            pl.ylim((-1., 1.))
            tmp = (u - u.mean(1)[:, None])
            sgn = np.sign(tmp.T.dot(u0.ravel('F')[:size_u] - u0.mean()))
            tmp *= sgn

            norm = tmp.max(0) - tmp.min(0)
            tmp = tmp / norm
            pl.plot(tmp)
            #                pl.ylim((-1, 1.2))
            pl.draw()
            pl.xlim((0, size_u))
            #pl.savefig('proj_%03d.png' % n_iter)
        if callback is not None:
            callback(w0)
    for j in range(n_task):
        #import ipdb; ipdb.set_trace()
        u_svd, s, vt_svd = linalg.svd(w_tmp[:, :, j], full_matrices=False)
        u[:, j], v[:, j] = u_svd[:, 0], s[0] * vt_svd[0]
    return u, v



from math import *

def cuberoot(x):
    if x >= 0:
        return x**(1/3.0)
    else: # negative argument!
        return -(-x)**(1/3.0)

def polyCubicRoots(a,b, c):
    aby3 = a / 3.0
    p = b - a*aby3
    q = (2*aby3**2- b)*(aby3) + c
    X =(p/3.0)**3
    Y = (q/2.0)**2
    Q = X + Y
    if Q >= 0:
        sqQ = sqrt(Q)
        # Debug January 11, 2013. Thanks to a reader!
        t = (-q/2.0 + sqQ)
        A = cuberoot(t)
        t = (-q/2.0 - sqQ)
        B = cuberoot(t)

        r1 = A + B- aby3
        re = -(A+B)/2.0-aby3
        im = sqrt(3.0)/2.0*(A-B)
        r2 = (re,im)
        r3 = (re,-im)
    else:
        # This part has been tested.
        p3by27= sqrt(-p**3/27.0)
        costheta = -q/2.0/ p3by27
        alpha = acos(costheta)
        mag = 2 * sqrt(-p/3.0)
        alphaby3 = alpha/3.0
        r1 = mag  * cos(alphaby3) - aby3
        r2 = -mag * cos(alphaby3+ pi/3)-aby3
        r3 = -mag * cos(alphaby3- pi/3) -aby3
    return r1, r2, r3

def rank_one_ecg(X, Y, size_u, u0=None, rtol=1e-3,
                        maxiter=100, verbose=False,
                        callback=None, v0=None, plot=False):
    """
    multi-target rank one model

        ||y - X vec(u v.T)||_2 ^2

    TODO: prior_u

    Parameters
    ----------
    X : array-like, shape (n, p)
    Y_train : array-like, shape (n, k)
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

    Reference
    ---------
    """


    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]
    size_v = X.shape[1] / size_u

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if plot:
        import pylab as pl

    if u0 is None:
        u0 = np.random.randn(size_u, 1)
    if u0.ndim == 1 or u0.shape[1] == 1:
        u = np.empty((u0.size, n_task))
        u[:, :] = u0
    else:
        u = u0

    if v0 is None:
        v = np.random.randn(size_v, n_task)
    else:
        v = v0

    if plot:
        fig = pl.figure()
        pl.show()

    X_ = splinalg.aslinearoperator(X)
    obj_old = np.inf
    a = Y - matmat2(X_, u, v, n_task)
    grad_u = - rmatmat1(X_, v, a, n_task)
    grad_v = - rmatmat2(X_, u, a, n_task)
    pk = [-grad_u, -grad_v]
    for n_iter in range(1, maxiter):
        deltak = (grad_u * grad_u).sum(0) + (grad_v * grad_v).sum(0)
        print('ITER: %s' % n_iter)
        print('PROJECTION')
        # projection step

        b = - (matmat2(X_, grad_u, v, n_task) + matmat2(X_, u, grad_v, n_task))
        c = - matmat2(X_, grad_u, grad_v, n_task)

        a0 = (b * a).sum(0)
        a1 = (b * b + 2 * c * a).sum(0)
        a2 = 3 * (b * c).sum(0)
        a3 = 2 * (c * c).sum(0)

        def ff(alpha):
            tmp = Y - matmat2(
                X_, u + alpha * pk[0],
                v + alpha * pk[1], n_task)
            return .5 * (linalg.norm(tmp,'fro') ** 2)

        q0 = a2 / a3
        q1 = a1 / a3
        q2 = a0 / a3
        step_size = np.empty(n_task)
        for i in range(n_task):
            root = polyCubicRoots(q0[i], q1[i], q2[i])
            step_size[i] = root[0]
        step_size = optimize.minimize_scalar(ff, tol=1e-32).x

        #print(.5 * linalg.norm(Y - matmat2(X_, u, v, n_task), 'fro') ** 2)
        u += step_size * pk[0]
        v += step_size * pk[1]
        #print(.5 * linalg.norm(Y - matmat2(X_, u, v, n_task), 'fro') ** 2)

        a = Y - matmat2(X_, u, v, n_task)
        new_grad_u = - rmatmat1(X_, v, a, n_task)
        new_grad_v = - rmatmat2(X_, u, a, n_task)
        yk_u = new_grad_u - grad_u
        yk_v = new_grad_v - grad_v
        beta_k = ((yk_u * new_grad_u).sum(0) + (yk_v * new_grad_v).sum(0))
        beta_k = max(0, beta_k / deltak)
        # if (grad_u * new_grad_u).sum() / (grad_u * grad_u).sum() >= .1:
        #     # restart
        #     print('RESTART')
        #     beta_k = 0.

        #import ipdb; ipdb.set_trace()
        pk = [- new_grad_u + beta_k * pk[0], - new_grad_v + beta_k * pk[1]]
        grad_u = new_grad_u
        grad_v = new_grad_v

        obj_new = .5 * linalg.norm(Y - matmat2(X_, u, v, n_task), 'fro') ** 2
        print('LOSS: %s' % obj_new)
        print('TOL: %s' % (np.abs(obj_old - obj_new) / obj_new))
        if np.abs(obj_old - obj_new) / obj_new < rtol:
            print('Converged')
            #break
        obj_old = obj_new
        # w_tmp = w0.reshape((size_u, size_v, n_task), order='F')
        # u, v = svd_power_method(w_tmp, v, 10)
        # w0 = khatri_rao(v, u)
        if plot:
            print('PLOT')
            pl.clf()
            pl.ylim((-1., 1.))
            tmp = (u - u.mean(0))
            sgn = np.sign(tmp.T.dot(u0.ravel('F')[:size_u] - u0.mean()))
            tmp *= sgn

            norm = tmp.max(0) - tmp.min(0)
            tmp = tmp / norm
            pl.plot(tmp)
            #                pl.ylim((-1, 1.2))
            pl.draw()
            pl.xlim((0, size_u))
            #pl.savefig('proj_%03d.png' % n_iter)
        if callback is not None:
            callback((u, v))

    return u, v




if __name__ == '__main__':
    np.random.seed(0)
    size_u, size_v = 9, 48
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .1 * np.random.randn(X.shape[0])
    #y = np.array([i * y for i in range(1, 3)]).T
    u, v = rank_one_ecg(X.A, y, size_u, verbose=True)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
