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


def khatri_rao(a, b):
    """
    Compute the Khatri-rao product, where the partition is taken to be
    the vectors along axis one.

    This is a helper function for rank_one

    Parameters
    ----------
    a : array, shape (n, p)
    b : array, shape (m, p)

    Returns
    -------

    """
    res = np.empty((a.shape[0] * b.shape[0], a.shape[1]), dtype=np.float)
    for i in range(a.shape[0]):
        res[i * b.shape[0]:(i + 1) * b.shape[0]] = a[i, np.newaxis] * b
    return res


    # res2 = []
    # for i in range(a.shape[1]):
    #     res2.append(np.kron(a[:, i], b[:, i]))
    # res2 = np.vstack(res2).T
    # assert np.allclose(res, res2)
    # return res


def rank_one(X, y, alpha, size_u, Z=None, prior_u=None, u0=None, v0=None, rtol=1e-6, maxiter=1000, verbose=False):
    """
    multi-target rank one

    Parameters
    ----------
    X : sparse matrix, shape (

    """

    s = splinalg.svds(X, 1)[1][0]

    X = splinalg.aslinearoperator(X)
    y = np.asarray(y)
    n_task = y.shape[1]


    def matvec_1(a, b, n_task):
        """
        (a.T kron I_n) b
        """

        B = b.reshape((n_task, -1, a.shape[1]), order='F')
        res = np.einsum("ijk, ik -> ij", B, a).T
        return res
        # return np.dot(b.reshape((-1, a.size), order='F'), a).ravel()


    def matvec_2(a, b, n_task):
        """
        (I kron a.T) b
        """
        B = b.reshape((n_task, -1, a.shape[1]), order='C')
        res = np.einsum("ijk, ik -> ij", B, a)
        #import ipdb; ipdb.set_trace()
        return res
        # return np.dot(b.reshape((-1, a.size), order='C'), a).ravel()

    if u0 is None:
        u0 = np.ones(size_u * n_task)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    Xy = X.rmatvec(y)


    def obj(a, b, n_task):
        uv0 = khatri_rao(b, a)
        return .5 * linalg.norm(y - X.matvec(uv0), 'fro') ** 2


    def grad_u(a, b, n_task):
        uv0 = khatri_rao(b, a)
        #import ipdb; ipdb.set_trace()
        tmp1 = Xy - X.rmatvec(X.matvec(uv0))
        res = -matvec_1(b.T, tmp1.T, n_task)#.ravel('F').reshape((-1, n_task))
        return res


    def grad_v(a, b, n_task):
        uv0 = khatri_rao(b, a)
        tmp1 = Xy - X.rmatvec(X.matvec(uv0))
        return - matvec_2(a.T, tmp1.T, n_task).T

    pobj = [np.inf]
    counter = 0
    u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    while counter < maxiter: # this allows to set maxiter to infinity
        counter += 1

        #print 'Check grad %s' % optimize.check_grad(lambda x: obj(x, v0[:, :1], n_task), lambda x: grad_u(x, v0[:, :1], n_task), u0[:, :1])
        #import ipdb; ipdb.set_trace()

        v0_norm2 = (v0 * v0).sum(0)
        step_size = 1. / (s * s * v0_norm2)

        for i in range(5):
            u0 -= step_size.reshape((1, n_task)) * grad_u(u0, v0, n_task)

        u0_norm2 = (u0 * u0).sum(0)
        step_size = 1. / (s * s * u0_norm2)
        for i in range(5):
            v0 -= step_size.reshape((1, n_task)) * grad_v(u0, v0, n_task)

        new_pobj = obj(u0, v0, n_task)
        print 'OBJ %s' % new_pobj
        gap = (pobj[-1] - new_pobj) / pobj[-1]
        if gap < rtol:
            break

        pobj.append(new_pobj)

    import pylab as pl
    pl.plot(pobj)
    pl.show()

    return u0, v0


if __name__ == '__main__':
    size_u, size_v = 10, 8
    X = sparse.csr_matrix(np.random.randn(1000, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .3 * np.random.randn(X.shape[0])
    # y = y.reshape((-1, 1))
    y = np.array([i * y for i in range(1, 10)]).T
    u, v = rank_one(X, y, .1, size_u, Z=None, verbose=True)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
