import numpy as np
from scipy import linalg, optimize
from scipy.sparse import linalg as splinalg
from datetime import datetime

def prox_l1(a, b):
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

def prox(X, t, v0, n_nonzero=190, n=0, algo='dense', n_svals=10):
    """prox operator for trace norm
    Algo: {sparse, dense}
    """

    if algo=='sparse':
        k = min(np.min(X.shape) - 1, n_nonzero)
        if v0 is None:
            v0_init = None
        else:
            v0_init = v0[0]
        if False: #X.shape[0] > X.shape[1]:
            raise NotImplementedError('Sorry (you need to pass u0 instead of v0)')
        u, s, vt = splinalg.svds(X, k=k, v0=v0_init, maxiter=500, tol=1e-6)
        u, s, vt = u[:, ::-1], s[::-1], vt[::-1]
    else:
        u, s, vt = linalg.svd(X, full_matrices=False)
        #u, s, vt = randomized_svd(X, n_svals)
    s[n:] = np.sign(s[n:]) * np.fmax(np.abs(s[n:]) - t, 0)
    low_rank = np.dot(u, np.dot(np.diag(s), vt))
    return low_rank, s, u, vt


def conj_loss(X, y, Xy, M, epsilon, sol0):
    # conjugate of the loss function
    n_features = X.shape[1]
    matvec = lambda z: X.rmatvec((X.matvec(z))) + epsilon * z
    K = splinalg.LinearOperator((n_features, n_features), matvec, dtype=X.dtype)
    sol = splinalg.cg(K, M.ravel(order='F') + Xy, maxiter=20, x0=sol0)[0]
    p = np.dot(sol, M.ravel(order='F')) - .5 * (linalg.norm(y - X.matvec(sol)) ** 2)
    p -= 0.5 * epsilon * (linalg.norm(sol) ** 2)
    return p, sol


def trace_pobj(X, y, B, alpha, epsilon, s_vals):
    n_samples, _ = X.shape
    bt = B.ravel(order='F')
    #s_vals = linalg.svdvals(B)
    return  0.5 * (linalg.norm(y - X.matvec(bt)) ** 2) +\
            0.5 * epsilon * (linalg.norm(bt) ** 2) +\
            alpha * linalg.norm(s_vals, 1)


def trace(X, y, alpha, beta, shape_B, rtol=1e-3, max_iter=1000, verbose=False,
          warm_start=None, n_svals=10, L=None, accelerated=False):
    """
    solve the model:

        ||y - X vec(B)||_2 ^2 + alpha ||B||_* + beta ||B||_F

    where vec = B.ravel('F')

    Parameters
    ----------
    X : LinearOperator
    L : None
        largest eigenvalue of X (optional)
    shape_B : tuple

    Returns
    -------
    B : array
    gap : float
    """
    X = splinalg.aslinearoperator(X)
    n_samples = X.shape[0]
    #alpha = alpha * n_samples
    beta = beta * n_samples

    if warm_start is None:
        # fortran ordered !!important when reshaping !!
        B = np.asfortranarray(np.zeros(shape_B))
    else:
        B = warm_start
    gap = []

    if L is None:
        def K_matvec(v):
            return X.rmatvec(X.matvec(v)) + beta * v
        K = splinalg.LinearOperator((X.shape[1], X.shape[1]), matvec=K_matvec, dtype=X.dtype)
        L = np.sqrt(splinalg.eigsh(K, 1, return_eigenvectors=False)[0])

    step_size = 1. / (L * L)
    Xy = X.rmatvec(y)
    v0 = None
    t = 1.
    conj0 = None
    time_vals = []
    obj_vals = []
    start = datetime.now()
    for n_iter in range(max_iter):
        time_vals.append((datetime.now() - start).total_seconds())
        obj_vals.append(0.5 * (linalg.norm(y - X.matvec(B.ravel('F'))) ** 2))
        b = B.ravel(order='F')
        grad_g = -Xy + X.rmatvec(X.matvec(b)) + beta * b
        tmp = (b - step_size * grad_g).reshape(*B.shape, order='F')
        xk, s_vals, u0, v0 = prox(tmp, step_size * alpha, v0, n_svals=n_svals)
        if accelerated:
            tk = (1 + np.sqrt(1 + 4 * t * t)) / 2.
            B = xk + ((t - 1.) / tk) * (xk - B)
            t = tk
        else:
            B = xk
        if n_iter % 2 == 1:
            tmp = grad_g.reshape(*B.shape, order='F')
            tmp = splinalg.svds(tmp, 1, tol=.1)[1][0]
            scale = min(1., alpha / tmp)
            M = grad_g * scale
            M = M.reshape(*B.shape, order='F')
            #assert linalg.norm(M, 2) <= alpha + 1e-7 # dual feasible
            pobj = trace_pobj(X, y, B, alpha, beta, s_vals)

            p, conj0 = conj_loss(X, y, Xy, M, beta, conj0)
            dual_gap = pobj + p
            # because we computed conj_loss approximately, dual_gap might happen to be negative
            dual_gap = np.abs(dual_gap)
            if verbose:
                print('Dual gap: %s' % dual_gap)
            gap.append(dual_gap)
            if np.abs(dual_gap) <= rtol:
                break


    return B, gap, obj_vals, time_vals

from low_rank_ import svd_power_method


def trace_frankwolfe(X, y, shape_B, max_iter=100):
    """
    trace norm constrained frank wolfe

    The model is:

        minimize ||y - Xb||^2_2

    subject to ||b.reshape(shape_B)||_* < 1

    TODO:
       * add regularization parameter
       * add duality gap
    """

    def f_obj(w):
        return .5 * (linalg.norm(y - X.dot(w)) ** 2)

    w = np.zeros(X.shape[1])
    obj_vals = []
    start = datetime.now()
    time_vals = []
    for i in range(max_iter):
        time_vals.append((datetime.now() - start).total_seconds())
        res = y - X.dot(w)
        obj_vals.append(0.5 * (linalg.norm(res) ** 2))
        grad = - X.T.dot(res)
        # u, v = svd_power_method(
        #     grad.reshape(shape_S, order='F'),
        #     s.reshape(shape_S, order='F'), 3)
        u, sv, v = splinalg.svds(grad.reshape(shape_B, order='F'), 1, tol=.1)
        #import ipdb; ipdb.set_trace()
        s = - np.outer(u, v, ).ravel('F')
        #q = X.dot(s)
        #alpha = q.dot(res) / q.dot(X.dot(s))
        alpha = 2. / (i + 2.)
        w = (1 - alpha) * w + alpha * s
    return w, obj_vals, time_vals


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(1000, 1000)
    shape_B = (100, 10)
    u, s, vt = linalg.svd(np.random.randn(*shape_B), full_matrices=False)
    w = np.outer(u[:, 0], vt[:, 0]).ravel('F') + \
        np.outer(u[:, 1], vt[:, 1]).ravel('F')
    w /= linalg.svdvals(w.reshape(shape_B, order='F')).sum()
    y = X.dot(w)
    y += (y.max() / 1.) * np.random.randn(X.shape[0])

    w_, obj_vals, time_vals = trace_frankwolfe(X, y, shape_B)
    obj_vals = [o - np.min(obj_vals) for o in obj_vals]
    alpha = 593
    import pylab as pl
    pl.plot(time_vals[:20], obj_vals[:20], color='orange',
            label='Frank Wolfe')
    pl.scatter(time_vals[:20], obj_vals[:20], color='orange')

    w2, _, obj_vals2, time_vals2 = trace(X, y, alpha, 0., shape_B,
                                         accelerated=False)
    obj_vals2 = [o - np.min(obj_vals2) for o in obj_vals2]
    pl.plot(time_vals2[:20], obj_vals2[:20], color='green',
            label='Proximal')
    pl.scatter(time_vals2[:20], obj_vals2[:20], color='green')


    w2, _, obj_vals2, time_vals2 = trace(X, y, alpha, 0., shape_B,
                                         accelerated=True)
    obj_vals2 = [o - np.min(obj_vals2) for o in obj_vals2]
    pl.plot(time_vals2[:20], obj_vals2[:20], color='blue',
            label='Proximal accelerated')
    pl.scatter(time_vals2[:20], obj_vals2[:20], color='blue')

    pl.legend()
    #pl.yscale('log')
    pl.axis('tight')
    pl.xlabel('Time (in seconds)')
    pl.ylabel('f(x_k)')
    pl.show()
