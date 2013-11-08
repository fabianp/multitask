
import tempfile
import numpy as np
from datetime import datetime

import scipy
import scipy.io
import pylab as pl
from scipy import linalg, signal, sparse

from scipy.sparse import linalg as splinalg
from nipy.modalities.fmri import hemodynamic_models as hdm

import multitask as mt
import hrf_estimation as he


DIR = tempfile.mkdtemp()
fir_length = 20
canonical = hdm.glover_hrf(1., 1., fir_length).reshape((-1, 1))


# load data -- 
print('Loading data')
ds = np.DataSource(DIR)
X_tmp = scipy.io.mmread(ds.open('X_train.mtx')).tocsr()
Y = np.load('Y_10000.npy')
X_session = X_tmp[:X_tmp.shape[0] / 4, :X_tmp.shape[1] / 5]
for n_session, Y_session in enumerate(np.split(Y, 5)):
    k = X_session.shape[0] / 7.
    X_train = X_session[:5 * k]
    X_test = X_session[5 * k:]
    Y_train = Y_session[:5 * k]
    Y_test = Y_session[5 * k:]
    n_task = 5
    print('Done')

    print('Detrending')
    from multitask.savitzky_golay import savgol_filter
    Y_train = Y_train.reshape((5, 672, -1))
    Y_train = Y_train[..., :n_task] - savgol_filter(
        Y_train[..., :n_task], 91, 4, axis=1)
    Y_train = scipy.signal.detrend(Y_train, axis=1)
    Y_test = Y_test.reshape((2, 672, -1))
    Y_test = Y_test[..., :n_task] - savgol_filter(
        Y_test[..., :n_task], 91, 4, axis=1)
    Y_test = scipy.signal.detrend(Y_test, axis=1)
    norm_train = np.sqrt((Y_train * Y_train).sum(1))
    norm_test = np.sqrt((Y_test * Y_test).sum(1))
    for i in range(5):
        Y_train[i] /= norm_train[i]
    for i in range(2):
        Y_test[i] /= norm_test[i]

    Y_train = Y_train.reshape((3360, n_task))
    Y_test = Y_test.reshape((1344, n_task))

    size_u = fir_length
    size_v = X_train.shape[1] / size_u
    if True:
        print('Precomputing initialization point')
        Iu = np.kron(np.eye(size_v, size_v), canonical)  # could be faster
        Q = X_train.dot(Iu)
        v0 = linalg.lstsq(Q, Y_train)[0]
        np.save('v0.npy', v0)
    else:
        v0 = np.load('v0.npy')


    print('Calling rank_one')
    u0 = canonical

    #print('GOING OBO')
    #start = datetime.now()
    #u0 = canonical
    #out = mt.rank_one_obo(
    #    X_train, Y_train, fir_length, u0=u0, v0=v0,
    #    verbose=False, plot=False, n_jobs=1)
    #print datetime.now() - start
    #u_obo, v = out

    print('RANK ONE CLASSIC SETTING')
    u = u0
    v = v0[:, 0]
    for i, y_train in enumerate(Y_train.T):
        pl.figure()
        y_train = y_train[:, None]
        # compute optimal point
        u0, v0 = u, v # the one from previous voxel
        u, v = he.rank_one(
                X_train, y_train, fir_length, u0=u0, v0=v0,
                verbose=1, method='TNC', rtol=1e-32)
        uv0 = he.khatri_rao(v, u)
        min_loss = .5 * (linalg.norm(y_train - X_train.dot(uv0)) ** 2)
        for solver in ('TNC', 'Newton-CG', 'L-BFGS-B', 'trust-ncg', 'CG'):
            loss = []
            timings = []
            def callback(w):
                W = w.reshape((-1, 1), order='F')
                u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
                uv0 = he.khatri_rao(v, u)
                loss.append(.5 * (linalg.norm(y_train - X_train.dot(uv0)) **
                                  2))
                timings.append((datetime.now() - start).total_seconds())
            start = datetime.now()
            u0 = canonical
            out = he.rank_one(
                X_train, y_train, fir_length, u0=u0, v0=v0,
                verbose=1, callback=callback, method=solver, rtol=1e-12)
            print datetime.now() - start
            u, v = out
            pl.plot(timings, loss - min_loss, label=solver, lw=4)
            #pl.scatter(timings, loss, marker='x', color='black', lw=.5)

        pl.legend(loc='lower left')
        #pl.xlim((0, .15))
        #
        pl.yscale('log')
        #pl.ylim((1e-6, 1))
        pl.axis('tight')
        pl.ylabel(r'$f(x_k) - f(x^{*})$', fontsize='x-large')
        pl.xlabel('Time (in seconds)', fontsize='x-large')
        pl.savefig('bench_r1.png', transparent=True)
        pl.show()
    import ipdb; ipdb.set_trace()

    residuals_rank_one = []
    for i in range(n_task):
        Iu = sparse.kron(sparse.eye(size_v, size_v), u[:, i][:, None])
        # could be faster
        Q = X_test.dot(Iu)
        v0 = splinalg.lsqr(Q, Y_test[:, i])[0]
        res = linalg.norm(Y_test[:, i] - Q.dot(v0))
        residuals_rank_one.append(res)
        print('Done %s out of %s' % (i, n_task))
    print('RESIDUALS RANK ONE: %s' % np.sum(residuals_rank_one))

    residuals_rank_one_obo = []
    for i in range(n_task):
        Iu = sparse.kron(sparse.eye(size_v, size_v), u_obo[:, i][:, None])
        # could be faster
        Q = X_test.dot(Iu)
        v0 = splinalg.lsqr(Q, Y_test[:, i])[0]
        res = linalg.norm(Y_test[:, i] - Q.dot(v0))
        residuals_rank_one_obo.append(res)
        print('Done %s out of %s' % (i, n_task))
    print('RESIDUALS RANK ONE OBO: %s' % np.sum(residuals_rank_one_obo))

    residuals_canonical = []
    Iu = np.kron(np.eye(size_v, size_v), canonical)
    # could be faster
    Q = X_test.dot(Iu)
    v_tmp = linalg.lstsq(Q, Y_test)[0]
    for i in range(n_task):
        res = linalg.norm(Y_test[:, i] - Q.dot(v_tmp[:, i]))
        residuals_canonical.append(res)
        print('Done %s out of %s' % (i, n_task))
    print('RESIDUALS CANONICAL: %s' % np.sum(residuals_canonical))


    pl.figure()
    pl.title('GLM rank one vs GLM canonical, session %s' % n_session)
    pl.scatter(residuals_rank_one, residuals_canonical)
    xx = np.linspace(min(pl.ylim()[0], pl.xlim()[0]),
                     max(pl.ylim()[1], pl.xlim()[1]))
    pl.plot(xx, xx)
    pl.xlabel('MSE glm rank one')
    pl.ylabel('MSE glm Canonical')
    pl.axis('tight')
    pl.savefig('mse_r1_vs_can_%03d.png' % n_session)
#    pl.show()

    pl.figure()
    pl.title('GLM rank one OBO vs GLM canonical, session %s' % n_session)
    pl.scatter(residuals_rank_one_obo, residuals_canonical)
    xx = np.linspace(min(pl.ylim()[0], pl.xlim()[0]),
                     max(pl.ylim()[1], pl.xlim()[1]))
    pl.plot(xx, xx)
    pl.xlabel('MSE glm rank one OBO')
    pl.ylabel('MSE glm Canonical')
    pl.axis('equal')
    pl.axis('tight')
    pl.savefig('mse_obo_vs_can_%03d.png' % n_session)
#    pl.show()


    pl.figure()
    pl.title('GLM rank one OBO vs GLM rank one, session %s' % n_session)
    pl.scatter(residuals_rank_one_obo, residuals_rank_one)
    xx = np.linspace(min(pl.ylim()[0], pl.xlim()[0]),
                     max(pl.ylim()[1], pl.xlim()[1]))
    pl.plot(xx, xx)
    pl.xlabel('MSE glm rank one OBO')
    pl.ylabel('MSE glm rank one')
    pl.axis('equal')
    pl.axis('tight')
    pl.savefig('mse_obo_vs_r1_%03d.png' % n_session)
#    pl.show()