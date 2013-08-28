import tempfile
import numpy as np
from datetime import datetime

import scipy
import scipy.io
import pylab as pl
from scipy import linalg, signal, sparse

from scipy.sparse import linalg as splinalg
from nipy.modalities.fmri import hemodynamic_models as hdm

DIR = tempfile.mkdtemp()
fir_length = 20
canonical = hdm.glover_hrf(1., 1., fir_length).reshape((-1, 1))


print('Loading data')
ds = np.DataSource(DIR)
print('X')
X_train = scipy.io.mmread(ds.open('X_train.mtx')).tocsr()
X_test = scipy.io.mmread(ds.open('X_test.mtx')).tocsr()
#Y_train = scipy.io.mmread(ds.open('Y_train.mtx.gz'))
Y = scipy.io.mmread(ds.open('Y.mtx'))
n_task = 10

# print('K_inv')
# K_inv = scipy.io.mmread(ds.open('K_inv.mtx')).tocsr()
#ridge_proj = lambda x: K_inv.dot(x)
# drifts = scipy.io.mmread(ds.open('drifts.mtx.gz'))
print('Done')



print('Detrending')
Y = scipy.signal.detrend(
    Y[:, :n_task], bp=np.linspace(0, X_train.shape[0], 7 * 5).astype(np.int),
    axis=0, type='linear')
Y_train = Y[:X_train.shape[0]]
Y_test = Y[X_train.shape[0]:]

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


import multitask as mt

print('Calling rank_one')


loss = []
timings = []


def callback(w):
    W = w.reshape((-1, 1), order='F')
    size_u = canonical.size
    size_v = X_train.shape[1] / size_u
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    tmp = Y_train - mt.low_rank_.matmat2(
        splinalg.aslinearoperator(X_train), u, v, n_task)
    loss.append(.5 * (linalg.norm(tmp) ** 2))
    timings.append((datetime.now() - start).total_seconds())

u0 = np.repeat(canonical, n_task).reshape((-1, n_task))

start = datetime.now()
out = mt.rank_one(
    X_train, Y_train, fir_length, u0=u0, v0=v0,
    rtol=1e-6, verbose=False, maxiter=500,
    callback=callback, plot=False, method='TNC')
print datetime.now() - start
u, v = out

norm_res = 0.
Iu = np.kron(np.eye(size_v, size_v), canonical)
# could be faster
Q = X_test.dot(Iu)
for i in range(n_task):
    v_tmp = linalg.lstsq(Q, Y_test[:, i])[0]
    res = linalg.norm(Y_test[:, i] - Q.dot(v_tmp))
    print((i, res))
    norm_res += res
print('RESIDUALS CANONICAL: %s' % norm_res)

norm_res = 0.
for i in range(n_task):
    Iu = np.kron(np.eye(size_v, size_v), u[:, i][:, None])
    # could be faster
    Q = X_test.dot(Iu)
    v0 = linalg.lstsq(Q, Y_test[:, i])[0]
    res = linalg.norm(Y_test[:, i] - Q.dot(v0))
    print((i, res))
    norm_res += res
print('RESIDUALS: %s' % norm_res)


# loss2 = []
# timings2 = []
# start = datetime.now()
# out = mt.rank_one(X, Y_train, 0, fir_length, u0=canonical, v0=v0, rtol=1e-12,
#                verbose=False, maxiter=500, callback=None)
#
# u, v = out
# norm_res = 0.
# for i in range(n_task):
#     Iu = np.kron(np.eye(size_v, size_v), u[:, i][:, None])
#     # could be faster
#     Q = X_test.dot(Iu)
#     v0 = linalg.lstsq(Q, Y_test[:, i])[0]
#     res = linalg.norm(Y_test[:, i] - Q.dot(v0))
#     print(res)
#     norm_res += res
# print('RESIDUALS: %s' % norm_res)

pl.figure()
pl.plot(timings, loss, label='Newton-CG', color='green')
pl.scatter(timings, loss, color='green')
# pl.plot(timings2, loss2, label='LBFGS', color='blue')
# pl.scatter(timings2, loss2, color='blue', marker='x')
pl.ylabel('Loss Function')
pl.xlabel('Time in seconds')
pl.legend()
pl.show()

# out = rank_one_proj2(X[train], Y_train[train], 0., fir_length,
#                      u0=canonical,
#                      maxiter=100, ls_proj=ls_proj, rtol=1e-6)

u_train, v_train = out

# now perform a GLM using the previous HRF
# total = 0.
# size_v = X.shape[1] / fir_length
# for i, u_t in enumerate(u_train.T):
#     u_t = u_t.reshape((-1, 1))
#     H = X[test].dot(np.kron(np.eye(size_v), u_t))
#     v_test = linalg.lstsq(H, Y_train[test, i])[0]
#     w_test = np.outer(u_t, v_test).ravel('F')
#     tmp = linalg.norm(Y_train[test, i] - X[test].dot(w_test))
#     print(tmp)
#     total += tmp
# print(total)


# fig = pl.figure()
# tmp = u_train.copy()
# sgn = np.sign(u_train.T.dot(canonical.ravel()))
# tmp *= sgn
# tmp = tmp / np.sqrt((tmp * tmp).sum(0))
# pl.plot(tmp)
# u = tmp
# pl.draw()


# from multitask.low_rank_ import rank_one
# out = rank_one(X, Y_train, 0, fir_length, u0=canonical, rtol=1e-6, verbose=False, maxiter=1000)
# u, v = out
#
# fig = pl.figure()
# tmp = u.copy()
# sgn = np.sign(u.T.dot(canonical.ravel()))
# tmp *= sgn
# tmp = tmp / np.sqrt((tmp * tmp).sum(0))
# pl.plot(tmp)
# u = tmp
# pl.draw()
#
#
# from sklearn import cluster
# km = cluster.KMeans(2)
# km.fit(u.T)
# np.save('cluster.npy', km.labels_)

