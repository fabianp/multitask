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
X = scipy.io.mmread(ds.open('X.mtx')).tocsr()
print('nullspace')
X_nullspace = scipy.io.mmread(ds.open('X_nullspace.mtx')).tocsr()
#Y = scipy.io.mmread(ds.open('Y.mtx.gz'))
Y = np.load('Y_10000.npy')
Y = Y[:, :500]
# print('K_inv')
# K_inv = scipy.io.mmread(ds.open('K_inv.mtx')).tocsr()
#ridge_proj = lambda x: K_inv.dot(x)
# drifts = scipy.io.mmread(ds.open('drifts.mtx.gz'))
print('Done')


print('Detrending')
Y = scipy.signal.detrend(
    Y, bp=np.linspace(0, X.shape[0], 7 * 5).astype(np.int),
    axis=0, type='linear')

print('Precomputing initialization point')
size_u = fir_length
size_v = X.shape[1] / size_u
K = X.T.dot(X)
K = K + sparse.eye(*K.shape)
tmp = np.kron(np.eye(size_v, size_v), canonical) # could be faster
Q = X.dot(tmp)
v0 = linalg.lstsq(Q, Y)[0]

# ls_sol = [splinalg.lsqr(X, Y[:, i])[0] for i in range(Y.shape[1])]
# ls_sol = np.array(ls_sol).T
# def ls_proj(x):
#     proj = ls_sol + X_nullspace.dot(X_nullspace.T.dot(x))
#     return proj

# .. standardize ..
#Y = Y.reshape((35, -1, Y.shape[1]))
#Y = (Y - Y.mean(axis=1)[:, np.newaxis, :]) / Y.std(axis=1)[:, np.newaxis, :]
#Y = Y.reshape((-1, Y.shape[2]))
print('Done')

from multitask.low_rank_ import rank_one_proj2, rank_one, rank_one_gradproj, khatri_rao

#from sklearn import cross_validation
#cv = cross_validation.KFold(Y.shape[0], n_folds=20, shuffle=False)
train = np.arange(X.shape[0] - 10)
test = np.arange(X.shape[0] - 10, X.shape[0])
#train, test = iter(cv).next()
print('Calling rank_one_proj2')


loss = []
timings = []

def callback(w):
    loss.append(.5 * (linalg.norm(Y - X.dot(w)) ** 2))
    timings.append((datetime.now() - start).total_seconds())

start = datetime.now()
out = rank_one_gradproj(X, Y, 0, fir_length, u0=canonical, v0=v0, rtol=1e-6,
                    verbose=False, maxiter=100, ls_proj=None,
                    callback=callback)


loss2 = []
timings2 = []
def cb2(w):
    n_task = Y.shape[1]
    size_u = fir_length
    size_v = X.shape[1] / fir_length
    W = w.reshape((-1, n_task), order='F')
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    w = khatri_rao(v, u)
    loss2.append(.5 * (linalg.norm(Y - X.dot(w)) ** 2))
    print('LOSS: %s' % loss2[-1])
    timings2.append((datetime.now() - start).total_seconds())
start = datetime.now()
out = rank_one(X, Y, 0, fir_length, u0=canonical, v0=v0, rtol=1e-6,
               verbose=False, maxiter=100, callback=cb2)

pl.figure()
pl.plot(timings, loss, label='Projected Nesterov-CG', color='green')
pl.scatter(timings, loss, color='green')
pl.plot(timings2, loss2, label='LBFGS', color='blue')
pl.scatter(timings2, loss2, color='blue', marker='x')
pl.ylabel('Loss Function')
pl.xlabel('Time in seconds')
pl.legend()
pl.show()

# out = rank_one_proj2(X[train], Y[train], 0., fir_length,
#                      u0=canonical,
#                      maxiter=100, ls_proj=ls_proj, rtol=1e-6)

u_train, v_train = out

# now perform a GLM using the previous HRF
# total = 0.
# size_v = X.shape[1] / fir_length
# for i, u_t in enumerate(u_train.T):
#     u_t = u_t.reshape((-1, 1))
#     H = X[test].dot(np.kron(np.eye(size_v), u_t))
#     v_test = linalg.lstsq(H, Y[test, i])[0]
#     w_test = np.outer(u_t, v_test).ravel('F')
#     tmp = linalg.norm(Y[test, i] - X[test].dot(w_test))
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
# out = rank_one(X, Y, 0, fir_length, u0=canonical, rtol=1e-6, verbose=False, maxiter=1000)
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

