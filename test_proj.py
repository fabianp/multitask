import tempfile
import numpy as np

import scipy
import scipy.io
import pylab as pl
from scipy import linalg, signal

from scipy.sparse import linalg as splinalg
from nipy.modalities.fmri import hemodynamic_models as hdm

DIR = tempfile.mkdtemp()
fir_length = 20
canonical = hdm.glover_hrf(1., 1., fir_length)


print('Loading data')
ds = np.DataSource(DIR)
print('X')
X = scipy.io.mmread(ds.open('X.mtx')).tocsr()
print('nullspace')
X_nullspace = scipy.io.mmread(ds.open('X_nullspace.mtx')).tocsr()
#Y = scipy.io.mmread(ds.open('Y.mtx.gz'))
Y = np.load('Y_10000.npy')
#Y = Y[:, :1000]
# print('K_inv')
# K_inv = scipy.io.mmread(ds.open('K_inv.mtx')).tocsr()
#ridge_proj = lambda x: K_inv.dot(x)
# drifts = scipy.io.mmread(ds.open('drifts.mtx.gz'))
print('Done')


print('Detrending')
Y = scipy.signal.detrend(
    Y, bp=np.linspace(0, X.shape[0], 7 * 5).astype(np.int),
    axis=0, type='linear')


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

from multitask.low_rank_ import rank_one_proj2, rank_one, rank_one_gradproj

#from sklearn import cross_validation
#cv = cross_validation.KFold(Y.shape[0], n_folds=20, shuffle=False)
train = np.arange(X.shape[0] - 10)
test = np.arange(X.shape[0] - 10, X.shape[0])
#train, test = iter(cv).next()
print('Calling rank_one_proj2')

out = rank_one_gradproj(X, Y, 0, fir_length, u0=canonical, rtol=1e-6,
                    verbose=False, maxiter=100)

# out = rank_one(X, Y, 0, fir_length, u0=canonical, rtol=1e-6, verbose=False,
#                maxiter=100)
#
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

