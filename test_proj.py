import tempfile

import scipy
import scipy.io
import pylab as pl
from scipy import linalg, signal

from scipy.sparse import linalg as splinalg

DIR = tempfile.mkdtemp()
fir_length = 20
canonical = hdm.glover_hrf(1., 1., fir_length)


print('Loading data')
ds = np.DataSource(DIR)
print('X')
X = scipy.io.mmread(ds.open('X.mtx')).tocsr()
print('nullspace')
X_nullspace = scipy.io.mmread(ds.open('X_nullspace.mtx')).tocsr()
# print('K_inv')
# K_inv = scipy.io.mmread(ds.open('K_inv.mtx')).tocsr()
#ridge_proj = lambda x: K_inv.dot(x)
print('Y')
Y = np.load('Y_10000.npy')
# drifts = scipy.io.mmread(ds.open('drifts.mtx.gz'))
print('Done')


print('Detrending')
Y = scipy.signal.detrend(Y, bp=np.linspace(0, X.shape[0], 7 * 5).astype(np.int),
                          axis=0, type='linear')
Y = Y[:, :500]

# .. standardize ..
#Y = Y.reshape((35, -1, Y.shape[1]))
#Y = (Y - Y.mean(axis=1)[:, np.newaxis, :]) / Y.std(axis=1)[:, np.newaxis, :]
#Y = Y.reshape((-1, Y.shape[2]))
print('Done')

from multitask.low_rank_ import rank_one_proj2

print('Calling rank_one_proj2')
out = rank_one_proj2(X, Y, 0., fir_length, u0=canonical, maxiter=100)
u, v = out

fig = pl.figure()
tmp = u.copy()
sgn = np.sign(u.T.dot(canonical.ravel()))
tmp *= sgn
tmp = tmp / np.sqrt((tmp * tmp).sum(0))
pl.plot(tmp)
u = tmp
pl.draw()


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

