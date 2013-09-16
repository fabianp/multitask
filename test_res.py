import sys
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


# load data -- 
print('Loading data')
ds = np.DataSource(DIR)
X_tmp = scipy.io.mmread(ds.open('X_train.mtx')).tocsr()
Y = np.load('Y_10000.npy')
X_session = X_tmp[:X_tmp.shape[0] / 4, :X_tmp.shape[1] / 5]
Y_session = Y[:X_session.shape[0]]
k = X_session.shape[0] / 7.
X_train = X_session[:5 * k]
X_test = X_session[5 * k:]
Y_train = Y_session[:5 * k]
Y_test = Y_session[5 * k:]
n_task = 500
print('Done')


print('Detrending')
from multitask.savitzky_golay import savgol_filter

if False:
    Y_train = scipy.signal.detrend(
        Y_train[:, :n_task], bp=np.linspace(0, X_train.shape[0], 5).astype(np.int),
        axis=0, type='linear')
    Y_test = scipy.signal.detrend(
        Y_test[:, :n_task], bp=np.linspace(0, X_test.shape[0], 2).astype(np.int),
        axis=0, type='linear')
else:
    Y_train = Y_train.reshape((5, 672, -1))
    Y_train = savgol_filter(Y_train[..., :n_task], 91, 4, axis=1)
    Y_test = Y_test.reshape((2, 672, -1))
    Y_test = savgol_filter(Y_test[..., :n_task], 91, 4, axis=1)

Y_train = Y_train.reshape((3360, -1))
Y_test = Y_test.reshape((1344, -1))

size_u = fir_length
size_v = X_train.shape[1] / size_u
if False:
    print('Precomputing initialization point')
    Iu = np.kron(np.eye(size_v, size_v), canonical)  # could be faster
    Q = X_train.dot(Iu)
    v0 = linalg.lstsq(Q, Y_train)[0]
    np.save('v0.npy', v0)
else:
    v0 = np.load('v0.npy')


import multitask as mt

print('Calling rank_one')

def classic_to_obo(classic_design, fir_length=1):
    """
    Will convert a classic or fir design to the one by one setting.
    Returns one matrix per event, containing event related regressors
    on the left and sum of remaining regressors on the right.
    """

    event_regressors = classic_design.reshape(
        len(classic_design), -1, fir_length)

    regressor_sum = event_regressors.sum(axis=1)

    remaining_regressors = (regressor_sum[:, np.newaxis, :] -
                            event_regressors)
    together = np.concatenate([event_regressors,
                               remaining_regressors], axis=2)

    return together.transpose(1, 0, 2)

loss = []
timings = []
def callback(w):
    W = np.asarray(w).reshape((-1, 1), order='F')
    size_u = canonical.size
    size_v = X_train.shape[1] / size_u
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    tmp = Y_train - mt.low_rank_.matmat2(
        splinalg.aslinearoperator(X_train), u, v, n_task)
    loss.append(.5 * (linalg.norm(tmp) ** 2))
    timings.append((datetime.now() - start).total_seconds())
u0 = np.repeat(canonical, n_task).reshape((-1, n_task))

going_obo = len(sys.argv) > 1 and sys.argv[1] == 'OBO'

print('GOING OBO')
start = datetime.now()
u0 = canonical
import multitask as mt
out = mt.rank_one_obo(
    X_train, Y_train, fir_length, u0=u0, v0=v0,
    verbose=False, plot=False, n_jobs=1)
print datetime.now() - start
u_obo, v = out

print('RANK ONE CLASSIC SETTING')
start = datetime.now()
u0 = canonical
import multitask as mt
import hrf_estimation as he
out = he.rank_one(
    X_train, Y_train, fir_length, u0=u0, v0=v0,
    verbose=True)
print datetime.now() - start
u, v = out

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
pl.scatter(residuals_rank_one, residuals_canonical)
xx = np.linspace(np.min(residuals_canonical) - 1,
                 np.max(residuals_canonical) + 1)
pl.plot(xx, xx)
pl.xlabel('MSE glm rank one')
pl.ylabel('MSE glm Canonical')
pl.xscale('log')
pl.yscale('log')
pl.axis('tight')
pl.show()

pl.figure()
pl.scatter(residuals_rank_one_obo, residuals_canonical)
xx = np.linspace(np.min(residuals_canonical) - 1,
                 np.max(residuals_canonical) + 1)
pl.plot(xx, xx)
pl.xlabel('MSE glm rank one OBO')
pl.ylabel('MSE glm Canonical')
pl.xscale('log')
pl.yscale('log')
pl.axis('tight')
pl.show()


pl.figure()
pl.scatter(residuals_rank_one_obo, residuals_rank_one)
xx = np.linspace(np.min(residuals_rank_one_obo) - 1,
                 np.max(residuals_rank_one_obo) + 1)
pl.plot(xx, xx)
pl.xlabel('MSE glm rank one OBO')
pl.ylabel('MSE glm rank one')
pl.xscale('log')
pl.yscale('log')
pl.axis('tight')
pl.show()