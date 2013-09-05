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


print('Loading data')
ds = np.DataSource(DIR)
X_train = scipy.io.mmread(ds.open('X_train.mtx')).tocsr()
# import ipdb; ipdb.set_trace()
# X_train = X_train[:X_train.shape[0] / 4, :X_train.shape[1] / 4]
X_test = scipy.io.mmread(ds.open('X_test.mtx')).tocsr()
#X_test = X_test[:, -X_train.shape[1]:]
#Y_train = scipy.io.mmread(ds.open('Y_train.mtx.gz'))
Y = np.load('Y_10000.npy')
n_task = 500
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


if len(sys.argv) > 1 and sys.argv[1] == 'OBO':
    print('GOING OBO')
    start = datetime.now()
    u0 = canonical
    import multitask as mt
    out = mt.rank_one_obo(
        X_train, Y_train, fir_length, u0=u0, v0=v0,
        verbose=False, plot=False, n_jobs=-1)
    print datetime.now() - start
    u, v = out

else:
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

pl.scatter(residuals_rank_one, residuals_canonical)
xx = np.linspace(np.min(residuals_canonical) - 1,
                 np.max(residuals_canonical) + 1)
pl.plot(xx, xx)
pl.xlabel('MSE glm Canonical')
pl.ylabel('MSE glm rank one OBO')
pl.xscale('log')
pl.yscale('log')
pl.show()
