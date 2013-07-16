
import tempfile

import scipy
import scipy.io
import pylab as pl
from scipy import linalg, signal
from sklearn import cross_validation

import hrf_estimation as he # pip install -U hrf_estimation
from nipy.modalities.fmri import hemodynamic_models as hdm

import numpy as np


DIR = tempfile.mkdtemp()
fir_length = 20
canonical = hdm.glover_hrf(1., 1., fir_length)

ds = np.DataSource(DIR)
X = scipy.io.mmread(ds.open('X.mtx.gz')).tocsr()
Y = scipy.io.mmread(ds.open('Y_1000.mtx.gz'))
drifts = scipy.io.mmread(ds.open('drifts_1000.mtx.gz'))

k = X.shape[0] / 5
Y = scipy.signal.detrend(Y, bp=np.linspace(0, X.shape[0], 5 * 5).astype(np.int), axis=0, type='linear')


from multitask.low_rank_ import rank_one_proj2

out = rank_one_proj2(X, Y[:, :100], 0., fir_length, u0=canonical)