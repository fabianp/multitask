import numpy as np
import multitask as mt
from scipy import linalg
import pylab as pl

n_samples, n_features = 1000, 100
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

res = []
for alpha in np.logspace(-1, 3, 400):
    print('alpha: %s' % alpha)
    B_, gap_ = mt.trace(X, y, alpha, 0., (10, 10), rtol=1e-10, verbose=True, max_iter=1000)
    s = linalg.svdvals(B_)
    pl.scatter(alpha * np.ones_like(s), s)
ax = pl.gca()
ax.set_xscale('log')
pl.show()