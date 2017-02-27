import numpy as np
import pandas as pd
from pca import PCA, flip_signs

# Because of numerical difference other datasets may need float tolerance tuning
# to pass all tests
X = pd.read_csv('/Users/evrial/Downloads/price_csv/yahooBars_SPY_1.csv')
X = X.drop('Date', axis=1).as_matrix()

n_samples, n_features = X.shape

pca = PCA()
eig_projected = pca.eigen(X)
svd_projected = pca.svd(X)
eig_projected, svd_projected = flip_signs(eig_projected, svd_projected)

np.testing.assert_allclose(eig_projected, svd_projected, rtol=1e-04)

assert pca.cov_mat.shape == (n_features, n_features)
np.testing.assert_allclose(pca.cov_mat, np.cov(pca.X, rowvar=False), rtol=1e-03)

np.testing.assert_almost_equal(np.sum(pca.eig_vals), len(pca.eig_vals))

np.testing.assert_allclose(pca.cov_mat.dot(pca.eig_vecs), pca.eig_vals * pca.eig_vecs, rtol=1e-05)
for ev in pca.eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

assert pca.U.shape == (n_samples, n_features)
assert pca.S.shape == (n_features, n_features)
assert pca.V.shape == (n_features, n_features)

np.testing.assert_allclose(pca.X, np.dot(pca.U, np.dot(pca.S, pca.Vt)))

# 1) then columns of V are principal directions/axes.
np.testing.assert_allclose(*flip_signs(pca.V, pca.eig_vecs))

# 2) columns of US are principal components
np.testing.assert_allclose(*flip_signs(pca.U.dot(pca.S), eig_projected), rtol=1e-04)

# 3) singular values are related to the eigenvalues of covariance matrix
np.testing.assert_allclose((pca.s ** 2) / (n_samples - 1), pca.eig_vals, rtol=1e-03)

print('Tests passed')
