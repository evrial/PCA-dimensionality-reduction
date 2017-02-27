# -*- coding: utf-8 -*-
import numpy as np


def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs


class PCA(object):
    """
    Principal Component Analysis

    Parameters
    ----------
    n_components : int
        Number of components to keep.
        if n_components is not set all components are kept

    Attributes
    ----------
    explained_variance : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_cumsum : array, [n_components]
        Cumulative sum of explained variance for every component.

    total_var : float
        Total variance explained. Equal to `sum(explained_variance)`

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components.

    Example usage
    -------------
    model = PCA(2)
    eig_projection = model.eigen(X)
    svd_projection = model.svd(X)
    svd_projection2D = model(X)  # callable class
    eig_projection2D = model(X, method='eig')  # Eigenvector method
    svd_projection2D[0]  # first component
    """
    def __init__(self, n_components=None):
        self.n_components = n_components

    def __call__(self, X, method='svd'):
        if method == 'svd':
            return self.svd(X)
        return self.eigen(X)

    @staticmethod
    def scale(X):
        """
        Standardize features by removing the mean and scaling to unit variance
        """
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X_std

    def eigen(self, X):
        """
        Compute the eigenvalues and right eigenvectors of a square array.
        Project them to new feature space
        """
        n_samples, n_features = X.shape
        # Estimate if n_components not specified
        self.n_components_ = self.n_components or n_features
        self.mean_ = np.mean(X, axis=0)

        k = self.n_components

        # Standardizing the data
        X = self.scale(X)

        # Eigendecomposition - Computing Eigenvectors and Eigenvalues

        # Covariance matrix after standardizing the data
        # equals to correlation matrix of raw or std data
        cov_mat = np.corrcoef(X, rowvar=False)

        print('Covariance matrix: \n%s' % cov_mat)

        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Sorting eigenpairs wrt. eigenvalues
        idx = eig_vals.argsort()[::-1]
        eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]

        # the eigenvalues in decreasing order
        print("Eigenvalues:\n", eig_vals[:k])
        # a matrix of eigenvectors (each column is an eigenvector)
        print("Eigenvectors:\n", eig_vecs[:k])

        # Explained Variance
        self.explained_variance = (eig_vals / np.sum(eig_vals))[:k]
        self.explained_variance_cumsum = np.cumsum(self.explained_variance)
        self.total_var = np.sum(self.explained_variance)

        print('Total variance explained:\n', self.total_var)
        print('Explained variance:\n', self.explained_variance)
        print('Explained variance cumsum:\n', self.explained_variance_cumsum)

        # Projection matrix and dimensionality reduction
        W = eig_vecs[:, 0:k]
        assert W.shape == (n_features, self.n_components_)

        # projections of X on the principal axes are called principal components
        PC_k = X.dot(W)
        assert PC_k.shape == (n_samples, self.n_components_)

        # expose variables as attributes for unit testing
        self.X = X
        self.cov_mat = cov_mat
        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs

        return PC_k

    def svd(self, X):
        """
        Singular Value Decomposition
        """
        n_samples, n_features = X.shape
        # Estimate if n_components not specified
        self.n_components_ = self.n_components or n_features
        self.mean_ = np.mean(X, axis=0)

        k = self.n_components

        # Standardizing the data
        X = self.scale(X)

        # we now perform singular value decomposition of X
        # "economy size" (or "thin") SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T
        S = np.diag(s)

        # 1) columns of V are principal directions/axes.
        # np.testing.assert_allclose(*flip_signs(V, principal_axes))

        # 2) columns of US are principal components
        # np.testing.assert_allclose(*flip_signs(U.dot(S), principal_components), rtol=1e-04)

        # 3) singular values are related to the eigenvalues of covariance matrix
        # np.testing.assert_allclose((s ** 2) / (n - 1), eig_vals)

        eig_vals = (s ** 2) / (n_features - 1)

        # Explained Variance
        self.explained_variance = (eig_vals / np.sum(eig_vals))[:k]
        self.explained_variance_cumsum = np.cumsum(self.explained_variance)
        self.total_var = np.sum(self.explained_variance)

        # dimensionality reduction
        US_k = U[:, 0:k].dot(S[0:k, 0:k])
        assert US_k.shape == (n_samples, self.n_components_)

        # expose variables as attributes for unit testing
        self.X = X
        self.U = U
        self.s = s
        self.Vt = Vt
        self.V = V
        self.S = S

        return US_k
