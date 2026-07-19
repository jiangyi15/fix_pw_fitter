"""Gaussian Process with RBF kernel for likelihood profiling."""

import numpy as np
from scipy.optimize import minimize


class GP:
    """Gaussian Process with RBF kernel and per-point noise."""

    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=1e-4):
        self.length_scale = np.atleast_1d(np.asarray(length_scale, float))
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self._X = None
        self._y = None
        self._L = None
        self._alpha = None

    def _k(self, x1, x2):
        sq = 0.0
        for d in range(x1.shape[1]):
            diff = x1[:, d:d+1] / self.length_scale[d] - x2[:, d:d+1].T / self.length_scale[d]
            sq += diff ** 2
        return self.sigma_f**2 * np.exp(-0.5 * sq)

    def fit(self, X, y, sigma_n_vec=None):
        """Train the GP. K = k(X,X) + diag(sigma_n_vec^2)."""
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        self._X = X
        self._y = y
        if sigma_n_vec is not None:
            sigma_n_vec = np.asarray(sigma_n_vec, float).ravel()
            K = self._k(X, X) + np.diag(sigma_n_vec**2)
        else:
            K = self._k(X, X) + self.sigma_n**2 * np.eye(len(X))
        K += np.eye(len(K)) * 1e-8  # numerical jitter
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        return self

    def predict(self, X, return_std=True):
        """Predict mean and std at new points.

        Uses O(n_pts * n_train) memory — no full N² matrix.
        """
        X = np.atleast_2d(np.asarray(X, float))
        K_s = self._k(X, self._X)
        mu = K_s @ self._alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self._L, K_s.T)
        var = self.sigma_f**2 - np.sum(v**2, axis=0)
        return mu, np.sqrt(np.maximum(var, 0))

    def optimize(self, X, y, sigma_n_vec=None):
        """MLE for length_scale and sigma_f (sigma_n fixed from per-point noise)."""
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        n = len(X)
        n_dims = X.shape[1]

        def neg_log_likelihood(hp):
            ls = np.exp(hp[:n_dims])
            sf = np.exp(hp[n_dims])
            old_ls, old_sf = self.length_scale.copy(), self.sigma_f
            self.length_scale = ls
            self.sigma_f = sf
            if sigma_n_vec is not None:
                K = self._k(X, X) + np.diag(sigma_n_vec**2)
            else:
                K = self._k(X, X) + self.sigma_n**2 * np.eye(n)
            K += np.eye(n) * 1e-8
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                self.length_scale, self.sigma_f = old_ls, old_sf
                return 1e12
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            self.length_scale, self.sigma_f = old_ls, old_sf
            return nll

        x0 = np.log(np.maximum(np.concatenate([self.length_scale, [self.sigma_f]]), 1e-6))
        bounds = [(-5, 5)] * n_dims + [(-2, 8)]
        res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.success:
            self.length_scale = np.exp(res.x[:n_dims])
            self.sigma_f = float(np.exp(res.x[n_dims]))
        self.fit(X, y, sigma_n_vec=sigma_n_vec)
        return self
