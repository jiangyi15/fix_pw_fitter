"""
fitter.py  –  Combined Parameters + FpwFitter for end-to-end fitting.

The Fitter class connects real-valued optimizer parameters x to
complex coupling coefficients c_k, and computes -log L and its gradient.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .parameters import Parameters
from .fit_fractions import FitFractions


class Fitter:
    """Combines Parameters + FpwFitter for end-to-end fitting.
    
    A thin wrapper that connects parameter conversion (x → c_k)
    to fitter evaluation (c_k → -log L, d(-log L)/dc*).

    Works with any fitter that has an evaluate(c) method returning
    (nll, grad) where grad = d(-log L)/dc*.

    After calling fit(), results are cached and accessible via
    convenience methods (couplings, uncertainties, correlations, etc.).
    """

    def __init__(self, parameters: Parameters, fitter: Any):
        """
        Args:
            parameters: Parameters instance defining the mapping x → c_k
            fitter: already-instantiated fitter (FpwFitter, FpwFitterChunked,
                    FpwFitterMP, NumpyFitter, etc.) — must have evaluate(c)
                    returning (nll, grad) where grad = d(-log L)/dc*.
        """
        self.parameters = parameters
        self.fitter = fitter
        self._result: Optional[OptimizeResult] = None

    def _get_cov_matrix(self) -> np.ndarray:
        """Retrieve or compute the covariance matrix (inverse Hessian)."""
        if hasattr(self, '_result') and self._result is not None:
            res = self._result
            if hasattr(res, 'hess_inv') and res.hess_inv is not None:
                return np.asarray(res.hess_inv)
        
        # Fallback to numerical calculation if not available
        try:
            return self.compute_hess_inv()
        except Exception:
            return np.eye(self.parameters.n_free)

    def _get_fit_fractions_obj(
        self,
        M: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> FitFractions:
        """Helper to create a FitFractions instance with current context."""
        if M is None:
            if hasattr(self.fitter, "_M"):
                M = self.fitter._M
            else:
                raise ValueError(
                    "M is not available. Please pass M explicitly."
                )

        if cov_matrix is None:
            cov_matrix = self._get_cov_matrix()

        return FitFractions(self.parameters, M, cov_matrix=cov_matrix)

    def get_couplings(self, x: np.ndarray) -> np.ndarray:
        """Get complex coupling coefficients c_k from real params x."""
        return self.parameters.build_c(x)

    def objective(self, x: np.ndarray) -> float:
        """Compute -log L from real parameters x."""
        c = self.parameters.build_c(x)
        nll, _ = self.fitter.evaluate(c)
        return float(nll)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute d(-log L)/dx from real parameters x."""
        c = self.parameters.build_c(x)
        _, g = self.fitter.evaluate(c)
        return self.parameters.gradient_chain_rule(g, x)

    def objective_and_gradient(
        self, x: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Compute both -log L and gradient efficiently."""
        c = self.parameters.build_c(x)
        nll, g = self.fitter.evaluate(c)
        grad = self.parameters.gradient_chain_rule(g, x)
        return float(nll), grad

    def fit(
        self,
        x0: Optional[np.ndarray] = None,
        method: str = "BFGS",
        options: Optional[dict] = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run optimization using scipy.optimize.minimize."""
        if x0 is None:
            n = self.parameters.n_free
            x0 = np.zeros(n)
            x0[0::2] = 1.0

        if options is None:
            options = {}

        self._result = minimize(
            fun=self.objective_and_gradient,
            x0=x0,
            method=method,
            jac=True,
            options=options,
            **kwargs,
        )
        return self._result

    def uncertainties(self, result: Optional[OptimizeResult] = None) -> np.ndarray:
        """Extract parameter uncertainties from fit result."""
        if result is None:
            result = self._require_fit()

        if not hasattr(result, "hess_inv") or result.hess_inv is None:
            raise ValueError(
                "No hess_inv in result. "
                "Use method='BFGS' or 'L-BFGS-B' (with options) "
                "to get uncertainty estimates."
            )

        hess_inv = np.asarray(result.hess_inv)
        hess_inv = (hess_inv + hess_inv.T) / 2.0
        variances = np.maximum(np.diag(hess_inv), 0.0)
        return np.sqrt(variances)

    # ---- Cached result convenience methods ----

    @property
    def result(self) -> OptimizeResult:
        """Cached fit result."""
        return self._require_fit()

    @property
    def best_nll(self) -> float:
        """Best -log L value from fit."""
        return float(self._require_fit().fun)

    @property
    def best_x(self) -> np.ndarray:
        """Best real parameter values from fit."""
        return self._require_fit().x.copy()

    def best_couplings(self) -> np.ndarray:
        """Complex coupling coefficients c_k at best fit."""
        return self.parameters.build_c(self._require_fit().x)

    def correlations(self, result: Optional[OptimizeResult] = None) -> np.ndarray:
        """Parameter correlation matrix from fit."""
        if result is None:
            result = self._require_fit()

        hess_inv = np.asarray(result.hess_inv)
        hess_inv = (hess_inv + hess_inv.T) / 2.0
        variances = np.maximum(np.diag(hess_inv), 0.0)
        sigma = np.sqrt(variances)
        sigma = np.where(sigma > 0, sigma, 1.0)
        
        corr = hess_inv / np.outer(sigma, sigma)
        return np.clip(corr, -1.0, 1.0)

    def predict_P(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute P_i values at given or best-fit parameters."""
        if x is None:
            x = self._require_fit().x
        c = self.parameters.build_c(x)
        _, _, P = self.fitter.evaluate(c, return_P=True)
        return P

    def compute_hess_inv(self, x: Optional[np.ndarray] = None, epsilon: float = 1e-4) -> np.ndarray:
        """Calculate the inverse Hessian matrix using the 3-point method."""
        if x is None:
            x = self._require_fit().x

        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = epsilon
            g_plus = self.gradient(x + dx)
            g_minus = self.gradient(x - dx)
            H[:, i] = (g_plus - g_minus) / (2.0 * epsilon)

        H = 0.5 * (H + H.T)
        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(H)

    def gradient_of_partial_R(
        self, component_indices: list[int], M: Optional[np.ndarray] = None, x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate the gradient of the partial intensity R = c^† M_sub c."""
        ff = self._get_fit_fractions_obj(M=M)
        if x is None:
            x = self._require_fit().x
        return ff._gradient_of_R_sub(component_indices, x)

    def compute_fit_fractions(
        self,
        component_groups: list[list[int]],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Calculate fit fractions and their uncertainties for groups of components."""
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute(x, component_groups)

    def compute_interference_fractions(
        self,
        component_pairs: list[list[int]],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Calculate interference fit fractions and their uncertainties."""
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute_interference(x, component_pairs)

    def compute_fit_fraction_matrix(
        self,
        component_groups: list[list[int]],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> dict:
        """Calculate a matrix of fit fractions and interferences.

        For input `[[0, 1], [2, 3]]`:
            - Diagonal[i, i] = FF of the i-th group (e.g., FF[0, 1]).
            - Off-diagonal[i, j] = Interference between groups i and j.
              Interference = FF[group_i + group_j] - FF[group_i] - FF[group_j].

        Args:
            component_groups: List of component index lists.
            M: Overlap matrix.
            x: Real parameter values. Defaults to best-fit x.
            cov_matrix: Covariance matrix V.

        Returns:
            Dictionary with "matrix", "errors", "groups".
        """
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute_matrix(x, component_groups)

    def compute_ratios(
        self,
        numerator_groups: list[list[int]],
        denominator_group: list[int],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Calculate ratios of fit fractions: R = FF_num / FF_den."""
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute_ratios(x, numerator_groups, denominator_group)

    def compute_fit_fraction_matrix(
        self,
        component_groups: list[list[int]],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> dict:
        """Calculate a matrix of fit fractions and interferences.

        For input `[[0, 1], [2, 3]]`:
            - Diagonal[i, i] = FF of the i-th group (e.g., FF[0, 1]).
            - Off-diagonal[i, j] = Interference between groups i and j.
              Interference = FF[group_i + group_j] - FF[group_i] - FF[group_j].

        Args:
            component_groups: List of component index lists.
            M: Overlap matrix.
            x: Real parameter values. Defaults to best-fit x.
            cov_matrix: Covariance matrix V.

        Returns:
            Dictionary with "matrix", "errors", "groups".
        """
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute_matrix(x, component_groups)

    def compute_ratios(
        self,
        numerator_groups: list[list[int]],
        denominator_group: list[int],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Calculate ratios of fit fractions: R = FF_num / FF_den."""
        ff = self._get_fit_fractions_obj(M=M, cov_matrix=cov_matrix)
        if x is None:
            x = self._require_fit().x
        return ff.compute_ratios(x, numerator_groups, denominator_group)

    def save_fit_fractions_csv(
        self,
        component_groups: list[list[int]],
        prefix: str = "fit_frac",
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Save fit fraction matrix and uncertainties to CSV files.

        Saves two files:
            - <prefix>.csv: Matrix of fit fraction values.
            - <prefix>_err.csv: Matrix of uncertainties.

        Args:
            component_groups: List of component groups defining the matrix rows/cols.
            prefix: Filename prefix (default: "fit_frac").
            M: Overlap matrix.
            x: Real parameter values. Defaults to best-fit x.
            cov_matrix: Covariance matrix.
        """
        import csv

        res = self.compute_fit_fraction_matrix(
            component_groups=component_groups, M=M, x=x, cov_matrix=cov_matrix
        )
        matrix = res["matrix"]
        errors = res["errors"]
        groups = res["groups"]

        labels = [str(g) for g in groups]

        def _write_csv(filename, data):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([""] + labels)
                for i, label in enumerate(labels):
                    writer.writerow([label] + list(data[i]))

        _write_csv(f"{prefix}.csv", matrix)
        _write_csv(f"{prefix}_err.csv", errors)

    def _require_fit(self) -> OptimizeResult:
        """Raise if fit() hasn't been called."""
        if self._result is None:
            raise RuntimeError(
                "No fit results available. Call fit() first."
            )
        return self._result

    def save_results(self, path: str) -> None:
        """Save fit results to a JSON file."""
        import json
        result = self._require_fit()
        x = result.x
        sigma = self.uncertainties(result)
        params = self.parameters

        value = {}
        error = {}
        for i, name in enumerate(params.free_params):
            value[f"{name}_r"] = float(x[2 * i])
            value[f"{name}_phi"] = float(x[2 * i + 1])
            error[f"{name}_r"] = float(sigma[2 * i])
            error[f"{name}_phi"] = float(sigma[2 * i + 1])

        output = {
            "value": value,
            "error": error,
            "status": {
                "NLL": float(result.fun),
                "Ndf": params.n_free,
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    @classmethod
    def load_results(cls, path: str) -> dict:
        """Load fit results from a JSON file."""
        import json
        with open(path) as f:
            return json.load(f)
