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


class Fitter:
    """Combines Parameters + FpwFitter for end-to-end fitting.

    A thin wrapper that connects parameter conversion (x → c_k)
    to fitter evaluation (c_k → -log L, d(-log L)/dc*).

    Works with any fitter that has an evaluate(c) method returning
    (nll, grad) where grad = d(-log L)/dc*.
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

    def get_couplings(self, x: np.ndarray) -> np.ndarray:
        """Get complex coupling coefficients c_k from real params x.

        Args:
            x: real parameter values [r_0, phi_0, r_1, phi_1, ...]

        Returns:
            c: complex128 array of shape (n_components,)
        """
        return self.parameters.build_c(x)

    def objective(self, x: np.ndarray) -> float:
        """Compute -log L from real parameters x.

        Args:
            x: real parameter values, shape (n_free_real,)

        Returns:
            nll: negative log-likelihood (float)
        """
        c = self.parameters.build_c(x)
        nll, _ = self.fitter.evaluate(c)
        return float(nll)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute d(-log L)/dx from real parameters x.

        Uses the chain rule:
            d(-log L)/dx_j = 2 Re( sum_k g_k * (dc_k/dx_j)* )
        where g_k = d(-log L)/dc*_k from the fitter.

        Args:
            x: real parameter values, shape (n_free_real,)

        Returns:
            grad: real gradient, shape (n_free_real,)
        """
        c = self.parameters.build_c(x)
        _, g = self.fitter.evaluate(c)  # g = d(-log L)/dc*
        return self.parameters.gradient_chain_rule(g, x)

    def objective_and_gradient(
        self, x: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Compute both -log L and gradient efficiently.

        Only calls fitter.evaluate once.

        Args:
            x: real parameter values, shape (n_free_real,)

        Returns:
            nll: float
            grad: real gradient, shape (n_free_real,)
        """
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
        """Run optimization using scipy.optimize.minimize.

        Args:
            x0: initial real parameter values [r_0, phi_0, ...].
                Defaults to [1.0, 0.0, 1.0, 0.0, ...] (r=1, phi=0).
            method: optimization method. Default: 'BFGS' (provides
                inverse Hessian for uncertainty estimation).
            options: passed to scipy.optimize.minimize.
            **kwargs: additional kwargs for scipy.optimize.minimize.

        Returns:
            OptimizeResult with x, fun, jac, hess_inv, etc.
        """
        if x0 is None:
            n = self.parameters.n_free
            x0 = np.zeros(n)
            x0[0::2] = 1.0  # r = 1
            # phi = 0 by default

        if options is None:
            options = {}

        result = minimize(
            fun=self.objective_and_gradient,
            x0=x0,
            method=method,
            jac=True,  # objective_and_gradient returns (f, grad)
            options=options,
            **kwargs,
        )
        return result

    def uncertainties(self, result: OptimizeResult) -> np.ndarray:
        """Extract parameter uncertainties from fit result.

        For BFGS (and other quasi-Newton methods), uses result.hess_inv
        (inverse Hessian at optimum).

        For -log L minimization, the Hessian at the minimum approximates
        the Fisher Information Matrix, so:
            Cov(θ) ≈ H^{-1} = hess_inv
            σ_j = sqrt(Cov[j,j])

        Args:
            result: OptimizeResult from fit()

        Returns:
            sigma: 1-sigma uncertainties for each real parameter,
                   shape (n_free_real,)
        """
        if not hasattr(result, "hess_inv") or result.hess_inv is None:
            raise ValueError(
                "No hess_inv in result. "
                "Use method='BFGS' or 'L-BFGS-B' (with options) "
                "to get uncertainty estimates."
            )

        # hess_inv is the inverse Hessian ≈ covariance matrix
        hess_inv = np.asarray(result.hess_inv)

        # Ensure symmetric
        hess_inv = (hess_inv + hess_inv.T) / 2.0

        # Covariance diagonal → variances → uncertainties
        variances = np.diag(hess_inv)

        # Guard against negative variances (numerical issues)
        variances = np.maximum(variances, 0.0)

        return np.sqrt(variances)
