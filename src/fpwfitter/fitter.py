"""
fitter.py  –  Combined Parameters + FpwFitter for end-to-end fitting.

The Fitter class connects real-valued optimizer parameters x to
complex coupling coefficients c_k, and computes -log L and its gradient.
"""

from __future__ import annotations

import json
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

        Results are cached and accessible via convenience methods
        (couplings, uncertainties, correlations, etc.).

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

        self._result = minimize(
            fun=self.objective_and_gradient,
            x0=x0,
            method=method,
            jac=True,  # objective_and_gradient returns (f, grad)
            options=options,
            **kwargs,
        )
        return self._result

    def _require_fit(self) -> OptimizeResult:
        """Raise if fit() hasn't been called."""
        if self._result is None:
            raise RuntimeError(
                "No fit results available. Call fit() first."
            )
        return self._result

    def uncertainties(self, result: Optional[OptimizeResult] = None) -> np.ndarray:
        """Extract parameter uncertainties from fit result.

        For BFGS (and other quasi-Newton methods), uses result.hess_inv
        (inverse Hessian at optimum).

        For -log L minimization, the Hessian at the minimum approximates
        the Fisher Information Matrix, so:
            Cov(θ) ≈ H^{-1} = hess_inv
            σ_j = sqrt(Cov[j,j])

        Args:
            result: OptimizeResult from fit(). Defaults to cached result.

        Returns:
            sigma: 1-sigma uncertainties for each real parameter,
                   shape (n_free_real,)
        """
        if result is None:
            result = self._require_fit()
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

    # ---- Cached result convenience methods ----

    @property
    def result(self) -> OptimizeResult:
        """Cached fit result. Raises if fit() hasn't been called."""
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
        """Complex coupling coefficients c_k at best fit.

        Returns:
            c: complex128 array of shape (n_components,)
        """
        return self.parameters.build_c(self._require_fit().x)

    def correlations(self, result: Optional[OptimizeResult] = None) -> np.ndarray:
        """Parameter correlation matrix from fit.

        Uses hess_inv to compute:
            corr[i,j] = cov[i,j] / (sigma_i * sigma_j)

        Args:
            result: OptimizeResult. Defaults to cached result.

        Returns:
            corr: correlation matrix, shape (n_free_real, n_free_real)
        """
        if result is None:
            result = self._require_fit()

        hess_inv = np.asarray(result.hess_inv)
        hess_inv = (hess_inv + hess_inv.T) / 2.0
        variances = np.maximum(np.diag(hess_inv), 0.0)
        sigma = np.sqrt(variances)

        # Avoid division by zero
        sigma = np.where(sigma > 0, sigma, 1.0)
        corr = hess_inv / np.outer(sigma, sigma)
        # Clamp to [-1, 1]
        corr = np.clip(corr, -1.0, 1.0)
        return corr

    def predict_P(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute P_i values at given or best-fit parameters.

        Args:
            x: real parameter values. Defaults to best-fit x.

        Returns:
            P: float64 array of shape (n_data,)
        """
        if x is None:
            x = self._require_fit().x
        c = self.parameters.build_c(x)
        _, _, P = self.fitter.evaluate(c, return_P=True)
        return P

    def compute_hess_inv(self, x: Optional[np.ndarray] = None, epsilon: float = 1e-4) -> np.ndarray:
        """Calculate the inverse Hessian matrix using the 3-point method.

        Computes the Hessian H_ij = d^2(-ln L) / dx_i dx_j 
        via finite difference of the gradients, then returns its inverse.

        Formula:
            H_ij = [g_j(x + e*e_i) - g_j(x - e*e_i)] / (2*e)

        Args:
            x: Real parameter values. Defaults to best-fit x.
            epsilon: Finite difference step size.

        Returns:
            hess_inv: Inverse Hessian matrix (covariance matrix approximation),
                      shape (n_free_real, n_free_real).
        """
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

        # Symmetrize to handle numerical noise
        H = 0.5 * (H + H.T)

        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(H)

    def gradient_of_partial_R(
        self, component_indices: list[int], M: Optional[np.ndarray] = None, x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate the gradient of the partial intensity R = c^† M_sub c.

        R represents the contribution to the signal normalization from a subset of 
        components. For k not in `component_indices`, the contribution is ignored.

        Args:
            component_indices: List of component indices (integers) to include.
            M: The overlap matrix (n_comp x n_comp). 
               If None, tries to retrieve from the fitter.
            x: Real parameter values. Defaults to best-fit x.

        Returns:
            grad: Real gradient vector of shape (n_free_real,).
        """
        if x is None:
            x = self._require_fit().x
            
        if M is None:
            # Attempt to retrieve M from the underlying fitter
            if hasattr(self.fitter, "_M"):
                M = self.fitter._M
            else:
                raise ValueError(
                    "M is not available. Please pass M explicitly or ensure "
                    "the fitter stores it as _M."
                )

        c = self.parameters.build_c(x)
        n_comp = len(c)
        
        # Create mask for active components
        mask = np.zeros(n_comp, dtype=bool)
        for idx in component_indices:
            if not (0 <= idx < n_comp):
                raise IndexError(f"Component index {idx} is out of range [0, {n_comp})")
            mask[idx] = True
            
        # Construct M_sub: only rows/cols in mask are non-zero
        M_sub = np.zeros_like(M)
        M_sub[np.ix_(mask, mask)] = M[np.ix_(mask, mask)]
        
        # Compute g_k = dR / dc_k*
        # R = sum_{p,q} c_p* M_sub_{pq} c_q
        # dR/dc_k* = sum_q M_sub_{kq} c_q = (M_sub c)_k
        g = M_sub @ c
        
        # Apply chain rule to get gradient wrt real parameters x
        return self.parameters.gradient_chain_rule(g, x)

    def compute_fit_fractions(
        self,
        component_groups: list[list[int]],
        M: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Calculate fit fractions and their uncertainties for groups of components.

        The fit fraction for a group of components $S$ is defined as:
            $FF_S = R_S / R_{total}$
        where $R = c^\\dagger M c$, and $R_S$ is the contribution from components in $S$.
        $R_S$ includes all intensity and interference terms within the group.

        Uncertainty is calculated using the error propagation formula:
            $\\sigma = \\sqrt{ g^T V g }$
        where $g = \\nabla_x FF_S$ and $V$ is the covariance matrix (inverse Hessian).

        Args:
            component_groups: List of component index lists, e.g., `[[0, 1], [2]]`.
            M: Overlap matrix. If None, tries to retrieve from the fitter.
            x: Real parameter values. Defaults to best-fit x.
            cov_matrix: Covariance matrix V. If None, uses `result.hess_inv` from the fit,
                        or falls back to `compute_hess_inv`.

        Returns:
            List of dictionaries, each containing:
                - "indices": The component indices for the group.
                - "value": The fit fraction value.
                - "error": The estimated uncertainty.
                - "gradient": The gradient of the fit fraction wrt real parameters x.
        """
        if x is None:
            x = self._require_fit().x

        if M is None:
            if hasattr(self.fitter, "_M"):
                M = self.fitter._M
            else:
                raise ValueError(
                    "M is not available. Please pass M explicitly."
                )

        c = self.parameters.build_c(x)
        n_comp = len(c)
        
        # Calculate Total R and its gradient
        R_total = np.real(c.conj() @ (M @ c))
        all_indices = list(range(n_comp))
        grad_R_total = self.gradient_of_partial_R(all_indices, M=M, x=x)

        # Determine Covariance Matrix V
        if cov_matrix is not None:
            V = np.asarray(cov_matrix)
        else:
            if hasattr(self, '_result') and self._result is not None:
                res = self._result
                if hasattr(res, 'hess_inv') and res.hess_inv is not None:
                    V = np.asarray(res.hess_inv)
                else:
                    V = self.compute_hess_inv(x=x)
            else:
                V = self.compute_hess_inv(x=x)

        # Symmetrize V
        V = 0.5 * (V + V.T)

        results = []
        for group in component_groups:
            indices_list = list(group)
            for idx in indices_list:
                if not (0 <= idx < n_comp):
                    raise IndexError(f"Component index {idx} is out of range [0, {n_comp})")

            # Calculate R_sub by zeroing components not in the group
            c_sub = np.zeros_like(c)
            c_sub[indices_list] = c[indices_list]
            R_sub = np.real(c_sub.conj() @ (M @ c_sub))
            
            # Gradient of R_sub
            grad_R_sub = self.gradient_of_partial_R(indices_list, M=M, x=x)
            
            # Fit Fraction
            FF = R_sub / R_total
            
            # Gradient of FF: d(R_sub/R_total) = (dR_sub - FF * dR_total) / R_total
            grad_FF = (grad_R_sub - FF * grad_R_total) / R_total
            
            # Uncertainty: sqrt( g^T V g )
            variance = grad_FF @ V @ grad_FF
            sigma = np.sqrt(max(0.0, np.real(variance)))
            
            results.append({
                "indices": indices_list,
                "value": FF,
                "error": sigma,
                "gradient": grad_FF.tolist(),
            })

        return results

    def save_results(self, path: str) -> None:
        """Save fit results to a JSON file.

        JSON structure:
            {
              "value": {"a_r": r_val, "a_phi": phi_val, ...},
              "error": {"a_r": sigma_r, "a_phi": sigma_phi, ...},
              "status": {"NLL": nll_value, "Ndf": n_free_real}
            }

        Args:
            path: output JSON file path
        """
        result = self._require_fit()
        x = result.x
        sigma = self.uncertainties(result)
        params = self.parameters

        # Build value and error dicts
        value = {}
        error = {}
        for i, name in enumerate(params.free_params):
            value[f"{name}_r"] = float(x[2 * i])
            value[f"{name}_phi"] = float(x[2 * i + 1])
            error[f"{name}_r"] = float(sigma[2 * i])
            error[f"{name}_phi"] = float(sigma[2 * i + 1])

        # Ndf is the number of free real parameters
        ndf = params.n_free

        output = {
            "value": value,
            "error": error,
            "status": {
                "NLL": float(result.fun),
                "Ndf": ndf,
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    @classmethod
    def load_results(cls, path: str) -> dict:
        """Load fit results from a JSON file.

        Args:
            path: input JSON file path

        Returns:
            dict with "value", "error", and "status" keys
        """
        with open(path) as f:
            return json.load(f)
