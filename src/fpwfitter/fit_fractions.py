"""
fit_fractions.py  –  Fit fraction, interference, and ratio calculations.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class FitFractions:
    """Calculates fit fractions, interferences, and ratios.

    Initialized with the model parameters, overlap matrix, and covariance matrix.
    """

    def __init__(
        self,
        parameters: Any,
        M: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
    ):
        """
        Args:
            parameters: A `Parameters` object (must have `build_c` and `gradient_chain_rule`).
            M: Overlap matrix (n_comp x n_comp).
            cov_matrix: Covariance matrix (n_free x n_free). If None, errors will be 0.
        """
        self.parameters = parameters
        self.M = np.asarray(M, dtype=np.complex128)
        self.cov_matrix = np.asarray(cov_matrix) if cov_matrix is not None else None

    def _get_V(self) -> np.ndarray:
        """Get symmetrized covariance matrix."""
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix is required for error calculation.")
        return 0.5 * (self.cov_matrix + self.cov_matrix.T)

    def _gradient_of_R_sub(self, indices: list[int], x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the partial intensity R_sub w.r.t real parameters x.

        R_sub = c_sub^† M c_sub, where c_sub has non-zero entries only at `indices`.
        """
        c = self.parameters.build_c(x)
        n_comp = len(c)

        # Construct c_sub
        c_sub = np.zeros_like(c)
        c_sub[indices] = c[indices]

        # Gradient of R_sub w.r.t c* is P_S M c_sub
        # But R_sub depends only on c_S, so gradient must be zero outside S.
        g_vec = self.M @ c_sub
        
        mask = np.zeros(n_comp, dtype=bool)
        mask[indices] = True
        g = np.zeros_like(c)
        g[mask] = g_vec[mask]

        return self.parameters.gradient_chain_rule(g, x)

    def compute(
        self,
        x: np.ndarray,
        component_groups: list[list[int]],
    ) -> list[dict]:
        """Calculate fit fractions for specified groups of components."""
        c = self.parameters.build_c(x)
        n_comp = len(c)
        R_total = np.real(c.conj() @ (self.M @ c))
        grad_R_total = self._gradient_of_R_sub(list(range(n_comp)), x)

        V = self._get_V() if self.cov_matrix is not None else None

        results = []
        for group in component_groups:
            indices = list(group)
            
            c_sub = np.zeros_like(c)
            c_sub[indices] = c[indices]
            R_sub = np.real(c_sub.conj() @ (self.M @ c_sub))

            grad_R_sub = self._gradient_of_R_sub(indices, x)

            FF = R_sub / R_total
            grad_FF = (grad_R_sub - FF * grad_R_total) / R_total

            if V is not None:
                variance = grad_FF @ V @ grad_FF
                sigma = np.sqrt(max(0.0, np.real(variance)))
            else:
                sigma = 0.0

            results.append({
                "indices": indices,
                "value": FF,
                "error": sigma,
                "gradient": grad_FF.tolist(),
            })

        return results

    def compute_interference(
        self,
        x: np.ndarray,
        component_pairs: list[list[int]],
    ) -> list[dict]:
        """Calculate interference fit fractions for pairs of components."""
        V = self._get_V() if self.cov_matrix is not None else None

        results = []
        for pair in component_pairs:
            if len(pair) != 2:
                raise ValueError("Each pair must contain exactly 2 indices.")
            i, j = pair[0], pair[1]

            def get_ff_res(idx_list):
                res = self.compute(x, [idx_list])[0]
                return res["value"], np.array(res["gradient"])

            val_ij, grad_ij = get_ff_res([i, j])
            val_i, grad_i = get_ff_res([i])
            val_j, grad_j = get_ff_res([j])

            ff_interference = val_ij - val_i - val_j
            grad_interference = grad_ij - grad_i - grad_j

            if V is not None:
                variance = grad_interference @ V @ grad_interference
                sigma = np.sqrt(max(0.0, np.real(variance)))
            else:
                sigma = 0.0

            results.append({
                "indices": [i, j],
                "value": ff_interference,
                "error": sigma,
                "gradient": grad_interference.tolist(),
            })

        return results

    def compute_matrix(
        self,
        x: np.ndarray,
        component_groups: list[list[int]],
    ) -> dict:
        """Calculate a matrix of fit fractions and interferences.

        For input `[[0, 1], [2, 3]]`:
            - Diagonal[i, i] = FF of the i-th group.
            - Off-diagonal[i, j] = Interference = FF[group_i + group_j] - FF[group_i] - FF[group_j].
        """
        n_groups = len(component_groups)
        matrix = np.zeros((n_groups, n_groups))
        errors = np.zeros((n_groups, n_groups))
        V = self._get_V() if self.cov_matrix is not None else None

        def get_ff_res(indices_list):
            res = self.compute(x, [indices_list])[0]
            return res["value"], res["error"], np.array(res["gradient"])

        group_ff = []
        group_grads = []
        
        # Diagonal
        for i, group in enumerate(component_groups):
            val, err, grad = get_ff_res(group)
            matrix[i, i] = val
            errors[i, i] = err
            group_ff.append(val)
            group_grads.append(grad)

        # Off-diagonal
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                combined = list(component_groups[i]) + list(component_groups[j])
                val_ij, _, grad_ij = get_ff_res(combined)

                ff_interference = val_ij - group_ff[i] - group_ff[j]
                grad_interference = grad_ij - group_grads[i] - group_grads[j]

                if V is not None:
                    variance = grad_interference @ V @ grad_interference
                    sigma = np.sqrt(max(0.0, np.real(variance)))
                else:
                    sigma = 0.0

                matrix[i, j] = ff_interference
                matrix[j, i] = ff_interference
                errors[i, j] = sigma
                errors[j, i] = sigma

        return {
            "matrix": matrix,
            "errors": errors,
            "groups": component_groups,
        }

    def compute_ratio(
        self,
        x: np.ndarray,
        numerator_groups: list[list[int]],
        denominator_group: list[int],
    ) -> list[dict]:
        """Calculate ratios of fit fractions: R = FF_num / FF_den.

        Args:
            x: Real parameter values.
            numerator_groups: List of groups for numerator.
            denominator_group: Group for denominator.
        """
        V = self._get_V() if self.cov_matrix is not None else None

        # Get denominator info
        den_res = self.compute(x, [denominator_group])[0]
        den_val = den_res["value"]
        den_grad = np.array(den_res["gradient"])

        results = []
        for group in numerator_groups:
            num_res = self.compute(x, [group])[0]
            num_val = num_res["value"]
            num_grad = np.array(num_res["gradient"])

            ratio = num_val / den_val
            # Gradient: d(N/D) = (D dN - N dD) / D^2
            grad_ratio = (den_val * num_grad - num_val * den_grad) / (den_val ** 2)

            if V is not None:
                variance = grad_ratio @ V @ grad_ratio
                sigma = np.sqrt(max(0.0, np.real(variance)))
            else:
                sigma = 0.0

            results.append({
                "numerator": group,
                "denominator": denominator_group,
                "value": ratio,
                "error": sigma,
                "gradient": grad_ratio.tolist(),
            })

        return results
