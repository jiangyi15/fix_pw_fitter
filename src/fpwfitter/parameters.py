"""
parameters.py  –  Real-to-complex parameter conversion for the fitter.

Converts a real-valued optimizer vector x into complex coupling coefficients c_k.

Pipeline:
    x (real)  →  y (complex, polar form)  →  c_k (products of y values)

Gradient chain rule (for real loss f, e.g. -log L):
    fitter returns  g_k = df/dc_k*
    We need  df/dx_j = 2 Re( sum_k g_k * (dc_k/dx_j)* )

    For c_k = prod(y_j for j in S_k), y_j = r_j * exp(i*phi_j):
        dc_k/dr_j   = c_k / r_j   (if j in S_k)
        dc_k/dphi_j = i * c_k     (if j in S_k)

    Therefore, let h_j = sum_{k: j in S_k} g_k * c_k*:
        df/dr_j   = 2 Re(h_j) / r_j
        df/dphi_j = 2 Im(h_j)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


class Parameters:
    """Converts real optimizer vector x → complex c_k, with gradient chain rule."""

    def __init__(
        self,
        fixed_table: Dict[str, complex],
        product_structure: List[List[str]],
    ):
        """
        Args:
            fixed_table: {param_name: complex_value} — fixed (non-optimized) parameters
            product_structure: list of lists of param names
                e.g. [["a", "b", "c"], ["a", "b", "d"]] means
                c = [y["a"]*y["b"]*y["c"], y["a"]*y["b"]*y["d"]]

        Free parameters are automatically inferred as all unique names
        in product_structure that are not in fixed_table.
        """
        self.fixed_table = dict(fixed_table)
        self.product_structure = [list(p) for p in product_structure]

        # ---- Determine free params from product_structure ----
        all_names_ordered = []
        seen = set()
        for prod in product_structure:
            for name in prod:
                if name not in seen:
                    seen.add(name)
                    all_names_ordered.append(name)

        free_names = [name for name in all_names_ordered if name not in self.fixed_table]
        fixed_names_ordered = [name for name in all_names_ordered if name in self.fixed_table]

        # Unified ordering: fixed first, then free
        all_names = fixed_names_ordered + free_names
        self.free_params = free_names
        self.name_to_idx = {name: i for i, name in enumerate(all_names)}
        self.n_total = len(all_names)
        self.n_fixed = len(self.fixed_table)
        self.n_free_complex = len(self.free_params)
        self.n_free_real = 2 * self.n_free_complex  # r and phi per free param

        # ---- Convert fixed values to complex array ----
        self.fixed_values = np.zeros(self.n_fixed, dtype=np.complex128)
        for i, name in enumerate(self.fixed_table.keys()):
            self.fixed_values[i] = complex(self.fixed_table[name])

        # ---- Convert product_structure to integer indices ----
        self.product_indices: List[np.ndarray] = []
        for prod in product_structure:
            indices = np.array([self.name_to_idx[name] for name in prod], dtype=np.int32)
            self.product_indices.append(indices)

        self.n_comp = len(self.product_indices)

        # ---- Build param → component mapping for gradient accumulation ----
        # For each free param index, which components depend on it?
        # free_param_idx 0..n_free_complex-1 maps to all_names index n_fixed + free_param_idx
        self._param_to_comps: List[List[int]] = [[] for _ in range(self.n_free_complex)]
        for k, indices in enumerate(self.product_indices):
            for idx in indices:
                if idx >= self.n_fixed:
                    free_idx = idx - self.n_fixed
                    self._param_to_comps[free_idx].append(k)

        # Precompute flat arrays for vectorized gradient accumulation
        # For each component k, store which free params it depends on
        self._comp_free_params: List[np.ndarray] = []
        for k, indices in enumerate(self.product_indices):
            free_idxs = indices[indices >= self.n_fixed] - self.n_fixed
            self._comp_free_params.append(free_idxs)

    @property
    def n_free(self) -> int:
        """Number of free real parameters (2 per complex free param: r and phi)."""
        return self.n_free_real

    @property
    def n_components(self) -> int:
        """Number of c_k components."""
        return self.n_comp

    def build_c(self, x: np.ndarray) -> np.ndarray:
        """Convert real vector x → complex c_k array.

        Args:
            x: real values [r_0, phi_0, r_1, phi_1, ...] for free params,
               shape (n_free_real,)

        Returns:
            c: complex128 array of shape (n_components,)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.shape[0] == self.n_free_real, (
            f"Expected x of length {self.n_free_real}, got {x.shape[0]}"
        )

        # Extract r, phi
        r = x[0::2]
        phi = x[1::2]

        # Polar → complex: y = r * exp(i*phi)
        y_free = r * (np.cos(phi) + 1j * np.sin(phi))

        # Combine fixed + free into full y array
        y = np.zeros(self.n_total, dtype=np.complex128)
        y[:self.n_fixed] = self.fixed_values
        y[self.n_fixed:] = y_free

        # Compute c_k = prod(y[indices_k]) for each component
        c = np.zeros(self.n_comp, dtype=np.complex128)
        for k, indices in enumerate(self.product_indices):
            c[k] = np.prod(y[indices])

        return c

    def gradient_chain_rule(self, g: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Convert complex gradient g = df/dc* → real gradient df/dx.

        Uses the identity:
            df/dx_j = 2 Re( sum_k g_k * (dc_k/dx_j)* )

        For c_k = prod(y_j for j in S_k), y_j = r_j * exp(i*phi_j):
            dc_k/dr_j   = c_k / r_j   (if j in S_k)
            dc_k/dphi_j = i * c_k     (if j in S_k)

        Therefore:
            h_j = sum_{k: j in S_k} g_k * c_k*
            df/dr_j   = 2 Re(h_j) / r_j
            df/dphi_j = 2 Im(h_j)

        Args:
            g: complex gradient from fitter, shape (n_components,)
            x: current real parameters [r_0, phi_0, r_1, phi_1, ...],
               shape (n_free_real,)

        Returns:
            dx: real gradient, shape (n_free_real,)
                [df/dr_0, df/dphi_0, df/dr_1, df/dphi_1, ...]
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        g = np.asarray(g, dtype=np.complex128).ravel()
        assert g.shape[0] == self.n_comp
        assert x.shape[0] == self.n_free_real

        # Compute c and y for chain rule
        r = x[0::2]
        phi = x[1::2]
        y_free = r * (np.cos(phi) + 1j * np.sin(phi))

        y = np.zeros(self.n_total, dtype=np.complex128)
        y[:self.n_fixed] = self.fixed_values
        y[self.n_fixed:] = y_free

        c = np.zeros(self.n_comp, dtype=np.complex128)
        for k, indices in enumerate(self.product_indices):
            c[k] = np.prod(y[indices])

        # Compute h_j = sum_{k: j in S_k} g_k * c_k*
        # For each free param j, sum over components that depend on it
        h = np.zeros(self.n_free_complex, dtype=np.complex128)
        for j in range(self.n_free_complex):
            comp_indices = self._param_to_comps[j]
            if comp_indices:
                h[j] = np.sum(g[comp_indices] * c[comp_indices].conj())

        # Chain rule
        # df/dr_j = 2 Re(h_j) / r_j
        # df/dphi_j = 2 Im(h_j)
        dr = 2.0 * np.real(h) / np.where(r != 0, r, 1.0)
        dphi = 2.0 * np.imag(h)

        # Interleave: [dr_0, dphi_0, dr_1, dphi_1, ...]
        dx = np.empty(self.n_free_real, dtype=np.float64)
        dx[0::2] = dr
        dx[1::2] = dphi

        return dx

    # ---- Factory methods ----

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
    ) -> "Parameters":
        """Create from a dictionary configuration.

        Expected keys:
            fixed: {name: value}  (optional)
            products: [[name_a, name_b, ...], ...]
        """
        fixed = {}
        for name, val in config.get("fixed", {}).items():
            fixed[name] = complex(val)

        return cls(
            fixed_table=fixed,
            product_structure=config["products"],
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Parameters":
        """Create from a YAML configuration file.

        Expected format:
            fixed:
              name1: "1.0+0.5j"
              name2: "0.5-0.3j"
            products:
              - [name1, name3]
              - [name2, name3]
              - [name4]

        Free parameters are automatically inferred from product_structure.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for from_yaml(). Install with: pip install pyyaml")

        path = Path(path)
        with open(path) as f:
            config = yaml.safe_load(f)

        fixed = {}
        for name, val in config.get("fixed", {}).items():
            fixed[name] = complex(val)

        return cls(
            fixed_table=fixed,
            product_structure=config["products"],
        )

    def __repr__(self) -> str:
        return (
            f"Parameters(n_fixed={self.n_fixed}, n_free={self.n_free_complex}, "
            f"n_comp={self.n_comp})"
        )
