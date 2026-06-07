"""Build the low-level Kernel config from a high-level physics description.

The high-level YAML describes the decay topology, particle properties,
and resonance content.  This module expands it into the flat index arrays,
interpolation tables, and mapping matrices that ``Kernel.__init__`` expects.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict


# ---------------------------------------------------------------------------
#  Internal representation of a decay chain
# ---------------------------------------------------------------------------

class _Decay:
    """A single two-body decay ``parent -> child1 + child2``."""
    __slots__ = ("parent", "child1", "child2")
    def __init__(self, parent: str, child1: str, child2: str):
        self.parent = parent
        self.child1 = child1
        self.child2 = child2


class _Chain:
    """A full decay chain from the top particle to all final-state particles."""
    def __init__(self, decays: list[_Decay]):
        self.decays = decays
        # list all intermediate resonances (non-final parents)
        self.resonances = [d.parent for d in decays
                           if d.parent != decays[0].parent]  # skip top

    def finals(self):
        """Return set of final-state particle names."""
        all_children = set()
        for d in self.decays:
            all_children.add(d.child1)
            all_children.add(d.child2)
        return all_children - {d.parent for d in self.decays}


# ---------------------------------------------------------------------------
#  Main builder
# ---------------------------------------------------------------------------

def build_config(physics: dict) -> dict:
    """Convert a high-level physics description into a Kernel config dict.

    Parameters
    ----------
    physics : dict
        A dict with ``decay`` and ``particle`` keys, as described in
        the README.  Example::

            decay:
              A: [[R, C], [Y, D]]
              R: [B, D]
              Y: [B, C]
            particle:
              $top: A
              $finals: [B, C, D]
              A:  {J: 0, P: -1, mass: 5.0}
              R:  [R1, R2]
              R1: {J: 0, P: 1, mass: 3.0, model: BW}
              R2: {J: 1, P: -1, mass: 3.0, model: BW}
              Y:  {J: 0, P: -1, mass: 3.0, model: BW}
              B:  {J: 0, P: -1, mass: 0.1}
              C:  {J: 0, P: -1, mass: 0.1}
              D:  {J: 0, P: -1, mass: 0.1}

    Returns
    -------
    dict
        Kernel config with flat index arrays, tables, and matrices.
    """
    part = dict(physics["particle"])
    top = part.pop("$top")
    finals = set(part.pop("$finals"))

    # Expand resonance aliases (e.g. R → [R1, R2])
    resonances: dict[str, list[str]] = {}
    for name, val in list(part.items()):
        if isinstance(val, list):
            resonances[name] = val
            part.pop(name)

    # Build decay trees from the ``decay`` section
    decay_raw = dict(physics["decay"])

    def _expand_tree(node: str) -> list[_Chain]:
        """Recursively expand a decay tree into flat decay chains."""
        if node in finals:
            return []  # no further decay
        branches = decay_raw[node]
        # branches is like [[R, C], [Y, D]] or [B, D]
        if isinstance(branches[0], str):
            branches = [branches]
        chains = []
        for b in branches:
            c1, c2 = b[0], b[1]
            d = _Decay(node, c1, c2)
            sub1 = _expand_tree(c1)
            sub2 = _expand_tree(c2)
            if not sub1 and not sub2:
                chains.append(_Chain([d]))
            elif sub1 and not sub2:
                for s in sub1:
                    chains.append(_Chain([d] + s.decays))
            elif sub2 and not sub1:
                for s in sub2:
                    chains.append(_Chain([d] + s.decays))
            else:
                for s1 in sub1:
                    for s2 in sub2:
                        chains.append(_Chain([d] + s1.decays + s2.decays))
        return chains

    base_chains = _expand_tree(top)

    # Substitute resonance aliases to generate all physical waves
    waves: list[dict] = []  # each entry: {chain_idx, res_map, label}
    for ci, chain in enumerate(base_chains):
        # Find which resonances in this chain have aliases
        subst_groups = []
        for res in chain.resonances:
            if res in resonances:
                subst_groups.append(resonances[res])
            else:
                subst_groups.append([res])
        # Cartesian product
        from itertools import product as iproduct
        for combo in iproduct(*subst_groups):
            res_map = {}
            for orig, chosen in zip(
                [r for r in chain.resonances if r in resonances], combo
            ):
                res_map[orig] = chosen
            # All actual resonance names for this wave
            actual_ress = []
            for r in chain.resonances:
                actual_ress.append(res_map.get(r, r))
            # Assign fixed BW parameters from particle props
            bw_masses = []
            bw_widths = []
            for r_name in actual_ress:
                p = part.get(r_name, part.get(r_name, {}))
                bw_masses.append(p.get("mass", 1.0))
                bw_widths.append(p.get("width", 0.1))
            waves.append({
                "res_map": res_map,
                "res_names": actual_ress,
                "n_res": len(actual_ress),
                "chain": chain,
                "bw_masses": bw_masses,
                "bw_widths": bw_widths,
            })

    nwaves = len(waves)
    nres = max(w["n_res"] for w in waves) if waves else 1
    ndecays = nres  # one form factor per resonance
    _meta = {
        "nwaves": nwaves, "nres": nres, "ndecays": ndecays,
        "n_m0": 0, "n_g0": 0, "n_gamma": 0, "n_bw": 0,
        "n_fl": 0, "nbasis": 0, "n_ang": 0, "n_per_group": 0,
    }

    # ---- Build flat index arrays -----------------------------------------

    # Count distinct BW parameters
    bw_set: list[tuple[float, float]] = []
    bw_to_idx: dict[tuple[float, float], int] = {}
    for w in waves:
        for m, g in zip(w["bw_masses"], w["bw_widths"]):
            key = (m, g)
            if key not in bw_to_idx:
                bw_to_idx[key] = len(bw_set)
                bw_set.append(key)

    n_bw = len(bw_set)
    m0_vals = np.array([m for m, _ in bw_set], dtype=np.float32)
    g0_vals = np.array([g for _, g in bw_set], dtype=np.float32)
    n_m0 = n_g0 = n_bw  # one mass+width per unique BW parameter

    bw_order = []
    for w in waves:
        for m, g in zip(w["bw_masses"], w["bw_widths"]):
            bw_order.append(bw_to_idx[(m, g)])
    bw_order = np.array(bw_order, dtype=np.int64)

    # BW indexers
    m0_index = np.arange(n_bw, dtype=np.int64)       # each BW term has its own m0
    bw_index = np.zeros(n_bw, dtype=np.int64)         # all use mass column 0
    bw_gamma_index = np.arange(n_bw, dtype=np.int64)  # each has its own gamma

    # Gamma interpolation (one per BW term)
    n_gamma = n_bw
    n_int = 200
    gamma_table = np.ones((n_gamma, n_int), dtype=complex)
    g0_index = np.arange(n_gamma, dtype=np.int64)

    gamma_index = np.zeros(n_gamma, dtype=np.int64)    # mass column 0
    gamma_type = np.zeros(n_gamma, dtype=np.int64)
    gamma_min = 0.0
    gamma_delta = 0.01

    matrix_gamma = np.eye(n_m0, n_gamma, dtype=float)  # identity

    # Form factors (one per BW term, same count)
    n_fl = n_bw
    fl_table = np.ones((n_fl, n_int), dtype=float)
    q_index = np.zeros(n_fl, dtype=np.int64)
    fl_type = np.zeros(n_fl, dtype=np.int64)
    fl_min = 0.0
    fl_delta = 0.01

    fl_order = np.zeros(nwaves * ndecays, dtype=np.int64)
    idx = 0
    for w in waves:
        for _ in range(w["n_res"]):
            fl_order[idx] = 0  # all use FL type 0
            idx += 1
        # pad with 0 for waves that have fewer resonances than nres
        for _ in range(nres - w["n_res"]):
            fl_order[idx] = 0
            idx += 1

    # Angular basis
    # Simplified: one angle per decay vertex
    n_ang = 2  # cos(theta) for each of two vertices
    angle_index = np.arange(n_ang, dtype=np.int64)
    angle_k = np.ones(n_ang, dtype=float)
    angle_b = np.zeros(n_ang, dtype=float)

    # Simple angular basis: one product per wave
    nbasis = nwaves
    n_per = 1
    # ensure all indices in [0, n_ang)
    ang_order = (np.arange(nbasis) % max(n_ang, 1)).astype(np.int64).reshape(-1, 1)

    # Mapping matrices
    matrix_ang = np.eye(nbasis, nwaves, dtype=complex)

    # Wave amplitudes split into two CP groups
    # Pad to even nwaves if needed (duplicate last wave's entries)
    if nwaves % 2 != 0:
        nwaves_eff = nwaves + 1
        # extend matrix_ang: add row AND column
        pad_m = np.zeros((nbasis, 1), dtype=complex)
        matrix_ang = np.hstack([matrix_ang, pad_m])
        pad_m2 = np.zeros((1, nwaves_eff), dtype=complex)
        matrix_ang = np.vstack([matrix_ang, pad_m2])
        pad_a = np.zeros((1, n_per), dtype=np.int64)
        ang_order = np.vstack([ang_order, pad_a])
        # extend bw_order, fl_order (append last wave's entries)
        bw_order = np.append(bw_order, bw_order[-nres:] if len(bw_order) >= nres else [0]*nres)
        fl_order = np.append(fl_order, fl_order[-ndecays:] if len(fl_order) >= ndecays else [0]*ndecays)
        nbasis = nwaves_eff
    else:
        nwaves_eff = nwaves

    _meta["_nwaves_eff"] = nwaves_eff
    _meta["nwaves_raw"] = nwaves
    n_per_group = nwaves_eff // 2

    return {
        "gamma_table": gamma_table,
        "fl_table": fl_table,
        "matrix_gamma": matrix_gamma,
        "matrix_ang": matrix_ang,
        "g0_index": g0_index,
        "gamma_index": gamma_index,
        "gamma_type": gamma_type,
        "gamma_min": gamma_min,
        "gamma_delta": gamma_delta,
        "m0_index": m0_index,
        "bw_index": bw_index,
        "bw_gamma_index": bw_gamma_index,
        "bw_order": bw_order,
        "q_index": q_index,
        "fl_type": fl_type,
        "fl_min": fl_min,
        "fl_delta": fl_delta,
        "fl_order": fl_order,
        "angle_index": angle_index,
        "angle_k": angle_k,
        "angle_b": angle_b,
        "ang_order": ang_order,
        "_meta": {
            "nwaves": _meta["_nwaves_eff"] if nwaves % 2 else nwaves,
            "nwaves_raw": nwaves,
            "nres": nres,
            "ndecays": ndecays,
            "n_m0": n_m0,
            "n_g0": n_g0,
            "n_gamma": n_gamma,
            "n_bw": n_bw,
            "n_fl": n_fl,
            "nbasis": nbasis if nwaves % 2 == 0 else nwaves+1,
            "n_ang": n_ang,
            "n_per_group": n_per_group,
        },
    }
