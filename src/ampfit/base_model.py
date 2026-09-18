"""Physical model: declarations -> structure, tables and kernel config.

:class:`BaseModel` is the interpreted physics: the :class:`~ampfit.decay_tree.DecayTree`,
the index/tables, and the predefined (base) kernel config.  It is built from
the raw declarations only and knows nothing about amplitude-model policy,
parameter surfaces or backends.

``Config`` (in :mod:`ampfit.config_loader`) owns the raw input and subclasses
``BaseModel`` for backward compatibility.
"""
import itertools
import math

import numpy as np

from .angular_formula import get_angle_formula
from .decay_tree import DecayTree


def row_block_factors(dic):
    """Event-row block duplication factors from the config ``data`` section.

    The number of identical-particle copies and the charge-conjugate (CP)
    block are derived from the declared particle symmetries — the
    *combination* of ``data.identical_particles`` and ``data.cp_particles``
    (NOT a hardcoded 8):

      * every ``identical_particles`` group of size ``g`` contributes
        ``g!`` identical-particle permutations,
      * ``cp_particles`` declares a global charge-conjugate map → ×2.

    Returns ``(n_perm, n_cp, n_blocks)`` where
    ``n_blocks = n_perm · n_cp``.  No declarations → 1 block (a
    single-flavour pure-PWA model, e.g. config_pwa.yml).
    """
    data_d = dic.get("data") or {}
    id_groups = data_d.get("identical_particles") or []
    cp_groups = data_d.get("cp_particles") or []
    n_perm = 1
    for grp in id_groups:
        n_perm *= math.factorial(len(grp))
    n_cp = 2 if cp_groups else 1
    return n_perm, n_cp, n_perm * n_cp


def _projection_duplicate(ret, n_proj):
    """p-major duplication of the per-wave arrays for ``n_proj`` projections.

    Every projection owns the same full wave list (the duplicated
    ``matrix_angle`` columns are identical copies until the spin/helicity
    angular generator lands — the angular difference per projection is the
    later, physics step).  The kernel config thereby gains a projection
    axis: ``n_wave → n_proj·n_wave`` while ``ck`` keeps length
    ``n_wave`` (shared across projections).

    Only the *per-wave* arrays are duplicated: ``matrix_angle`` (columns)
    and ``bw_order`` / ``fl_order`` (rows).  The index/unique arrays
    (``m0_index``, ``mass_index``, gamma/fl tables, …) are shared by all
    projections and stay untouched.
    """
    ret = dict(ret)
    for key in ("matrix_angle",):
        if key in ret:
            ret[key] = np.concatenate([ret[key]] * n_proj, axis=1)
    for key in ("bw_order", "fl_order"):
        if key in ret:
            ret[key] = np.concatenate([ret[key]] * n_proj, axis=0)
    return ret


class BaseModel:
    def __init__(self, dic, config_path=""):
        self.dic = dic
        self._config_path = config_path
        # The decay tree is interpreted by a standalone value object; the
        # legacy attribute names below stay as aliases for compatibility.
        self.decay_tree = DecayTree(self.dic["decay"], self.dic["particle"])
        self.top = self.decay_tree.top
        self.finals = self.decay_tree.finals
        self.decay_struct = self.decay_tree.struct
        self.decay_chains_lst = self.decay_tree.chains
        self.full_decay = self.decay_tree.full
        self.n_decay = self.decay_tree.n_decay
        self.n_res = self.decay_tree.n_res
        self.n_angles = self.decay_tree.n_angles
        self.topo_index = self.decay_tree.topo_index
        self.n_topo = self.decay_tree.n_topo
        self.m0_phys_name = []
        self.g0_phys_name = []
        self.unique_l = []
        self.unique_bw = []
        self.unique_gamma = []
        self.unique_fl = []
        self.unique_angle_basis = []
        self.n_interp_gamma = 2000  # gamma table interpolation points
        self._param_display_map = None

        # ── Config-derived views ───────────────────────────────────
        # The amplitude model lives on the Fitter, not on Config.  n_proj
        # and angle_formula are config options, computed here.
        self.n_proj = self._compute_n_proj()
        self.angle_formula_mode = self._compute_angle_formula()

    def _spin_state_count(self, name):
        d = self.dic.get("particle", {}).get(name)
        if not isinstance(d, dict):
            return 1
        spins = d.get("spins")
        if spins is not None:
            return max(1, len(list(spins)))
        return int(2 * d.get("J", 0)) + 1

    def _compute_n_proj(self):
        """Projection count: explicit ``n_proj`` else external spin states."""
        explicit = self.dic.get("n_proj")
        if explicit is not None:
            return int(explicit)
        n = self._spin_state_count(self.top)
        for f in self.finals:
            n *= self._spin_state_count(f)
        return max(1, n)

    def _compute_angle_formula(self):
        mode = self.dic.get("angle_formula", "helicity")
        if mode not in ("helicity", "cache"):
            raise ValueError(
                f"angle_formula must be 'helicity' or 'cache', got {mode!r}")
        return mode

    def topo_index_from_name(self, name):
        """Topology slot of structural decay-section name(s).

        Thin delegate to :meth:`DecayTree.topo_index_from_name`, kept on
        ``Config`` for backward compatibility.
        """
        return self.decay_tree.topo_index_from_name(name)

    def get_max_mass_range(self):
        m_center = self.dic["particle"][self.top]["mass"]
        m_finals = sorted([self.dic["particle"][i]["mass"] for i in self.finals])
        m_max = m_center - m_finals[0]
        m_min = m_finals[0]+ m_finals[1]
        return m_min, m_max

    def build_gamma_table(self, n_interp=None):
        if n_interp is None:
            n_interp = self.n_interp_gamma
        gamma_table = {}
        m_min, m_max = self.get_max_mass_range()
        m = np.linspace(m_min-0.01, m_max + 0.01, n_interp)
        for chain in self.full_decay.chains:
            for decay in chain.decays[1:]:
                g0 = decay.core._model.get_gamma_name()
                if g0[0] not in gamma_table:
                    gamma = decay.core._model.gamma(m)
                    for i,j in zip(g0, gamma):
                        gamma_table[i] = j
        return gamma_table, m[0], m[1]-m[0]


    def get_max_q_range(self):
        m_center = self.dic["particle"][self.top]["mass"]
        m_finals = sorted([self.dic["particle"][i]["mass"] for i in self.finals])
        m_max = m_center - m_finals[0]
        m1, m2 = m_finals[0:2]
        q_max = math.sqrt( (m_max**2 - (m1+m2)**2)*(m_max**2-(m1+m2)**2) )/2/m_max
        return 0., q_max * 1.2  # 20% safety margin for sub-decay q values

    def build_fl_table(self, l_list, n_interp=2000):
        """
        Build Blatt-Weisskopf form factor table with TFPWA normalization.

        Delegates to ``ampfit.utils.bw_form_factor`` for the per-point
        calculation.

        TFPWA normalizes form factors so that F(q0) = q0 at reference
        momentum q0 = 1 GeV.  This ensures consistency with TFPWA
        amplitudes.
        """
        from ampfit.bw_form_factor import form_factor as bw_form_factor

        d = 3.0   # Barrier radius (GeV⁻¹)
        q0 = 1.0  # Reference momentum (GeV)

        q_min, q_max = self.get_max_q_range()
        q = np.linspace(q_min, q_max, n_interp)
        q = np.clip(q, 0, np.inf)

        rows = [bw_form_factor(L, q, q0_ref=q0, d=d) for L in l_list]
        return np.stack(rows, axis=0), q[0], q[1] - q[0]





    def _helicity_top_states(self):
        """External top-helicity projection states (single for a J=0 top)."""
        if getattr(self, "angle_formula_mode", "helicity") != "helicity":
            return [0]
        from ampfit.helicity_angle import to_spin
        top_d = self.dic["particle"].get(self.top, {})
        spins = top_d.get("spins")
        if spins is not None:
            return [to_spin(x) for x in spins]
        J = top_d.get("J", 0)
        if int(J) == 0:
            return [to_spin(0)]
        return [to_spin(m) for m in range(-int(J), int(J) + 1)]

    def _final_names(self):
        """Final-state particle names in canonical (config $finals) order."""
        return list(self.dic.get("particle", {}).get("$finals", []) or [])

    def _helicity_external_states(self):
        """Projection states = top helicity x each final's helicity.

        Each state is ``(lam_top, leaf_lams_in_final_order)``; the
        incoherent |A|^2 sum runs over all of them (spin-averaged finals).
        With spin-0 finals this is exactly the old top-only list.
        """
        from fractions import Fraction
        from ampfit.helicity_angle import to_spin
        tops = self._helicity_top_states()
        per_fin = []
        for nm in self._final_names():
            d = self.dic["particle"].get(nm, {})
            sp = d.get("spins")
            if sp is not None:
                per_fin.append([to_spin(x) for x in sp])
            else:
                j = to_spin(d.get("J", 0))
                j2 = int(2 * j)
                per_fin.append([to_spin(Fraction(v, 1))
                                for v in range(-j2, j2 + 1)])
        import itertools
        out = []
        for lt in tops:
            for leaf in (itertools.product(*per_fin) if per_fin else [()]):
                out.append((lt, leaf))
        return out


    def _n_active_topologies(self):
        """Number of topologies that actually produce partial waves."""
        if getattr(self, "_active_topo_cache", None) is None:
            pw = list(self.full_decay.get_partial_waves())
            self._active_topo_cache = len({dc.topo_id() for _, dc in pw})
        return self._active_topo_cache

    def _wave_angle_terms(self, decaychain, ls, lam):
        """Angular terms of one partial wave for one projection lambda.

        Returns ``[{'coeffs', 'k', 'b'}, ...]`` over the canonical
        phi-first layout (n_angles columns).  Gauge drops the top-J=0
        rotation; a spinful top keeps all 2·n_vertices columns and uses
        the requested external helicity *lam*.

        When the process has SPINFUL FINAL states, each spinful final
        additionally contributes three alignment variables and the wave is
        rotated to a common final axis:

            A_{lambda'} = Sum_{lambda} A_tree(lambda) * D^{j*}_{lambda',lambda}

        (identical to the un-aligned amplitude for alpha=beta=gamma=0).
        Spin-0-final processes take the original code path unchanged.
        """
        if getattr(self, "angle_formula_mode", "helicity") != "helicity":
            return get_angle_formula(decaychain, ls)
        from itertools import product
        from fractions import Fraction
        from ampfit.helicity_angle import (
            to_spin, helicity_values, decay_chain_to_tree, tree_vertices,
            decay_chain_leaves, amplitude_monomials, _reduce_layout,
            alignment_D_parts)
        fin_names = self._final_names()
        finJ = [to_spin(self.dic["particle"].get(nm, {}).get("J", 0))
                for nm in fin_names]
        aligned_idx = [i for i, J in enumerate(finJ) if J != 0]

        top = decaychain.decays[0].core
        leaves = decay_chain_leaves(decaychain)
        top_j0 = to_spin(top.J) == 0
        tree = decay_chain_to_tree(decaychain)
        nv = len(tree_vertices(tree))

        if isinstance(lam, tuple):
            lam_top_in, new_canon = lam
        else:
            lam_top_in, new_canon = lam, None
        lam_top = 0 if top_j0 else to_spin(lam_top_in)

        def _mono(canon):
            """Reduced monomial dict for one canonical leaf-lambda tuple."""
            if canon is None:
                leaf_lams = tuple(0 for _ in leaves)
            else:
                want = dict(zip(fin_names, (to_spin(x) for x in canon)))
                leaf_lams = tuple(to_spin(want[o.name]) if o.name in want
                                  else to_spin(0) for o in leaves)
            _, m = amplitude_monomials(tree, tuple(ls), lam_top, leaf_lams)
            _, m = _reduce_layout(m, nv, (0, 1, 2) if top_j0 else (),
                                  phi_first=True)
            return m

        # ---- no alignment needed (spinless finals, legacy int lam, or a
        # ---- single active topology): original per-state tree monomial ----
        need_align = (aligned_idx and new_canon is not None
                      and self._n_active_topologies() > 1)
        if not need_align:
            mono = _mono(new_canon)
            terms = []
            for key, coef in mono.items():
                if abs(coef) < 1e-12:
                    continue
                k = []
                b = []
                for (kind, f) in key:
                    fr = float(f)
                    k.append(int(round(fr)) if abs(fr - round(fr)) < 1e-9
                             else fr)
                    b.append('cos' if kind == 'c' else 'sin')
                terms.append({'coeffs': complex(round(coef.real, 14),
                                                round(coef.imag, 14)),
                              'k': k, 'b': b})
            return terms

        # ---- spinful finals: rotate to the common final axis ------------
        base_len = None
        align_terms = []          # per aligned final: list of (coef, 3 entries)
        for g, pos in enumerate(aligned_idx):
            jf = finJ[pos]
            lam_new = to_spin(new_canon[pos])
            per = []
            for lam_old in helicity_values(jf):
                a, b, g3 = alignment_D_parts(jf, lam_new, lam_old)
                for ca, (ka, fa) in a:
                    for cb, (kb, fb) in b:
                        for cg, (kg, fg) in g3:
                            per.append((lam_old,
                                        ca * cb * cg,
                                        ((ka, fa), (kb, fb), (kg, fg))))
            align_terms.append(per)

        acc = {}
        for combo in product(*align_terms):
            old_leaf = tuple(it[0] for it in combo)
            coef_align = 1.0
            entries = []
            for (_, cf, ent) in combo:
                coef_align *= cf
                entries.extend(ent)
            old_canon = list(new_canon)
            for pos, val in zip(aligned_idx, old_leaf):
                old_canon[pos] = val
            mono = _mono(tuple(old_canon))
            if base_len is None:
                base_len = len(next(iter(mono)))
            for key, cv in mono.items():
                extkey = key + tuple(entries)
                acc[extkey] = acc.get(extkey, 0.0) + coef_align * cv

        terms = []
        for key, coef in acc.items():
            if abs(coef) < 1e-12:
                continue
            k = []
            b = []
            for (kind, f) in key:
                fr = float(f)
                k.append(int(round(fr)) if abs(fr - round(fr)) < 1e-9
                         else fr)
                b.append('cos' if kind == 'c' else 'sin')
            terms.append({'coeffs': complex(round(coef.real, 14),
                                            round(coef.imag, 14)),
                          'k': k, 'b': b})
        return terms

    def build_single_index(self):
        """Generic kernel-config base (any topology, any angle count, any
        number of spin projections p-major).

        * cache mode / J=0 top: one projection (lambda 0), identical to the
          original output;
        * helicity mode with a spinful top: one *distinct* p-major column
          per external top helicity (columns = p·N + k), shared ck.
        """
        topo_id_map = self.topo_index
        bw_gamma = {}
        waves = list(self.full_decay.get_partial_waves())
        N = len(waves)
        states = self._helicity_external_states()
        P = len(states)
        helicity = getattr(self, "angle_formula_mode", "helicity") == "helicity"

        # ── pass 1: unique bw/gamma/fl + basis union over (wave, proj) ──
        for ls, decaychain in waves:
            for li in ls:
                if li[0] not in self.unique_l:
                    self.unique_l.append(li[0])
            topo = topo_id_map[decaychain.topo_id()]
            for idx, decay in enumerate(decaychain.decays):
                if idx != 0:  # sub-decay resonance
                    m_name = decay.core.name + "_mass"
                    if m_name not in self.m0_phys_name:
                        self.m0_phys_name.append(m_name)
                    m_idx = self.n_res * topo + idx - 1
                    bw_id = (m_name, m_idx)
                    if bw_id not in self.unique_bw:
                        self.unique_bw.append(bw_id)
                    tmp = []
                    for g0 in decay.core._model.get_gamma_name():
                        if g0 not in self.g0_phys_name:
                            self.g0_phys_name.append(g0)
                        g_id = (g0, m_idx)
                        if g_id not in self.unique_gamma:
                            self.unique_gamma.append(g_id)
                        tmp.append(g_id)
                    bw_gamma[bw_id] = tmp
                fl_id = (ls[idx][0], self.n_decay * topo + idx)
                if fl_id not in self.unique_fl:
                    self.unique_fl.append(fl_id)
            projs = states if helicity else [0]
            for lam in projs:
                ang_terms = self._wave_angle_terms(decaychain, ls, lam)
                for ang in ang_terms:
                    basis_key = (topo, tuple(ang["k"]), tuple(ang["b"]))
                    if basis_key not in self.unique_angle_basis:
                        self.unique_angle_basis.append(basis_key)

        # ── build ret ────────────────────────────────────────────────────
        ret = {}
        matrix_gamma = np.zeros((len(self.unique_gamma), len(self.unique_bw)))
        for idx, k in enumerate(self.unique_bw):
            for j in bw_gamma[k]:
                matrix_gamma[self.unique_gamma.index(j), idx] = 1.0
        ret["matrix_gamma"] = matrix_gamma
        bw_order = []
        fl_order = []
        # per-wave rows (entry-major), duplicated P times below
        for ls, decaychain in waves:
            topo = topo_id_map[decaychain.topo_id()]
            for idx, decay in enumerate(decaychain.decays):
                if idx != 0:
                    m_name = decay.core.name + "_mass"
                    m_idx = self.n_res * topo + idx - 1
                    bw_order.append(self.unique_bw.index((m_name, m_idx)))
                fl_id = (ls[idx][0], self.n_decay * topo + idx)
                fl_order.append(self.unique_fl.index(fl_id))
        # columns p-major: col = p*N + kk
        matrix_cols = []
        for p in range(P):
            for ls, decaychain in waves:
                topo = topo_id_map[decaychain.topo_id()]
                lam = states[p] if helicity else 0
                ang_terms = self._wave_angle_terms(decaychain, ls, lam)
                col = np.zeros(len(self.unique_angle_basis)) + 0j
                for ang in ang_terms:
                    basis_key = (topo, tuple(ang["k"]), tuple(ang["b"]))
                    col[self.unique_angle_basis.index(basis_key)] = ang["coeffs"]
                matrix_cols.append(col)
        ret["matrix_angle"] = np.stack(matrix_cols, axis=-1)
        ret["bw_order"] = np.array(bw_order * P)   # p-major row duplication
        ret["fl_order"] = np.array(fl_order * P)

        angle_index = []
        angle_k = []
        angle_b = []
        for key in self.unique_angle_basis:
            angle_index.append(key[0])
            angle_k.append(key[1])
            angle_b.append([(0 if k == "cos" else -np.pi / 2) for k in key[2]])
        ret["angle_index"] = np.stack(angle_index)
        ret["angle_k"] = np.stack(angle_k)
        ret["angle_b"] = np.stack(angle_b)
        ret["n_wave_base"] = N
        ret["n_proj_base"] = P

        m0_index = []
        mass_index = []
        for key in self.unique_bw:
            m0_index.append(self.m0_phys_name.index(key[0]))
            mass_index.append(key[1])
        ret["mass_index"] = np.stack(mass_index)
        ret["m0_index"] = np.stack(m0_index)
        g0_index = []
        g0_mass_index = []
        for key in self.unique_gamma:
            g0_index.append(self.g0_phys_name.index(key[0]))
            g0_mass_index.append(key[1])
        ret["g0_mass_index"] = np.stack(g0_mass_index)
        ret["g0_index"] = np.stack(g0_index)
        fl_type = []
        fl_q_index = []
        for key in self.unique_fl:
            fl_type.append(self.unique_l.index(key[0]))
            fl_q_index.append(key[1])
        ret["fl_type"] = np.stack(fl_type)
        ret["fl_q_index"] = np.stack(fl_q_index)

        gamma_table, g_min, g_delta = self.build_gamma_table()
        ret["gamma_table"] = np.stack(
            [gamma_table[i] for i in self.g0_phys_name], axis=0)
        ret["gamma_min"] = g_min
        ret["gamma_delta"] = g_delta
        ret["fl_table"], ret["fl_min"], ret["fl_delta"] = self.build_fl_table(self.unique_l)
        ret["ck_map"] = list(self.full_decay.get_partial_waves_params())
        # canonical per-event angle columns (phi-first) of the first chain —
        # the layout the event data builder must fill for pure-PWA models
        try:
            from ampfit.helicity_angle import (decay_chain_to_tree,
                                               tree_vertices, canonical_variables,
                                               to_spin)
            first_chain = waves[0][1]
            _nv = len(tree_vertices(decay_chain_to_tree(first_chain)))
            _topj0 = to_spin(first_chain.decays[0].core.J) == 0
            ret["variables"] = canonical_variables(_nv, top_j0=_topj0)
            ret["top_j0"] = bool(_topj0)
        except Exception:
            pass
        return ret

    def build_all_index(self):
        """Kernel config (compatibility shim → :meth:`build_base_kernel_config`)."""
        return self.build_base_kernel_config()

    def build_base_kernel_config(self):
        # identical-particle × CP row blocks, from the config declarations
        # (legacy B→4π: 4 permutations × 2 CP = 8; pure PWA without
        # declarations → 1).
        n_perm, n_cp, C = row_block_factors(self.dic)
        # loop and shift based on block (single generic base; blocks only
        # for declared identical/CP partners; pure PWA -> C == 1).
        base = self.build_single_index()
        ck = self.full_decay.get_partial_waves_params()
        P = base.get("n_proj_base", 1) or 1

        def _repeat(arr, count=C):
            return np.concatenate([arr]*count, axis=0)

        def _shift_repeat(arr, strip, count=C):
            all_arr = []
            for i in range(count):
                all_arr.append(arr + strip * i)
            return np.concatenate(all_arr,axis=0)

        def _matrix_repeat(arr, count=C):
            all_arr = []
            for i in range(count):
                all_tmp_arr = []
                for j in range(count):
                    if i==j:
                        all_tmp_arr.append(arr)
                    else:
                        all_tmp_arr.append(np.zeros_like(arr))
                all_arr.append(np.concatenate(all_tmp_arr, axis=1))
            return np.concatenate(all_arr,axis=0)

        ret = {}
        ret["m0_index"] = _repeat(base["m0_index"])
        ret["g0_index"] = _repeat(base["g0_index"])
        ret["fl_type"] = _repeat(base["fl_type"])
        ret["angle_k"] = _repeat(base["angle_k"])
        ret["angle_b"] = _repeat(base["angle_b"])
        ret["fl_type"] = _repeat(base["fl_type"])
        ret["mass_index"] = _shift_repeat(base["mass_index"], self.n_topo* self.n_res)
        ret["g0_mass_index"] = _shift_repeat(base["g0_mass_index"], self.n_topo* self.n_res)
        ret["fl_q_index"] = _shift_repeat(base["fl_q_index"], self.n_topo* self.n_decay)
        ret["bw_order"] = _shift_repeat(base["bw_order"], len(self.unique_bw))
        ret["fl_order"] = _shift_repeat(base["fl_order"], len(self.unique_fl))
        ret["angle_index"] = _shift_repeat(base["angle_index"], self.n_topo)
        ret["matrix_angle"] = _matrix_repeat(base["matrix_angle"])
        ret["matrix_gamma"] = _matrix_repeat(base["matrix_gamma"])
        ret["n_perm"] = n_perm
        ret["n_cp"] = n_cp
        ret["n_blocks"] = C

        for name in ["gamma_table","fl_table", "gamma_min", "gamma_delta", "fl_min", "fl_delta"]:
            ret[name] = base[name]

        # ── cuda_v4_pwa: p-major projection duplication ─────────────────
        # When the config has more than one incoherent projection (spin
        # projections of top/final particles), the per-wave arrays are
        # duplicated p-major (n_wave → P·N) so every projection owns all
        # waves.  All projections share one ck of length N = n_wave/P.
        # PLACEHOLDER: until the spin/helicity angular generator lands, the
        # duplicated matrix_angle columns are identical copies (the angular
        # difference per projection is filled in later).
        ret["n_proj"] = P
        for meta in ("ck_map", "variables", "top_j0", "n_proj_base"):
            if meta in base:
                ret[meta] = base[meta]

        return ret

    def get_ck_map(self):
        cks = self.full_decay.get_partial_waves_params()
        n_perm, n_cp, C = row_block_factors(self.dic)
        ret = []
        for cp in range(n_cp):
            for _ in range(n_perm):
                for j in cks:
                    if cp == 0:
                        ret.append(j)
                    else:
                        ret.append((j[0], j[1].replace("g_ls", "g_lsbar"), *j[2:]))
        return ret

    def chain_ck_ranges(self):
        """Per-chain ck index ranges in the base (pre-block) layout.

        Returns ``[(base_start, base_end, chain), ...]``: each full decay
        chain owns ``len(chain.get_gls_combination())`` consecutive ck slots,
        and the last ``base_end`` is the total base ck count (equal to
        ``len(full_decay.get_partial_waves_params())``).  Row-block expansion
        (``n_perm · n_cp``) is applied separately by ``_expand_to_blocks``.
        """
        idx = 0
        ranges = []
        for chain in self.full_decay.chains:
            n = len(chain.get_gls_combination())
            ranges.append((idx, idx + n, chain))
            idx += n
        return ranges

    def _expand_to_blocks(self, base_indices, n_base):
        """Repeat base_indices across all row blocks.

        The block count is the config-derived ``n_perm · n_cp`` from
        :func:`row_block_factors` (identical-particle permutations × the
        B0/B0bar CP map) — 8 for the legacy four-body configs, 1 for a
        single-flavour pure-PWA config.
        """
        n_blocks = row_block_factors(self.dic)[2]
        result = []
        for block in range(n_blocks):
            offset = block * n_base
            for i in sorted(base_indices):
                result.append(offset + i)
        return result

    def get_ck_indices(self, resonance_names):
        """Return ck indices for partial waves involving given resonance(s).

        Any chain where a decay's ``core.name`` matches one of the
        resonance names contributes all its partial-wave ck indices.

        Args:
            resonance_names: str or list of str — particle names from config,
                             e.g. ``"f0(500)"`` or ``["a1(1260)p", "a1(1260)m"]``.

        Returns:
            list[int] — ck indices covering all row blocks (n_perm · n_cp).
        """
        if isinstance(resonance_names, str):
            resonance_names = [resonance_names]
        target = set(resonance_names)

        chain_ranges = self.chain_ck_ranges()
        n_base = chain_ranges[-1][1] if chain_ranges else 0

        matching = set()
        for start, end, chain in chain_ranges:
            for decay in chain.decays:
                if decay.core.name in target:
                    matching.update(range(start, end))
                    break

        return self._expand_to_blocks(matching, n_base)

    def get_decay_ck_indices(self, decay_pairs, wave_idx=None):
        """Return ck indices for chains matching specific decay relationships.

        A chain matches when *every* ``(parent, child)`` pair in
        *decay_pairs* is found in its decay tree.

        Examples::

            # All chains where a1(1260)p → f0(500) + X
            cfg.get_decay_ck_indices([("a1(1260)p", "f0(500)")])

            # Chains with both a1(1260)p → f0(500) and f0(500) → pip1
            cfg.get_decay_ck_indices([("a1(1260)p", "f0(500)"),
                                      ("f0(500)", "pip1")])

            # Only the S-wave (first LS combination) of a1 → ρπ
            cfg.get_decay_ck_indices([("a1(1260)p", "rhoA")], wave_idx=0)

            # Only the D-wave (second LS combination) of a1 → ρπ
            cfg.get_decay_ck_indices([("a1(1260)p", "rhoA")], wave_idx=1)

        Args:
            decay_pairs: list of ``(parent_name, child_name)`` tuples.
                         All pairs must be satisfied by the same chain.
            wave_idx: int or None.  If None, include all partial waves
                      (all base indices in the chain range).  If an int,
                      include only the *wave_idx*-th partial wave
                      (i.e. ``start + wave_idx`` within each matching
                      chain).  This separates individual LS combinations
                      within a decay, e.g., S-wave (wave_idx=0) from
                      D-wave (wave_idx=1) for a1 → ρπ.

        Returns:
            list[int] — ck indices covering all row blocks (n_perm · n_cp).
        """
        chain_ranges = self.chain_ck_ranges()
        n_base = chain_ranges[-1][1] if chain_ranges else 0

        matching = set()
        for start, end, chain in chain_ranges:
            # Check every pair is satisfied somewhere in this chain
            ok = True
            for parent, child in decay_pairs:
                found = False
                for decay in chain.decays:
                    if decay.core.name == parent:
                        out_names = [o.name for o in decay.outs]
                        if child in out_names:
                            found = True
                            break
                if not found:
                    ok = False
                    break
            if ok:
                if wave_idx is not None:
                    matching.add(start + wave_idx)
                else:
                    matching.update(range(start, end))

        return self._expand_to_blocks(matching, n_base)

    # ── Display helpers ──────────────────────────────────────────

    def name_display_map(self):
        """Map particle config names to display names.

        Returns a dict ``{config_name: display_name}``, where
        *display_name* is the LaTeX-formatted string (with ``$``
        delimiters) from :attr:`Particle.display`.

        Values with the same display name merge naturally (e.g. both
        ``rhoA`` and ``rhoB`` map to the same key).
        """
        seen = {}
        for chain in self.full_decay.chains:
            for decay in chain.decays:
                p = decay.core
                if p.name not in seen:
                    seen[p.name] = p.display
                for out in decay.outs:
                    if out.name not in seen:
                        seen[out.name] = out.display
        return seen

    def display_decay(self, decay):
        """LaTeX display string for a decay: ``parent → child1 child2``.

        Uses :attr:`Particle.display` for each particle name.
        """
        parent = decay.core.display
        children = [o.display for o in decay.outs]
        return rf"{parent} \to {children[0]}\,{children[1]}"

    def display_chain(self, chain):
        """LaTeX display string for an entire decay chain."""
        parts = [self.display_decay(d) for d in chain.decays]
        return r" \quad ".join(parts)

    _L_LABEL = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}

    def display_g_ls(self, decay):
        """Display names for each ``g_ls`` partial wave of a decay.

        Returns a list of LaTeX strings, one per LS combination::

            g^{{\\rho \\to \\pi\\pi}}_{{S}},  g^{{\\rho \\to \\pi\\pi}}_{{D}},  …
        """
        sup = self.display_decay(decay).replace("$", "")
        ls_list = decay.get_ls_list()
        return [rf"$g^{{{sup}}}_{{{self._L_LABEL.get(l, str(l))}}}$"
                for l, s in ls_list]

    def display_g_lsbar(self, decay):
        """Display names for each ``\\bar{{g}}_{{ls}}`` partial wave."""
        sup = self.display_decay(decay).replace("$", "")
        ls_list = decay.get_ls_list()
        return [rf"$\bar{{g}}^{{{sup}}}_{{{self._L_LABEL.get(l, str(l))}}}$"
                for l, s in ls_list]

    def display_a_total(self, chain):
        """Display name for the total amplitude: ``a_{{\\mathrm{{total}}}}^{{decay[0]}}``."""
        sup = self.display_decay(chain.decays[0]).replace("$", "") if chain.decays else ""
        return rf"$a_{{\mathrm{{total}}}}^{{{sup}}}$"

    def _build_param_display_map(self):
        """Pre-build mapping of all config-known parameter names to LaTeX display strings."""
        n_map = self.name_display_map()

        # Scalar names (fixed)
        sc = {
            'gamma':       r'$\Gamma$',
            'delta_gamma': r'$\Delta\Gamma$',
            'delta_m':     r'$\Delta m$',
            'A_prod':      r'$A_{\mathrm{prod}}$',
            'poqr':        r'$\mathrm{poq}_r$',
            'poqi':        r'$\mathrm{poq}_i$',
        }

        # Particle mass / width
        for pname, pdisp in n_map.items():
            b = pdisp.strip("$")
            sc[f"{pname}_mass"]  = rf"$m_{{{b}}}$"
            sc[f"{pname}_width"] = rf"$\Gamma_{{{b}}}$"

        # Decay-level g_ls and total
        for chain in self.full_decay.chains:
            cs = str(chain).replace("+", ".")
            for decay in chain.decays:
                ds = str(decay).replace("+", ".")
                g_disps = self.display_g_ls(decay)
                gb_disps = self.display_g_lsbar(decay)
                for idx, gd in enumerate(g_disps):
                    base = gd.strip("$")
                    sc[f"{ds}_g_ls_{idx}r"] = rf"$|{base}|$"
                    sc[f"{ds}_g_ls_{idx}i"] = rf"$\arg({base})$"
                for idx, gd in enumerate(gb_disps):
                    base = gd.strip("$")
                    sc[f"{ds}_g_lsbar_{idx}r"] = rf"$|{base}|$"
                    sc[f"{ds}_g_lsbar_{idx}i"] = rf"$\arg({base})$"
            # total amplitude for each chain
            base = self.display_a_total(chain).strip("$")
            sc[f"{cs}_total_0r"] = rf"$|{base}|$"
            sc[f"{cs}_total_0i"] = rf"$\arg({base})$"

        return sc

    def param_display(self, name):
        """Map a parameter name to its LaTeX display string.

        Uses a pre-built map covering mass, width, g_ls, g_lsbar, total,
        and scalar parameters.  Falls back to ``$\\mathrm{{name}}$`` with
        underscores escaped so ``_`` does not create unwanted subscripts.
        """
        if self._param_display_map is None:
            self._param_display_map = self._build_param_display_map()
        if name in self._param_display_map:
            return self._param_display_map[name]
        # Fallback: escape underscores for safe math-mode rendering
        safe = name.replace("_", r"\_")
        return rf"$\mathrm{{{safe}}}$"
