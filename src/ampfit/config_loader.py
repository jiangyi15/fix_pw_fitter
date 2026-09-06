import yaml
import itertools
import numpy as np
from .particle_model import build_particle
from .angular_formula import get_angle_formula
import math


def load_config(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename) as f:
        ret = yaml.safe_load(f)
    return ret


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


# Legacy time/mixing scalar parameters (D0-D0bar flavour-tagged mixing).
LEGACY_SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod",
                       "poqr", "poqi"]
LEGACY_SCALAR_DEFAULTS = {"gamma": 0.0, "delta_gamma": 0.0, "delta_m": 0.506,
                          "A_prod": 0.0, "poqr": 1.0, "poqi": 0.0}
# data.amp_model values that imply the flavour-tag mixing model ⇒ the legacy
# six scalars are used.  ``flavour_tag_mix`` is the canonical tag-mix model
# (configs may also spell it ``flour_tag_mix``); ``p4_directly`` kept for
# backward compatibility of the old B→4π configs.
MIXING_AMP_MODELS = ("flavour_tag_mix", "flour_tag_mix", "p4_directly")


class Particle:
    def __init__(self, name, **kwargs):
        self.name = name
        self._decays = []
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'display'):
            from ampfit.utils import fmt_particle
            self.display = fmt_particle(name, full=True)
    def add_decay(self, decay):
        """Register a Decay in this particle if no identical decay exists."""
        out_names = tuple(o.name for o in decay.outs)
        for d in self._decays:
            if tuple(o.name for o in d.outs) == out_names:
                return  # already registered
        self._decays.append(decay)
    def __str__(self):
        return self.name


def s_range(l1, l2):
    a = l1
    while a < l2:
        yield a
        a += 1

class Decay:
    def __init__(self, core, outs, **kwargs):
        self.core = core
        self.outs = outs
        core.add_decay(self)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_ls_list(self):
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        pa = getattr(self.core, "P", None)
        pb = getattr(self.outs[0], "P", None)
        pc = getattr(self.outs[1], "P", None)
        ls_list = []
        for s in s_range(abs(jb-jc), jb+jc +1):
            for l in range(int(abs(ja-s)), int(ja + s)+1):
                if not getattr(self, "p_break", False):
                    if (pa is not None and pb is not None and pc is not None):
                        if l % 2 == (1 if pa*pb*pc == 1 else 0):
                            continue
                ls_list.append((l,s))
        return ls_list

    def get_ls_names(self):
        prefix = str(self).replace("+", ".")
        ret = []
        for i in range(len(self.get_ls_list())):
            ret.append(f"{prefix}_g_ls_{i}")
        return ret

    def __str__(self):
        return f"{self.core}->" + "+".join([str(i) for i in self.outs])

class DecayChain:
    def __init__(self, decays):
        self.decays = decays
        decay_particles = [i.core.name for i in self.decays]
        out_particles = []
        for i in self.decays:
            for j in i.outs:
                out_particles.append(j.name)
        self.finals = [i for i in out_particles if i not in decay_particles]
        self.top = [i for i in decay_particles if i not in out_particles][0]
        self.inner = [i for i in out_particles if i in decay_particles]

    def get_ls_combination(self):
        ls_lists = [i.get_ls_list() for i in self.decays]
        # print(ls_lists)
        ret = list(itertools.product(*ls_lists))
        return ret

    def get_gls_combination(self):
        ls_lists = [i.get_ls_names() for i in self.decays]
        # print(ls_lists)
        total = str(self).replace("+",".") + "_total_0"
        ret = list(itertools.product([total], *ls_lists))
        return ret

    def get_topo_map(self):
        topo_map = {k: [k] for k in self.finals}
        while self.top not in topo_map:
            for j in self.decays:
                if all([k.name in topo_map for k in j.outs]):
                    tmp = []
                    for k in j.outs:
                        tmp += topo_map[k.name]
                    topo_map[j.core.name] = tmp
        return topo_map

    def topo_id(self):
        topo_map = self.get_topo_map()
        topo_id = tuple(sorted([tuple(sorted(topo_map[i]))  for i in self.inner   ]))
        return topo_id

    def __str__(self):
        names = [str(i) for i in self.decays]
        return "".join(names)



class DecayGroup:
    def __init__(self, chains):
        self.chains = chains

    def __iter__(self):
        return iter(self.chains)

    def get_partial_waves(self):
        ret = []
        for i in self.chains:
            for j in i.get_ls_combination():
                ret.append((j, i))
        return ret

    def get_partial_waves_params(self):
        ret = []
        for i in self.chains:
            for j in i.get_gls_combination():
                ret.append(j)
        return ret



class Config:
    def __init__(self, filename):
        self.dic = load_config(filename)
        self._config_path = filename if isinstance(filename, str) else ""
        top = self.dic["particle"]["$top"]
        finals = list(self.dic["particle"]["$finals"])
        self.top = top
        self.finals = finals

        self.decay_struct = self.build_decay_struct(self.dic["decay"], top, finals)
        self.decay_chains_lst = self.get_decay_chains(self.decay_struct, self.dic["particle"])
        self.full_decay = self.build_decay_chains(self.decay_chains_lst, self.dic["particle"])

        self.n_decay = len(self.decay_struct[0])
        self.n_res = self.n_decay - 1
        self.n_angles = 2 * self.n_decay
        if self.dic["particle"][top]["J"] == 0:
            self.n_angles = self.n_angles - 3 # 3d rotaion is not need
        self.topo_index = self.get_topo_index(self.full_decay)
        self.n_topo = len(self.topo_index)
        self.m0_phys_name = []
        self.g0_phys_name = []
        self.unique_l = []
        self.unique_bw = []
        self.unique_gamma = []
        self.unique_fl = []
        self.unique_angle_basis = []
        self.n_interp_gamma = 2000  # gamma table interpolation points
        self._param_display_map = None

        # ── cuda_v4_pwa / scalar-free extensions ───────────────────────
        # n_proj: number of incoherent projections (helicity / spin
        # projections of the EXTERNAL particles).  Every projection shares
        # the same ck; wave entries are stored p-major (n_wave = n_proj·N).
        # Explicit ``n_proj`` in the config wins; otherwise auto-computed as
        #   n_proj = ∏_i n_i   over i ∈ {top} ∪ {finals}
        #   n_i    = len(spins) if the particle declares ``spins`` else 2J+1
        # (intermediate resonances are NOT counted — they only shape the
        # per-wave angular formula).
        if self.dic.get("n_proj") is not None:
            self.n_proj = int(self.dic["n_proj"])
        else:
            self.n_proj = self._auto_n_proj()
        # scalar_names / scalar_defaults: config-driven fit scalars.  Absent
        # → legacy six mixing scalars (kept so old time/mixing configs work);
        # ``scalar_names: []`` removes scalars entirely (v4 PWA model).
        self.scalar_names = self._resolve_scalar_names()
        self.scalar_defaults = self.dic.get("scalar_defaults")
        # ── angular-formula implementation switch ──────────────────────
        # 'helicity' (default) → the numeric helicity-angle engine
        #   (amplitude_monomials / chain_angular_table).
        # 'cache' → the predefined angular_formula.cache_formula tables
        #   (kept for cross-checking the two implementations).
        mode = self.dic.get("angle_formula", "helicity")
        if mode not in ("helicity", "cache"):
            raise ValueError(
                f"angle_formula must be 'helicity' or 'cache', got {mode!r}")
        self.angle_formula_mode = mode

    # ── scalar decision (data.amp_model) ──────────────────────────────
    def _data_amp_model(self):
        """The ``data.amp_model`` model name (string).

        The YAML value may be a plain string (legacy, e.g. ``p4_directly``)
        or a dict keyed by the model name, e.g.
        ``{'flavour_tag_mix': {base_model: time_dep_cp, ...}}`` — in that
        case the single dict key is the model name.  Returns None if unset.
        """
        data_d = self.dic.get("data") or {}
        m = data_d.get("amp_model")
        if isinstance(m, dict):
            m = next(iter(m)) if len(m) else None
        elif isinstance(m, (list, tuple)):
            m = m[0] if len(m) else None
        if isinstance(m, str):
            m = m.strip() or None
        return m

    def _resolve_scalar_names(self):
        """Config-driven scalar list (legacy time/mixing scalars or none).

        Decided by ``data.amp_model`` (see :meth:`_data_amp_model`):
          * explicit top-level ``scalar_names:`` list → used as-is
          * ``data.amp_model`` in ``MIXING_AMP_MODELS`` (e.g.
            ``flavour_tag_mix``) → the legacy six mixing scalars
          * anything else (incl. no amp_model) → ``[]`` — scalar-free,
            i.e. the pure-PWA / non-tag-mixing models get no scalars.
        """
        explicit = self.dic.get("scalar_names")
        if explicit is not None:
            return list(explicit)
        amp_model = self._data_amp_model()
        if amp_model in MIXING_AMP_MODELS:
            return list(LEGACY_SCALAR_NAMES)
        return []

    def _spin_state_count(self, name):
        """Number of spin states of an external particle.

        ``n_i = len(spins)`` when the particle declares a ``spins`` list
        (e.g. ``[-1, 1]`` = transverse-only spin-1), else ``2J+1``.
        """
        d = self.dic["particle"].get(name)
        if not isinstance(d, dict):
            return 1            # composite / non-dict entry — not external
        spins = d.get("spins")
        if spins is not None:
            return max(1, len(list(spins)))
        return int(2 * d.get("J", 0)) + 1

    def _auto_n_proj(self):
        """n_proj from the spin multiplicities of top + final particles."""
        n = self._spin_state_count(self.top)
        for f in self.finals:
            n *= self._spin_state_count(f)
        return max(1, n)

    def get_topo_index(self, decay):
        topo_id = {}
        for i in decay.chains:
            tmp = i.topo_id()
            if tmp not in topo_id:
                topo_id[tmp] = len(topo_id)
        return topo_id


    def build_decay_struct(self, decay, top, finals):
        # loop to find the chains that top -> a + b, a-> ..., b -> ...
        # filter with finals

        def get_sub_decays(dic, top):

            if top not in dic:
                return [[]]
            res_decay = dic[top]
            if not isinstance(res_decay[0], list):
                res_decay = [ res_decay ]

            ret = []
            for i in res_decay:
                outs = [j for j in i if isinstance(j, str)]
                kwargs_list = [k for k in i if isinstance(k, dict)]
                kwargs = {}
                for kw in kwargs_list:
                    kwargs.update(kw)
                outdecay = [get_sub_decays(dic, outi) for outi in outs]
                # print(outdecay)
                for prod in itertools.product(*outdecay):
                    # print("prod", "prod")
                    table = [(top, outs, kwargs)]
                    for i in prod:
                        table += i
                    ret.append(table)

            return ret

        return get_sub_decays(decay, top)


    def get_decay_chains(self, decay_struct, res_map):
        ret = []
        for decay_chain in decay_struct:
            used_res = []
            for i in decay_chain:
                if i[0] not in used_res:
                        used_res.append(i[0])
                for j in i[1]:
                    if j not in used_res:
                        used_res.append(j)
            combinations = []
            replace_res = []
            for j in used_res:
                if isinstance(res_map[j], list):
                    combinations.append(res_map[j])
                    replace_res.append(j)

            for c in itertools.product(*combinations):
                tmp = decay_chain
                for ci,ji in zip(c, replace_res):
                    tmp = replace_chain(tmp, ji, ci)
                ret.append(tmp)
        return ret

    def build_decay_chains(self, lst, dic):
        all_particles = []
        for i in lst:
            for j in i:
                if j[0] not in all_particles:
                    all_particles.append(j[0])
                for k in j[1]:
                    if k not in all_particles:
                        all_particles.append(k)
        particles = {}
        for k in all_particles:
            particles[k] = Particle(k, **dic[k])
            particles[k]._model = build_particle(k, **dic[k])
            particles[k]._model.register_parent(particles[k])
        ret = []
        for i in  lst:
            tmp = []
            for j in i:
                tmp.append(Decay(particles[j[0]], [particles[k] for k in j[1]], **j[2]))
            chain = DecayChain(tmp)
            ret.append(chain)
        return DecayGroup(ret)

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

    def _wave_angle_terms(self, decaychain, ls, lam):
        """Angular terms of one partial wave for one projection lambda.

        Returns ``[{'coeffs', 'k', 'b'}, ...]`` over the canonical
        phi-first layout (n_angles columns).  Gauge drops the top-J=0
        rotation; a spinful top keeps all 2·n_vertices columns and uses
        the requested external helicity *lam*.
        """
        if getattr(self, "angle_formula_mode", "helicity") != "helicity":
            return get_angle_formula(decaychain, ls)
        from ampfit.helicity_angle import (
            to_spin, decay_chain_to_tree, tree_vertices,
            decay_chain_leaves, amplitude_monomials, _reduce_layout)
        top = decaychain.decays[0].core
        leaves = decay_chain_leaves(decaychain)
        top_j0 = to_spin(top.J) == 0
        if not top_j0 and any(to_spin(o.J) != 0 for o in leaves):
            raise NotImplementedError(
                "helicity mode: spinful final states not supported")
        tree = decay_chain_to_tree(decaychain)
        nv = len(tree_vertices(tree))
        lam_top = 0 if top_j0 else to_spin(lam)
        _, mono = amplitude_monomials(tree, tuple(ls), lam_top,
                                      tuple(0 for _ in leaves))
        _, mono = _reduce_layout(mono, nv, (0, 1, 2) if top_j0 else (),
                                 phi_first=True)
        terms = []
        for key, coef in mono.items():
            if abs(coef) < 1e-12:
                continue
            k = []
            b = []
            for (kind, f) in key:
                fr = float(f)
                k.append(int(round(fr)) if abs(fr - round(fr)) < 1e-9 else fr)
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
        states = self._helicity_top_states()
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
        ret["fl_table"], ret["fl_min"], ret["fl_delta"] =             self.build_fl_table(self.unique_l)
        ret["ck_map"] = list(self.full_decay.get_partial_waves_params())
        return ret

    def _build_pwa_index(self):
        """Pure-PWA kernel arrays (C == 1, helicity mode) via pwa_build.

        Also syncs the Fitter-facing Config attributes (m0/g0 parameter-name
        lists) that are normally populated by ``build_single_index``.
        """
        from ampfit.pwa_build import build_pwa_kernel_config
        kc = build_pwa_kernel_config(self)
        self.m0_phys_name = list(kc.get("m0_names", self.m0_phys_name))
        self.g0_phys_name = list(kc.get("g0_names", self.g0_phys_name))
        return kc

    def build_all_index(self):
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
        if "ck_map" in base:
            ret["ck_map"] = base["ck_map"]

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

    def _chain_ranges(self):
        """Build list of (base_start, base_end, chain) for all chains."""
        idx = 0
        ranges = []
        for chain in self.full_decay.chains:
            n = len(chain.get_gls_combination())
            ranges.append((idx, idx + n, chain))
            idx += n
        return ranges

    def _expand_to_blocks(self, base_indices, n_base):
        """Repeat base_indices across all 8 topology blocks."""
        result = []
        for block in range(8):
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
            list[int] — ck indices covering all 8 topology blocks.
        """
        if isinstance(resonance_names, str):
            resonance_names = [resonance_names]
        target = set(resonance_names)

        chain_ranges = self._chain_ranges()
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
            list[int] — ck indices covering all 8 topology blocks.
        """
        chain_ranges = self._chain_ranges()
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
        import re
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




def replace_chain(chain, res, res2):
    ret = []
    for decay in chain:
        core = res2 if decay[0] == res else decay[0]
        outs = [res2 if i == res else i for i in decay[1]]
        ret.append( (core, outs, *decay[2:]))
    return ret


if __name__=="__main__":
    a = Config("config_angle.yml")
    c = a.build_all_index()
    from ampfit.numpy_kernel import NumpyKernel
    kernel  = NumpyKernel(c)
    ck = a.get_ck_map()
    # print(c)
    n_events = 7
    n = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }
    )
    l = kernel._compute(
    {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    },
    {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    },
    norm=1.0
    )

    params = {"ck": np.random.random(len(ck)) + 1j*np.random.random(len(ck)),
     "m0": np.random.random(len(a.m0_phys_name)) + 2,
     "g0": np.random.random(len(a.g0_phys_name)) + 0.01,
     "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    data =     {
     "mass": np.random.random((n_events, 2*3*8)),
     "q": np.random.random((n_events, 3*3*8)),
     "angle": np.random.random((n_events, 3*8, 3)),
     "frac": np.random.random((n_events, )),
     "time": np.random.random((n_events, )),
     "bkg": np.random.random((n_events, )),
     "weight": np.ones_like(np.random.random((n_events, ))),
    }


    l = kernel._compute(params,
    data,
    norm=1.0
    )
    l2 = [kernel._compute(params,
    {k: v[i:i+1] for k, v in data.items()},
    norm=1.0
    ) for i in range(n_events)]
    print(l[0], sum(i[0] for i in l2))
    print(l[2], [i[2] for i in l2])
