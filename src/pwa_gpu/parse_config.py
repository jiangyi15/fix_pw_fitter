"""
Parser: tf_pwa YAML config → pwa_gpu kernel arrays.

Convention:
  A_wave = ck[w] * (∏ 1/bwall_r) * (∏ bf_d) * ag[w]

Each physical decay chain with N LS combos → N kernel waves (one per LS choice).
CPV doubles this (B / Bbar in first/second half_waves).

No gamma/bf interpolation tables yet — only structural arrays.
"""

import yaml
import itertools
import math
import numpy as np
from collections import OrderedDict

# ====================================================================
# 1. Data structures
# ====================================================================

class ResonantState:
    """A resonant particle with quantum numbers."""
    def __init__(self, name, J=0, P=1, mass=0.0, width=0.0,
                 model='BW', **extra):
        self.name = name
        self.J = J
        self.P = P
        self.mass = float(mass)
        self.width = float(width)
        self.model = model
        self.extra = extra  # gamma_file, g_*, mass_list, etc.

    def __repr__(self):
        return f"{self.name}(J={self.J},P={self.P:+d})"


class DecayStep:
    """One decay: parent → [daughters] with model info."""
    def __init__(self, parent, daughters, model=None, bf_L=None, p_break=False):
        self.parent = parent        # str
        self.daughters = list(daughters)  # [str, ...]
        self.model = model          # e.g. 'gls_cpv_aabar' or None
        self.bf_L = bf_L            # orbital L for barrier factor
        self.p_break = p_break      # if True, all LS combos allowed (no parity filter)

    def __repr__(self):
        return f"{self.parent}→{self.daughters}"


class PhysicalWave:
    """
    One physical decay chain with specific resonance assignment.

    Contains:
      chain: [DecayStep, DecayStep, ...]  — decay tree
      resonances: [ResonantState, ...]     — in order along chain
      topology: str
      topology_key: (topology, J1, P1, J2, P2)  — for angular formula
      n_ls_per_decay: [int, ...]                — LS count per decay
    """
    def __init__(self, wid):
        self.id = wid
        self.name = ""
        self.chain = []           # [DecayStep, ...]
        self.resonances = []      # [ResonantState, ...]
        self.topology = ""
        self.topology_key = None
        self.n_ls_per_decay = []  # one per DecayStep
        self.has_cpv = False

    def n_ls_counts(self):
        """Get list of LS counts only."""
        return [c if isinstance(c, int) else c[0] for c in self.n_ls_per_decay]

    def get_ls_list(self, decay_idx):
        """Get [(L,S), ...] for a given decay index."""
        if decay_idx < len(self.n_ls_per_decay):
            entry = self.n_ls_per_decay[decay_idx]
            if isinstance(entry, tuple):
                return entry[1]  # the list of (L,S)
        return [(0, 0)]

    def __repr__(self):
        return (f"PW[{self.id}] {self.name} topo={self.topology} "
                f"ls={self.n_ls_counts()}")


class KernelWave:
    """
    One kernel wave: one LS combo of one decay in one CPV sector.

      ck_formula: list of (param_type, param_name, ls_idx)
                  describing how to compute ck = product of values
      ag_matrix_entry: list of (basis_idx, coeff)  for matrix_ang row
      bw_order_entries: [res_idx, ...]  indices into bwall array
      bf_order_entries: [bf_type_idx, ...]
      phys_parent: int  (physical wave id)
      is_bar: bool  (True = Bbar, False = B)
    """
    def __init__(self, kwid, pw_id):
        self.id = kwid
        self.pw_id = pw_id
        self.ck_formula = []          # [(type, name, ls_idx), ...]
        self.ag_matrix_entry = []     # [(basis_idx, complex_coeff), ...]
        self.bw_order_entries = []    # [res_idx, ...]
        self.bf_order_entries = []    # [bf_type_idx, ...]
        self.is_bar = False

    def __repr__(self):
        return f"KW[{self.id}] PW={self.pw_id} bar={self.is_bar} ck={self.ck_formula}"


# ====================================================================
# 2. Config loading helpers
# ====================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_decay(cfg, name):
    """Look up decay daughters in 'decay' section."""
    d = cfg.get('decay', {})
    if name in d:
        raw = d[name]
        daughters = [x for x in raw if isinstance(x, str)]
        opts = {}
        for x in raw:
            if isinstance(x, dict):
                opts.update(x)
        return daughters, opts
    return [], {}


def get_particle(cfg, name):
    """Look up particle properties in 'particle' section."""
    return cfg.get('particle', {}).get(name, None)


def get_finals(cfg):
    return cfg.get('particle', {}).get('$finals', [])


def get_top(cfg):
    return cfg.get('particle', {}).get('$top', 'B')


# ====================================================================
# 3. Resolve particle → ResonantState
# ====================================================================

def resolve_resonance(cfg, name):
    """
    Given a particle name, return list of ResonantState options.
    Handles multiplets like pipi1: [rhoA] or pipipi: [a2, pi1].
    """
    props = get_particle(cfg, name)
    if props is None:
        return [ResonantState(name)]

    if isinstance(props, list):
        # Multiplet: [rhoA] or [f0(980), f0(500)]
        states = []
        for p in props:
            if isinstance(p, str):
                states.extend(resolve_resonance(cfg, p))
            elif isinstance(p, dict):
                for k, v in p.items():
                    st = resolve_resonance(cfg, k)
                    for s in st:
                        for kk, vv in v.items():
                            setattr(s, kk, vv)
                    states.extend(st)
        return states

    if isinstance(props, dict):
        J = props.get('J', 0)
        P = props.get('P', 1)
        mass = props.get('mass', 0.0)
        width = props.get('width', 0.0)
        model = props.get('model', 'BW')
        extra = {k: v for k, v in props.items()
                 if k not in ('J', 'P', 'mass', 'width', 'model')}
        return [ResonantState(name, J, P, mass, width, model, **extra)]

    return [ResonantState(name)]


# ====================================================================
# 4. Count LS combinations per decay
# ====================================================================

def count_ls(J_parent, J_d1, J_d2, P_parent=1, P_d1=1, P_d2=1, p_break=False):
    """
    Count LS combinations for decay parent(J,P) → d1(J1,P1) + d2(J2,P2).
    
    LS coupling: tensor product of spins J_parent = L + S, where S = J1 + J2.
    
    If p_break=True: ALL mathematically allowed (L,S) are returned.
    If p_break=False: parity-conserving combos only:
        P_parent = P_d1 * P_d2 * (-1)^L
    
    Returns list of (L, S).
    """
    allowed = []
    S_min = abs(J_d1 - J_d2)
    S_max = J_d1 + J_d2
    for S in range(int(S_min), int(S_max) + 1):
        L_min = abs(J_parent - S)
        L_max = J_parent + S
        for L in range(int(L_min), int(L_max) + 1):
            if p_break:
                allowed.append((L, S))
            else:
                P_prod = P_d1 * P_d2
                if P_parent == P_prod * ((-1) ** L):
                    allowed.append((L, S))
    if not allowed:
        allowed = [(0, 0)]
    return allowed


def count_ls_decay(cfg, parent_name, daughters):
    """Count LS combos for a decay given the parent and daughter names."""
    # Get parent J,P
    props = get_particle(cfg, parent_name)
    if props is None or not isinstance(props, dict):
        return [(0, 0)]

    J_parent = props.get('J', 0)
    P_parent = props.get('P', 1)

    # Get daughter quantum numbers (use first resonance if multiplet)
    d1_props = resolve_resonance(cfg, daughters[0])[0] if daughters else ResonantState('')
    d2_props = resolve_resonance(cfg, daughters[1])[0] if len(daughters) > 1 else ResonantState('')

    return count_ls(J_parent, d1_props.J, d2_props.J)


# ====================================================================
# 5. Expand config → PhysicalWave list
# ====================================================================

def expand_physical_waves(cfg):
    """
    Parse config and return list of PhysicalWave objects.
    One per unique resonance assignment through a decay tree.
    """
    finals = set(get_finals(cfg))
    top = get_top(cfg)
    decay_section = cfg.get('decay', {})
    top_decays = decay_section.get(top, [])

    # ---------- helper: recursively build decay chains ----------
    def resolve_intermediate(name):
        """
        For a particle name (like 'pipi1', 'pipipi'), return list of
        (resonance_name, [daughter_names]) options.
        """
        daughters, _ = get_decay(cfg, name)
        props = get_particle(cfg, name)

        if isinstance(props, list):
            # multiplet: each entry is a possible resonance
            options = []
            for p in props:
                if isinstance(p, str):
                    # Check if this resonance has its own decay definition
                    rd, _ = get_decay(cfg, p)
                    if rd:
                        options.append((p, rd))
                    else:
                        options.append((p, daughters))
            if not options:
                options = [(name, daughters)]
            return options

        # single resonance or no particle section
        return [(name, daughters)]

    def expand_to_chains(name):
        """
        Recursively expand a particle into all possible chains.
        Each chain is [(resonance, daughter_names, is_final), ...].
        is_final=True means this is a final state (no further decay).
        """
        options = resolve_intermediate(name)
        result = []
        for res_name, res_daughters in options:
            sub_opts = []
            all_final = True
            for d in res_daughters:
                if d in finals:
                    sub_opts.append([(d, [], True)])
                else:
                    subs = expand_to_chains(d)
                    all_final = False
                    if subs:
                        sub_opts.append(subs)
                    else:
                        sub_opts.append([(d, [], True)])

            if all_final or not res_daughters:
                result.append([(res_name, res_daughters, False)])
            else:
                for combo in itertools.product(*sub_opts):
                    # Build daughter list: extract resonance name from each sub-chain
                    daughter_names = []
                    for sub_chain in combo:
                        if isinstance(sub_chain, tuple):
                            daughter_names.append(sub_chain[0])
                        elif sub_chain:
                            daughter_names.append(sub_chain[0][0])
                    chain = [(res_name, daughter_names, False)]
                    for sub_chain in combo:
                        if isinstance(sub_chain, tuple):
                            chain.append(sub_chain)
                        else:
                            chain.extend(sub_chain)
                    result.append(chain)
        return result

    # ---------- main: process each B decay line ----------
    waves = []
    pw_id = 0

    for decay_line in top_decays:
        b_daughters = [x for x in decay_line if isinstance(x, str)]
        b_opts = {}
        for x in decay_line:
            if isinstance(x, dict):
                b_opts.update(x)
        b_model = b_opts.get('model', 'gls_cpv_aabar')
        b_p_break = b_opts.get('p_break', False)

        # Expand each B daughter
        per_daughter = []
        for d in b_daughters:
            if d in finals:
                per_daughter.append([(d, [], True)])
            else:
                chains = expand_to_chains(d)
                if chains:
                    per_daughter.append(chains)
                else:
                    per_daughter.append([(d, [], True)])

        # Cross-product across B daughters
        for combo in itertools.product(*per_daughter):
            pw = PhysicalWave(pw_id)

            # Skip non-resonant waves: intermediates with empty particle section
            # e.g. pipi1S: [], pipi1f: [] mean no resonance defined
            has_res = False
            for d_chain in combo:
                if isinstance(d_chain, list):
                    for entry in d_chain:
                        rname, rd, is_final = entry[:3] if len(entry) >= 3 else ("", [], False)
                        if not is_final:
                            props = get_particle(cfg, rname)
                            # Accept if props is a dict with J, or non-empty list (multiplet)
                            if (isinstance(props, dict) and 'J' in props) or \
                               (isinstance(props, list) and len(props) > 0):
                                has_res = True
            if not has_res:
                continue  # skip non-resonant chains

            pw_id += 1
            pw.has_cpv = (b_model == 'gls_cpv_aabar')

            # Build the chain
            full_chain = []
            all_resonances = []

            for d_chain in combo:
                if isinstance(d_chain, tuple) and len(d_chain) == 3 and d_chain[2] is True:
                    # Simple final state: (name, [], True)
                    full_chain.append(DecayStep(d_chain[0], [], model=None))
                else:
                    # Chain of (res_name, daughter_names, is_final) entries
                    for entry in d_chain:
                        rname, rd, is_final = entry
                        if not is_final:
                            res = resolve_resonance(cfg, rname)
                            if res:
                                all_resonances.append(res[0])
                            sub_daughters = [d for d in rd if isinstance(d, str)]
                            full_chain.append(DecayStep(rname, sub_daughters,
                                                        model=b_model if rname == b_daughters[0] else None))

            # Handle the B decay step itself
            b_step = DecayStep(top, b_daughters, model=b_model, p_break=b_p_break)
            pw.chain = [b_step] + full_chain
            pw.resonances = all_resonances

            # Build name
            res_names = [r.name for r in pw.resonances]
            pw.name = "B->" + ".".join(res_names)

            # Classify topology
            pw.topology = classify_topology(pw.chain, finals)
            pw.topology_key = build_topology_key(pw)

            # Count LS per decay
            pw.n_ls_per_decay = count_ls_for_chain(cfg, pw, finals)

            waves.append(pw)

    return waves


# ====================================================================
# 6. Topology classification
# ====================================================================

def classify_topology(chain, finals):
    """
    Determine topology string from the chain.
    """
    if len(chain) < 3:
        return "unknown"

    # Check if chain[1] and chain[2] both decay directly to 2 finals each
    # (pim1, pip1)+(pim2, pip2) topology
    d1 = set(chain[1].daughters) if len(chain) > 1 else set()
    d2 = set(chain[2].daughters) if len(chain) > 2 else set()
    fin_set = set(finals)

    if d1 and d2 and d1.issubset(fin_set) and d2.issubset(fin_set):
        # Both sub-decays go directly to finals
        if len(d1) == 2 and len(d2) == 2:
            return "(pim1, pip1)+(pim2, pip2)"

    # Cascade topology: chain[1] has a non-final daughter
    if chain[1].daughters:
        nd1 = [d for d in chain[1].daughters if d not in fin_set]
        if nd1:
            return "(pim1, pip1, pip2)"

    # Fallback
    return "unknown"


def build_topology_key(pw):
    """
    Build (topology, J1, P1, J2, P2) key for angular formula lookup.
    chain[1].core = first sub-decay parent
    chain[2].core = second sub-decay parent
    """
    topo = pw.topology
    if topo == "unknown":
        return (topo, 0, 1, 0, 1)

    # Get J,P from resonances
    if topo == "(pim1, pip1)+(pim2, pip2)":
        # chain[1] → first resonance, chain[2] → second resonance
        if len(pw.resonances) >= 2:
            J1, P1 = pw.resonances[0].J, pw.resonances[0].P
            J2, P2 = pw.resonances[1].J, pw.resonances[1].P
        else:
            J1, P1 = 0, 1
            J2, P2 = 0, 1
    else:
        # Cascade: chain[1] → cascade resonance, chain[2] → sub-resonance
        if len(pw.resonances) >= 2:
            J1, P1 = pw.resonances[0].J, pw.resonances[0].P
            J2, P2 = pw.resonances[1].J, pw.resonances[1].P
        elif len(pw.resonances) == 1:
            J1, P1 = pw.resonances[0].J, pw.resonances[0].P
            J2, P2 = 0, 1
        else:
            J1, P1 = 0, 1
            J2, P2 = 0, 1

    return (topo, J1, P1, J2, P2)


def get_particle_jp(cfg, name):
    """Get (J, P) for a particle name from config."""
    props = get_particle(cfg, name)
    if isinstance(props, dict):
        return props.get('J', 0), props.get('P', 1)
    return 0, 1


def count_ls_for_chain(cfg, pw, finals):
    """
    Count LS combinations for each decay in the chain.
    Uses p_break flag: if True, ALL LS combos are allowed (no parity filter).
    """
    n_ls = []
    for i, step in enumerate(pw.chain):
        if not step.daughters:
            n_ls.append((1, [(0, 0)]))
            continue

        # Get parent J,P
        if step.parent == cfg.get('particle', {}).get('$top', 'B'):
            Jp, Pp = get_particle_jp(cfg, step.parent)
        else:
            parent_res = [r for r in pw.resonances if r.name == step.parent]
            if parent_res:
                Jp, Pp = parent_res[0].J, parent_res[0].P
            else:
                Jp, Pp = 0, 1

        # Get daughter J,P values
        d_info = []
        for d in step.daughters:
            if d in finals:
                d_info.append((0, -1))  # pion: J=0, P=-1
            else:
                d_res = [r for r in pw.resonances if r.name == d]
                if d_res:
                    d_info.append((d_res[0].J, d_res[0].P))
                else:
                    d_props = resolve_resonance(cfg, d)
                    if d_props:
                        d_info.append((d_props[0].J, d_props[0].P))
                    else:
                        d_info.append((0, 1))

        if len(d_info) >= 2:
            J1, P1 = d_info[0]
            J2, P2 = d_info[1]
            ls_list = count_ls(Jp, J1, J2, Pp, P1, P2, p_break=step.p_break)
        else:
            ls_list = [(0, 0)]

        n_ls.append((len(ls_list), ls_list))
    return n_ls


# ====================================================================
# 7. Angular basis functions (pure trig, no g_ls)
# ====================================================================

class AngularBasis:
    """
    Container for pure trigonometric angular basis functions.

    Each basis function: ∏_j cos(k_j · θ_j + b_j)
    where θ = (θ₁, θ₂, φ) — 3 angles.

    Terms are grouped by LS component:
      ls_groups = [(ls_idx, [(ks, bs, coeff), ...]), ...]
    """
    N_ANGLES = 3  # θ₁, θ₂, φ

    def __init__(self):
        self.terms = []          # flat list: [(ks, bs, coeff), ...]
        self.ls_groups = []      # list of (ls_idx, n_terms_in_group)

    def start_ls(self, ls_idx):
        """Start a new LS component group."""
        self.ls_groups.append((ls_idx, 0))

    def add_term(self, ks, bs, coeff):
        assert len(ks) == self.N_ANGLES and len(bs) == self.N_ANGLES
        self.terms.append((tuple(ks), tuple(bs), complex(coeff)))
        if self.ls_groups:
            # increment count for the current (last) group
            g = self.ls_groups[-1]
            self.ls_groups[-1] = (g[0], g[1] + 1)

    def add_cos(self, angle_idx, coeff=1.0):
        ks = [0]*self.N_ANGLES; bs = [0.0]*self.N_ANGLES
        ks[angle_idx] = 1
        self.add_term(ks, bs, coeff)

    def add_sin(self, angle_idx, coeff=1.0):
        rs = [0]*self.N_ANGLES; bs = [0.0]*self.N_ANGLES
        rs[angle_idx] = 1
        bs[angle_idx] = -math.pi/2
        self.add_term(rs, bs, coeff)

    def add_product(self, specs, coeff=1.0+0j):
        """specs = [(angle_idx, 'cos'|'sin', k), ...]"""
        ks = [0]*self.N_ANGLES; bs = [0.0]*self.N_ANGLES
        for idx, trig, k in specs:
            ks[idx] = k
            if trig == 'sin':
                bs[idx] = -math.pi/2
        self.add_term(ks, bs, coeff)

    def n_basis(self):
        return len(self.terms)

    def get_terms_for_ls(self, ls_idx):
        """Get list of (term_idx, coeff) for a given LS component."""
        result = []
        term_start = 0
        for g_ls, g_n in self.ls_groups:
            if g_ls == ls_idx:
                for i in range(g_n):
                    result.append((term_start + i, complex(self.terms[term_start + i][2])))
            term_start += g_n
        return result

    def to_arrays(self):
        """Return (ang_k, ang_b, coeffs)."""
        n = self.n_basis()
        if n == 0:
            return None, None, np.array([], dtype=np.complex128)
        ak = np.zeros((n, self.N_ANGLES), dtype=np.float64)
        ab = np.zeros((n, self.N_ANGLES), dtype=np.float64)
        co = np.zeros(n, dtype=np.complex128)
        for i, (ks, bs, c) in enumerate(self.terms):
            ak[i] = ks; ab[i] = bs; co[i] = c
        return ak, ab, co


# ------------------------------------------------------------------
# All 14 registered angular formulas (pure trig, no g_ls/bf)
# ------------------------------------------------------------------

def build_angular_basis(key):
    """
    Build AngularBasis for (topology, J1, P1, J2, P2).
    Returns AngularBasis with pure-trigonometric basis functions.
    Each term is one f_j(θ₁,θ₂,φ) that gets paired with a g_ls[j].
    """
    topo, J1, P1, J2, P2 = key
    b = AngularBasis()

    # --- Topology A: (pim1,pip1)+(pim2,pip2) ---
    if topo == "(pim1, pip1)+(pim2, pip2)":
        if J1 == 1 and P1 == -1 and J2 == 1 and P2 == -1:
            # VV case: 3 LS components (L=0,1,2; S=1)
            b.start_ls(0)  # LS₀: f₀ = √⅓(B0 − B1)
            b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 1)], math.sqrt(1/3))
            b.add_product([(0, 'cos', 1), (1, 'cos', 1), (2, 'cos', 0)], -math.sqrt(1/3))
            b.start_ls(1)  # LS₁: f₁ = −i/√2·B2
            b.add_product([(2, 'sin', 1), (0, 'sin', 1), (1, 'sin', 1)], -1j / math.sqrt(2))
            b.start_ls(2)  # LS₂: f₂ = √⅙(B0 + 2·B1)
            b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 1)], math.sqrt(1/6))
            b.add_product([(0, 'cos', 1), (1, 'cos', 1), (2, 'cos', 0)], 2*math.sqrt(1/6))
            return b

        if J1 == 0 and P1 == 1 and J2 == 1 and P2 == -1:
            b.start_ls(0)  # −cosθ₂
            b.add_cos(1, -1.0)
            return b

        if J1 == 1 and P1 == -1 and J2 == 0 and P2 == 1:
            b.start_ls(0)  # −cosθ₁
            b.add_cos(0, -1.0)
            return b

    # --- Topology B: (pim1,pip1,pip2) ---
    if topo == "(pim1, pip1, pip2)":
        if J1 == 0 and P1 == -1:
            if J2 == 0 and P2 == 1:
                b.start_ls(0)
                b.add_product([(0, 'cos', 0), (1, 'cos', 0), (2, 'cos', 0)], 1.0)
                return b
            elif J2 == 1 and P2 == -1:
                b.start_ls(0)
                b.add_cos(1, -1.0)
                return b
            elif J2 == 2 and P2 == 1:
                b.start_ls(0)
                b.add_product([(0, 'cos', 0), (1, 'cos', 0), (2, 'cos', 0)], 0.25)
                b.add_product([(1, 'cos', 2), (0, 'cos', 0), (2, 'cos', 0)], 0.75)
                return b

        if J1 == 1 and P1 == -1:
            if J2 == 1 and P2 == -1:
                b.start_ls(0)
                b.add_product([(2, 'sin', 1), (0, 'sin', 1), (1, 'sin', 1)], -1j / math.sqrt(2))
                return b
            elif J2 == 2 and P2 == 1:
                b.start_ls(0)
                b.add_product([(2, 'sin', 1), (0, 'sin', 1), (1, 'sin', 2)], -1j * math.sqrt(6) / 4)
                return b

        if J1 == 1 and P1 == 1:
            if J2 == 0 and P2 == 1:
                b.start_ls(0)
                b.add_cos(0, -1.0)
                return b
            elif J2 == 1 and P2 == -1:
                # 2 LS: a1→rhoA+π
                b.start_ls(0)  # f₀ = √⅓(B0 − B1)
                b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 1)], math.sqrt(1/3))
                b.add_product([(0, 'cos', 1), (1, 'cos', 1), (2, 'cos', 0)], -math.sqrt(1/3))
                b.start_ls(1)  # f₁ = √⅙(B0 + 2·B1)
                b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 1)], math.sqrt(1/6))
                b.add_product([(0, 'cos', 1), (1, 'cos', 1), (2, 'cos', 0)], 2*math.sqrt(1/6))
                return b
            elif J2 == 2 and P2 == 1:
                # 2 LS: a1→f2+π (complex trig polynomials)
                b.start_ls(0)
                b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 3)], -24 * 0.25 * math.sqrt(10)/80)
                b.add_product([(0, 'cos', 1), (1, 'cos', 0), (2, 'cos', 0)], -24 * (-0.5) * math.sqrt(10)/80)
                b.add_product([(0, 'cos', 1), (1, 'cos', 2), (2, 'cos', 0)], -24 * 0.5 * math.sqrt(10)/80)
                b.add_product([(0, 'cos', 1), (1, 'cos', 0), (2, 'cos', 0)], 16 * math.sqrt(10)/80)
                b.start_ls(1)
                b.add_product([(2, 'cos', 1), (0, 'sin', 1), (1, 'sin', 3)], -8 * 0.25 * math.sqrt(15)/40)
                b.add_product([(0, 'cos', 1), (1, 'cos', 0), (2, 'cos', 0)], 12 * 0.5 * math.sqrt(15)/40)
                b.add_product([(0, 'cos', 1), (1, 'cos', 2), (2, 'cos', 0)], -12 * 0.5 * math.sqrt(15)/40)
                b.add_product([(0, 'cos', 1), (1, 'cos', 0), (2, 'cos', 0)], -8 * math.sqrt(15)/40)
                return b

        if J1 == 2 and P1 == -1:
            if J2 == 0 and P2 == 1:
                b.start_ls(0)
                b.add_product([(0, 'cos', 0), (1, 'cos', 0), (2, 'cos', 0)], 0.25)
                b.add_product([(0, 'cos', 2), (1, 'cos', 0), (2, 'cos', 0)], 0.75)
                return b
            elif J2 == 1 and P2 == -1:
                # 2 LS: pi2→rhoA+π
                b.start_ls(0)
                b.add_product([(1, 'cos', 1), (0, 'cos', 0), (2, 'cos', 0)], -12 * math.sqrt(10)/80)
                b.add_product([(1, 'cos', 1), (0, 'cos', 2), (2, 'cos', 0)], 12 * math.sqrt(10)/80)
                b.add_product([(2, 'cos', 1), (0, 'sin', 2), (1, 'sin', 1)], -12 * math.sqrt(10)/80)
                b.add_product([(1, 'cos', 1), (0, 'cos', 0), (2, 'cos', 0)], 16 * math.sqrt(10)/80)
                b.start_ls(1)
                b.add_product([(1, 'cos', 1), (0, 'cos', 0), (2, 'cos', 0)], 12 * 0.5 * math.sqrt(15)/40)
                b.add_product([(1, 'cos', 1), (0, 'cos', 2), (2, 'cos', 0)], -12 * 0.5 * math.sqrt(15)/40)
                b.add_product([(2, 'cos', 1), (0, 'sin', 2), (1, 'sin', 1)], -8 * 0.5 * math.sqrt(15)/40)
                b.add_product([(1, 'cos', 1), (0, 'cos', 0), (2, 'cos', 0)], -8 * math.sqrt(15)/40)
                return b
            elif J2 == 2 and P2 == 1:
                # Full expansion of pi2→f2+π formula (3 LS)
                # Basis: B0=1, B1=cos(2φ), B2=cos(2θ₁), B3=cos(2θ₂),
                #        B4=cos(2φ)cos(2θ₁), B5=cos(2φ)cos(2θ₂),
                #        B6=cos(2θ₁)cos(2θ₂), B7=cos(2φ)cos(2θ₁)cos(2θ₂),
                #        B8=cos(φ)sin(2θ₁)sin(2θ₂)
                # Expand a1-a6 in terms of B0-B8, then apply coeffs
                # a1 = ⅛(B0-B1-B2-B3+B4+B5+B6-B7)
                # a2 = ¼(B0-B2-B3+B6)
                # a3 = ½(B0-B2)
                # a4 = ¼·B8
                # a5 = ½(B0-B3)
                # a6 = B0
                # f(j) = Σᵢ aᵢ · coeffs[j,i] for j=0,1,2
                c0 = math.sqrt(5)/320    # coeff for LS₀
                c1 = math.sqrt(14)/448   # coeff for LS₁
                c2 = math.sqrt(70)*3/4480  # coeff for LS₂
                coeff_mat = [
                    [-96, 192, -96, -192, -96, 64],
                    [-96, -96, 96, 96, 96, -64],
                    [-32, 304, -192, 256, -192, 128],
                ]
                # a1-coeffs for each basis term:
                a1_coeffs = {0: 1/8, 1: -1/8, 2: -1/8, 3: -1/8, 4: 1/8, 5: 1/8, 6: 1/8, 7: -1/8}
                a2_coeffs = {0: 1/4, 2: -1/4, 3: -1/4, 6: 1/4}
                a3_coeffs = {0: 1/2, 2: -1/2}
                a4_coeffs = {8: 1/4}
                a5_coeffs = {0: 1/2, 3: -1/2}
                a6_coeffs = {0: 1.0}
                all_a = [a1_coeffs, a2_coeffs, a3_coeffs, a4_coeffs, a5_coeffs, a6_coeffs]
                # basis specs for each basis function type
                basis_specs = {
                    0: [(0, 'cos', 0), (1, 'cos', 0), (2, 'cos', 0)],
                    1: [(2, 'cos', 2), (0, 'cos', 0), (1, 'cos', 0)],
                    2: [(0, 'cos', 2), (1, 'cos', 0), (2, 'cos', 0)],
                    3: [(1, 'cos', 2), (0, 'cos', 0), (2, 'cos', 0)],
                    4: [(2, 'cos', 2), (0, 'cos', 2), (1, 'cos', 0)],
                    5: [(2, 'cos', 2), (1, 'cos', 2), (0, 'cos', 0)],
                    6: [(0, 'cos', 2), (1, 'cos', 2), (2, 'cos', 0)],
                    7: [(2, 'cos', 2), (0, 'cos', 2), (1, 'cos', 2)],
                    8: [(2, 'cos', 1), (0, 'sin', 2), (1, 'sin', 2)],
                }
                for ls_idx in range(3):
                    b.start_ls(ls_idx)
                    for bi_idx, a_i in enumerate(all_a):
                        a_coeff = coeff_mat[ls_idx][bi_idx] * [c0, c1, c2][ls_idx]
                        for base_idx, base_coeff in a_i.items():
                            if base_idx in basis_specs:
                                b.add_product(basis_specs[base_idx], a_coeff * base_coeff)
                return b

        if J1 == 2 and P1 == 1:
            if J2 == 1 and P2 == -1:
                b.start_ls(0)
                b.add_product([(2, 'sin', 1), (0, 'sin', 2), (1, 'sin', 1)], 1j * math.sqrt(6) / 4)
                return b

    # Fallback: constant 1
    b.start_ls(0)
    b.add_product([(0, 'cos', 0), (1, 'cos', 0), (2, 'cos', 0)], 1.0)
    return b


# ====================================================================
# 8. PhysicalWave → KernelWave expansion
# ====================================================================

def expand_to_kernel_waves(pw_list, cfg):
    """
    Expand PhysicalWaves into KernelWaves.
    Each physical wave → n_kernel = ∏ n_ls_per_decay × (2 if cpv else 1)
    """
    kernel_waves = []
    kw_id = 0

    for pw in pw_list:
        # Get LS per decay: list of [(L,S), ...] (the actual LS values)
        ls_counts = pw.n_ls_counts()

        # Build all LS index combinations across decays
        ls_ranges = [range(c) for c in ls_counts]
        ls_combos = list(itertools.product(*ls_ranges)) if ls_ranges else [()]

        # For CPV: B and Bbar
        cpv_sectors = [False, True] if pw.has_cpv else [False]

        # Kernel splits at half_waves: first half is B (g_ls), second half is Bbar (g_lsbar)
        for is_bar in cpv_sectors:
            for ls_combo in ls_combos:
                kw = KernelWave(kw_id, pw.id)
                kw_id += 1
                kw.is_bar = is_bar

                # --- ck formula ---
                # ck = total × ∏ g_ls_decay[ls_idx] for all decays with LS params
                for di, (step, ls_idx) in enumerate(zip(pw.chain, ls_combo)):
                    if step.daughters:
                        if step.model == 'gls_cpv_aabar':
                            if is_bar:
                                kw.ck_formula.append(('g_lsbar', step.parent, ls_idx))
                            else:
                                kw.ck_formula.append(('g_ls', step.parent, ls_idx))
                        else:
                            # Standard HelicityDecay (single g_ls, no B/Bbar split)
                            kw.ck_formula.append(('g_ls', step.parent, ls_idx))

                # Add total coupling (always first)
                kw.ck_formula.insert(0, ('total', pw.name, 0))

                # --- bw_order: index for each resonance in chain order ---
                # All resonances get a bwall entry (even non-BW, approximate for now)
                kw.bw_order_entries = list(range(len(pw.resonances)))

                # --- bf_order: barrier factor types ---
                # One per actual decay; bf type = L value from the LS combo × d=3.0
                bf_entries = []
                d_idx = 0
                for di, step in enumerate(pw.chain):
                    if not step.daughters:
                        continue
                    # Get L from this decay's LS combo
                    ls_list = pw.get_ls_list(di)
                    ls_idx = ls_combo[di] if di < len(ls_combo) else 0
                    if ls_idx < len(ls_list):
                        L, S = ls_list[ls_idx]
                    else:
                        L = 0
                    # Barrier key: (L, d=3.0)
                    bf_entries.append(L)  # store L value; key is (L, 3.0)
                    d_idx += 1
                kw.bf_order_entries = bf_entries

                # --- ag matrix entry: which basis function for this LS combo ---
                # Get the angular basis
                basis = build_angular_basis(pw.topology_key)

                # For this LS combo, we need the LS index that corresponds to
                # the decay that has multiple LS combinations
                # The angular function f_j is the j-th term in the basis
                # which corresponds to the LS index of the relevant decay

                # The formula in fixed_ls_chaint combines ls_amp from multiple decays.
                # For VV case: sum over B decay LS components
                # For cascade: sum over chain[1] LS components
                #
                # The LS combo determines WHICH f_j is active.
                # The angular basis terms come in groups (one per LS component).
                # We find the basis index range for the relevant decay.
                n_basis = basis.n_basis()

                # For VV: n_ls[0] = 3 (B decay), each has its own f_j
                # For cascade: n_ls[1] = n (chain[1] decay), each has its own f_j
                #
                # The f_j terms are the first n basis terms, then the next n, etc.
                # Actually, looking at build_angular_basis: terms are added per LS component.
                # So: f₀ = terms[0..m₀-1], f₁ = terms[m₀..m₀+m₁-1], ...

                # Find which decay's LS drives the angular formula
                # The formula has structure: formula(ls_amp[0], ls_amp[1]) * ls_amp[2][0]
                # where ls_amp[0] = B decay (chain[0]), ls_amp[1] = chain[1], ls_amp[2] = chain[2]
                # The multi-LS sum comes from the decay with >1 LS that is
                # summed over in the formula.
                #
                # For VV: chain[0] has 3 LS, formula sums over them
                # For cascade: chain[1] has multi-LS, formula sums over them
                # We find the first decay (excluding terminal finals) with >1 LS
                ls_decay_idx = 0
                for di, n in enumerate(ls_counts):
                    if n > 1 and di < 3 and di < len(pw.chain) and pw.chain[di].daughters:
                        ls_decay_idx = di
                        break

                # The selected LS index for this specific kernel wave
                selected_ls = ls_combo[ls_decay_idx] if ls_decay_idx < len(ls_combo) else 0

                # Get only the basis terms for this LS component
                ls_terms = basis.get_terms_for_ls(selected_ls)
                for term_idx, coeff in ls_terms:
                    kw.ag_matrix_entry.append((term_idx, coeff))

                kernel_waves.append(kw)

    return kernel_waves


# ====================================================================
# 9. Build kernel config dict
# ====================================================================

def get_resonance_descendants(pw, res_name, finals):
    """Get final-state particle descendants of a resonance in a decay chain."""
    descendants = []
    for step in pw.chain:
        if step.parent == res_name and step.daughters:
            for d in step.daughters:
                if d in finals:
                    descendants.append(d)
                else:
                    descendants.extend(get_resonance_descendants(pw, d, finals))
            break
    return sorted(descendants)


def get_mass_key(res_name, pw, finals, ident_groups=None):
    """
    Get a unique mass key for a resonance based on its descendant finals.
    Normalizes identical particles so that (pip1,pim1) and (pip2,pim2)
    produce the same key when pip1≡pip2 and pim1≡pim2.
    """
    desc = get_resonance_descendants(pw, res_name, finals)
    if ident_groups is None:
        return tuple(desc)
    # Normalize: map each particle to its group leader
    norm_map = {}
    for group in ident_groups:
        leader = group[0]
        for p in group:
            norm_map[p] = leader
    # Normalize identical particles but KEEP duplicates (different count = different mass)
    # e.g. (pip1, pim1) ≠ (pip1, pim1, pip1) in terms of invariant mass
    norm_desc = tuple(sorted(norm_map.get(p, p) for p in desc))
    return norm_desc


def build_kernel_config(pw_list, kw_list, cfg):
    """
    Build the config dict for PWAGPU from PhysicalWaves + KernelWaves.
    """
    if not kw_list:
        raise ValueError("No kernel waves!")

    n_waves = len(kw_list)
    finals = set(get_finals(cfg))

    # --- Deduplicate by (mass_key, m0, width) ---
    # RhoA and rhoB share the same mass_key (π⁺π⁻ after identical particle normalization),
    # same m0=0.769, same width=0.1506 → they share ONE bwall entry.
    bwall_key_to_idx = OrderedDict()
    res_name_to_bwall = {}

    for pw in pw_list:
        for res in pw.resonances:
            if res.name in res_name_to_bwall:
                continue
            mk = get_mass_key(res.name, pw, finals, ident_groups=None)
            # Deduplication key: (mass_key, m0, width, model)
            # Gamma table differs per model (BW, Bugg, one, FlatteC all have different
            # mass-dependent width behavior) — so model must be part of the key.
            key = (mk, res.mass, res.width, res.model)
            if key not in bwall_key_to_idx:
                bwall_key_to_idx[key] = len(bwall_key_to_idx)
            res_name_to_bwall[res.name] = bwall_key_to_idx[key]

    n_m0_base = len(bwall_key_to_idx)
    BW_MODELS = {'BW', 'GS_rho', 'width_linear_npy', 'GS_rho_omega'}

    # --- Identical particle permutations ---
    import itertools
    ident_particles = cfg.get('data', {}).get('identical_particles', [])
    perm_groups = [list(itertools.permutations(g)) for g in ident_particles]
    n_perm = max(1, len(list(itertools.product(*perm_groups)))) if perm_groups else 1
    if perm_groups:
        n_perm = 1
        for g in perm_groups:
            n_perm *= len(g)
    # Each permutation creates independent copies of ALL entries:
    n_m0 = n_m0_base * n_perm
    bw_index = np.arange(n_m0, dtype=np.int32)  # identity, permuted blocks
    n_mass_columns = n_m0

    # --- gamma_index & n_g0: count all g0 parameters per perm ---
    g0_base = []
    for res_name, bwall_idx in res_name_to_bwall.items():
        res = None
        for pw in pw_list:
            for r in pw.resonances:
                if r.name == res_name:
                    res = r; break
            if res: break
        if res and res.model == 'FlatteC':
            props = get_particle(cfg, res_name)
            n_g = sum(1 for k in props if k.startswith('g_')) if isinstance(props, dict) else 4
            for _ in range(n_g):
                g0_base.append(bwall_idx)
        else:
            g0_base.append(bwall_idx)
    n_g0_base = len(g0_base)
    n_g0 = n_g0_base * n_perm
    gamma_index = np.arange(n_g0, dtype=np.int32)  # identity, permuted blocks

    # --- q_stride (base, will be per-permutation) ---
    q_entries = OrderedDict()
    for pw in pw_list:
        for step in pw.chain:
            if step.daughters:
                qkey = (step.parent, tuple(sorted(step.daughters)))
                if qkey not in q_entries:
                    q_entries[qkey] = len(q_entries)
    q_stride_base = len(q_entries)
    q_stride = q_stride_base * n_perm

    # --- phys indices (base values, same across perms) ---
    m0_value_groups = OrderedDict()
    m0_phys_base = np.zeros(n_m0_base, dtype=np.int32)
    for key, idx in bwall_key_to_idx.items():
        mk, m0_val, w_val, model = key
        pk = (m0_val, model)
        if pk not in m0_value_groups: m0_value_groups[pk] = len(m0_value_groups)
        m0_phys_base[idx] = m0_value_groups[pk]
    m0_phys_index = np.tile(m0_phys_base, n_perm)
    n_m0_phys = len(m0_value_groups)

    g0_value_groups = OrderedDict()
    g0_phys_base = np.zeros(n_g0_base, dtype=np.int32)
    # Compute per-base-g0 phys index
    gi = 0
    for res_name, bwall_idx in res_name_to_bwall.items():
        res = None
        for pw in pw_list:
            for r in pw.resonances:
                if r.name == res_name: res = r; break
            if res: break
        w_val = res.width if res else 0
        model = res.model if res else 'BW'
        if model == 'FlatteC':
            props = get_particle(cfg, res_name) if res else {}
            n_g = sum(1 for k in props if k.startswith('g_')) if isinstance(props, dict) else 4
            for _ in range(n_g):
                pk = (w_val, model)
                if pk not in g0_value_groups: g0_value_groups[pk] = len(g0_value_groups)
                g0_phys_base[gi] = g0_value_groups[pk]
                gi += 1
        else:
            pk = (w_val, model)
            if pk not in g0_value_groups: g0_value_groups[pk] = len(g0_value_groups)
            g0_phys_base[gi] = g0_value_groups[pk]
            gi += 1
    g0_phys_index = np.tile(g0_phys_base, n_perm)
    n_g0_phys = len(g0_value_groups)

    # Non-standard lineshapes count
    n_special = 0
    for pw in pw_list:
        for res in pw.resonances:
            if res.model not in BW_MODELS:
                n_special += 1
                break

    # --- bw_order: which bwall entries per wave (base, single perm) ---
    n_waves_base = len(kw_list)
    max_res = max(len(kw.bw_order_entries) for kw in kw_list) if kw_list else 1
    bw_order_base = np.zeros(n_waves_base * max_res, dtype=np.int32)

    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if pw:
            for j, res in enumerate(pw.resonances):
                if j < max_res:
                    bwall_idx = res_name_to_bwall.get(res.name, 0)
                    bw_order_base[kw.id * max_res + j] = bwall_idx

    # --- bf_index: unique barrier factor types ---
    q_column_map = OrderedDict()
    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if pw:
            for step in pw.chain:
                if step.daughters:
                    qkey = (step.parent, tuple(sorted(step.daughters)))
                    if qkey not in q_column_map:
                        q_column_map[qkey] = len(q_column_map)

    # map (L, d, decay_qkey) → bf_type index
    unique_bf = OrderedDict()
    bf_q_column = []  # for each bf_type, which q_flat column to use
    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if pw:
            d_idx = 0
            for step in pw.chain:
                if step.daughters:
                    qkey = (step.parent, tuple(sorted(step.daughters)))
                    L = kw.bf_order_entries[d_idx] if d_idx < len(kw.bf_order_entries) else 0
                    bf_key = (int(L), 3.0, qkey)
                    if bf_key not in unique_bf:
                        unique_bf[bf_key] = len(unique_bf)
                        bf_q_column.append(q_column_map[qkey])
                    d_idx += 1

    n_bf_types = len(unique_bf)
    n_bf_types_base = n_bf_types
    bf_index = np.array(bf_q_column, dtype=np.int32)

    # --- bf_order: per wave × decay position → bf_type index ---
    max_decays = max(len(kw.bf_order_entries) for kw in kw_list) if kw_list else 1
    bf_order = np.zeros(n_waves * max_decays, dtype=np.int32)

    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if pw:
            d_idx = 0
            for step in pw.chain:
                if step.daughters:
                    if d_idx < max_decays:
                        qkey = (step.parent, tuple(sorted(step.daughters)))
                        L = kw.bf_order_entries[d_idx] if d_idx < len(kw.bf_order_entries) else 0
                        bf_key = (int(L), 3.0, qkey)
                        bf_order[kw.id * max_decays + d_idx] = unique_bf.get(bf_key, 0)
                        d_idx += 1

    # --- Angular basis ---
    # Build all angular bases and collect unique trig terms into a global registry
    # Each unique (k_θ1, k_θ2, k_φ, b_θ1, b_θ2, b_φ) = one global basis function
    global_terms = OrderedDict()   # (k1,k2,k3,b1,b2,b3) → global_idx
    global_coeffs = []             # coeff list (for matrix_ang column reference)

    # Build bases for all formula keys
    formula_bases = {}  # key → AngularBasis
    for pw in pw_list:
        k = pw.topology_key
        if k not in formula_bases:
            formula_bases[k] = build_angular_basis(k)
            # Register all trig terms from this basis
            basis = formula_bases[k]
            for i, (ks, bs, _) in enumerate(basis.terms):
                key = (ks[0], ks[1], ks[2], bs[0], bs[1], bs[2])
                if key not in global_terms:
                    global_terms[key] = len(global_terms)

    n_basis = len(global_terms)
    n_angles = 3

    # Build shared ang_k, ang_b, ang_index
    ang_k = np.zeros((n_basis, 3), dtype=np.float64)
    ang_b = np.zeros((n_basis, 3), dtype=np.float64)
    # ang_index[i][j]: which angles_flat column for angle j of basis i
    # We use 3 fixed columns: θ₁=0, θ₂=1, φ=2
    ang_index = np.tile(np.array([0, 1, 2], dtype=np.int32), (n_basis, 1))

    for (k1, k2, k3, b1, b2, b3), idx in global_terms.items():
        ang_k[idx] = [k1, k2, k3]
        ang_b[idx] = [b1, b2, b3]

    # Build matrix_ang by mapping per-wave basis terms to global indices
    matrix_ang = np.zeros((n_waves, n_basis), dtype=np.complex128)

    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        if not pw or pw.topology_key not in formula_bases:
            continue
        basis = formula_bases[pw.topology_key]

        # If kw has explicit ag_matrix_entry (per-LS basis terms), use them
        if kw.ag_matrix_entry:
            for local_idx, coeff in kw.ag_matrix_entry:
                # Map local term index to global index
                if local_idx < len(basis.terms):
                    ks, bs, _ = basis.terms[local_idx]
                    gkey = (ks[0], ks[1], ks[2], bs[0], bs[1], bs[2])
                    gidx = global_terms.get(gkey, None)
                    if gidx is not None:
                        matrix_ang[kw.id, gidx] = coeff

    # --- matrix_gamma: width mixing (n_m0 × n_g0) ---
    # Identity per permutation block, repeated n_perm times
    matrix_gamma = np.zeros((n_m0, n_g0), dtype=np.float64)
    for perm in range(n_perm):
        for g_idx, bwall_idx in enumerate(g0_base):
            row = bwall_idx + perm * n_m0_base
            col = g_idx + perm * n_g0_base
            matrix_gamma[row, col] = 1.0

    # --- Placeholder tables ---
    # gamma_table and bf_table need discussion - use minimal defaults
    n_gamma_points = 2
    gamma_table = np.zeros((n_g0, n_gamma_points), dtype=np.complex128)
    n_bf_points = 2
    bf_table = np.ones((n_bf_types, n_bf_points), dtype=np.float64)

    # --- Grid parameters (placeholders) ---
    g_min = 0.0
    g_delta = 0.01
    q_min = 0.0
    q_delta = 0.01

    # --- Permutation expansion: tile each B/Bbar group separately ---
    n_waves_base = len(kw_list)
    n_half = n_waves_base // 2  # first half = B, second half = Bbar

    def split_and_tile(arr, chunk_size, shift_fn):
        """Split array at half_waves, tile each half by n_perm with per-perm shift."""
        if len(arr) == 0: return np.array([], dtype=arr.dtype)
        b_part = arr[:n_half * chunk_size]
        bar_part = arr[n_half * chunk_size:]
        parts = []
        for perm in range(n_perm):
            parts.append(shift_fn(b_part, perm))
        for perm in range(n_perm):
            parts.append(shift_fn(bar_part, perm))
        return np.concatenate(parts) if parts else arr

    # Split kw_list into B and Bbar groups
    kw_b = [kw for kw in kw_list if not kw.is_bar]
    kw_bar = [kw for kw in kw_list if kw.is_bar]

    if n_perm > 1:
        def shift_bw(x, perm): return x + perm * n_m0_base
        def shift_bf(x, perm): return x + perm * n_bf_types_base
        def shift_ang(x, perm): return x + perm * 3
        def shift_q(x, perm): return x + perm * q_stride_base
        def same(x, perm): return x

        bw_order = split_and_tile(bw_order_base, max_res, shift_bw)
        bf_order_arr = split_and_tile(bf_order, max_decays, shift_bf)
        bf_index = split_and_tile(bf_index, 1, shift_q)
        ang_index_arr = split_and_tile(ang_index, n_angles, shift_ang)
        matrix_ang = split_and_tile(matrix_ang, matrix_ang.shape[1], same)

        # wave_info: group by B/Bbar then by perm
        wi_b = [{'name': next((p.name for p in pw_list if p.id == kw.pw_id), "?"),
                 'pw_id': kw.pw_id, 'is_bar': kw.is_bar,
                 'ck_formula': kw.ck_formula, 'ag_nnz': len(kw.ag_matrix_entry)}
                for kw in kw_b]
        wi_bar = [{'name': next((p.name for p in pw_list if p.id == kw.pw_id), "?"),
                   'pw_id': kw.pw_id, 'is_bar': kw.is_bar,
                   'ck_formula': kw.ck_formula, 'ag_nnz': len(kw.ag_matrix_entry)}
                  for kw in kw_bar]
        wave_info = []
        for perm in range(n_perm):
            for wi in wi_b:
                w = dict(wi); w['perm'] = perm; wave_info.append(w)
        for perm in range(n_perm):
            for wi in wi_bar:
                w = dict(wi); w['perm'] = perm; wave_info.append(w)

        ang_stride_total = 3 * n_perm
        n_waves = len(wave_info)
        n_bf_types = len(bf_index)
    else:
        bw_order = bw_order_base
        bf_order_arr = bf_order
        ang_index_arr = ang_index
        wi_b = [{'name': next((p.name for p in pw_list if p.id == kw.pw_id), "?"),
                 'pw_id': kw.pw_id, 'is_bar': kw.is_bar,
                 'ck_formula': kw.ck_formula, 'ag_nnz': len(kw.ag_matrix_entry)}
                for kw in kw_b]
        wi_bar = [{'name': next((p.name for p in pw_list if p.id == kw.pw_id), "?"),
                   'pw_id': kw.pw_id, 'is_bar': kw.is_bar,
                   'ck_formula': kw.ck_formula, 'ag_nnz': len(kw.ag_matrix_entry)}
                  for kw in kw_bar]
        wave_info = wi_b + wi_bar
        ang_stride_total = 3
        n_waves = n_waves_base

    config = {
        'bw_index': bw_index,
        'gamma_index': gamma_index,
        'm0_phys_index': m0_phys_index,
        'g0_phys_index': g0_phys_index,
        'bw_order': bw_order,
        'bf_index': bf_index,
        'bf_order': bf_order,
        'ang_index': ang_index,
        'ang_k': ang_k,
        'ang_b': ang_b,
        'matrix_ang': matrix_ang,
        'matrix_gamma': matrix_gamma,
        'gamma_table': gamma_table,
        'bf_table': bf_table,
        'g_min': g_min,
        'g_delta': g_delta,
        'q_min': q_min,
        'q_delta': q_delta,
        # Metadata
        'n_waves': n_waves,
        'n_m0': n_m0,
        'n_g0': n_g0,
        'n_mass_columns': n_mass_columns,
        'q_stride': q_stride,
        'ang_stride': ang_stride_total,
        'n_special': n_special,
        'n_m0_phys': n_m0_phys,
        'n_g0_phys': n_g0_phys,
        'n_res_per_wave': max_res,
        'n_decays_per_wave': max_decays,
        'n_bf_types': n_bf_types,
        'n_basis': n_basis,
        'n_ang_per_basis': n_angles,
        'n_gamma_points': n_gamma_points,
        'n_bf_points': n_bf_points,
        'n_perm': n_perm,
        'wave_info': wave_info,
    }

    return config


# ====================================================================
# 10. Main entry point
# ====================================================================

def parse_config(config_path):
    """Full pipeline: config → physical waves → kernel config."""
    cfg = load_config(config_path)

    print("[1] Expanding physical waves...")
    pw_list = expand_physical_waves(cfg)
    print(f"    Found {len(pw_list)} physical waves")

    print("[2] Expanding to kernel waves (LS × CPV)...")
    kw_list = expand_to_kernel_waves(pw_list, cfg)
    print(f"    Found {len(kw_list)} kernel waves")

    print("[3] Building kernel config arrays...")
    kernel_config = build_kernel_config(pw_list, kw_list, cfg)

    print(f"    n_waves={kernel_config['n_waves']}, n_m0={kernel_config['n_m0']}, "
          f"n_g0={kernel_config['n_g0']}")
    print(f"    n_basis={kernel_config['n_basis']}, "
          f"n_bf_types={kernel_config['n_bf_types']}")

    return kernel_config, pw_list, kw_list


if __name__ == '__main__':
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python -m pwa_gpu.parse_config <config.yml>")
        sys.exit(1)

    cfg, pw_list, kw_list = parse_config(sys.argv[1])

    print("\nPhysical waves:")
    for pw in pw_list:
        print(f"  PW[{pw.id}] {pw.name}")
        print(f"       topo={pw.topology} key={pw.topology_key}")
        print(f"       resonances={[r.name for r in pw.resonances]}")
        print(f"       chain={[(s.parent, s.daughters) for s in pw.chain]}")
        n_kw = np.prod(pw.n_ls_counts()) * (2 if pw.has_cpv else 1)
        print(f"       n_ls={pw.n_ls_counts()} cpv={pw.has_cpv} → {n_kw} KW")

    print(f"\nKernel waves ({len(kw_list)}):")
    for kw in kw_list:
        pw = next((p for p in pw_list if p.id == kw.pw_id), None)
        name = pw.name if pw else "?"
        print(f"  KW[{kw.id}] PW={kw.pw_id} '{name}' bar={kw.is_bar}")
        print(f"       ck={kw.ck_formula}")
        print(f"       bw={kw.bw_order_entries} bf={kw.bf_order_entries}")
        print(f"       ag_nnz={len(kw.ag_matrix_entry)}")
