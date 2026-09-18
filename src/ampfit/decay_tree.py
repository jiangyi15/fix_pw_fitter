"""Decay-tree interpretation.

Turns the raw ``decay:`` / ``particle:`` declarations into an object: the
decay structure, the resonance-resolved chains, and the stable topology
index.

:class:`DecayTree` is a pure value object built from the declarations only —
it depends on neither the kernel config, nor the parameter surface, nor any
backend.  :class:`~ampfit.config_loader.Config` composes it and keeps the
legacy attribute names (``full_decay``, ``decay_struct``, ``topo_index``, …)
as aliases for backward compatibility.
"""
import itertools
import math

from .particle_model import build_particle

__all__ = [
    "Particle", "Decay", "DecayChain", "DecayGroup",
    "DecayTree", "build_decay_struct", "get_decay_chains",
    "build_decay_chains", "get_topo_index", "replace_chain",
]


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
        for s in s_range(abs(jb - jc), jb + jc + 1):
            for l in range(int(abs(ja - s)), int(ja + s) + 1):
                if not getattr(self, "p_break", False):
                    if (pa is not None and pb is not None and pc is not None):
                        if l % 2 == (1 if pa * pb * pc == 1 else 0):
                            continue
                ls_list.append((l, s))
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
        ret = list(itertools.product(*ls_lists))
        return ret

    def get_gls_combination(self):
        ls_lists = [i.get_ls_names() for i in self.decays]
        total = str(self).replace("+", ".") + "_total_0"
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
        topo_id = tuple(sorted([tuple(sorted(topo_map[i])) for i in self.inner]))
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


def replace_chain(chain, res, res2):
    ret = []
    for decay in chain:
        core = res2 if decay[0] == res else decay[0]
        outs = [res2 if i == res else i for i in decay[1]]
        ret.append((core, outs, *decay[2:]))
    return ret


def build_decay_struct(decay, top, finals):
    # loop to find the chains that top -> a + b, a-> ..., b -> ...
    # filter with finals
    def get_sub_decays(dic, top):
        if top not in dic:
            return [[]]
        res_decay = dic[top]
        if not isinstance(res_decay[0], list):
            res_decay = [res_decay]

        ret = []
        for i in res_decay:
            outs = [j for j in i if isinstance(j, str)]
            kwargs_list = [k for k in i if isinstance(k, dict)]
            kwargs = {}
            for kw in kwargs_list:
                kwargs.update(kw)
            outdecay = [get_sub_decays(dic, outi) for outi in outs]
            for prod in itertools.product(*outdecay):
                table = [(top, outs, kwargs)]
                for i in prod:
                    table += i
                ret.append(table)

        return ret

    return get_sub_decays(decay, top)


def get_decay_chains(decay_struct, res_map):
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
            for ci, ji in zip(c, replace_res):
                tmp = replace_chain(tmp, ji, ci)
            ret.append(tmp)
    return ret


def build_decay_chains(lst, dic):
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
    for i in lst:
        tmp = []
        for j in i:
            tmp.append(Decay(particles[j[0]], [particles[k] for k in j[1]], **j[2]))
        chain = DecayChain(tmp)
        ret.append(chain)
    return DecayGroup(ret)


def get_topo_index(decay):
    topo_id = {}
    for i in decay.chains:
        tmp = i.topo_id()
        if tmp not in topo_id:
            topo_id[tmp] = len(topo_id)
    return topo_id


def symmetry_factors(data_spec):
    """``(n_perm, n_cp, n_blocks, identical_groups, cp_groups)`` from the
    config ``data`` section.  Each identical group of size ``g`` contributes
    ``g!`` permutations; a declared CP map contributes a second (conjugate)
    copy -> ``n_blocks = n_perm * n_cp``.
    """
    data = data_spec or {}
    id_groups = [list(g) for g in (data.get("identical_particles") or [])]
    cp_groups = [list(g) for g in (data.get("cp_particles") or [])]
    n_perm = 1
    for g in id_groups:
        n_perm *= math.factorial(len(g))
    n_cp = 2 if cp_groups else 1
    return n_perm, n_cp, n_perm * n_cp, id_groups, cp_groups


def block_column_orders(finals, id_groups=(), cp_groups=()):
    """Column orders for the identical permutations x CP blocks.

    Returns ``[(order, is_cp), ...]``: ``momenta[:, order]`` gives that
    block's configuration; ``is_cp`` marks a CP block (conjugate columns +
    reversed 3-momentum).  Order is arbitrary (all blocks share one ``ck``).
    """
    idx = {f: j for j, f in enumerate(finals)}
    n = len(finals)
    group_cols = [[idx[g] for g in grp] for grp in (id_groups or [])]
    if group_cols:
        perms = []
        for combo in itertools.product(*[list(itertools.permutations(c))
                                         for c in group_cols]):
            order = list(range(n))
            for cols, perm in zip(group_cols, combo):
                for dst, src in zip(cols, perm):
                    order[dst] = src
            perms.append(tuple(order))
    else:
        perms = [tuple(range(n))]

    cp_col = None
    if cp_groups:
        cp_col = list(range(n))
        for pair in cp_groups:
            ia, ib = idx[pair[0]], idx[pair[1]]
            cp_col[ia], cp_col[ib] = ib, ia

    blocks = [(p, False) for p in perms]
    if cp_col is not None:
        for p in perms:
            blocks.append((tuple(cp_col[p[i]] for i in range(n)), True))
    return blocks


class DecayTree:
    """The decay/particle declarations as a pure value object.

    Built from the raw ``decay`` spec and ``particle`` spec only; it holds the
    structural view (``struct``), the resolved chains (``full``) and the
    stable topology index.
    """

    def __init__(self, decay_spec, particle_spec, data_spec=None):
        self.decay_spec = decay_spec
        self.particle_spec = particle_spec
        self.data = dict(data_spec or {})
        self.top = particle_spec["$top"]
        self.finals = list(particle_spec["$finals"])
        # The particle/column ORDER is defined by ``data.dat_order`` (``$finals``
        # is just the list of finals): order ``finals`` by it when it is a
        # permutation of the declared finals.
        dat_order = list(self.data.get("dat_order") or [])
        if dat_order and set(dat_order) == set(self.finals):
            self.finals = list(dat_order)

        self.struct = build_decay_struct(decay_spec, self.top, self.finals)
        # intermediate structural chain specs are not kept on the object:
        # `struct` is the structural view, `full` the resolved chains.
        struct_chains = get_decay_chains(self.struct, particle_spec)
        self.full = build_decay_chains(struct_chains, particle_spec)

        self.n_decay = len(self.struct[0])
        self.n_res = self.n_decay - 1
        self.n_angles = 2 * self.n_decay
        if particle_spec[self.top]["J"] == 0:
            self.n_angles = self.n_angles - 3  # 3d rotation is not needed
        self.topo_index = self._build_topo_from_struct()
        self.n_topo = len(self.topo_index)

        # decay-symmetry declarations (data section): identical-particle
        # permutations x CP duplicate the event rows.
        (self.n_perm, self.n_cp, self.n_blocks,
         self.identical_groups, self.cp_groups) = symmetry_factors(self.data)

    # -- queries ---------------------------------------------------------
    def partial_waves(self):
        """``(ls, chain)`` pairs over every resonance-resolved chain."""
        return self.full.get_partial_waves()

    def block_orders(self):
        """``[(column order, is_cp), ...]`` for the declared blocks."""
        return block_column_orders(self.finals, self.identical_groups,
                                   self.cp_groups)

    # -- display (particle / decay-chain labels) -------------------------
    def name_display_map(self):
        """Map particle config names to display names.

        Covers every particle appearing in a chain (as a decay core or an
        out).  Values with the same display merge naturally (e.g. ``rhoA``
        and ``rhoB`` both map to the same label).
        """
        seen = {}
        for chain in self.full.chains:
            for decay in chain.decays:
                p = decay.core
                if p.name not in seen:
                    seen[p.name] = p.display
                for out in decay.outs:
                    if out.name not in seen:
                        seen[out.name] = out.display
        return seen

    def display_decay(self, decay):
        """LaTeX display string for a decay: ``parent -> child1 child2``."""
        parent = decay.core.display
        children = [o.display for o in decay.outs]
        return rf"{parent} \to {children[0]}\,{children[1]}"

    def display_chain(self, chain):
        """LaTeX display string for an entire decay chain."""
        parts = [self.display_decay(d) for d in chain.decays]
        return r" \quad ".join(parts)

    def _build_topo_from_struct(self):
        """Stable topology index over ``struct`` structural paths.

        The topology axis is enumerated from the DECAY STRUCTURE (all final
        pairings declared in ``decay:`` — including pairings that currently
        have no resonance candidates), not from the resonance-resolved
        chains.  Every structure path gets a fixed slot, so later adding a
        resonance to another pairing does not renumber the existing
        topologies and does not break the built data/kernel indices.

        A path is keyed by the canonical final-partition of its inner cores
        (identical to ``DecayChain.topo_id()``).  As a safety net, if a
        chain topology is not covered by the structural enumeration the
        previous chain-derived map is returned unchanged.
        """
        struct_keys = []
        for path in self.struct:
            outs_all = {}
            for p, outs, _kw in path:
                outs_all[p] = list(outs)

            def _leaves(n):
                if n not in outs_all:
                    return [n]
                out = []
                for o in outs_all[n]:
                    out += _leaves(o)
                return out

            cores = [p for p in outs_all if p != self.top]
            key = tuple(sorted(tuple(sorted(_leaves(c))) for c in cores))
            if key not in struct_keys:
                struct_keys.append(key)
        struct_map = {k: i for i, k in enumerate(struct_keys)}

        # all chain topologies must be covered by the structural paths
        chain_keys = {ch.topo_id() for ch in self.full.chains}
        if not chain_keys.issubset(struct_map):
            return get_topo_index(self.full)
        return struct_map

    def topo_index_from_name(self, name):
        """Topology slot of structural decay-section name(s).

        *name* may be a single intermediate pairing label (e.g. ``'pipeta'``)
        or a *list* of the internal core labels of a chain (for models with
        any number of decays, e.g. B→4π ``['rhoA', 'rhoB']``).  The slot is
        found by matching the combined final-leaf grouping of those cores
        against ``self.topo_index``.  Raises KeyError if nothing matches.
        """
        finals = self.finals
        d = self.decay_spec

        def _outs_of(n):
            entry = d.get(n)
            if entry is None:
                return []
            if not isinstance(entry, list):
                entry = [entry]
            outs = []
            for item in entry:
                if isinstance(item, str):
                    outs.append(item)
                elif isinstance(item, (list, tuple)):
                    outs += [k for k in item if isinstance(k, str)]
            return outs

        def _leaves(n):
            if n in finals:
                return [n]
            out = []
            for o in _outs_of(n):
                out += _leaves(o)
            return out

        names = [name] if isinstance(name, str) else list(name)
        groups = [tuple(sorted(set(_leaves(nm)))) for nm in names]
        key = tuple(sorted(groups))
        if key in self.topo_index:
            return self.topo_index[key]
        raise KeyError(
            f"no topology for decay names {names!r} (groups {groups}); "
            f"available: {self.topo_index}")
