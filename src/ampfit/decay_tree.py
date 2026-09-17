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


class DecayTree:
    """The decay/particle declarations as a pure value object.

    Built from the raw ``decay`` spec and ``particle`` spec only; it holds the
    resolved structure, the chains, and the stable topology index.
    """

    def __init__(self, decay_spec, particle_spec):
        self.decay_spec = decay_spec
        self.particle_spec = particle_spec
        self.top = particle_spec["$top"]
        self.finals = list(particle_spec["$finals"])

        self.struct = build_decay_struct(decay_spec, self.top, self.finals)
        self.chains = get_decay_chains(self.struct, particle_spec)
        self.full = build_decay_chains(self.chains, particle_spec)

        self.n_decay = len(self.struct[0])
        self.n_res = self.n_decay - 1
        self.n_angles = 2 * self.n_decay
        if particle_spec[self.top]["J"] == 0:
            self.n_angles = self.n_angles - 3  # 3d rotation is not needed
        self.topo_index = self._build_topo_from_struct()
        self.n_topo = len(self.topo_index)

    # -- queries ---------------------------------------------------------
    def partial_waves(self):
        """``(ls, chain)`` pairs over every resonance-resolved chain."""
        return self.full.get_partial_waves()

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
