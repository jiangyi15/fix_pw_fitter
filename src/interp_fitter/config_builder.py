"""Parse a high-level physics description into an intermediate representation.

The YAML describes the decay topology, particle properties, and resonance
content.  This module expands it into a structured form (``PhysicsModel``)
that can later be converted into the ``Kernel`` config.

Particles carry arbitrary key-value properties defined by each model
(``model`` key selects the lineshape).  Decays support N children
(two-body, three-body, …).
"""

from __future__ import annotations

from itertools import product as iproduct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Intermediate representation
# ---------------------------------------------------------------------------

@dataclass
class Decay:
    """``parent -> children[0] + children[1] + ...`` (N-body)."""
    parent: str
    children: list[str]
    p_break: bool = False                # parity violation flag
    parent_particle: Particle | None = None
    child_particles: list[Particle] | None = None

    def get_ls_list(self) -> list[tuple[int, float]]:
        """Return valid (L, S) pairs for this two-body decay."""
        if len(self.children) != 2 or self.parent_particle is None:
            return []
        pp = self.parent_particle.props
        c1p = self.child_particles[0].props if self.child_particles else {}
        c2p = self.child_particles[1].props if self.child_particles else {}
        return get_ls_list(
            parent_J=pp.get("J", 0), parent_P=pp.get("P", 1),
            child1_J=c1p.get("J", 0), child1_P=c1p.get("P", 1),
            child2_J=c2p.get("J", 0), child2_P=c2p.get("P", 1),
            p_break=self.p_break,
        )

    def get_g_ls(self) -> list[str]:
        """Coupling parameter names for each (L, S) pair.

        Format: ``{parent}->{child1}.{child2}_g_ls_{idx}``.
        Example::

            Decay A -> B + C  with LS = [(1, 1.0), (3, 1.0)]
            →  [\"A->B.C_g_ls_0\", \"A->B.C_g_ls_1\"]
        """
        if len(self.children) != 2:
            return []
        c1, c2 = self.children[0], self.children[1]
        return [f"{self.parent}->{c1}.{c2}_g_ls_{i}"
                for i in range(len(self.get_ls_list()))]
        """Return valid (L, S) pairs for this two-body decay.

        Requires exactly two children and that the ``parent_particle``
        and ``child_particles`` have been set.  Returns an empty list
        for N-body decays with ``N != 2``.  The ``p_break`` attribute
        controls whether parity conservation is enforced.
        """
        if len(self.children) != 2 or self.parent_particle is None:
            return []
        pp = self.parent_particle.props
        c1p = self.child_particles[0].props if self.child_particles else {}
        c2p = self.child_particles[1].props if self.child_particles else {}
        return get_ls_list(
            parent_J=pp.get("J", 0), parent_P=pp.get("P", 1),
            child1_J=c1p.get("J", 0), child1_P=c1p.get("P", 1),
            child2_J=c2p.get("J", 0), child2_P=c2p.get("P", 1),
            p_break=self.p_break,
        )


@dataclass
class _Chain:
    """Internal: a decay chain before alias substitution."""
    decays: list[Decay]

    @property
    def resonances(self) -> list[str]:
        if not self.decays:
            return []
        top = self.decays[0].parent
        return [d.parent for d in self.decays if d.parent != top]


@dataclass
class Particle:
    """A particle with arbitrary model-defined properties.

    Common keys (interpreted by each model): ``J``, ``P``, ``mass``,
    ``width``, ``model``.
    """
    name: str
    props: dict = field(default_factory=dict)


@dataclass
class DecayChain:
    """A specific decay path from top to finals, with concrete resonances."""
    decays: list[Decay]
    resonances: list[str]          # resolved resonance names
    resonance_map: dict[str, str]  # alias → concrete name
    particles: list[Particle]      # properties for each resonance in order

    @property
    def finals(self) -> set[str]:
        parents = {d.parent for d in self.decays}
        children = set()
        for d in self.decays:
            children.update(d.children)
        return children - parents

    @property
    def topo_id(self) -> tuple[tuple[str, ...], ...]:
        """Sorted tuple of resonance final-state tuples — a unique topology key.

        Each intermediate resonance contributes its ``topo_map`` value.
        The collection is sorted so that identical topologies compare equal
        regardless of resonance ordering.
        """
        tm = self.topo_map
        return tuple(sorted(tm[r] for r in self.resonances))

    def ls_combinations(self) -> list[tuple[tuple[int, float], ...]]:
        """Cartesian product of (L, S) pairs across all decays.

        Each element is a tuple of ``(L, S)`` pairs, one per decay
        in the chain.
        """
        lists = [d.get_ls_list() for d in self.decays]
        if not lists:
            return []
        return [combo for combo in iproduct(*lists)]

    def g_ls_combinations(self) -> list[tuple[str, ...]]:
        """Cartesian product of ``g_ls`` names across all decays.

        Each element is a tuple of parameter names, one per decay.
        """
        lists = [d.get_g_ls() for d in self.decays]
        if not lists:
            return []
        return [combo for combo in iproduct(*lists)]

    @property
    def topo_map(self) -> dict[str, tuple[str, ...]]:
        """Map each particle to its final-state tuple.

        For a final-state particle the tuple is ``(name,)``.
        For a parent it is the concatenation of its children's tuples,
        recursively flattened to finals.

        Example::

            DecayChain: A→R1+C  R1→B+D
            topo_map = {
                "A":  ("B", "C", "D"),
                "R1": ("B", "D"),
                "B":  ("B",),
                "C":  ("C",),
                "D":  ("D",),
            }
        """
        children_of: dict[str, list[str]] = {}
        for d in self.decays:
            children_of.setdefault(d.parent, []).extend(d.children)
        all_parts = set(children_of.keys())
        for ch in children_of.values():
            all_parts.update(ch)

        memo: dict[str, tuple[str, ...]] = {}

        def _resolve(name: str) -> tuple[str, ...]:
            if name in memo:
                return memo[name]
            if name not in children_of:  # final state
                memo[name] = (name,)
                return memo[name]
            result: list[str] = []
            for c in children_of[name]:
                result.extend(_resolve(c))
            memo[name] = tuple(sorted(result))
            return memo[name]

        for p in all_parts:
            _resolve(p)
        return memo


@dataclass
class PhysicsModel:
    """Full expanded physics description."""
    top: str
    finals: list[str]
    decay_chains: list[DecayChain]  # all concrete decay paths (the DecayGroup)
    particles: dict[str, Particle]

    def __post_init__(self):
        """Validate that all chains share the same top and finals."""
        if not self.decay_chains:
            return
        ref = self.decay_chains[0].topo_map.get(self.top, ())
        for dc in self.decay_chains[1:]:
            other = dc.topo_map.get(self.top, ())
            if other != ref:
                raise ValueError(
                    f"Inconsistent top decay: {self.top} → {ref} vs {other}. "
                    "All DecayChains must map the top to the same finals."
                )


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

def get_ls_list(parent_J: float, parent_P: int,
                child1_J: float, child1_P: int,
                child2_J: float, child2_P: int,
                p_break: bool = False) -> list[tuple[int, float]]:
    r"""Return all valid (L, S) pairs for a two-body decay.

    Spins may be half-integers (0, 0.5, 1, 1.5, …).  *L* is always an
    integer; *S* can be half-integer.

    Constraints:

    * **Spin addition** — total spin *S* of the two daughters::

        |J₁ − J₂| ≤ S ≤ J₁ + J₂   (step 1)

    * **Angular momentum** — L couples with S to form the parent *J*::

        |L − S| ≤ J_parent ≤ L + S

    * **Parity** — unless ``p_break=True``::

        P_parent = P₁ · P₂ · (−1)^{L}

    Parameters
    ----------
    parent_J, parent_P : float, int
        Spin and parity of the decaying particle.
    child1_J, child1_P, child2_J, child2_P : float, int
        Spin and parity of the two daughters.
    p_break : bool
        If True, the parity conservation rule is skipped.

    Returns
    -------
    list[tuple[int, float]]
        All allowed (orbital angular momentum *L*, total spin *S*) pairs.
    """
    # Work in units of 1/2 to handle half-integer spins
    def _to_halves(v: float) -> int:
        return int(round(2 * v))

    Jp2 = _to_halves(parent_J)
    J12 = _to_halves(child1_J)
    J22 = _to_halves(child2_J)

    S_min = abs(J12 - J22)
    S_max = J12 + J22

    results: list[tuple[int, float]] = []
    for S2 in range(S_min, S_max + 1, 2):  # step 2 in half-units = step 1
        # L range: |Jp - S| ≤ L ≤ Jp + S
        L_min = int(abs(Jp2 - S2) // 2)
        L_max = int((Jp2 + S2) // 2)
        for L in range(L_min, L_max + 1):
            if p_break:
                results.append((L, S2 / 2.0))
            else:
                if parent_P == child1_P * child2_P * ((-1) ** L):
                    results.append((L, S2 / 2.0))
    return results


def parse_physics(physics: dict) -> PhysicsModel:
    """Parse a high-level physics dict into a ``PhysicsModel``.

    Parameters
    ----------
    physics : dict
        Dict with ``decay`` and ``particle`` keys::

            decay:
              A: [[R, C], [Y, D, E]]
              R: [B, D]
              Y: [B, C]

            particle:
              $top: A
              $finals: [B, C, D, E]
              A:  {J: 0, P: -1, mass: 5.0}
              R:  [R1, R2]
              R1: {J: 0, P: 1, mass: 3.0, model: BW, width: 0.15}
              R2: {J: 1, P: -1, mass: 3.0, model: BW, width: 0.10}
              Y:  {J: 0, P: -1, mass: 3.0, model: BW, width: 0.12}
              B:  {J: 0, P: -1, mass: 0.1}
              C:  {J: 0, P: -1, mass: 0.1}
              D:  {J: 0, P: -1, mass: 0.1}
              E:  {J: 0, P: -1, mass: 0.1}

        ``R: [R1, R2]`` means *R* is an alias that expands to two distinct
        resonances, generating separate physics waves.
    """
    raw_part = dict(physics["particle"])
    top = raw_part.pop("$top")
    finals = set(raw_part.pop("$finals"))
    decay_raw = dict(physics["decay"])

    # Separate alias lists from particle definitions
    particles: dict[str, Particle] = {}
    aliases: dict[str, list[str]] = {}

    for name, val in list(raw_part.items()):
        if isinstance(val, list):
            aliases[name] = val
        elif isinstance(val, dict):
            particles[name] = Particle(name=name, props=dict(val))

    def _parse_branch(branch: list) -> tuple[list[str], dict]:
        """Split a decay branch into child names and merged options dict."""
        children: list[str] = []
        opts: dict = {}
        for item in branch:
            if isinstance(item, dict):
                opts.update(item)
            else:
                children.append(str(item))
        return children, opts

    # ------------------------------------------------------------------
    #  Decay tree expansion  (unexpanded chains with aliases)
    # ------------------------------------------------------------------
    def _expand(node: str) -> list[_Chain]:
        if node in finals:
            return []
        branches = decay_raw[node]
        if isinstance(branches[0], str):
            branches = [branches]
        result: list[_Chain] = []
        for branch in branches:
            children, opts = _parse_branch(branch)
            top_decay = Decay(node, children, p_break=opts.get("p_break", False))
            sub_lists = [_expand(c) for c in children]
            non_empty = [sl for sl in sub_lists if sl]
            if not non_empty:
                result.append(_Chain([top_decay]))
            else:
                for combo in iproduct(*non_empty):
                    combined = [top_decay]
                    for sc in combo:
                        combined.extend(sc.decays)
                    result.append(_Chain(combined))
        return result

    chains = _expand(top)

    # ------------------------------------------------------------------
    #  Build DecayChains  (substitute resonance aliases)
    # ------------------------------------------------------------------
    decay_chains: list[DecayChain] = []
    for chain in chains:
        subst_groups = [aliases.get(r, [r]) for r in chain.resonances]
        for combo in iproduct(*subst_groups):
            res_map: dict[str, str] = {}
            idx = 0
            for r in chain.resonances:
                if r in aliases:
                    res_map[r] = combo[idx]
                    idx += 1
            actual = [res_map.get(r, r) for r in chain.resonances]
            parts = [particles.get(n, Particle(n)) for n in actual]

            # Build decays with aliases resolved in parent/children names
            def _resolve(name: str) -> str:
                return res_map.get(name, name)

            resolved_decays = []
            for d in chain.decays:
                parent_name = _resolve(d.parent)
                child_names = [_resolve(c) for c in d.children]
                resolved_decays.append(Decay(
                    parent=parent_name,
                    children=child_names,
                    p_break=d.p_break,
                    parent_particle=particles.get(parent_name),
                    child_particles=[particles.get(c) for c in child_names],
                ))

            # Skip chains where any two-body decay has no valid (L,S)
            if any(len(d.get_ls_list()) == 0 for d in resolved_decays
                   if len(d.children) == 2):
                continue

            decay_chains.append(DecayChain(
                decays=resolved_decays, resonances=actual,
                resonance_map=res_map, particles=parts))

    return PhysicsModel(top=top, finals=list(finals),
                        decay_chains=decay_chains, particles=particles)
