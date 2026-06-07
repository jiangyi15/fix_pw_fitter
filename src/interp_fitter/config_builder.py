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
            memo[name] = tuple(result)
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
        top_sets = set()
        for dc in self.decay_chains:
            tm = dc.topo_map
            top_sets.add(frozenset(tm.get(self.top, ())))
        if len(top_sets) != 1:
            raise ValueError(
                f"Inconsistent top decay: {self.top} → "
                f"{[sorted(s) for s in top_sets]}. "
                "All DecayChains must produce the same set of final states."
            )


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

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
            children = [str(c) for c in branch]
            top_decay = Decay(node, children)
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
                resolved_decays.append(Decay(
                    parent=_resolve(d.parent),
                    children=[_resolve(c) for c in d.children],
                ))

            decay_chains.append(DecayChain(
                decays=resolved_decays, resonances=actual,
                resonance_map=res_map, particles=parts))

    return PhysicsModel(top=top, finals=list(finals),
                        decay_chains=decay_chains, particles=particles)
