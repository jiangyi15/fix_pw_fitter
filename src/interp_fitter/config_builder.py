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


@dataclass
class PhysicsModel:
    """Full expanded physics description."""
    top: str
    finals: list[str]
    decay_chains: list[DecayChain]  # all concrete decay paths (the DecayGroup)
    particles: dict[str, Particle]


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
            decay_chains.append(DecayChain(
                decays=chain.decays, resonances=actual,
                resonance_map=res_map, particles=parts))

    return PhysicsModel(top=top, finals=list(finals),
                        decay_chains=decay_chains, particles=particles)
