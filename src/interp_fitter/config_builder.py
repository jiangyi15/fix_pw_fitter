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
class Chain:
    """A full decay chain from the top particle to all final states."""
    decays: list[Decay]

    @property
    def resonances(self) -> list[str]:
        """Non-top intermediate particles that could be substituted."""
        if not self.decays:
            return []
        top = self.decays[0].parent
        return [d.parent for d in self.decays if d.parent != top]

    @property
    def finals(self) -> set[str]:
        """Particles that never appear as a parent in this chain."""
        parents = {d.parent for d in self.decays}
        children = set()
        for d in self.decays:
            children.update(d.children)
        return children - parents


@dataclass
class Particle:
    """A particle with arbitrary model-defined properties.

    Common keys (interpreted by each model): ``J``, ``P``, ``mass``,
    ``width``, ``model``.
    """
    name: str
    props: dict = field(default_factory=dict)


@dataclass
class Wave:
    """A single physics wave after alias substitution."""
    chain: Chain
    resonances: list[str]          # resolved resonance names
    resonance_map: dict[str, str]  # alias → concrete name
    particles: list[Particle]      # properties for each resonance in order


@dataclass
class PhysicsModel:
    """Full expanded physics description."""
    top: str
    finals: list[str]
    chains: list[Chain]
    waves: list[Wave]
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
    #  Decay tree expansion
    # ------------------------------------------------------------------
    def _expand(node: str) -> list[Chain]:
        """Return all chains rooted at *node*."""
        if node in finals:
            return []
        branches = decay_raw[node]
        # Normalise: always list of lists
        if isinstance(branches[0], str):
            branches = [branches]
        chains: list[Chain] = []
        for branch in branches:
            children = [str(c) for c in branch]
            top_decay = Decay(node, children)

            # Expand each child; ``None`` means child is final
            sub_lists = [_expand(c) for c in children]
            non_empty = [sl for sl in sub_lists if sl]

            if not non_empty:
                chains.append(Chain([top_decay]))
            else:
                for combo in iproduct(*non_empty):
                    combined = [top_decay]
                    for sc in combo:
                        combined.extend(sc.decays)
                    chains.append(Chain(combined))
        return chains

    chains = _expand(top)

    # ------------------------------------------------------------------
    #  Wave generation  (substitute resonance aliases)
    # ------------------------------------------------------------------
    waves: list[Wave] = []
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
            wave_parts = [particles.get(n, Particle(n)) for n in actual]
            waves.append(Wave(chain=chain, resonances=actual,
                              resonance_map=res_map, particles=wave_parts))

    return PhysicsModel(top=top, finals=list(finals), chains=chains,
                        waves=waves, particles=particles)
