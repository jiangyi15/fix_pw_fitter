"""Parse a high-level physics description into an intermediate representation.

The YAML describes the decay topology, particle properties, and resonance
content.  This module expands it into a structured form (``PhysicsModel``)
that can later be converted into the ``Kernel`` config.
"""

from __future__ import annotations

from itertools import product as iproduct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Data classes  (the intermediate representation)
# ---------------------------------------------------------------------------

@dataclass
class Decay:
    """A single two-body decay ``parent -> child1 + child2``."""
    parent: str
    child1: str
    child2: str


@dataclass
class Chain:
    """A full decay chain from the top to all final-state particles."""
    decays: list[Decay]

    @property
    def resonances(self) -> list[str]:
        return [d.parent for d in self.decays
                if d.parent != self.decays[0].parent]

    @property
    def finals(self) -> set[str]:
        children = set()
        for d in self.decays:
            children.add(d.child1)
            children.add(d.child2)
        return children - {d.parent for d in self.decays}


@dataclass
class Particle:
    name: str
    J: float = 0.0
    P: int = 1
    mass: float = 0.0
    width: float = 0.0
    model: str = ""


@dataclass
class Wave:
    """A single physics wave: one chain with all alias substitutions applied."""
    chain: Chain
    resonances: list[str]          # resolved resonance names for this wave
    resonance_map: dict[str, str]  # alias → concrete name
    particles: list[Particle]      # particle properties for each resonance


@dataclass
class PhysicsModel:
    """The full expanded physics description."""
    top: str
    finals: list[str]
    chains: list[Chain]
    waves: list[Wave]
    particles: dict[str, Particle]  # all particles (including finals)


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

def parse_physics(physics: dict) -> PhysicsModel:
    """Parse a high-level physics dict into a ``PhysicsModel``.

    Parameters
    ----------
    physics : dict
        Dict with ``decay`` and ``particle`` keys as described in the README::

            decay:
              A: [[R, C], [Y, D]]
              R: [B, D]
              Y: [B, C]
            particle:
              $top: A
              $finals: [B, C, D]
              A:  {J: 0, P: -1, mass: 5.0}
              R:  [R1, R2]
              R1: {J: 0, P: 1, mass: 3.0, model: BW, width: 0.15}
              R2: {J: 1, P: -1, mass: 3.0, model: BW, width: 0.10}
              Y:  {J: 0, P: -1, mass: 3.0, model: BW, width: 0.12}
              B:  {J: 0, P: -1, mass: 0.1}
              C:  {J: 0, P: -1, mass: 0.1}
              D:  {J: 0, P: -1, mass: 0.1}
    """
    raw_part = dict(physics["particle"])
    top = raw_part.pop("$top")
    finals = list(raw_part.pop("$finals"))
    decay_raw = dict(physics["decay"])

    # Separate particle lists (resonance aliases) from definitions
    particles: dict[str, Particle] = {}
    resonances: dict[str, list[str]] = {}

    for name, val in list(raw_part.items()):
        if isinstance(val, list):
            resonances[name] = val
        elif isinstance(val, dict):
            particles[name] = Particle(name=name, **val)

    # Build decay chains
    def _expand(node: str) -> list[Chain]:
        if node in finals:
            return []
        branches = decay_raw[node]
        if isinstance(branches[0], str):
            branches = [branches]
        chains = []
        for b in branches:
            c1, c2 = str(b[0]), str(b[1])
            d = Decay(node, c1, c2)
            sub1 = _expand(c1)
            sub2 = _expand(c2)
            if not sub1 and not sub2:
                chains.append(Chain([d]))
            elif sub1 and not sub2:
                chains.extend(Chain([d] + s.decays) for s in sub1)
            elif sub2 and not sub1:
                chains.extend(Chain([d] + s.decays) for s in sub2)
            else:
                for s1 in sub1:
                    for s2 in sub2:
                        chains.append(Chain([d] + s1.decays + s2.decays))
        return chains

    chains = _expand(top)

    # Build waves by substituting resonance aliases
    waves: list[Wave] = []
    for chain in chains:
        subst_groups = []
        for res in chain.resonances:
            subst_groups.append(resonances.get(res, [res]))
        for combo in iproduct(*subst_groups):
            res_map: dict[str, str] = {}
            for orig, chosen in zip(
                [r for r in chain.resonances if r in resonances], combo
            ):
                res_map[orig] = chosen
            actual = [res_map.get(r, r) for r in chain.resonances]
            wave_parts = [particles.get(n, Particle(name=n)) for n in actual]
            waves.append(Wave(
                chain=chain,
                resonances=actual,
                resonance_map=res_map,
                particles=wave_parts,
            ))

    return PhysicsModel(
        top=top,
        finals=finals,
        chains=chains,
        waves=waves,
        particles=particles,
    )
