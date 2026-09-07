#!/usr/bin/env python3
"""Regenerate the tf-pwa reference output used by validate_aligned_pwa.py.

Run inside the tf-pwa environment, e.g.:

    conda run -n base python Validation/gen_tfpwa_reference.py

It feeds the same three embedded CM events to tf-pwa's
cal_angle_from_particle (align_ref="center_mass") and writes
/tmp/tfpwa_phys.npz with the aligned euler of the spin-1/2 final `p` per
chain, then checks the recorded tf-pwa numbers embedded in
validate_aligned_pwa.py are reproduced.

tf-pwa source is only used HERE to (re)produce those numbers; the
validation runner compares against the embedded constants only.

Environment:
    TFPWA_PATH   override the tf-pwa source path (default /mnt/e/github/tf-pwa)
"""
import ast
import os
import sys

TFPWA = os.environ.get("TFPWA_PATH", "/mnt/e/github/tf-pwa")
if TFPWA not in sys.path:
    sys.path.insert(0, TFPWA)

import numpy as np
from tf_pwa.particle import BaseParticle, BaseDecay, DecayChain, DecayGroup
from tf_pwa.cal_angle import cal_angle_from_particle

HERE = os.path.dirname(os.path.abspath(__file__))


def _const(name, path):
    """Read a module-level np.array(...) literal from *path* via AST (no
    imports -> the tf-pwa env does not need the project package)."""
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    v = node.value
                    if isinstance(v, ast.Call):        # np.array(...)
                        v = v.args[0]
                    return np.array(ast.literal_eval(v))
    raise RuntimeError(f"cannot find {name} in {path}")


_MOM = _const('_MOM', os.path.join(HERE, 'validate_aligned_pwa.py'))
TF_EULER = _const('TF_EULER', os.path.join(HERE, 'validate_aligned_pwa.py'))
n = _MOM.shape[0]
print("events", n)
mom = {nm: _MOM[:, i] for i, nm in enumerate(['p', 'pim', 'pip', 'eta'])}


def P(name, J, Ppar, mass):
    return BaseParticle(name, J=J, P=Ppar, mass=mass)


top = P("Lc", 0.5, +1, 2.28646)
sig = P("Sig1385p", 1.5, +1, 1.3828)
a0 = P("a098", 0.0, +1, 0.98)
lam = P("Lambdap", 0.5, +1, 1.11568)
p = P("p", 0.5, +1, 0.938272)
pip = P("pip", 0.0, -1, 0.13957)
pim = P("pim", 0.0, -1, 0.13957)
eta = P("eta", 0.0, -1, 0.54786)

chain0 = DecayChain([BaseDecay(top, [sig, eta]),
                     BaseDecay(sig, [lam, pip]),
                     BaseDecay(lam, [p, pim])])
chain1 = DecayChain([BaseDecay(top, [a0, lam]),
                     BaseDecay(a0, [pip, eta]),
                     BaseDecay(lam, [p, pim])])
dg = DecayGroup([chain0, chain1])
top_struct = dg.topology_structure()
print("n standard chains", len(top_struct))

data = {}
for chain in top_struct:
    for node, finals in chain.sorted_table().items():
        total = None
        for f in finals:
            total = mom[str(f)] if total is None else total + mom[str(f)]
        if node not in data:
            data[node] = {"p": total}


def run(final_rest):
    res = cal_angle_from_particle(
        dict(data), dg, using_topology=True, random_z=False, r_boost=True,
        final_rest=final_rest, align_ref="center_mass")
    out = []
    for chain in top_struct:
        got = None
        for dec in chain:
            for child in dec.outs:
                if str(child) == "p":
                    got = res[chain][dec][child].get("aligned_angle")
        out.append(np.stack([np.asarray(got["alpha"]).reshape(-1),
                             np.asarray(got["beta"]).reshape(-1),
                             np.asarray(got["gamma"]).reshape(-1)], axis=-1))
    return np.stack(out, axis=1)


euler = run(False)
euler_rest = run(True)
np.savez('/tmp/tfpwa_phys.npz', euler=euler, euler_rest=euler_rest)
print("saved /tmp/tfpwa_phys.npz", euler.shape)

# tf-pwa stores (gamma, beta, alpha) of the physical rotation; mapped to our
# standard (alpha, beta, gamma) = tf(gamma, beta, alpha):
exp = TF_EULER[..., [2, 1, 0]]
diff = float(np.abs(euler[..., [2, 1, 0]] - exp).max())
print(f"max |d| vs embedded tf-pwa reference (final_rest=False): {diff:.2e}")
print("OK" if diff < 1e-8 else "MISMATCH")
