"""ONE case-1 table for the flat ``pwa_gen`` kernels: a copy, its parent, and
the source digests.

A case-1 copy is standalone: its own module, its own ``.so``, its own symbols,
and the kernel it was copied FROM left byte-identical.  That is one property
with three faces, so it is one row here and one test — not a per-family file.

WHICH FAMILIES CARRY A ROW IS A DECISION, and it is written here:

* the PINNED-coefficient caches (``ampcache``, ``ampcache32``);
* the fixed-coefficient caches (``s0f``, ``s0g``);
* the s0e tile family (``s0e`` is the original, ``s0e2``/``s0e3``/``s0e4`` copy
  it and ``s0e5`` copies ``s0e2``);
* ``s0c2``, a copy of ``s0c``.

Those are exactly the kernels whose source is a copy under the kernel policy.
The dense leaves (``gen``/``cache``/``gen2``/``1bw``/``s0c``) have no case-1
claim to make, and the C/OpenMP CPU leaves make theirs in their own files (this
table's module/symbol/digest derivation is CUDA-only, so a CPU row cannot be
expressed here) — ``NO_ISOLATION`` says so for each, with its reason, so a
newly registered leaf has to be added to ONE of the two sets instead of being
silently uncovered by both.

``SOURCE_SHA256`` holds ONE digest per source file (the parents are shared:
``s0f.cu`` is the parent of ``s0g`` and of ``ampcache``), so a digest literal
can never disagree with itself between two rows.
"""
import ast
import hashlib
import importlib
import os

import numpy as np

import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend
from tabpwa.cuda import __file__ as _cuda_dir_file

from _flat_compute import PINNED, SPECS as LEAVES
from _flat_grid import CFG_KM

#: ``sha256`` of every source a row names (its own, and the parent it copies).
#: An edit to a working kernel fails HERE first: a copy is never a reason to
#: change the kernel it copies, and a kernel is never edited in place.
SOURCE_SHA256 = {
    "kernels_flat_pwa_gen_s0e.cu":
        "ff15fc9116676b0fe39ec7891df4a4188e62bf4c616b1f0c5b86b210011d3962",
    "kernels_flat_pwa_gen_s0e2.cu":
        "2e93919448341cbeec0f89620a34be2471cd9b0840bec12ab2d6781da341acb4",
    "kernels_flat_pwa_gen_s0e3.cu":
        "ef881f2330120b249f9594e9aa2fa140fb58d49e352e05de4bb3eeadd5c27116",
    "kernels_flat_pwa_gen_s0e4.cu":
        "7d1976c5b1343dc280429b17bbc46373f8b8f0feb3916fe103390c9dfaccf995",
    "kernels_flat_pwa_gen_s0e5.cu":
        "0abc4f5c3641fc87e38cea2dda37b7a7cd5a7cf6d9f69da4372fe9b497463090",
    "kernels_flat_pwa_gen_s0c.cu":
        "a16cabd68ca8bf6b8c6cd9535125e6aec2b061a736fbf619634453d67f30a192",
    "kernels_flat_pwa_gen_s0c2.cu":
        "a406a25868e3429796a30845578743e03e4ee45df52c422e18d6cc21a18e68b3",
    "kernels_flat_pwa_gen_s0f.cu":
        "a108ba867b1ce1375066812c69b3d613293bdd4020dade35545650146d4e7744",
    "kernels_flat_pwa_gen_s0g.cu":
        "7308ca87a126dfe7b2578b484a7d57d2708d2c2073c15042864771fe491df6a5",
    "kernels_flat_pwa_gen_ampcache.cu":
        "a2b53fa491f7180f58665b82c0835b1803fead5401514716cc8c165ae9efd69a",
    "kernels_flat_pwa_gen_ampcache32.cu":
        "3bbbb5ce2afd9f7477e4ccf3b10b692d2d89509dcc7703b6112498c40d7bc8a3",
}

#: name -> the source this kernel was copied FROM (``None`` = the family's
#: original).  ``module``/``symbol_prefix``/its own source are DERIVED from the
#: name, so a row cannot disagree with itself.
ISOLATION = {
    "cuda_flat_pwa_gen_s0e":      {"parent": None},
    "cuda_flat_pwa_gen_s0e2":     {"parent": "kernels_flat_pwa_gen_s0e.cu"},
    "cuda_flat_pwa_gen_s0e3":     {"parent": "kernels_flat_pwa_gen_s0e.cu"},
    "cuda_flat_pwa_gen_s0e4":     {"parent": "kernels_flat_pwa_gen_s0e.cu"},
    "cuda_flat_pwa_gen_s0e5":     {"parent": "kernels_flat_pwa_gen_s0e2.cu"},
    "cuda_flat_pwa_gen_s0c2":     {"parent": "kernels_flat_pwa_gen_s0c.cu"},
    "cuda_flat_pwa_gen_s0f":      {"parent": "kernels_flat_pwa_gen_s0e2.cu"},
    "cuda_flat_pwa_gen_s0g":      {"parent": "kernels_flat_pwa_gen_s0f.cu"},
    "cuda_flat_pwa_gen_ampcache": {"parent": "kernels_flat_pwa_gen_s0f.cu"},
    "cuda_flat_pwa_gen_ampcache32": {
        "parent": "kernels_flat_pwa_gen_ampcache.cu"},
}

#: leaves with NO case-1 claim, each with the reason.  Explicit, so the
#: completeness assert below is a partition rather than a subset.
NO_ISOLATION = {
    "cuda_flat_pwa_gen":
        "the dense family's original — no parent to be a copy of",
    "cuda_flat_cache_pwa_gen":
        "dense family: predates the case-1 rule and its symbols are not the "
        "uniform stem, so it makes no copy claim",
    "cuda_flat_pwa_gen2":
        "case-1 copy of cuda_flat_cache_pwa_gen, but it predates the rule, so "
        "it makes no copy claim here; its config/dedup contract is "
        "test_flat_dedup.py's",
    "cuda_flat_pwa_gen_1bw":
        "case-1 copy of cuda_flat_pwa_gen2, but it predates the rule, so it "
        "makes no copy claim here; its config/dedup contract is "
        "test_flat_dedup.py's",
    "cuda_flat_pwa_gen_s0c":
        "the s0c original — its copy (s0c2) carries the row",
    "cpu_flat_pwa_gen":
        "C/OpenMP leaf: no CUDA module/.so and no case-1 copy",
    "cpu_flat_pwa_gen_cached":
        "C/OpenMP case-1 copy of cpu_flat_pwa_gen, but this table is the CUDA "
        "one (its module/symbol/digest derivation is CUDA-only); the CPU pair's "
        "standalone claim is asserted in its own file",
    "cpu_flat_pwa_gen_cached2":
        "C/OpenMP CASE-1 COPY of cpu_flat_pwa_gen_cached (own .c/.so/symbols, "
        "see kernels_cpu_flat_pwa_gen_cached2.c and "
        "test_cpu_flat_pwa_gen_cached2.py), but this table is the CUDA one — "
        "its module/symbol/digest derivation is CUDA-only, so CPU digests are "
        "out of scope here; the CPU pair's standalone claim is asserted in its "
        "own file",
}

assert not set(ISOLATION) & set(NO_ISOLATION), (
    f"a leaf cannot both carry and disclaim a case-1 row: "
    f"{sorted(set(ISOLATION) & set(NO_ISOLATION))}")
assert set(ISOLATION) | set(NO_ISOLATION) == set(LEAVES) | set(PINNED), (
    "every flat leaf must be in exactly one of ISOLATION / NO_ISOLATION — a "
    "newly registered leaf fails here instead of being silently uncovered: "
    f"missing {sorted((set(LEAVES) | set(PINNED)) - set(ISOLATION) - set(NO_ISOLATION))}, "
    f"stale {sorted((set(ISOLATION) | set(NO_ISOLATION)) - set(LEAVES) - set(PINNED))}")


def module_of(name):
    """``cuda_flat_pwa_gen_s0f`` -> ``tabpwa.cuda._flat_pwa_gen_s0f``."""
    return "tabpwa.cuda._" + name.removeprefix("cuda_")


def prefix_of(name):
    """The symbol/source stem: ``cuda_flat_pwa_gen_s0f`` -> ``flat_pwa_gen_s0f``."""
    return name.removeprefix("cuda_")


def _source_path(fname):
    return os.path.join(os.path.dirname(_cuda_dir_file), fname)


def _digest(fname):
    with open(_source_path(fname), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _imported_modules(path):
    with open(path) as f:
        tree = ast.parse(f.read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


#: A case-1 copy carries verbatim copies of the build-time helpers it needs
#: ("is STANDALONE: it must not import another backend's module").  The policy
#: requires them to be pinned by an ``np.array_equal`` test against the
#: originals, and that pin is ONE table too: ``{copy: {helper: original}}``.
#: The originals are listed separately, because THEY carry the helpers by
#: definition and must not be mistaken for unguarded copies.
HELPER_COPIES = {
    "_flat_pwa_gen_s0f": {
        "uslot_map": "_flat_pwa_gen_s0e",
        "amp_slot_map": "_flat_pwa_gen_s0e",
        "index_runs": "_flat_pwa_gen_s0e",
    },
    "_flat_pwa_gen_s0g": {
        "uslot_map": "_flat_pwa_gen_s0e",
        "amp_slot_map": "_flat_pwa_gen_s0e",
        "index_runs": "_flat_pwa_gen_s0e",
    },
    "_flat_pwa_gen_s0c2": {
        "uslot_map": "_flat_pwa_gen_s0e",
        "amp_slot_map": "_flat_pwa_gen_s0e",
        "index_runs": "_flat_pwa_gen_1bw",
    },
    "_flat_pwa_gen_ampcache": {
        "uslot_map": "_flat_pwa_gen_s0e",
        "amp_slot_map": "_flat_pwa_gen_s0e",
    },
    "_flat_pwa_gen_ampcache32": {
        "uslot_map": "_flat_pwa_gen_s0e",
        "amp_slot_map": "_flat_pwa_gen_s0e",
    },
}

#: Modules that carry a helper as the FAMILY's own implementation (the s0e tile
#: chain defines them, ``1bw`` defines the ``n_wave``-checked ``index_runs``),
#: so they are originals rather than copies.  A module here that stops defining
#: a helper fails the derivation below; a module in NEITHER set fails it too.
HELPER_ORIGINALS = (
    "_flat_pwa_gen_s0e", "_flat_pwa_gen_s0e2", "_flat_pwa_gen_s0e3",
    "_flat_pwa_gen_s0e4", "_flat_pwa_gen_s0e5", "_flat_pwa_gen_1bw",
)

#: helper -> (arguments, keywords) on a real kernel config, derived the way the
#: wrappers derive them (``n_wave`` is ``fl_table``'s row count, ``n_proj`` is
#: ``matrix_angle``'s columns over it), so this is the production call.
_HELPER_ARGS = {
    "uslot_map": lambda kc: (
        (np.asarray(kc["basis_table"], np.complex128),
         np.asarray(kc["basis_m_index"], np.int64)), {}),
    "amp_slot_map": lambda kc: (
        (kc["matrix_angle"], kc["fl_table"], kc["fl_q_index"],
         kc["num_table"], kc["num_m_index"],
         int(np.asarray(kc["fl_table"]).shape[0]),
         max(1, int(np.asarray(kc["matrix_angle"]).shape[1])
             // int(np.asarray(kc["fl_table"]).shape[0]))), {}),
    # ``s0e``/``s0f``/``s0g`` spell it positional-first, ``1bw``/``s0c2`` put
    # ``n_wave`` in front of it: the keyword is the one call that suits both.
    "index_runs": lambda kc: (
        (np.asarray(kc["gamma_matrix"], np.float64),),
        {"n_basis": int(np.asarray(kc["basis_table"]).shape[0])}),
}


def _helper_config():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    return f.kernel_config


def _carried_helpers():
    """``{module: {helper names}}`` for every loaded flat wrapper that has any."""
    import sys

    out = {}
    for mod in list(sys.modules):
        if "tabpwa.cuda._flat_pwa_gen" not in mod or mod.endswith("__"):
            continue
        path = os.path.splitext(sys.modules[mod].__file__)[0] + ".py"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            tree = ast.parse(fh.read())
        names = {n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in _HELPER_ARGS}
        if names:
            out[mod.rsplit(".", 1)[-1]] = names
    return out


def test_every_helper_copy_has_a_row():
    """A module that carries a helper is a COPY (row here) or an ORIGINAL.

    Derived from the source, so a copy that travels with a helper and no row
    fails here instead of going unguarded — the failure mode the deleted
    per-kernel file left behind.
    """
    carried = _carried_helpers()
    assert set(HELPER_ORIGINALS) <= set(carried), (
        f"declared original(s) that no longer carry a helper: "
        f"{sorted(set(HELPER_ORIGINALS) - set(carried))}")
    for mod in HELPER_ORIGINALS:
        assert mod not in HELPER_COPIES, mod
    stray = set(carried) - set(HELPER_COPIES) - set(HELPER_ORIGINALS)
    assert not stray, (
        "a module carries a build-time helper with no HELPER_COPIES row and no "
        f"HELPER_ORIGINALS entry: {sorted(stray)}")
    for mod, names in HELPER_COPIES.items():
        assert set(names) == carried[mod], (mod, sorted(names), sorted(carried[mod]))


@pytest.mark.parametrize("mod", sorted(HELPER_COPIES))
def test_a_helper_copy_matches_its_original(mod):
    """``np.array_equal`` on EVERY output, on a real kernel config.

    A copy may differ in its error messages (they name its own backend) and
    nowhere else: same dedup, same CSR, same ordering.  HONEST ABOUT STRENGTH:
    on these fixtures every basis row is distinct and every gamma row is dense,
    so ``uslot_map`` is the identity and ``index_runs`` emits the full span —
    the comparison therefore covers the key construction, the dtype handling
    and the emit order of every element, not a folding case (a fold is gated
    numerically by the production-shape tests).
    """
    kc = _helper_config()
    nb = int(np.asarray(kc["basis_table"]).shape[0])
    for helper, origin in HELPER_COPIES[mod].items():
        args, kw = _HELPER_ARGS[helper](kc)
        mine = getattr(importlib.import_module("tabpwa.cuda." + mod), helper)
        theirs = getattr(importlib.import_module("tabpwa.cuda." + origin), helper)
        got, want = mine(*args, **kw), theirs(*args, **kw)
        got = got if isinstance(got, tuple) else (got,)
        want = want if isinstance(want, tuple) else (want,)
        assert len(got) == len(want), (mod, helper)
        for i, (g, w) in enumerate(zip(got, want)):
            assert np.array_equal(np.asarray(g), np.asarray(w)), (
                f"{mod}.{helper} output {i} differs from {origin}.{helper}")


def test_the_helper_comparison_covers_every_element():
    """The arrays the comparison above runs on are fully populated.

    Guards the one way that test could go quiet: an empty or degenerate kernel
    config would make every output trivially equal.
    """
    from tabpwa.cuda._flat_pwa_gen_s0e import index_runs, uslot_map

    kc = _helper_config()
    nb = int(np.asarray(kc["basis_table"]).shape[0])
    _rows, _cols, slot = uslot_map(
        np.asarray(kc["basis_table"], np.complex128),
        np.asarray(kc["basis_m_index"], np.int64))
    assert slot.size == nb > 0
    lo, hi, _nz_k, nz_m = index_runs(
        np.asarray(kc["gamma_matrix"], np.float64), n_basis=nb)
    assert nz_m.size > 0
    assert int(hi[-1]) == nz_m.size          # every row's run is closed
    assert np.all(hi > lo)                   # no empty wave
@pytest.mark.parametrize("name", sorted(ISOLATION))
def test_case1_isolation(name):
    """A standalone copy: own module/``.so``/symbols, parent source intact.

    Everything but the row's ``parent`` is DERIVED from the registry name, so
    the four faces of the property cannot disagree: the copy does not import
    the kernel it copies, it loads its own ``.so`` with its own symbols, its
    parent's ``.so`` does not carry them, a second instance shares the one
    loaded library, and every source involved matches ``SOURCE_SHA256``.
    """
    row = ISOLATION[name]
    prefix = prefix_of(name)
    mod = importlib.import_module(module_of(name))
    mine = "cuda_compute_" + prefix
    own_src = "kernels_" + prefix + ".cu"

    # own module: the copy must not import the kernel it copies
    py = os.path.splitext(mod.__file__)[0] + ".py"
    imported = _imported_modules(py)
    parent = row["parent"]
    if parent is not None:
        parent_prefix = parent[len("kernels_"):-len(".cu")]
        parent_module = "tabpwa.cuda._" + parent_prefix
        assert mod is not importlib.import_module(parent_module)
        assert not any(parent_module in m for m in imported), (
            f"a copy must not import the kernel it copies: {sorted(imported)}")

    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    a = create_backend(name, kc, model="pwa_gen")
    try:
        lib = a.kernel._lib
        assert hasattr(lib, mine), f"{name}'s .so has no {mine}"
        if parent is not None:
            b = create_backend("cuda_" + parent_prefix, kc, model="pwa_gen")
            try:
                theirs = "cuda_compute_" + parent_prefix
                assert b.kernel._lib is not lib, "the copy shares the parent's .so"
                assert hasattr(b.kernel._lib, theirs)
                assert not hasattr(lib, theirs), (
                    f"{name}'s library carries the parent's {theirs}")
                assert not hasattr(b.kernel._lib, mine), (
                    f"the parent's library carries {name}'s {mine}")
            finally:
                b.free()
        # ONE library per process for THIS kernel: the module-level handle
        a2 = create_backend(name, kc, model="pwa_gen")
        try:
            assert a2.kernel._lib is lib
        finally:
            a2.free()
    finally:
        a.free()

    # the sources: this kernel's own, and the one it copied
    assert own_src in SOURCE_SHA256, (
        f"{name}: its own source {own_src} is not pinned — add the digest")
    assert _digest(own_src) == SOURCE_SHA256[own_src], (
        f"{own_src} changed — a working kernel is never edited in place")
    if parent is not None:
        assert parent in SOURCE_SHA256, parent
        assert _digest(parent) == SOURCE_SHA256[parent], (
            f"{parent} changed — it IS the parent {name} was copied from, so a "
            "change here is a change to the parent kernel")
