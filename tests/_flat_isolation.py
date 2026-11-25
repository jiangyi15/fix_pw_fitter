"""ONE case-1 table for the flat ``pwa_gen`` kernels: a copy, its parent, and
the source digests.

A case-1 copy is standalone: its own module, its own ``.so``, and the kernel
it was copied FROM left byte-identical.  The C IDENTIFIERS are not part of that
identity: a copy made under the current rule keeps the parent's verbatim
(``cffi.dlopen`` is ``RTLD_LOCAL`` and each kernel is its own ``.so``, so
identical names cannot be interposed) — ``fslot`` is the first such, declared
per row as ``symbols: "parent"``.  The older copies export their own prefixes;
that is HISTORY, recorded per row as ``symbols: "own"``, not a convention.

WHICH FAMILIES CARRY A ROW IS A DECISION, and it is written here:

* the PINNED-coefficient caches (``ampcache``, ``ampcache32``);
* the fixed-coefficient caches (``s0f``, ``s0g``);
* the s0e tile family (``s0e`` is the original, ``s0e2``/``s0e3``/``s0e4`` copy
  it and ``s0e5`` copies ``s0e2``);
* ``s0c2``, a copy of ``s0c``;
* ``fslot``, a copy of ``gen2`` — the first made WITHOUT RENAMING its C
  identifiers, which the policy now requires of every copy; the table says so
  per row (``symbols: "parent"``), because the older copies do carry their own.

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
        "29ecf5b37490240877b11d464de8df84b7dd95c62ff0999bac3ebea761d5c34e",
    "kernels_flat_pwa_gen_s0e2.cu":
        "8043687a357c70ad053c35d0172e397d409feca14f82e55007593750029baa5e",
    "kernels_flat_pwa_gen_s0e3.cu":
        "479d967a5fc55412d82081019bbccaa44405e01b7dbf4ed904862b4a98007796",
    "kernels_flat_pwa_gen_s0e4.cu":
        "89f7be0181ec9c56461ae568b8ead6d9bbcd88550fd4f474e3817df1751aa22a",
    "kernels_flat_pwa_gen_s0e5.cu":
        "0a7e20c4d1133e984c89276ba8168157bc1f7bd72507976ca284e93a22b704d1",
    "kernels_flat_pwa_gen_s0c.cu":
        "93eb5f8b58bf84420d394c8f98158499d8e00c1fcda36e0fa43b8e668c99add5",
    "kernels_flat_pwa_gen_s0c2.cu":
        "759952f30ab7c26da0aa9e1e386a38db43195240a49e1ffcb2430dc22d674e73",
    "kernels_flat_pwa_gen_s0f.cu":
        "f8e757b06387df98a962211b64c7ac5e155f40069a225bbdf6a4b078655a8e14",
    "kernels_flat_pwa_gen_s0g.cu":
        "9902b65cc19bc50552fa30af17174ee1f6540f209ed95a6ec16e4b0b13164d2b",
    "kernels_flat_pwa_gen_ampcache.cu":
        "ca9b486141dd5b0c7545255556d4ff61c7b9a2b899b13fc227a789b19a8f9297",
    "kernels_flat_pwa_gen_ampcache32.cu":
        "bbc75ef510c24526794bce7bbeac5e66522f0846b945533c53d3d4288dfe58a7",
    # the parent of `cuda_flat_pwa_gen_fslot` (no row of its own: gen2 predates
    # the rule, see NO_ISOLATION) — a copy may not edit what it copies from
    "kernels_flat_pwa_gen2.cu":
        "137391316d581f33ed9e9526bb33c596454716eb73e8c668852eeca5681bfe6f",
    "kernels_flat_pwa_gen_fslot.cu":
        "9439109829826849ef16ecda1fc6665d5ba0b8f3531bd8a2594e025752d92595",
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
    # the FACTOR-LEVEL dedup: a copy of gen2 whose backward sweep runs over
    # distinct (wave, factor) rows instead of wave blocks.  `symbols` is the
    # ROW's declaration of the copy's C API: "parent" = made WITHOUT RENAMING
    # (kernel policy §1), i.e. its library exports the parent's own
    # `cuda_*_flat_pwa_gen2` names and is kept apart by being a separate
    # `dlopen` object.  Omitted = its own names (the older rows).
    "cuda_flat_pwa_gen_fslot": {"parent": "kernels_flat_pwa_gen2.cu",
                                "symbols": "parent"},
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
    "_flat_pwa_gen_fslot": {
        "slot_map": "_flat_pwa_gen2",
    },
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
    # the wave-level ``slot_map`` the `fslot` copy travels with: `gen2` DEFINES
    # it, and `1bw` carries its OWN variant (its body differs from gen2's, so
    # neither is an unguarded copy of the other)
    "_flat_pwa_gen2",
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
    # the wave-level dedup helper; ``fslot`` travels with it VERBATIM and calls
    # it on the reshaped `(wave, factor)` matrix as well
    "slot_map": lambda kc: (
        (np.asarray(kc["gamma_matrix"], np.float64),), {}),
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
    """A standalone copy: own module/``.so``, parent source intact.

    Everything but the row's ``parent`` (and its declared symbol face) is
    DERIVED from the registry name, so the faces of the property cannot
    disagree: the copy does not import the kernel it copies, it loads its own
    ``.so`` — a SEPARATE object from the parent's — a second instance shares
    the one loaded library, and every source involved matches
    ``SOURCE_SHA256``.

    THE SYMBOL FACE IS THE ROW'S.  A copy made WITHOUT RENAMING (the policy's
    form: the new ``.cu`` is the parent's source apart from the change)
    exports the parent's ``cuda_*_flat_pwa_gen2`` names, so the old
    mutual-exclusion pair — "my library has no parent symbol and the parent's
    has none of mine" — is vacuous there and is asserted only for the rows
    that DID take their own names.  What keeps the two apart in either case is
    the OBJECT: ``cffi.ffi.dlopen`` is ``RTLD_NOW | RTLD_LOCAL`` and each
    kernel is its own ``.so``, so identical names cannot be interposed.
    """
    row = ISOLATION[name]
    prefix = prefix_of(name)
    mod = importlib.import_module(module_of(name))
    parent = row["parent"]
    verbatim = row.get("symbols") == "parent"
    parent_prefix = parent[len("kernels_"):-len(".cu")] if parent else None
    mine = "cuda_compute_" + (parent_prefix if verbatim else prefix)
    own_src = "kernels_" + prefix + ".cu"
    if verbatim:
        assert parent is not None, f"{name}: a verbatim face needs a parent"

    # own module: the copy must not import the kernel it copies
    py = os.path.splitext(mod.__file__)[0] + ".py"
    imported = _imported_modules(py)
    if parent is not None:
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
                if verbatim:
                    # the copy's C API IS the parent's: same names, different
                    # object — that is the whole contract, so check the name is
                    # really there rather than that it is absent
                    assert hasattr(lib, theirs)
                else:
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
