"""The compute suite every flat ``pwa_gen`` CUDA backend shares (registry-driven).

These backends implement ONE function — ``load_data(data)`` returning a handle
and ``compute(params, handle, norm=...) -> (Q, grads, P)`` — so the tests for
what that function must DO are written once, parametrised over the REGISTRY:
``backends_to_test()`` derives the list from ``backends_for_model("pwa_gen")``,
so a backend registered later is covered without touching this file.  Nothing
about a variant's internals belongs here.

Shared: Q/P/grads vs the fp64 ``numpy_flat`` reference (both norms, rsize 1 and
17), the grouped objective, ``rsize = 1`` == the per-event NLL, FD gradients,
chunk invariance (multi-chunk == single-chunk, a chunk boundary does not move
the fill), handle independence / freed handle, and the input errors.

The list of backends comes from the REGISTRY and its TAGS (``reference`` /
``delegate`` / ``n_res1`` / ``coeff_pinned`` / ``probe``, declared at
registration) — not from a name list, so a backend registered later is covered
automatically and an exclusion is a capability, never a silent skip.  The
RULE is the pure function ``_select(registry, tags_of)``, so it can be proved
on a synthetic registry; ``coeff_pinned`` (every coefficient declared fixed, so
``grads["coeff"] == 0`` and a moved ``coeff`` raises) is subtracted exactly
like ``reference``/``delegate``: the free-coefficient contract below does not
apply to it, and its own fixture gates the pinned one.  ``probe`` is a
TEST-SIDE STUB (``test_amp_model``'s ``_model_probe``, ``test_config_backend``'s
``_kwarg_probe``) registered model-independently, so it lands in every model's
registry; tagging it is what keeps a stub out of the sweep without writing a
test module's private name into it.

``SPECS`` is fixed at import (a parametrisation must be), but the COMPLETENESS
check is not: ``test_the_live_registry_is_fully_swept`` recomputes the rule
against ``backends_for_model`` AS IT IS NOW and reports anything registered
after this module was imported.  A snapshot taken at import cannot — it
asserts that the suite swept what it happened to see.

Local (the family files): dedup/index pins, CSR, tile/smem, path A/B, the
parent A/B, this family's own-module/`.so`/symbols, registration, and the
sparse-row edge cases.  Which file owns which leaf is ``CONFIG_OWNER`` below.
Tolerances are the ones the per-variant files used: the fp32 cache floor is
ABSOLUTE, so ``_floor_ok`` accepts either the relative or the absolute bound.

``test_flat_compute.py`` imports this module so pytest collects the suite under
``[<backend name>]`` ids.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import (backend_tags, backends_for_model,
                             create_backend)
from _flat_grid import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
#: every fixture with exactly one BW per chain (a `_mb` chain has two)
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]

# ── the fp32 cache floor ────────────────────────────────────────────────
# The one load-time cache is the numerator `amp = fa·fl·num` as float2, so
# agreement with the fp64 `numpy_flat` reference is limited by fp32, not by the
# algebra.  The floor is therefore a MEASURED per-backend number, never a
# family maximum: the dense backends reach 1e-6, the tile family 3e-5 on P.
# For `s0c`/`s0c2` the `_ident`/`_cp` fixtures span the widest P dynamic range,
# so their file also gated the ABSOLUTE error (1e-4) — `abs_ok` records that,
# and `_floor_ok` is used only for those two.
CACHE_FLOOR_ABS = 1e-4
NH = 1e-6

#: Per-backend agreement floors, copied VERBATIM from each variant's own file
#: (HEAD), plus whether that file used `_floor_ok` (rel OR the absolute floor)
#: or a plain `_rel`, and the CONFIG SET its anchor was measured on.  A
#: consolidation may not weaken a gate, and the family genuinely differs:
#:
#:   * `cuda_flat_pwa_gen` / `cuda_flat_cache_pwa_gen` gated at 1e-6 pure
#:     `_rel` on KM+MB only (their files never ran `_ident`/`_cp`).  On the
#:     wider qualifying set these two MEASURE P 2.97e-6 / 3.54e-6 and coeff
#:     4.97e-6 / 4.70e-6 (rsize 1, widest of rs 1/17 x norm None/1.3), i.e. the
#:     1e-6 claim is shape-bound — so the record keeps the shapes it was
#:     measured on instead of inventing a wider claim.
#:   * `cuda_flat_cache_pwa_gen` never compared the raw `coeff` vector in its
#:     anchor; its own coeff gate is PER NAME at 1e-4 (the deduped raw vector
#:     reads 1.8e-6 on KM, which is why 1e-6 would be a NEW, unmet gate).
#:   * the `s0e`/`gen2`/`1bw` family: 2e-5 / P 3e-5 / coeff 1e-4, strict `_rel`.
#:   * `s0c`/`s0c2`: 2e-5 / P 4e-5 / coeff 2e-4, `_floor_ok` (rel OR abs 1e-4).
#:
#:   name -> (rel, p, coeff, abs_ok, cfgs)
FLOORS = {
    "cuda_flat_cache_pwa_gen": (1e-6, 1e-6, 1e-4, False, (CFG_KM, CFG_MB)),
    "cuda_flat_pwa_gen": (1e-6, 1e-6, 1e-6, False, (CFG_KM, CFG_MB)),
    "cuda_flat_pwa_gen2": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_1bw": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0c": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0c2": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e2": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e3": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e4": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    # case-1 copies of s0e2: same floors (FMA re-association + the deduped
    # denominator move the numbers by <= 1 ulp, measured in their own files)
    "cuda_flat_pwa_gen_s0e5": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    # the fixed-wave cache kernel is registered MASK-OFF here: with no mask it
    # runs the PARENT's code path by construction (identity permutation, full
    # CSR, no cache, `n_fix + k` with `n_fix == 0`), so its exact coefficient
    # gradient holds and the family's s0e2 floors apply unchanged.  The MASKED
    # behaviour is the SHARED mask suite, tests/_flat_masked.py (selected by the
    # `fixed_coeff` tag: a fixed wave's columns are exactly 0 by contract, and
    # the strict floors below apply to the masked run too); the mask-off
    # comparison to s0e2 is tests/_flat_masked.py's
    # test_with_no_mask_the_numbers_are_the_parent_s0e2.
    "cuda_flat_pwa_gen_s0f": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    # the UN-STAGED copy of s0f (registered mask-off here, as s0f is): the
    # only cache is the same fp32 `amp`, and the free block's denominator is
    # walked instead of staged — fp64 either way.  MEASURED over QUALIFYING at
    # rsize 1/17 x norm None/1.3: Q 9.2e-7, P 3.4e-6, dck 1.3e-6, dcoeff
    # 4.7e-6 (dnorm 2.9e-7), i.e. the s0e2 numbers; its masked behaviour is the
    # same shared suite (tests/_flat_masked.py).
    "cuda_flat_pwa_gen_s0g": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    # the CPU leaf is fp64 end to end, so its gap to the reference is the
    # Catmull-Rom-vs-linear interpolation difference, not accumulation:
    # MEASURED over QUALIFYING at rsize 1/17 x norm None/1.3 -> Q 9.3e-7,
    # P 3.0e-6, dck 1.4e-6, dcoeff 5.0e-6.  Gated at the s0c numbers
    # (2e-5 / 4e-5 / 2e-4, rel-OR-abs) — a >5x margin on every quantity.
    "cpu_flat_pwa_gen": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    # the CACHED cpu leaf is a case-1 copy of the one above, bit-identical to
    # it (gated exactly, in its own file), so it keeps the same floor.
    "cpu_flat_pwa_gen_cached": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    # the PRODUCT-cache copy of the cached leaf: same floors as the uncached
    # kernel (it is NOT bit-identical to the split cache — its own file
    # reports the <= 16 ulp drift against it)
    "cpu_flat_pwa_gen_cached2": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
}

#: the tag vocabulary — DECLARED at registration in ``src/tabpwa/backends``,
#: never guessed from a name here.
#:   reference — the NumPy oracle the suite compares against
#:   delegate  — wraps another backend instead of implementing the math
#:   n_res1    — REFUSES a multi-BW chain (``n_res != 1`` raises)
#:   fixed_coeff — takes ``fixed_cols`` + ``fixed_values`` and caches the
#:                 columns a fixed wave reads (the mask suite's selector,
#:                 ``tests/_flat_masked.py``)
#:   coeff_pinned — every coefficient is PINNED: the whole per-wave amplitude
#:                 is a load-time cache, ``grads["coeff"] == 0`` by contract and
#:                 a compute whose ``coeff`` moved RAISES.  Subtracted from this
#:                 suite like ``reference``/``delegate``: the free-coefficient
#:                 contract this file tests (FD on ``coeff``, the mask-off
#:                 anchor, an arbitrary ``params["coeff"]``) is exactly what a
#:                 pinned kernel refuses to do — its own file gates the pinned
#:                 contract.
#:   probe     — a TEST-SIDE STUB, not a backend: ``test_amp_model``'s
#:                 ``_model_probe`` and ``test_config_backend``'s
#:                 ``_kwarg_probe`` declare one at module level to exercise
#:                 ``create_backend``'s kwarg/model injection.  A declared
#:                 capability, not a name prefix: a stub that leaked into this
#:                 suite would be swept as if it computed, and excluding it by
#:                 name here would put a test module's private name into the
#:                 suite's rule.
TAG_VOCABULARY = frozenset({"reference", "delegate", "n_res1", "fixed_coeff",
                            "coeff_pinned", "probe"})

#: tags that REMOVE a name from this suite — by capability, never by name.
#: `reference`/`delegate`/`coeff_pinned` are not this suite's contract (their
#: own fixtures gate them) and `probe` is not a backend at all.
EXCLUDED_TAGS = frozenset({"reference", "delegate", "coeff_pinned", "probe"})


def _select(registry, tags_of):
    """THE RULE, on any registry: names in, ``(tested, excluded)`` out.

    *registry* is a collection of backend names and *tags_of(name)* returns
    that name's declared tags — so the rule is a PURE function of the two, and
    can be exercised on a synthetic registry (see
    ``test_the_rule_excludes_a_probe_and_reports_a_late_backend``) without
    touching the process-wide one.

    It reads the registry it is GIVEN, never a snapshot: a backend registered
    after this module was imported is either in the result or absent from it,
    and the caller's ledger (``_missing_from_suite``) then says so out loud.
    That is the whole point — a frozen set cannot notice anything.
    """
    names = set(registry)
    excluded = {n for n in names if frozenset(tags_of(n)) & EXCLUDED_TAGS}
    return sorted(names - excluded), sorted(excluded)


def _missing_from_suite(specs, registry, tags_of):
    """Testable leaves the registry holds that *specs* does not cover.

    *specs* is what the suite actually swept (fixed when the parametrisation
    was built).  Empty means complete; anything in it was registered after that
    point and is therefore NOT tested — the condition that must be reported
    rather than silently tolerated.
    """
    tested, _ = _select(registry, tags_of)
    return sorted(set(tested) - set(specs))


def _load_test_module(mod):
    """Import a sibling test module BY PATH, whatever pytest called it.

    ``tests/`` is a package, so the same file is ``test_x`` under one rootdir
    and ``tests.test_x`` under another.  Skip if either name is already loaded;
    otherwise load the file and publish it under its bare name (a second
    execution would only re-define the same names).
    """
    import importlib.util
    import os
    import sys

    for key in (mod, f"tests.{mod}"):
        if key in sys.modules:
            return sys.modules[key]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mod + ".py")
    spec = importlib.util.spec_from_file_location(mod, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod, None)
        raise
    return module


def backends_to_test():
    """Every registered ``pwa_gen`` backend that IMPLEMENTS the function.

    Derived from the registry and its capability tags: references (the oracle),
    delegates (a forwarding wrapper — ``auto``/``shard`` — whose own tests pin
    the forwarding), PINNED-coefficient kernels (``coeff_pinned``: their
    ``coeff`` cannot move, so the free-coefficient contract below — FD on
    ``coeff``, the mask-off anchor, an arbitrary ``params["coeff"]`` — does not
    apply to them) and test-side stubs (``probe``) are subtracted, so no name
    is written out here and a backend registered later is covered
    automatically.
    """
    reg = set(backends_for_model("pwa_gen"))

    def tags_of(name):
        return backend_tags(name, "pwa_gen")

    leaves, _excluded = _select(reg, tags_of)
    # per-tag views, asserted below: an exclusion must be a DECLARED capability
    ref = set(backends_for_model("pwa_gen", tag="reference"))
    deleg = set(backends_for_model("pwa_gen", tag="delegate"))
    pinned = set(backends_for_model("pwa_gen", tag="coeff_pinned"))
    probe = set(backends_for_model("pwa_gen", tag="probe"))
    assert ref and deleg, "the tag vocabulary is not declared at registration"
    assert ref <= reg and deleg <= reg and pinned <= reg and probe <= reg
    # a pinned-coefficient kernel computes, it is not an oracle and it does not
    # forward: those tags are mutually exclusive, so a subtraction order can
    # never decide whether a backend is tested
    assert not pinned & (ref | deleg), sorted(pinned & (ref | deleg))
    assert set(leaves) | ref | deleg | pinned | probe == reg, (
        "registry names neither tested nor tagged: "
        f"{sorted(reg - set(leaves) - ref - deleg - pinned - probe)}")
    assert not ref & deleg
    unknown = {t for n in reg for t in tags_of(n)} - TAG_VOCABULARY
    assert not unknown, f"undeclared tags in the registry: {sorted(unknown)}"
    nres1 = set(backends_for_model("pwa_gen", tag="n_res1"))
    assert nres1 <= set(leaves) | pinned, sorted(nres1 - (set(leaves) | pinned))
    return leaves


#: the registry loop — the list is DERIVED, never written out by hand.
#: Fixed at import because a parametrisation is; the COMPLETENESS check is not
#: (``test_the_live_registry_is_fully_swept`` recomputes it live), so a backend
#: registered after this module is imported fails that check instead of falling
#: silently outside the suite.
SPECS = backends_to_test()
#: WHICH MODULE OWNS EACH LEAF'S CONFIG CONTRACT.  The compute suite above is
#: registry-driven, so nothing has to be written down for it; the config
#: contract (accepted layouts, refusals, the layout/objective A/Bs) is
#: partitioned across the family files, and this map is the rule they check
#: themselves against.  A leaf registered later lands in NO value here, so the
#: equality below FAILS instead of the leaf being silently uncovered by every
#: table — the same round-trip ``_flat_masked``/``_flat_pinned`` apply by tag.
CONFIG_OWNER = {
    "cuda_flat_pwa_gen": "test_flat_dedup.py",
    "cuda_flat_cache_pwa_gen": "test_flat_dedup.py",
    "cuda_flat_pwa_gen2": "test_flat_dedup.py",
    "cuda_flat_pwa_gen_1bw": "test_flat_dedup.py",
    "cuda_flat_pwa_gen_s0c": "test_flat_dedup.py",
    "cuda_flat_pwa_gen_s0c2": "test_flat_dedup.py",
    "cuda_flat_pwa_gen_s0e": "test_flat_tile.py",
    "cuda_flat_pwa_gen_s0e2": "test_flat_tile.py",
    "cuda_flat_pwa_gen_s0e3": "test_flat_tile.py",
    "cuda_flat_pwa_gen_s0e4": "test_flat_tile.py",
    "cuda_flat_pwa_gen_s0e5": "test_flat_tile.py",
    "cuda_flat_pwa_gen_s0f": "_flat_masked.py",
    "cuda_flat_pwa_gen_s0g": "_flat_masked.py",
    "cuda_flat_pwa_gen_ampcache": "_flat_pinned.py",
    "cuda_flat_pwa_gen_ampcache32": "_flat_pinned.py",
    "cpu_flat_pwa_gen": "test_cpu_flat_pwa_gen.py",
    "cpu_flat_pwa_gen_cached": "test_cpu_flat_pwa_gen_cached.py",
    "cpu_flat_pwa_gen_cached2": "test_cpu_flat_pwa_gen_cached2.py",
}
_ALL_LEAVES = set(SPECS) | set(backends_for_model("pwa_gen", tag="coeff_pinned"))
assert set(CONFIG_OWNER) == _ALL_LEAVES, (
    "every flat leaf needs a CONFIG contract owner (a register-only addition "
    "must fail here, not vanish from every family table): missing "
    f"{sorted(_ALL_LEAVES - set(CONFIG_OWNER))}, stale "
    f"{sorted(set(CONFIG_OWNER) - _ALL_LEAVES)}")



#: the multi-BW guard, by tag, so the skip names the capability
N_RES1 = frozenset(backends_for_model("pwa_gen", tag="n_res1"))
#: PINNED-coefficient kernels are subtracted from SPECS (their own fixture
#: gates the free-coefficient contract) but the multi-BW guard is a SHAPE
#: question, not a contract one: they take a multi-BW chain and are swept by
#: `test_multi_bw_chain_is_rejected_or_accepted` too.
PINNED = frozenset(backends_for_model("pwa_gen", tag="coeff_pinned"))
MULTI_BW_SWEEP = sorted(set(SPECS) | PINNED)


def _floor_ok(a, b, rel_bound, abs_bound=CACHE_FLOOR_ABS):
    """The fp32 floor is ABSOLUTE; ``_rel`` divides by the value.

    Accepting either bound keeps the gate meaningful for a near-zero Q without
    loosening it for the ordinary ones (the absolute bound IS the measured
    shared-cache gap).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    worst = float(np.abs(a - b).max())
    return worst / max(np.abs(b).max(), 1e-300) < rel_bound or worst < abs_bound


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _pid_alive(pid):
    """Whether ``pid`` exists right now (``PermissionError`` means it does)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:                 # another user's process: it exists
        return True
    except OSError:
        return False
    return True


def other_gpu_users():
    """PIDs holding the device right now, excluding this process.

    ``free_bytes()`` is DEVICE-WIDE: any other process allocating on the card
    moves the number inside a before/after trace, so a byte-level memory gate
    would be measuring someone else's work — that is not a wider tolerance to
    invent, it is a precondition that does not hold.  ``cudaMemGetInfo`` cannot
    attribute memory to a PID, so the honest check is "no one else is on the
    card".

    ``nvidia-smi --query-compute-apps`` can list a PID whose process has already
    exited (MEASURED: a call returned a pid that ``ps`` did not know, and ``[]``
    seconds later), so every entry is checked with ``os.kill(pid, 0)`` before it
    is reported.  A GHOST is not a user: skipping a valid gate because of one —
    or failing it — is the false signal this precondition exists to avoid.  A
    real process may still appear AFTER the query; a gate therefore re-queries
    after its measurement and treats a changed set as an invalid window rather
    than asserting on it.

    The query is a PRECONDITION, never the oracle: when ``nvidia-smi`` or
    ``--query-compute-apps`` is unavailable this returns ``[]`` and the gate
    RUNS — a missing tool must not silently skip a leak gate.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mine = os.getpid()
    pids = set()
    for line in out.splitlines():
        cell = line.split(",")[0].strip()
        try:
            pid = int(cell)
        except ValueError:                  # blank, [N/A], a header
            continue
        if pid != mine and _pid_alive(pid):
            pids.add(pid)
    return sorted(pids)


def _gate(name, a, b, which="rel", cfg=None):
    """The floor for THAT backend on THAT fixture class.

    Inside the configs its own file measured, the backend's own bound applies
    (the dense ones reach 1e-6, the tile family 3e-5 — one family-wide number
    would loosen some of them).  Outside that set, the family's measured
    wide-range floor applies, so the case is still GATED rather than skipped.
    """
    rel, p, coeff, abs_ok, cfgs = FLOORS[name]
    if cfg is not None and cfg not in cfgs:
        return _rel(a, b) < WIDE[which]
    # "ck" shares the generic rel bound; WIDE carries it separately
    bound = {"rel": rel, "p": p, "ck": rel, "coeff": coeff}[which]
    return _floor_ok(a, b, bound) if abs_ok else _rel(a, b) < bound


def _cfgs(name):
    """The config set THIS backend's own anchor was measured on."""
    return FLOORS[name][4]


#: On a fixture OUTSIDE a backend's recorded set the gate is the family's own
#: measured wide-range floor, never a skip: dropping the case would test the
#: backend on nothing at all, and lifting its 1e-6 there would be a gate that
#: cannot pass.  MEASURED over the whole family on ``_ident``/``_cp``,
#: rsize 1/17, worst of norm 1.3 and norm None: Q 9.3e-7, P 3.5e-6,
#: ck 1.4e-6, coeff 5.0e-6 — the 2-5x margins below.  Q and ck are the
#: norm-dependent pair (norm=None is the raw sum, so its gradient is not
#: divided by the normalisation).
WIDE = {"rel": 2e-6, "p": 1e-5, "ck": 5e-6, "coeff": 2e-5}
#: The grouped-objective scenario (34 events, seed 5, rsize 17, un-normalised
#: weights) is its own: MEASURED worst 1.3e-6 across the family -> 4x margin.
GROUPED = 5e-6


assert set(FLOORS) == set(SPECS), (
    "every registered leaf needs its own measured floor - a missing entry "
    f"must not inherit another backend's: {sorted(set(SPECS) - set(FLOORS))}")


def _replicate(data, rs):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def test_backend_tags_are_declared_metadata():
    """Tags are DECLARED at registration — the suite selects on them."""
    assert backend_tags("numpy_flat", "pwa_gen") == frozenset({"reference"})
    assert backend_tags("cuda_flat_pwa_gen_s0c2", "pwa_gen") \
        == frozenset({"n_res1"})
    assert backend_tags("auto", "pwa_gen") == frozenset({"delegate"})
    # a pinned-coefficient kernel is a LEAF with its own contract, never an
    # oracle and never a forwarding wrapper.  It carries NO `n_res1`: the
    # denominator is the product over the factor axis, so a multi-BW chain is
    # computed by it like any other leaf (the sweep below covers it).
    assert backend_tags("cuda_flat_pwa_gen_ampcache", "pwa_gen") \
        == frozenset({"coeff_pinned"})
    # an untagged backend returns an empty frozenset, not None
    assert backend_tags("cuda_v4_pwa", "pwa") == frozenset()


def test_the_pinned_tag_is_out_of_this_suite_and_nobody_else_is():
    """``coeff_pinned`` is subtracted here — and it is the ONLY such backend.

    The pinned tag means "``coeff`` cannot move", which is the opposite of this
    file's free-coefficient contract, so those kernels are gated by their own
    fixture.  Anything else that stops appearing here would be a silent skip,
    which is why the exclusion is asserted by NAME through its tag rather than
    trusted to the subtraction.
    """
    # the LIVE registry, read now: a snapshot taken at import is what let a
    # backend registered later fall outside the suite unnoticed
    reg = set(backends_for_model("pwa_gen"))
    pinned = set(backends_for_model("pwa_gen", tag="coeff_pinned"))
    assert "coeff_pinned" in TAG_VOCABULARY
    assert pinned and pinned <= reg
    assert not pinned & set(backends_for_model("pwa_gen", tag="reference"))
    assert not pinned & set(backends_for_model("pwa_gen", tag="delegate"))
    assert not pinned & set(SPECS)          # subtracted, not tested here
    assert set(SPECS) == reg - set(backends_for_model("pwa_gen",
                                                     tag="reference")) \
        - set(backends_for_model("pwa_gen", tag="delegate")) - pinned \
        - set(backends_for_model("pwa_gen", tag="probe"))


def test_the_live_registry_is_fully_swept():
    """Completeness against the REGISTRY AS IT IS NOW, not at import.

    ``SPECS`` is frozen when the parametrisation is built (a parametrisation
    must be).  If a backend registers afterwards — a new module imported later
    in the session, a plugin, a future leaf — that frozen list cannot contain
    it, and the only honest answer is to SAY SO.  Comparing against the live
    rule is what makes the two cases distinguishable:

      * registered before this module was imported -> already in ``SPECS``;
      * registered after -> ``_missing_from_suite`` names it, and this fails.

    A snapshot (``_REG_AT_IMPORT``, deleted) made the check pass in both cases,
    i.e. it asserted that the suite had swept what it happened to see.
    """
    registry = set(backends_for_model("pwa_gen"))

    def tags_of(name):
        return backend_tags(name, "pwa_gen")

    expected, excluded = _select(registry, tags_of)
    assert set(SPECS) == set(expected), sorted(set(expected) ^ set(SPECS))
    assert _missing_from_suite(SPECS, registry, tags_of) == [], (
        "registered after this module was imported and therefore NOT swept: "
        f"{_missing_from_suite(SPECS, registry, tags_of)}")
    # the subtraction is by CAPABILITY: every excluded name carries one
    # (a `probe` stub is excluded too, and it is not a backend at all)
    assert set(excluded) == registry - set(expected)
    assert all(set(tags_of(n)) & EXCLUDED_TAGS for n in excluded)


def test_the_probe_tag_is_a_declared_capability():
    """A test-side stub is TAGGED, not excluded by a name prefix.

    ``test_amp_model`` and ``test_config_backend`` declare stubs at import to
    exercise ``create_backend``'s kwarg/model injection.  They are
    model-independent, so they land in EVERY model's registry — including
    ``pwa_gen``'s — and an untagged one is indistinguishable from a real leaf.

    Loading the two modules is the whole point of the case: they are imported
    AFTER ``SPECS`` was computed (a sibling file's module object), which is
    exactly the registry state that a frozen snapshot could not describe.
    """
    amps = _load_test_module("test_amp_model")
    cfgs = _load_test_module("test_config_backend")
    assert "probe" in TAG_VOCABULARY
    for name in (amps.MODEL_PROBE, cfgs.KWARG_PROBE):
        assert backend_tags(name, "pwa_gen") == frozenset({"probe"}), name
        assert name not in SPECS, name
        assert name not in backends_to_test(), name
    # and the LIVE completeness check tolerates them: excluded by capability,
    # not forgotten — a stub registered after the import is the exact case
    # that must not read as "missing"
    registry = set(backends_for_model("pwa_gen"))
    assert _missing_from_suite(SPECS, registry,
                               lambda n: backend_tags(n, "pwa_gen")) == []


def test_the_rule_excludes_a_probe_and_reports_a_late_backend():
    """THE PROPERTY, proved on a synthetic registry — not asserted about ours.

    Two failures are possible when a frozen snapshot replaces the live rule,
    and each is shown here on a registry built by hand:

      (a) a ``probe`` stub is swept as if it were a backend (not excluded);
      (b) a backend registered LATER is silently outside the suite (not
          reported).

    The rule is a pure function of ``(registry, tags_of)``, so this needs no
    registration, no CUDA, and no kernel config.
    """
    tags = {
        "reference_backend": {"reference"},
        "delegate_backend": {"delegate"},
        "pinned_kernel": {"coeff_pinned"},
        "_probe_stub": {"probe"},
        "a_kernel": set(),          # swept at import
        "late_backend": set(),      # registered AFTER import, untagged
    }
    registry = set(tags)
    specs_at_import = ["a_kernel"]          # what the suite actually swept

    tested, excluded = _select(registry, tags.__getitem__)
    # (a) the probe is excluded BY ITS TAG, not by being named "_probe_stub"
    assert "_probe_stub" in excluded
    assert "_probe_stub" not in tested
    # the three not-this-suite capabilities are excluded the same way
    assert excluded == ["_probe_stub", "delegate_backend", "pinned_kernel",
                        "reference_backend"]
    # (b) the late untagged backend IS a candidate — the rule does not hide it
    assert tested == ["a_kernel", "late_backend"]
    # ... and against what was swept it is REPORTED, which is the whole point
    assert _missing_from_suite(specs_at_import, registry, tags.__getitem__) \
        == ["late_backend"]
    # once swept, the report is empty (and a probe was never a candidate, so
    # tagging it excluded is not the same as "forgetting" it)
    assert _missing_from_suite(specs_at_import + ["late_backend"], registry,
                               tags.__getitem__) == []


@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_anchor_matches_numpy_flat(name, cfg, norm, rs):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    # the sibling fixtures' anchor shape (64 events, seed 0), so the floors
    # below are the same measurements they were calibrated on
    base = _in_grid(kc, 64, 0)
    data = _replicate(base, rs) if rs > 1 else base
    params, _ = f.build_params(f.initial_values(seed=1))
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=rs)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _gate(name, Pg, Pr, "p", cfg=cfg), "P"
        assert _gate(name, Qg, Qr, cfg=cfg), "Q"
        assert _gate(name, gg["ck"], gr["ck"], "ck", cfg=cfg), "dck"
        assert _gate(name, gg["coeff"], gr["coeff"], "coeff", cfg=cfg), (
            "dcoeff")
        if norm is not None:
            assert _gate(name, gpu.kernel._last_dnorm, ref._last_dnorm,
                         cfg=cfg), "dnorm"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("name", SPECS)
def test_one_group_per_event_reduces_to_the_per_event_nll(name, cfg):
    """``rsize = 1`` is the per-event log, so it must match ``rsize = None``.

    Also pins that the GROUPED objective is the flat family's group-TOTAL
    weight form (`-W_g·log(S_g/W_g)`, `numpy_flat`'s), not s0c's plain
    `-log(Pb/norm + Bb)` — the two differ by ``W_g`` whenever a group's weights
    do not sum to 1.  Un-normalised weights make that difference visible.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 34, 5)              # 34 events: a short final group
    params, _ = f.build_params(f.initial_values(seed=1))
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=17)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=17)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, _gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < GROUPED, "P"
        assert _rel(Qg, Qr) < GROUPED, "Q"
    finally:
        ref.free()
        gpu.free()
    # and with group-NORMALISED weights both forms coincide, which is what
    # makes the per-event value a valid cross-check of the grouped one
    d = _replicate(data, 1)
    w = np.asarray(d["weight"], float)
    n = (w.size // 17) * 17
    wg = w[:n].reshape(-1, 17)
    d["weight"] = w.copy()
    d["weight"][:n] = (wg / wg.sum(axis=1, keepdims=True)).reshape(-1)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=17)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=17)
    try:
        hr = ref.load_data(d)
        hg = gpu.load_data(d)
        Qr = ref.compute(params, hr, norm=1.3)[0]
        Qg = gpu.compute(params, hg, norm=1.3)[0]
        assert _rel(Qg, Qr) < GROUPED, "Q (normalised weights)"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_IDENT, CFG_CP])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_coeff_gradient_matches_central_differences(name, cfg, rs):
    """``dcoeff`` must be ∂Q/∂coeff of THIS kernel's Q, per basis row.

    This is the gate that catches a wrong CSR or a wrong m-major
    transposition: a map that included the wrong columns would still be
    repeatable (both computes wrong the same way), so only an independent
    derivative can catch it.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, 48)
    data = _replicate(base, rs) if rs > 1 else base
    gpu = create_backend(name, kc, resolution_size=rs, model="pwa_gen")
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        coeff = np.asarray(params["coeff"], np.float64).copy()
        for i in range(coeff.size):
            up, dn = coeff.copy(), coeff.copy()
            up[i] += NH
            dn[i] -= NH
            pu, pn = dict(params), dict(params)
            pu["coeff"], pn["coeff"] = up, dn
            fd = (gpu.compute(pu, h, norm=None)[0]
                  - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                f"{cfg} rs={rs} row {i}: kernel {g0['coeff'][i]!r} "
                f"vs FD {fd!r}")
    finally:
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_CP])
@pytest.mark.parametrize("name", SPECS)
def test_ck_gradient_matches_fd(name, cfg):
    """``dck`` after the u-merge — the same independent check."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 48)
    gpu = create_backend(name, kc, model="pwa_gen")
    h = gpu.load_data(data)
    try:
        ck = np.asarray(params["ck"], np.complex128).copy()
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        for i in range(ck.size):
            for part in (0, 1):
                d = np.zeros_like(ck)
                d[i] = NH if part == 0 else 1j * NH
                pu, pn = dict(params), dict(params)
                pu["ck"], pn["ck"] = ck + d, ck - d
                fd = (gpu.compute(pu, h, norm=None)[0]
                      - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
                # the kernel's dck is the raw Wirtinger half; the real
                # parameter convention downstream is 2·Re / -2·Im
                dres = (2.0 * np.real(g0["ck"][i]) if part == 0
                        else -2.0 * np.imag(g0["ck"][i]))
                assert np.isclose(dres, fd, rtol=1e-4, atol=1e-8), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        gpu.free()


@pytest.mark.parametrize("name", SPECS)
def test_a_multi_chunk_call_equals_a_single_chunk_call(name, ):
    """Batching is an inner loop: the numbers must not know about it.

    Exercised with and without the grouped objective, and at batch sizes that
    straddle both guards (1/5 are below one group at rsize 17; 1e9 is capped).
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 500, 3)

    def run(rs, bs):
        be = create_backend(name, kc, resolution_size=rs, batch_size=bs, model="pwa_gen")
        try:
            h = be.load_data(data)
            Q, g, P = be.compute(params, h, norm=1.3)
            h.free()
            return Q, g["ck"], g["coeff"], float(g["norm"]), np.asarray(P)
        finally:
            be.free()

    for rs in (1, 17):
        base = run(rs, 10 ** 9)             # one chunk
        for bs in (1, 5, 1000, 5000, 50000):
            got = run(rs, bs)
            for label, a, b in (("Q", got[0], base[0]),
                               ("dck", got[1], base[1]),
                               ("dcoeff", got[2], base[2]),
                               ("dnorm", got[3], base[3]),
                               ("P", got[4], base[4])):
                rel = _rel(a, b)
                assert rel <= 1e-14, f"rsize={rs} batch={bs}: {label} {rel:.2e}"


@pytest.mark.parametrize("name", SPECS)
def test_a_chunk_boundary_does_not_move_the_fill(name, ):
    """The load-time fill is chunked too — so a chunk must read ITS events.

    Regression gate: the fill's `event_idx` is chunk-local, so passing the
    sample's first events to every chunk leaves the cache of every chunk after
    the first computed from the wrong masses (measured: Q off by 8.5 on 16
    events at batch_size 1).  Pinned by a sample much longer than the chunk.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 37, 11)
    k = create_backend(name, kc, batch_size=4, model="pwa_gen")
    h = k.load_data(data)
    k2 = create_backend(name, kc, batch_size=10 ** 6, model="pwa_gen")
    h2 = k2.load_data(data)
    try:
        Qa, _ga, Pa = k.compute(params, h, norm=1.3)
        Qb, _gb, Pb = k2.compute(params, h2, norm=1.3)
        assert _rel(Pa, Pb) <= 1e-14
        assert abs(Qa - Qb) <= 1e-12 * max(abs(Qb), 1.0)
        # a per-event quantity: the LAST events prove the fill ran per chunk
        assert abs(float(Pa[-1]) - float(Pb[-1])) <= 1e-13 * max(
            abs(float(Pb[-1])), 1.0)
    finally:
        k.free()
        k2.free()


@pytest.mark.parametrize("name", SPECS)
def test_two_live_handles_are_independent(name, ):
    """The fitter holds phsp AND data at once — each load owns its arrays."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    params, _ = f.build_params(f.initial_values(seed=1))
    a = _in_grid(kc, 16, 1)
    b = _in_grid(kc, 16, 2)
    try:
        h1 = k.load_data(a)
        h2 = k.load_data(b)
        assert h1.ptr != h2.ptr
        Qa1, _g, Pa = k.compute(params, h1, norm=None)
        Qb, _g, Pb = k.compute(params, h2, norm=None)
        Qa2 = k.compute(params, h1, norm=None)[0]
        # h2's load did not clobber h1's
        assert abs(Qa1 - Qa2) <= 1e-12 * max(abs(Qa1), 1.0)
        assert _rel(Pa, Pb) > 1e-6            # ... and they are different data
        h1.free()
        h2.free()
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_a_freed_handle_is_reported_not_crashed(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    h = k.load_data(_in_grid(kc, 8))
    params, _ = f.build_params(f.initial_values(seed=1))
    try:
        h.free()
        h.free()                              # idempotent
        with pytest.raises((RuntimeError, ValueError),
                           match="already freed"):
            k.compute(params, h, norm=None)
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_rejects_an_already_deduped_config(name, ):
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        create_backend(name, dd, model="pwa_gen")


@pytest.mark.parametrize("name", SPECS)
def test_wrong_parameter_lengths_are_rejected(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    h = k.load_data(_in_grid(kc, 8))
    params, _ = f.build_params(f.initial_values(seed=1))
    try:
        bad = dict(params)
        bad["ck"] = np.zeros(k.kernel.n_wave + 1, complex)
        with pytest.raises(ValueError, match="n_wave"):
            k.compute(bad, h, norm=None)
        bad = dict(params)
        bad["coeff"] = np.zeros(k.kernel.n_basis + 1)
        with pytest.raises(ValueError, match="n_basis"):
            k.compute(bad, h, norm=None)
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_resolution_size_below_one_is_rejected(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        create_backend(name, f.kernel_config, resolution_size=0, model="pwa_gen")

@pytest.mark.parametrize("name", SPECS)
def test_resolution_size_one_is_the_per_event_nll(name, ):
    """`rsize = 1` is the default AND the per-event objective."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 48, 4)
    params, _ = f.build_params(f.initial_values(seed=1))
    dflt = create_backend(name, kc, model="pwa_gen")
    one = create_backend(name, kc, model="pwa_gen", resolution_size=1)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=1)
    try:
        Qa, _ga, Pa = dflt.compute(params, dflt.load_data(data), norm=1.3)
        Qb, _gb, Pb = one.compute(params, one.load_data(data), norm=1.3)
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _rel(Qa, Qb) <= 1e-14
        assert _rel(Pa, Pb) <= 1e-14
        assert _gate(name, Qb, Qr) and _gate(name, Pb, Pr, "p")
    finally:
        dflt.free()
        one.free()
        ref.free()


@pytest.mark.parametrize("name", SPECS)
def test_norm_none_and_a_scalar_norm_are_different_objectives(name, ):
    """`norm=None` is the raw sum; `norm=v` is the normalised NLL."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 48, 6)
    params, _ = f.build_params(f.initial_values(seed=1))
    be = create_backend(name, kc, model="pwa_gen")
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    try:
        h = be.load_data(data)
        hr = ref.load_data(data)
        Qn = be.compute(params, h, norm=None)[0]
        Qv = be.compute(params, h, norm=1.3)[0]
        Rn = ref.compute(params, hr, norm=None)[0]
        Rv = ref.compute(params, hr, norm=1.3)[0]
        assert _gate(name, Qn, Rn) and _gate(name, Qv, Rv)
        assert abs(Qn - Qv) > 1e-9 * max(abs(Qn), 1.0)
    finally:
        be.free()
        ref.free()


@pytest.mark.parametrize("name", MULTI_BW_SWEEP)
def test_multi_bw_chain_is_rejected_or_accepted(name, ):
    """`n_res1`-tagged backends guard it; every other leaf takes the chain.

    A PINNED-coefficient kernel is swept here as well: the multi-BW question is
    about the shape of the denominator (a product over the factor axis), which
    the pinned cache computes like any other leaf.
    """
    if name in N_RES1:
        pytest.skip(f"{name}: tagged n_res1 — its wrapper REFUSES a multi-BW "
                    f"chain (n_res != 1 raises, naming cuda_flat_pwa_gen2); "
                    f"that refusal is gated in the file that owns its config "
                    f"contract, {CONFIG_OWNER[name]}, not here")
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    create_backend(name, f.kernel_config, model="pwa_gen").free()
