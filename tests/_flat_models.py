"""The two AXES of the flat ``pwa_gen`` suite, in one place.

The suite must cover two different things, and multiplying them is a mistake
(``|A| x leaves + |B| x 2`` (declaration + FD), never ``models x leaves``):

    A  kc STRUCTURE  x  every pwa_gen leaf
       one fixture per mechanism the BACKEND code can distinguish (block
       rebasing, the factor product, a wide basis, a unit denominator, ...),
       swept over every leaf by ``_flat_compute.py`` + the family files.

    B  every particle MODEL, as a UNIT OF A CONFIG
       what a model DECLARES (its row names, its heads, its expansion) is
       checked once with NO backend at all, then its numbers go through the
       family's reference path (``Fitter.get_nll`` -> ``numpy_flat``) and its
       gradient is checked by central differences there.  No "typical
       backends" — a model is not a backend case — and no FIXTURE either: each
       entry is rendered into ``TEMPLATE``, a minimal in-memory config, and
       ``test_the_template_renders_the_fixture_for_the_default_basis`` pins
       that the default basis renders the same kernel config as
       ``tests/config_pwa_gen.yml`` key by key.

Why not the cross product: a leaf never sees a model, only the kernel config
``build_from_groups`` produced (``create_backend(name, kernel_config, model=)``
takes the kc and the model NAME, and the kernels read arrays).  Two models that
yield the same structure therefore exercise the SAME backend code on the same
shapes and the same kind of numbers — a second copy of the test, not a second
case.  A structure no model can produce is still worth a leaf case, which is
exactly axis A.  So the suite costs ``|A| x leaves + |B| x 3`` instead of
``models x leaves``.

Two things this file does NOT do, both deliberate:

* It does not move the model-only fixtures (``_km``, ``_cidx``) out of the leaf
  sweep.  Their per-leaf ``FLOORS`` in ``_flat_compute.py`` were MEASURED on
  those exact config sets, so re-pointing the sweep is a re-measurement of
  every floor — a separate change with its own evidence, not a file move.
  What is enforced here is the boundary: a NEW model gets a ``MODELS`` entry
  (no fixture), and ``AXIS_A`` must keep stating the mechanism each remaining
  fixture contributes (``test_axis_a_is_one_fixture_per_kc_class``).
* It does not skip the parity case for a model whose kc class axis A already
  sweeps (most of them: the 3-row default basis).  That case is not a second
  leaf-correctness claim — the leaf's claim is axis A's — it is a
  CROSS-VALIDATION of the model's own recipe (transform -> basis rows -> kc)
  against the fp64 reference on a real leaf: a wrong transform shows up as a
  non-finite or displaced P, which no declaration check can see.

The live gate is ``test_flat_models.py::test_every_registered_model_is_covered``:
a particle model registered later is neither swept nor silently ignored — it
must be named here, as a ``MODELS`` entry or in ``EXCLUDED`` with the reason
AND the place it is covered instead (the ``CONFIG_OWNER`` pattern).

Measured facts about the axis boundary (``kc_class`` below), so a future
fixture cannot be added "because it is a new model":

    tests/config_pwa_gen.yml            (1,1,3,1)    the default basis
    tests/config_pwa_gen_km.yml         (2,1,5,2)    a 5-row denominator + 2 heads
    tests/config_pwa_gen_ident.yml      (4,1,10,4)   identical-particle blocks
    tests/config_pwa_gen_same.yml       (4,1,10,4)   SAME signature as _ident:
                                                     two chains sharing one
                                                     resonance (different mass
                                                     columns), a different
                                                     mechanism with the same
                                                     shape
    tests/config_pwa_gen_cp.yml         (8,1,20,8)   identical + CP blocks
    tests/config_pwa_gen_mb.yml         (4,2,8,4)    the factor product
    tests/config_pwa_gen_mb2.yml        (12,2,12,12) factor product x topologies
    tests/config_pwa_gen_4pi_mb.yml     (32,2,40,32) blocks x factor product
    tests/config_pwa_gen_cidx.yml       (4,1,1,4)    unit denominator, 4 heads
    tests/config_pwa_gen_cidx_2f.yml    (6,2,4,6)    unit denominator in a
                                                     factor product
"""
import copy
from typing import NamedTuple

import numpy as np

#: The TEMPLATE every ``MODELS`` entry is rendered into — a minimal in-memory
#: ``pwa_gen`` config (three finals, one two-body sub-decay), NOT a fixture
#: file: a model is a unit of a config, so all it needs is a topology and the
#: resonance slot.  The masses are the ones ``config_pwa_gen.yml`` uses, so the
#: kernel config this renders for the DEFAULT basis is identical to that
#: fixture's (``test_the_template_renders_the_fixture_for_the_default_basis``
#: pins it key by key) — the fixture stays as the CROSS-CHECK, never the source.
#: NOTE: no `n_proj` key.  On this path the projection count is DERIVED from
#: the decay's spin structure — `len(_helicity_external_states())`, i.e.
#: ``(2J_top+1) x PROD_finals (2J_f+1)`` (spin-averaged finals), which
#: `_pg_host_records` puts in `kc["n_proj"]` and `matrix_angle` carries as
#: `(n_angle_basis, n_proj·n_wave)` p-major.  An explicit `n_proj:` is
#: IGNORED here (measured: declaring 1 or 3 on this template both give 3 —
#: every final is spin-0, so the product IS the top's multiplicity); the key
#: only means something on the plain `pwa` path (`BaseModel._compute_n_proj`).
TEMPLATE = {
    "amp_model": "pwa_gen",
    "data": {"dat_order": ["pip", "pim", "eta"]},
    "decay": {"top": [["res", "eta"]], "res": ["pip", "pim"]},
    "particle": {
        "$top": "top",
        "$finals": ["pip", "pim", "eta"],
        "res": ["k2"],
        "top": {"J": 1, "P": -1, "mass": 3.096813522559038},
        "pip": {"J": 0, "P": -1, "mass": 0.139567},
        "pim": {"J": 0, "P": -1, "mass": 0.139567},
        "eta": {"J": 0, "P": -1, "mass": 0.5478621409593663},
        "k2": {"J": 1, "P": -1, "mass": 0.80, "width": 0.30,
               "model": "basis"},
    },
}

#: The resonance every ``MODELS`` entry replaces.
RESONANCE = "k2"

class Fixture(NamedTuple):
    """A flat config fixture: what it REQUIRES of a leaf, and what it adds.

    ``needs`` is the ONLY selector.  A leaf declares what it refuses as a
    capability tag (``n_res1``), so the fixture states the fact that meets that
    tag; the sweep is then ``needs x tags`` and a new fixture needs no edit to
    any per-leaf list.  The mechanism sentence is documentation — it says which
    backend code the fixture reaches, which is why the fixture exists at all.
    """
    needs: frozenset
    mechanism: str


MULTI_BW = frozenset({"multi_bw"})

#: The ONLY mapping a fixture's need needs: a need is a property of the kc, and
#: a backend tag can REFUSE it.  Declared, so the vocabulary stays closed (a new
#: need without an entry is a silent no-op, so the gate below checks it).
REFUSES = {"multi_bw": "n_res1"}

#: Axis A, as DATA: every flat fixture, what a leaf must TOLERATE to take it,
#: and the mechanism it contributes.  ``test_the_declared_needs_match_the_built_
#: kernel_config`` checks the declaration against the built kc, so a config that
#: grows a second BW without saying so fails instead of being silently swept by
#: leaves that cannot take it.
CONFIGS = {
    "tests/config_pwa_gen.yml": Fixture(
        frozenset(), "the default 3-row basis, one wave"),
    "tests/config_pwa_gen_km.yml": Fixture(
        frozenset(), "a 5-row denominator over 2 waves (2 free heads)"),
    "tests/config_pwa_gen_ident.yml": Fixture(
        frozenset(), "identical-particle blocks: n_blocks > 1, one `ck`"),
    "tests/config_pwa_gen_same.yml": Fixture(
        frozenset(),
        "two chains sharing one resonance (different mass columns per wave); "
        "same kc_class as _ident, a DIFFERENT index mechanism"),
    "tests/config_pwa_gen_cp.yml": Fixture(
        frozenset(), "identical particles + CP blocks (n_blocks 4)"),
    "tests/config_pwa_gen_mb.yml": Fixture(
        MULTI_BW, "the factor product (n_res 2)"),
    "tests/config_pwa_gen_mb2.yml": Fixture(
        MULTI_BW, "the factor product x several topologies (wave dedup)"),
    "tests/config_pwa_gen_4pi_mb.yml": Fixture(
        MULTI_BW, "blocks x the factor product (32 waves, 40 basis rows)"),
}

#: Fixtures that are NOT structural classes, and the file that owns them.
#:
#: ``_cidx``/``_cidx_2f`` carry the spline resonance: a unit denominator and N
#: numerator heads, i.e. one factor with a `num_table` of N rows.  A kernel
#: branches on none of that — `n_wave`, `n_basis` and the CSR are read
#: generically, and the denominator being `one` is a BUILD-TIME decision
#: (`drop_unit_factors`, gated in ``test_skip_model.py``).  So the fixture adds
#: no leaf case; it is a MODEL case (``MODELS["spline_c_idx"]``) and its own
#: file's numbers.
OWNED_ELSEWHERE = {
    "tests/config_pwa_gen_cidx.yml": "tests/test_spline_c_idx.py",
    "tests/config_pwa_gen_cidx_2f.yml": "tests/test_skip_model.py",
    # a SPIN case (`1/2 -> 1/2 0 0`), not a kc structure: the leaf reads
    # `n_proj` as a generic column count, and the projection count is derived
    "tests/config_pwa_gen_spin_half.yml": "tests/test_spin_half.py",
}

#: Back-compat view for the tests that only need the sentence.
AXIS_A = {path: fix.mechanism for path, fix in CONFIGS.items()}

#: Axis-A fixtures whose ANCHOR belongs to another file: the multi-BW pair is
#: already gated (tighter: 2e-5/3e-5/1e-4) by
#: ``test_flat_dedup.py::test_the_multi_bw_anchor`` on the same sample, params
#: and quantities, so the shared sweep must NOT hold a second copy of it.
ANCHOR_OWNER = {
    "tests/config_pwa_gen_mb.yml": "tests/test_flat_dedup.py",
    "tests/config_pwa_gen_mb2.yml": "tests/test_flat_dedup.py",
    "tests/config_pwa_gen_4pi_mb.yml": "tests/test_flat_dedup.py",
}


def needs_of(kc):
    """The needs a BUILT kernel config actually has (the declaration's check)."""
    n_res = int(np.asarray(kc["gamma_matrix"]).shape[1])
    return MULTI_BW if n_res > 1 else frozenset()


def configs_for(leaf, model="pwa_gen"):
    """The fixtures this leaf can take — decided by its TAGS, not by its size.

    ``n_res1`` is the one capability a kernel refuses today, so it is the one
    fact a fixture can state.  The one exception is ownership, not accuracy:
    the multi-BW fixtures' ANCHOR belongs to ``test_flat_dedup.py`` (which
    gates them tighter), so they are not swept here.
    """
    from tabpwa.backends import backend_tags

    tags = backend_tags(leaf, model)
    return [path for path, fix in sorted(CONFIGS.items())
            if not refused(fix, tags) and path not in ANCHOR_OWNER]


def refused(fix, tags):
    """Whether a leaf's tags refuse a fixture's needs (see ``REFUSES``)."""
    unknown = set(fix.needs) - set(REFUSES)
    assert not unknown, f"need(s) {sorted(unknown)} have no tag in REFUSES"
    return any(REFUSES[need] in tags for need in fix.needs)


class Model(NamedTuple):
    """One particle model, and the kc its ONE-FACTOR template must assemble.

    ``block`` is merged into the base config's ``particle.k2`` (so ``J``/``P``
    and the mass/width stay).  The expectations are written out rather than
    derived from the model's own interface: deriving them would compare the
    model with itself.

    ``n_wave``/``n_basis``/``num_shape`` are the CONFIG's numbers, not the
    model's: ``n_basis`` is the assembled config's row table — a SUM over the
    config's factors, one row per (factor, basis row), deduplicated by name
    (MEASURED on ``_cidx_2f``: names ``['one', 'c_s0', 'one', 'c_width_t']`` —
    ``one`` appears once per factor), and ``n_wave`` is the sum over
    ``wave_expansion``.  A model contributes only its ``den_names`` (its row
    names) and its free numerator heads; a second factor adds rows, it does not
    change the model.  ``test_flat_models`` therefore checks this table against
    a ONE-FACTOR config, where the two coincide by construction.
    """
    block: dict
    n_wave: int          # the template's len(ck)
    n_basis: int         # the template's denominator row count
    den_names: tuple     # the weight names the config shows, in row order
    num_shape: tuple     # num_table.shape = (n_wave, n_res, n_interp)
    expansion: list      # wave_expansion() = [(base, offset, count), ...]


def _m(block, n_wave, n_basis, den_names, num_shape, expansion):
    return Model(block, n_wave, n_basis, tuple(den_names), tuple(num_shape),
                 list(expansion))


#: Axis B.  One entry per registered particle model that can be a ``pwa_gen``
#: resonance from config keys alone.  The parameter values are chosen so the
#: lineshape OVERLAPS the sample window (``_flat_grid._in_grid`` puts the
#: events near the low end of the mass grid): a Gaussian far outside it
#: underflows to ~1e-181 and the fp32 cache has nothing left to carry, which
#: is a parameterisation problem, not a kernel one (measured).
MODELS = {
    # -- the two built-in bases --------------------------------------------
    "basis": _m({}, 1, 3, ("k2_s0", "one", "k2_width_t"), (1, 1, 2000),
                [(0, 0, 1)]),
    # a plain running-width resonance through the BASIS path: the generic
    # branch of `_pg_chain_factors` (its weights are derived from mass/width)
    "BW": _m({"model": "BW"}, 1, 3, ("k2_s0", "one", "k2_width_t"),
             (1, 1, 2000), [(0, 0, 1)]),
    "BWR": _m({"model": "BWR"}, 1, 3, ("k2_s0", "one", "k2_width_t"),
              (1, 1, 2000), [(0, 0, 1)]),
    "BWRS": _m({"model": "BWRS"}, 1, 3, ("k2_s0", "one", "k2_width_t_t"),
               (1, 1, 2000), [(0, 0, 1)]),
    "GS_rho": _m({"model": "GS_rho"}, 1, 3, ("k2_s0", "one", "k2_width_t"),
                 (1, 1, 2000), [(0, 0, 1)]),
    "Bugg": _m({"model": "Bugg"}, 1, 3, ("k2_s0", "one", "k2_width_t"),
               (1, 1, 2000), [(0, 0, 1)]),
    "linear": _m({"model": "linear"}, 1, 3, ("k2_s0", "one", "k2_width_t"),
                 (1, 1, 2000), [(0, 0, 1)]),
    "pipi_swave": _m({"model": "pipi_swave"}, 1, 3,
                     ("k2_s0", "one", "k2_width_t"), (1, 1, 2000),
                     [(0, 0, 1)]),
    # fixed-shape rows: the numerator IS the lineshape (mu/sigma inside the
    # sample window; see the MODELS docstring)
    "GaussianBasis": _m({"model": "GaussianBasis", "mu": 1.0, "sigma": 0.3},
                        1, 3, ("k2_s0", "one", "k2_width_t"), (1, 1, 2000),
                        [(0, 0, 1)]),
    "BSplineBasis": _m({"model": "BSplineBasis", "mu": 1.0, "sigma": 0.3},
                       1, 3, ("k2_s0", "one", "k2_width_t"), (1, 1, 2000),
                       [(0, 0, 1)]),
    # -- the degenerate / unit-denominator shapes --------------------------
    "one": _m({"model": "one"}, 1, 1, ("one",), (1, 1, 2000), [(0, 0, 1)]),
    "skip": _m({"model": "skip"}, 1, 1, ("one",), (1, 1, 2000), [(0, 0, 1)]),
    # -- multi-row denominators built from their own polynomial rows --------
    "RhoOmega": _m({"model": "RhoOmega"}, 1, 3,
                   ("k2_s0", "one", "k2_width_t"), (1, 1, 2000),
                   [(0, 0, 1)]),
    "RhoOmega2": _m({"model": "RhoOmega2"}, 1, 8,
                    ("k2_s0", "one") + tuple(f"k2_w{i}_t" for i in range(6)),
                    (1, 1, 2000), [(0, 0, 1)]),
    "RhoOmegaBWR": _m({"model": "RhoOmegaBWR"}, 1, 10,
                      ("k2_s0", "one") + tuple(f"k2_w{i}_t" for i in range(8)),
                      (1, 1, 2000), [(0, 0, 1)]),
    "RhoOmegaGS": _m({"model": "RhoOmegaGS"}, 1, 10,
                     ("k2_s0", "one") + tuple(f"k2_w{i}_t" for i in range(8)),
                     (1, 1, 2000), [(0, 0, 1)]),
    "Exp": _m({"model": "Exp", "pure_exp": True}, 1, 52,
              ("k2_s0", "one") + tuple(f"k2_gk{i}_t" for i in range(50)),
              (1, 1, 2000), [(0, 0, 1)]),
    "ExpSpline": _m({"model": "ExpSpline", "pure_exp": True}, 1, 52,
                    ("k2_s0", "one") + tuple(f"k2_gk{i}_t" for i in range(50)),
                    (1, 1, 2000), [(0, 0, 1)]),
    "ExpSplineSVD": _m({"model": "ExpSplineSVD"}, 1, 10,
                       ("k2_s0", "one") + tuple(f"k2_sr{i}_t" for i in range(8)),
                       (1, 1, 2000), [(0, 0, 1)]),
    "Exp2DSplineSVD": _m({"model": "Exp2DSplineSVD"}, 1, 20,
                         ("k2_s0", "one")
                         + tuple(f"k2_sr{i}_t" for i in range(18)),
                         (1, 1, 2000), [(0, 0, 1)]),
    # -- K-matrix: rows AND heads from one polynomial ----------------------
    "kmatrix_n": _m({"model": "kmatrix_n", "mass_list": [0.8],
                     "width_list": [0.3]},
                    1, 3, ("k2_s0", "k2_poly", "k2_rho"), (1, 1, 2000),
                    [(0, 0, 1)]),
    "kmatrix_n_gs": _m({"model": "kmatrix_n_gs", "mass_list": [0.8],
                        "width_list": [0.3]},
                       1, 3, ("k2_s0", "k2_poly", "k2_rho"), (1, 1, 2000),
                       [(0, 0, 1)]),
    "kmatrix_n_ieps": _m({"model": "kmatrix_n_ieps", "mass_list": [0.8],
                          "width_list": [0.3]},
                         1, 3, ("k2_s0", "k2_poly", "k2_rho"), (1, 1, 2000),
                         [(0, 0, 1)]),
    "kmatrix_n_gs_ieps": _m({"model": "kmatrix_n_gs_ieps", "mass_list": [0.8],
                             "width_list": [0.3]},
                            1, 3, ("k2_s0", "k2_poly", "k2_rho"), (1, 1, 2000),
                            [(0, 0, 1)]),
    "kmatrix_n_2pole": _m({"model": "kmatrix_n", "mass_list": [0.8, 1.4],
                           "width_list": [0.3, 0.2]},
                          2, 5, ("k2_s0", "k2_poly", "k2_m4", "k2_rho",
                                 "k2_srho"), (2, 1, 2000), [(0, 0, 2)]),
    # -- a numerator expansion on a UNIT denominator -----------------------
    "ExpCheb": _m({"model": "ExpCheb", "mass": 0.8, "k": 2.0},
                  15, 1, ("k2_one",), (15, 1, 2000), [(0, 0, 15)]),
    "spline_c_idx": _m({"model": "spline_c_idx",
                        "points": [0.4, 0.6, 0.8, 1.0, 1.3, 1.7]},
                       4, 1, ("one",), (4, 1, 2000), [(0, 0, 4)]),
}

#: Registered models this axis does NOT build, each with the reason and the
#: place it is covered instead.  Both fields are checked by the live gate.
EXCLUDED = {
    "width_linear_npy": (
        "needs `file:` — a measured running-width table, i.e. DATA, so it has "
        "no config-only form",
        "tests/config_amp.yml"),
    "ck_matrix_v2": (
        "needs `gamma_file:` (a partial-wave matrix on disk)",
        "tests/config_amp.yml"),
    "ck_matrix_v2_sqrt": (
        "needs `gamma_file:`",
        "tests/config_amp.yml"),
    "ck_matrix_disp_v2": (
        "needs `gamma_file:`",
        "tests/config_amp.yml"),
    "FlatteC": (
        "its `mass_list` is a per-channel daughter-pair block this table does "
        "not model",
        "tests/config_pwa_flatte.yml"),
    "FlatteGen": (
        "its `mass_list` is a nested per-channel daughter-pair block",
        "tests/test_flatte_gen.py"),
    "BSpline": (
        "requires a `knots` / (`x_range`, `n_free`) block AND edits a "
        "`{name}_mass` input the basis transform does not publish",
        "tests/test_spline_gamma_model.py"),
}

#: THE GATE IS A CLASS, DECLARED BY TAG.  A leaf that carries the amplitude in
#: a float2 cache is accurate to ~1e-7..2.5e-5 (MEASURED over every fixture and
#: model), an fp64 leaf to ~1e-15; those are the only two classes and the tag
#: says which one a leaf is.  1e-4 is 4x the worst measured cached gap and four
#: orders below an algebra change; 1e-12 is the fp64 summation-order floor.
CORRECT_TOL = {"fp32_cache": 1e-4, "fp64": 1e-12}


def tol_for(leaf, model="pwa_gen"):
    """The bound a leaf's ACCURACY TAG declares (``fp32_cache`` or fp64)."""
    from tabpwa.backends import backend_tags

    tags = backend_tags(leaf, model)
    return CORRECT_TOL["fp32_cache" if "fp32_cache" in tags else "fp64"]


#: The model axis runs on the family's REFERENCE path — ``Fitter.get_nll``,
#: whose default ``pwa_gen`` backend is this one — because a model is a UNIT OF
#: A CONFIG, not a backend case: whether a *leaf* is right on a structure is
#: axis A's question, and asking it again per model would measure the leaf.
#: Two MEASUREMENTS say the same thing from the other side (both found while
#: this axis did sweep every leaf): a fixed-shape row far outside the sample
#: window puts P at 1e-181, where the fp32 cache has nothing left and the
#: grouped log-sum NaNs (``GaussianBasis``, mu 0.775 / sigma 0.1 — the fp64
#: reference is fine), and ``ExpSplineSVD``'s 10-row basis reaches 2.5e-5
#: against 1e-12 on the fp64 leaf.  Both are properties of the LEAF's floor,
#: not of the model, so the model axis has no per-leaf floor at all.
REFERENCE = "numpy_flat"


def config(name):
    """``TEMPLATE`` with ``MODELS[name]``'s block rendered into ``RESONANCE``.

    No file is read and no path is resolved: the sample is synthesised by
    ``_flat_grid._in_grid`` and handed to ``set_data`` / ``set_phsp``, and the
    template carries only the geometry (``dat_order``).
    """
    d = copy.deepcopy(TEMPLATE)
    block = dict(d["particle"][RESONANCE])
    block.update(MODELS[name].block)
    d["particle"][RESONANCE] = block
    return d


def kc_class(kc):
    """The structural signature a leaf can distinguish (axis A's unit).

    Shapes only: two fixtures whose kc differs here can need different backend
    code; two whose kc agrees here cannot.  ``same`` and ``ident`` share it
    (stated in the module docstring) — the fixture, not the signature, is what
    separates their mechanisms.
    """
    nt = np.asarray(kc.get("num_table"))
    gm = np.asarray(kc["gamma_matrix"])
    return (int(gm.shape[0]), int(gm.shape[1]),
            int(np.asarray(kc["basis_table"]).shape[0]),
            int(nt.shape[0]) if nt.ndim == 3 else -1,
            int(kc.get("n_proj", 1)))
