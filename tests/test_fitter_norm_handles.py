"""The ``Fitter``-level contract: two handles must coexist.

``Fitter`` loads phase space (``set_phsp`` → ``_phsp_handle``) and THEN data
(``set_data`` → ``_data_handle``), and keeps BOTH alive for the whole fit:
``_compute_norm_batched`` integrates over the phase-space handle while
``get_nll_raw`` evaluates the data handle.  A backend that keeps one mutable
"current data" slot inside its context — so the second ``load_data`` frees the
first — silently returns the DATA integral where the phase-space integral was
asked for.

That failure is invisible to a test that calls ``backend.compute`` with
explicit handles (every kernel test does), so it is pinned here at the layer
that actually uses the backend: the values are compared against an
independently constructed reference backend.

Gate:
    norm computed through the fitter == norm computed directly over the
    phase-space sample, for every pwa_gen backend that can be loaded twice.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model

CFG = "tests/config_pwa_gen_km.yml"


def _samples(n, mass, q, weight_total):
    return {
        "mass": np.full((n, 1), mass),
        "q": np.full((n, 2), q),
        "angle": np.zeros((n, 1, 4)),
        # the fitter's convention: phase-space weights sum to 1
        "weight": np.full(n, weight_total / n),
        "bkg": np.full(n, 0.1),
    }


def _leaf_backends():
    # test-side stubs are tagged `probe` at registration (``_model_probe``,
    # ``_kwarg_probe``): a capability, not a ``_``-prefix, so a stub that is
    # named differently is excluded too.
    probes = frozenset(backends_for_model("pwa_gen", tag="probe"))
    # `external` = needs a server/launcher in another process (`remote`, `mpi`);
    # a delegate that forwards IN-process (`auto`, `integrated_pwa_gen`,
    # `fixed_coeff`) is constructible and belongs in this sweep.  `shard` spawns
    # its workers from a queue, which is a different lifecycle.
    external = frozenset(backends_for_model("pwa_gen", tag="external"))
    return sorted(n for n in backends_for_model("pwa_gen")
                  if n != "default" and n not in probes
                  and n not in external and n != "shard")


def test_reference_backend_gets_the_phase_space_norm():
    """Sanity: the numpy reference is right, so it can anchor the rest."""
    f = Fitter(CFG)
    f.apply_constrains()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)

    phsp = _samples(50, 0.9, 0.8, 1.0)
    data = _samples(20, 1.5, 1.2, 20.0)      # deliberately different kinematics
    f.set_phsp(phsp)
    f.set_data(data)

    norm_fit, _ = f._compute_norm_batched(params)
    b = create_backend("numpy_flat", f.kernel_config, model="pwa_gen")
    norm_phsp, _, _ = b.compute(params, b.load_data(phsp), norm=None,
                                return_p=False)
    assert float(norm_fit) == pytest.approx(float(norm_phsp), rel=1e-10)
    b.free()


def _handle_id(holder):
    """The identity a ``load_data`` return value exposes, whatever its shape.

    ``numpy_flat`` returns the data dict itself (zero copy), the CUDA kernels
    return ``{"handle": ptr, ...}`` and the established kernels return a
    ``DataHandle`` with ``.ptr``/``.ne``.
    """
    if holder is None:
        return None
    if isinstance(holder, dict):
        for key in ("handle", "ptr"):
            if key in holder:
                return holder[key]
        return id(holder)
    return getattr(holder, "ptr", None) or id(holder)


@pytest.mark.parametrize("name", _leaf_backends())
def test_two_handles_coexist_through_the_fitter(name):
    """``set_data`` after ``set_phsp`` must not clobber the phsp handle."""
    f = Fitter(CFG, backend=name)
    f.apply_constrains()
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    kc = f.kernel_config

    phsp = _samples(50, 0.9, 0.8, 1.0)
    data = _samples(20, 1.5, 1.2, 20.0)
    f.set_phsp(phsp)
    f.set_data(data)

    # the two handles must be INDEPENDENT pieces of state.  A backend that
    # hands back its own context as the handle makes them the same pointer,
    # and then every load_data() overwrites the other sample in place.
    assert f._phsp_holder is not f._data_holder
    h_ph = _handle_id(f._phsp_holder)
    h_da = _handle_id(f._data_holder)
    assert h_ph != h_da, (
        f"{name}: the phase-space and data handles are the SAME object "
        f"({h_ph!r}) — load_data() overwrites the previous sample in place, "
        f"so the norm is integrated over the wrong events")

    norm_fit, _ = f._compute_norm_batched(params)

    # the same backend family, a FRESH instance: no shared mutable state, so
    # this is the norm of the phase-space sample by construction
    ref = create_backend(name, kc, model="pwa_gen")
    norm_phsp, _, _ = ref.compute(params, ref.load_data(phsp), norm=None,
                                  return_p=False)
    ref.free()
    # and the data integral, which is what a clobbered handle would return
    ref2 = create_backend(name, kc, model="pwa_gen")
    norm_data, _, _ = ref2.compute(params, ref2.load_data(data), norm=None,
                                   return_p=False)
    ref2.free()

    got = float(norm_fit)
    assert got == pytest.approx(float(norm_phsp), rel=1e-6), (
        f"{name}: norm is {got!r}, the phase-space integral is "
        f"{float(norm_phsp)!r} (the DATA integral is {float(norm_data)!r}) — "
        f"the phase-space handle was overwritten by load_data(data)")

    # the same backend family, a FRESH instance: no shared mutable state, so
    # this is the norm of the phase-space sample by construction
    ref = create_backend(name, kc, model="pwa_gen")
    norm_phsp, _, _ = ref.compute(params, ref.load_data(phsp), norm=None,
                                  return_p=False)
    ref.free()
    # and the data integral, which is what a clobbered handle would return
    ref2 = create_backend(name, kc, model="pwa_gen")
    norm_data, _, _ = ref2.compute(params, ref2.load_data(data), norm=None,
                                   return_p=False)
    ref2.free()

    got = float(norm_fit)
    assert got == pytest.approx(float(norm_phsp), rel=1e-6), (
        f"{name}: norm is {got!r}, the phase-space integral is "
        f"{float(norm_phsp)!r} (the DATA integral is {float(norm_data)!r}) — "
        f"the phase-space handle was overwritten by load_data(data)")
