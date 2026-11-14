#!/usr/bin/env python3
"""Fast kernel evaluation straight from a kernel config + data.

No fitter, no amplitude model, no constraints at evaluation time: the raw
kernel API is ``K(kernel_config) -> load_data(data) -> compute(params)``.
The kernel config is a plain dict of numpy arrays, so it can be dumped once
(``--dump-kc``) and reloaded later (``--kc``) with **zero** framework
imports — that is the "fast evaluation" path.

Build the kernel config once (the only step that needs the framework)::

    python scripts/eval_kernel.py --config tests/config_pwa.yml \\
        --backend cuda_v5_pwa_s0e --dump-kc /tmp/kr/k.npz

Evaluate from the dumped config (no framework at all)::

    python scripts/eval_kernel.py --kc /tmp/kr/k.npz \\
        --kernel cuda_v5_pwa_s0e --events 200000 --repeat 5

``--compare`` runs a second kernel on the same config/parameters and prints
the relative difference (e.g. ``--kernel cuda_v5_pwa_s0e --compare cpu_v5_pwa``).

Data: synthesized with the shapes the kernel expects by default, or real
arrays with ``--data`` (an ``.npz`` with ``mass``, ``q``, ``angle``,
``weight``, ``bkg``).
"""
import argparse
import importlib
import inspect
import sys
import time

import numpy as np

_ALIASES = {"cuda": "cuda_v3", "cuda64": "cuda_v3"}

_CPU_MODULES = {
    "cpu_v5_pwa": "tabpwa.cpu._pwa_v5",
    "cpu_pwa_s0c": "tabpwa.cpu._pwa_s0c",
    "cpu_pwa_gen": "tabpwa.cpu._pwa_gen",
    "cpu_v4_pwa": "tabpwa.cpu._pwa_gen",
    "cpu_v3": "tabpwa.cpu._v3",
    "cpu64_v3": "tabpwa.cpu._v3",
}


def _kernel_class(name):
    """Kernel wrapper class for a backend name."""
    name = _ALIASES.get(name, name)
    if name in _CPU_MODULES:
        mod_name = _CPU_MODULES[name]
    elif name.startswith("cuda_"):
        mod_name = "tabpwa.cuda._" + name[len("cuda_"):]
    else:
        raise SystemExit(f"{name!r}: no kernel module known for this name")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(f"{name!r}: cannot import {mod_name} ({exc})")
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != mod.__name__:
            continue
        if hasattr(cls, "load_data") and hasattr(cls, "compute"):
            return cls
    raise SystemExit(f"{mod_name}: no kernel class found")


# ── kernel config: build once / load later ────────────────────────────
def _build_kc(config, backend):
    from tabpwa import Fitter

    return Fitter(config, backend=backend).kernel_config


def _save_kc(path, kc):
    np.savez(path, **{k: np.asarray(v) for k, v in kc.items()})
    print(f"kernel config -> {path} ({len(kc)} entries)")


def _load_kc(path):
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


# ── data ───────────────────────────────────────────────────────────────
def _dims(kc):
    n_at = int(np.max(np.asarray(kc["angle_index"]))) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(np.asarray(kc["mass_index"]))) + 1
    n_q = int(np.max(np.asarray(kc["fl_q_index"]))) + 1
    return n_at, nac, n_mass, n_q


def _synth(kc, n, seed=1, lo=0.6, hi=1.7):
    n_at, nac, n_mass, n_q = _dims(kc)
    r = np.random.RandomState(seed)
    return {
        "mass": lo + (hi - lo) * r.rand(n, n_mass),
        "q": np.abs(r.randn(n, n_q)) + 0.3,
        "angle": r.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + r.rand(n),
        "bkg": np.abs(r.randn(n)) * 0.1 + 0.05,
    }


def _params(kern, kc, rng):
    """Random but correctly sized parameters (sizes from the kernel itself)."""
    N = int(getattr(kern, "n_wave_base",
                    np.asarray(kc["angle_k"]).shape[-2]))
    n_m0 = int(getattr(kern, "n_m0_params",
                       np.max(np.asarray(kc["m0_index"])) + 1))
    n_g0 = int(getattr(kern, "n_g0_params",
                       np.max(np.asarray(kc["g0_index"])) + 1))
    params = {"ck": rng.normal(size=N) + 1j * rng.normal(size=N),
              "m0": 0.7 + 0.4 * rng.random(n_m0),
              "g0": 0.1 + 0.2 * rng.random(n_g0)}
    if getattr(kern, "_generic", False):
        # pwa_gen generic denominator: explicit (n_wave, n_basis) coefficients
        nw = int(getattr(kern, "n_wave"))
        nf = int(getattr(kern, "n_basis", 0))
        params["coeff"] = rng.normal(size=nw * nf)
    return params


# ── timing ─────────────────────────────────────────────────────────────
def _bench(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    best, mean = np.inf, 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        mean += dt
    return best, mean / max(repeat, 1)


def _run(name, kc, data, params, args, norm=None, label="compute"):
    cls = _kernel_class(name)
    t0 = time.perf_counter()
    kern = cls(kc, batch_size=args.batch_size)
    t_create = time.perf_counter() - t0
    t0 = time.perf_counter()
    dh = kern.load_data(data)
    t_load = time.perf_counter() - t0
    best, mean = _bench(
        lambda: kern.compute(params, dh, norm=norm, return_p=False),
        args.warmup, args.repeat)
    Q, grads, _ = kern.compute(params, dh, norm=norm, return_p=False)
    ne = data["mass"].shape[0]
    print(f"{name:28s} create {t_create * 1e3:7.3f} ms | "
          f"load {t_load * 1e3:7.3f} ms | {label} best {best * 1e3:8.3f} ms "
          f"mean {mean * 1e3:8.3f} ms | {ne / best / 1e6:6.2f} Mev/s | "
          f"Q {Q:.10g}")
    kern.free()
    return Q, grads


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("kernel config (one of)")
    src.add_argument("--config", help="YAML config; builds the kernel config "
                                      "through the model (build time only)")
    src.add_argument("--kc", help="dumped kernel config (.npz) — no framework")
    ap.add_argument("--dump-kc", help="write the built kernel config here")
    ap.add_argument("--kernel", help="kernel to evaluate (requires --kc); "
                                     "defaults to --backend")
    ap.add_argument("--backend", default="cuda_v5_pwa_s0e",
                    help="backend used to build the kernel config")
    ap.add_argument("--compare", help="run a second kernel on the same "
                                      "config/parameters and report the "
                                      "relative difference")
    ap.add_argument("--events", type=int, default=50000)
    ap.add_argument("--phsp-events", type=int, default=0,
                    help="also time the norm pass (norm=None) on phsp data")
    ap.add_argument("--data", help="real data .npz (mass/q/angle/weight/bkg)")
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-imports", action="store_true",
                    help="report whether the fitter/model/backends modules "
                         "were imported at all")
    args = ap.parse_args(argv)

    if not args.kc and not args.config:
        ap.error("need --kc or --config")

    kc = _load_kc(args.kc) if args.kc else _build_kc(args.config, args.backend)
    if args.dump_kc:
        _save_kc(args.dump_kc, kc)

    name = args.kernel or (args.backend if args.kc is None else None)
    if name is None:
        ap.error("--kernel is required with --kc")

    data = (dict(np.load(args.data, allow_pickle=True)) if args.data
            else _synth(kc, args.events, args.seed))
    ne = data["mass"].shape[0]
    print(f"kernel config : {args.kc or args.config}   events {ne}")

    # parameters are sized from a kernel instance (N, n_m0, n_g0)
    probe = _kernel_class(name)(kc, batch_size=args.batch_size)
    params = _params(probe, kc, np.random.default_rng(args.seed))
    probe.free()

    norm = float(ne) * 0.5            # a plausible pdf denominator
    Q, grads = _run(name, kc, data, params, args, norm=norm)

    if args.phsp_events:
        ph = _synth(kc, args.phsp_events, args.seed + 1)
        _run(name, kc, ph, params, args, norm=None, label="norm pass")
        del ph

    if args.compare:
        Q2, _ = _run(args.compare, kc, data, params, args, norm=norm)
        den = max(abs(Q), 1e-30)
        print(f"compare       |dQ|/|Q| = {abs(Q2 - Q) / den:.3e}")

    print("grads         " + ", ".join(
        f"{k}:{np.size(v)}" for k, v in grads.items()))
    if args.show_imports:
        loaded = [m for m in ("tabpwa.fitter", "tabpwa.amp_model",
                              "tabpwa.base_model", "tabpwa.backends")
                  if m in sys.modules]
        print("framework     " + ("imported: " + ", ".join(loaded)
                                  if loaded else "none imported"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
