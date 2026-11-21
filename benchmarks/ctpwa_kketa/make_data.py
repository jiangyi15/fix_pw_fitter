#!/usr/bin/env python3
"""CTPWA benchmark samples, built by the framework itself.

    python make_data.py --n-data 1000000 --n-phsp 5000000

Two steps, both framework code — no hand-written kinematics:

1. `pwa_build.generate_pwa_phsp(model, chain, n)` — flat 3-body phase space of
   `jpsi -> (K+ K-) + eta`, sampled with the exact three-body density and
   boosted up the chain (the inverse of `decay_angles_from_momenta`).  Its
   four-vectors are **(E, px, py, pz)**, which is what the loaders below read.
2. `model.build_event_data(momenta, kc)` — the same conversion
   `Fitter.load_momenta_conf` uses, so `mass` / `q` / `angle` are guaranteed
   to be laid out the way the kernels index them.

The phase space is flat (unit weights), which is what a *timing* sample needs;
`toy_pwa_generate` would be the choice for a model toy, but its rejection
acceptance on this model is ~0.2 %, i.e. ~5x10^8 candidates for 10^6 events.

The `pwa_gen` config reads the same arrays.  The grid assertion below is on the
config THIS script loaded; the two configs happen to share both grids
(mass [0.9774, 2.6132], q [0, 1.3372]), so run it once per config if in doubt.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--n-phsp", type=int, default=5_000_000)
    ap.add_argument("--batch", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()

    from tabpwa import Fitter
    from tabpwa.pwa_build import generate_pwa_phsp

    # the reference backend is model-scoped: `numpy_pwa` is registered for
    # `amp_model: pwa`, `numpy_flat` for `pwa_gen` (asking for the wrong one is a
    # hard ValueError, so pick it from the config rather than guessing)
    from tabpwa.config_loader import RawConfig
    amp = str(RawConfig(a.config).dic.get("amp_model", "pwa"))
    ref = "numpy_flat" if amp in ("pwa_gen", "pwa_basis") else "numpy_pwa"
    f = Fitter(a.config, backend=ref)
    kc = f.kernel_config
    model = f.model
    chain = model.full_decay.chains[0]

    def convert(n, seed):
        t0 = time.perf_counter()
        chunks = []
        for k in range(0, n, a.batch):
            m = min(a.batch, n - k)
            chunks.append(generate_pwa_phsp(model, chain, m, seed=seed + k))
        mom = np.concatenate(chunks)
        ev = model.build_event_data(mom, kc)
        ev = {k: np.asarray(v) for k, v in ev.items()}
        ev["weight"] = np.ones(len(mom))
        # `build_event_data` already emits `bkg` (ones).  The loader does NOT
        # read that key -- `Fitter.load_npz` wants `bkg_raw` and otherwise
        # defaults to ones -- and a background-free sample cannot carry
        # `bkg_raw = 0` either (set_data requires N_b > 0).  The background term
        # is inert because `data.purity: 1.0` zeroes its scale, not because of
        # anything written here.
        print(f"  {n:,} events: momenta {mom.shape} -> mass {ev['mass'].shape} "
              f"q {ev['q'].shape} angle {ev['angle'].shape} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return ev

    os.makedirs(a.out, exist_ok=True)
    data = convert(a.n_data, a.seed)
    phsp = convert(a.n_phsp, a.seed + 7919)

    # validity gates: finite, and inside every interpolation grid
    for tag, ev in (("data", data), ("phsp", phsp)):
        for k in ("mass", "q"):
            v = ev[k]
            assert np.isfinite(v).all(), f"{tag}.{k} has non-finite values"
    m = phsp["mass"]
    m_lo = float(kc.get("gamma_min", kc.get("m_min")))
    m_hi = m_lo + float(kc.get("gamma_delta", kc.get("m_delta"))) * (
        np.asarray(kc.get("gamma_table", kc.get("basis_table"))).shape[-1] - 1)
    q_t = np.asarray(kc["fl_table"])
    q_lo = float(kc.get("fl_min", kc.get("q_min")))
    q_hi = q_lo + float(kc.get("fl_delta", kc.get("q_delta"))) * (q_t.shape[-1] - 1)
    print(f"  grids: mass [{m_lo:.4f}, {m_hi:.4f}] q [{q_lo:.4f}, {q_hi:.4f}]")
    for tag, ev in (("data", data), ("phsp", phsp)):
        print(f"  {tag}: mass [{ev['mass'].min():.4f}, {ev['mass'].max():.4f}] "
              f"q [{ev['q'].min():.4f}, {ev['q'].max():.4f}]")
        assert m_lo <= ev["mass"].min() and ev["mass"].max() <= m_hi, "mass out of grid"
        assert q_lo <= ev["q"].min() and ev["q"].max() <= q_hi, "q out of grid"
        # every momentum column must be alive for the whole sample: a topology
        # whose top vertex resolves to the wrong leaf gives q = 0 for ALL events
        # of that chain, which silently zeroes its waves (the same-charge trap)
        med = np.median(ev["q"], axis=0)
        dead = np.flatnonzero(med <= 0)
        assert dead.size == 0, (
            f"{tag}: q column(s) {dead.tolist()} are zero for >=50 % of events -- "
            f"a decay chain collapsed (check the recoil lepton/kaon names)")
        print(f"  {tag}: all {ev['q'].shape[1]} q columns alive "
              f"(min median {med.min():.4g})")

    np.savez(os.path.join(a.out, "data.npz"), **data)
    np.savez(os.path.join(a.out, "phsp.npz"), **phsp)
    print(f"wrote {a.out}/data.npz ({a.n_data:,}) and phsp.npz ({a.n_phsp:,})")


if __name__ == "__main__":
    main()
