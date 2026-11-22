#!/usr/bin/env python3
"""PWACG benchmark samples, built by the framework itself.

    python make_data.py --n-data 10000 --n-phsp 10000
    python make_data.py --n-data 100000 --n-phsp 100000     # 100k
    python make_data.py --n-data 1000000 --n-phsp 5000000   # the paper's 3e6-scale run

Two steps, both framework code — no hand-written angle kinematics:

1. `generate_cascade_phsp(...)` — the **flat four-body phase space** of
   `psi(3686) -> (phi -> K+ K-) + (R -> K+ K-)`.  `pwa_build.generate_pwa_phsp`
   is three-body only (`top -> (sub -> a b) + c`), so this file writes the 2+2
   cascade itself.  It is still "the framework's kinematics": the density uses
   `pwa_build._two_body_p`, the boost is `pwa_build._boost_vec`, and the
   construction is the exact inverse of the angle builder the configs feed —
   every two-body vertex is isotropic in its own rest frame, then boosted into
   the top rest frame.

   The density in the two di-kaon masses is the factorised four-body phase
   space (see the derivation in `generate_cascade_phsp`):
   `dPhi4 ~ p_top(M; m12, m34) * p12(m12) * p34(m34) dm12 dm34 dOmega...`,
   i.e. rejection sampling in the (m12, m34) plane with that weight and
   isotropic, independent directions at each vertex.

3. `model.build_event_data(momenta, kc)` — the same conversion
   `Fitter.load_momenta_conf` uses, so `mass` / `q` / `angle` are laid out the
   way the kernels index them.

It also writes the **reference-framework contract** so both sides time the
same events (`data/Momentum_kk.npz`, and `data/Momentum_kk_phsp.npz` for the
phase-space sample):

    Kp, Km, Pip, Pim   each (n_events, 4), columns (px, py, pz, E), float64

Spatial first, energy last, because the reference flips the spatial part and
dots with `diag(-1, -1, -1, 1)` (`Tensor/CacheTensor.py:42-44`).  `Kp`/`Km`
are OUR `Kp1`/`Km1` (the phi pair) and `Pip`/`Pim` are our `Kp2`/`Km2` (the
R pair) — the pi names are a misnomer inherited from a pi-pi model and are kept
only because the reference reader expects those keys.  Event `i` is the same
event in both files.  `--no-ref-format` skips these two files.

**No symmetry declaration** (see `config_gen.yml`): this is the matched
sample/model pair for PWACG, whose formalism has no identical-particle/CP
block expansion — 24 amplitude components, not 4 x 24.

Four-vectors are **(E, px, py, pz)** in `data.dat_order` order, which is
`cfg.finals` order: Kp1, Km1 are the phi pair and Kp2, Km2 the R pair.

The phase space is flat (unit weights), which is what a *timing* sample needs.
The grid assertions below are the contract: every sampled mass and momentum
must lie inside the tables the kernels interpolate.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import extra_amp  # noqa: E402,F401  registers the benchmark line shapes


def _unit_vec(rng, n):
    """Isotropic unit vectors (n, 3): uniform on the sphere."""
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def generate_cascade_phsp(M_top, m_k, n_events, seed=None, batch=200_000):
    """Flat four-body phase space of `top -> (sub1 -> a b) + (sub2 -> c d)`.

    All four daughters have mass `m_k` here (K+ K- K+ K-), the two subsystems
    an arbitrary `m_R` each (the phi pair's mass does not enter the flat density).

    Factorised four-body phase space (standard sequential-decay form)

        dPhi4 = dPhi2(P; Q12, Q34) * ds12/2pi * dPhi2(Q12)
                                    * ds34/2pi * dPhi2(Q34)

    with `dPhi2(Q; b, c) = p/(16 pi^2 m_Q) dOmega`.  Writing `ds = 2 m dm`
    the `m` denominators cancel,

        dPhi4 ~ p_top(M; m12, m34) * p12(m12) * p34(m34) dm12 dm34 dOmega_top dOmega12 dOmega34

    so uniform (m12, m34) with the weight `p_top * p12 * p34` and isotropic
    directions is exactly flat four-body phase space.  Rejection sampling below
    is that weight, with `wmax` on a grid; the density is smooth and positive on
    `m12 + m34 < M`, so the acceptance is O(10 %) and exact.

    Returns `(momenta, masses)` — momenta `(n, 4, 4)` in (E, px, py, pz) with
    columns `[phi_K+, phi_K-, R_K+, R_K-]`, and the sampled `(m12, m34)`.
    """
    from tabpwa.pwa_build import _boost_vec, _two_body_p

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    lo = 2.0 * m_k
    hi = M_top - 2.0 * m_k

    # wmax over the allowed triangle m12 + m34 <= M_top
    g = np.linspace(lo, hi, 400)
    m12g, m34g = np.meshgrid(g, g, indexing="ij")
    ok = (m12g + m34g) <= M_top
    wg = _two_body_p(np.full_like(m12g, M_top), m12g, m34g) \
        * _two_body_p(m12g, m_k, m_k) * _two_body_p(m34g, m_k, m_k)
    wmax = float(wg[ok].max()) * 1.0001

    m12 = np.empty(n_events)
    m34 = np.empty(n_events)
    got = 0
    cand_total = 0
    while got < n_events:
        want = min(batch, n_events - got)
        m = max(int(want / 0.08), 4 * want)          # ~8 % acceptance
        m_int = int(m)
        m12c = rng.uniform(lo, hi, m_int)
        m34c = rng.uniform(lo, hi, m_int)
        inside = (m12c + m34c) <= M_top
        w = np.zeros(m_int)
        w[inside] = _two_body_p(np.full(inside.sum(), M_top), m12c[inside], m34c[inside]) \
            * _two_body_p(m12c[inside], m_k, m_k) * _two_body_p(m34c[inside], m_k, m_k)
        keep = rng.uniform(0.0, wmax, m_int) < w
        k = int(keep.sum())
        n = min(k, want)
        m12[got:got + n] = m12c[keep][:n]
        m34[got:got + n] = m34c[keep][:n]
        got += n
        cand_total += m_int
    acc = n_events / max(cand_total, 1)

    # ── orientations, isotropic at every vertex ───────────────────────────
    u_top = _unit_vec(rng, n_events)     # direction of the phi system in the top rest frame
    u_phi = _unit_vec(rng, n_events)     # K+ direction in the phi rest frame
    u_R = _unit_vec(rng, n_events)       # K+ direction in the R rest frame

    top4 = np.tile([M_top, 0.0, 0.0, 0.0], (n_events, 1))
    p_top = _two_body_p(np.full(n_events, M_top), m12, m34)
    E_phi = np.sqrt(p_top ** 2 + m12 ** 2)
    phi4 = np.concatenate([E_phi[:, None], p_top[:, None] * u_top], axis=-1)
    R4 = top4 - phi4

    def two_body(m_pair, direction, m_a):
        """`m_pair -> a a` in `m_pair`'s rest frame: (a+, a-) 4-vectors."""
        p = _two_body_p(m_pair, m_a, m_a)
        E = np.sqrt(p ** 2 + m_a ** 2)
        a_plus = np.concatenate([E[:, None], p[:, None] * direction], axis=-1)
        a_minus = np.concatenate([E[:, None], -p[:, None] * direction], axis=-1)
        return a_plus, a_minus

    Kp1_rest, Km1_rest = two_body(m12, u_phi, m_k)
    Kp2_rest, Km2_rest = two_body(m34, u_R, m_k)

    # boost each subsystem's daughters from its rest frame into the top rest frame
    b_phi = phi4[:, 1:] / phi4[:, 0:1]
    b_R = R4[:, 1:] / R4[:, 0:1]
    mom = np.stack([
        _boost_vec(Kp1_rest, b_phi), _boost_vec(Km1_rest, b_phi),
        _boost_vec(Kp2_rest, b_R), _boost_vec(Km2_rest, b_R),
    ], axis=1)
    print(f"  cascade phsp: m12 [{m12.min():.4f}, {m12.max():.4f}] "
          f"m34 [{m34.min():.4f}, {m34.max():.4f}]  acceptance {acc:.1%}",
          flush=True)
    return mom, (m12, m34)


def write_ref_format(mom, path, M_top):
    """Write `mom` `(n, 4, 4)` in `(E, px, py, pz)` as the reference contract.

    Four keys, `(n_events, 4)`, columns **(px, py, pz, E)** — the reference
    flips the spatial part and dots with `diag(-1,-1,-1,1)`.  `Kp`/`Km` are the
    phi pair (our Kp1/Km1) and `Pip`/`Pim` the R pair (our Kp2/Km2); the pi
    names are the reference's inherited misnomer.
    """
    Kp, Km, Pip, Pim = (np.ascontiguousarray(mom[:, j][:, [1, 2, 3, 0]],
                                             dtype=np.float64) for j in range(4))
    np.savez(path, Kp=Kp, Km=Km, Pip=Pip, Pim=Pim)
    return Kp, Km, Pip, Pim


def _mass4(mom):
    """Invariant mass from (E, px, py, pz) rows."""
    return np.sqrt(np.maximum(mom[:, 0] ** 2 - np.sum(mom[:, 1:] ** 2, axis=-1), 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--n-data", type=int, default=10_000)
    ap.add_argument("--n-phsp", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--phi-mass", type=float, default=1.02,
                    help="phi mass for the MC |BW|^2 weight (they hard-code 1.02)")
    ap.add_argument("--phi-width", type=float, default=0.004,
                    help="phi width for the MC |BW|^2 weight (their 0.004)")
    ap.add_argument("--no-ref-format", action="store_true",
                    help="skip data/Momentum_kk*.npz (the reference framework's "
                         "px,py,pz,E contract)")
    a = ap.parse_args()

    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    # the reference backend is model-scoped: `numpy_flat` for `pwa_gen`,
    # `numpy_pwa` for `pwa` (asking for the wrong one is a hard ValueError)
    dic = RawConfig(a.config).dic
    amp = str(dic.get("amp_model", "pwa"))
    ref = "numpy_flat" if amp in ("pwa_gen", "pwa_basis") else "numpy_pwa"
    f = Fitter(a.config, backend=ref)
    kc = f.kernel_config
    model = f.model
    # the matched (no-symmetry) model must carry exactly the reference's 24
    # amplitude components (2 + 5 + 2 + 5 + 5 + 5, one chain per resonance);
    # a symmetry declaration would multiply this by 4
    for i, c in enumerate(model.full_decay.chains):
        n_ls = len(c.get_ls_combination())
        print(f"  chain {i}: " + " x ".join(d.core.name for d in c.decays[1:])
              + f"   g_ls {n_ls}", flush=True)
    n_wave_geom = len(model.get_ck_map())
    print(f"  waves (len(get_ck_map())): {n_wave_geom}", flush=True)
    if n_wave_geom != 24:
        raise SystemExit(
            f"make_data.py: expected 24 waves (the reference's 24 amplitude "
            f"components: f0(980) 2, f0(1710) 2, f2(1270) 5, f2'(1525) 5, "
            f"f2(2150) 5, f2(2550) 5), got {n_wave_geom}.  Check "
            f"`identical_particles`/`cp_particles` in {a.config} — a symmetry "
            f"declaration multiplies the wave axis by n_perm * n_cp (and makes "
            f"the comparison 4x unfair).")

    part = dic["particle"]
    M_top = float(part[part["$top"]]["mass"])
    m_k = float(part["Kp1"]["mass"])
    m_phi = float(part["phires"]["mass"])

    def convert(n, seed):
        t0 = time.perf_counter()
        chunks = []
        for k in range(0, n, a.batch):
            m = min(a.batch, n - k)
            mom_k, _masses = generate_cascade_phsp(M_top, m_k, m, seed=seed + k)
            chunks.append(mom_k)
        mom = np.concatenate(chunks)
        ev = model.build_event_data(mom, kc)
        ev = {k: np.asarray(v) for k, v in ev.items()}
        ev["weight"] = np.ones(len(mom))
        # `build_event_data` already emits `bkg` (ones); the loader does not read
        # that key (`Fitter.load_npz` wants `bkg_raw`, else ones).  The background
        # term is inert because `data.purity: 1.0` zeroes its scale.
        print(f"  {n:,} events: momenta {mom.shape} -> mass {ev['mass'].shape} "
              f"q {ev['q'].shape} angle {ev['angle'].shape} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return ev, mom

    os.makedirs(a.out, exist_ok=True)
    data, mom_data = convert(a.n_data, a.seed)
    phsp, mom_phsp = convert(a.n_phsp, a.seed + 7919)

    # ── the phi, as an MC WEIGHT rather than an amplitude factor ───────────
    #
    # The reference's amplitude carries a FIXED phi BW, so
    #     log|phi·F|² = log|phi|² + log|F|²
    # and the phi contributes a parameter-free constant to the DATA term.  It
    # does not vanish from the normalisation, though: the integral is
    # Σ_phsp |phi|²|F(c)|², a fixed per-event WEIGHT on |F(c)|².  So folding
    # |phi(m12)|² into the phase-space weights lets the amplitude drop it while
    # the likelihood surface stays bit-for-bit the same (verified: gradient
    # cosine 0.99999999997, |Δg|/|g| 4.1e-6 against the config that keeps it).
    #
    # Without the weight the model genuinely changes (cosine −0.13 there).
    # the config's phi block is INERT under `model: one`, so nothing else would
    # notice if it disagreed with the weight's phi — assert it here
    _phi = part["phires"]
    assert (abs(float(_phi["mass"]) - a.phi_mass) < 1e-12
            and abs(float(_phi["width"]) - a.phi_width) < 1e-12), (
        f"{a.config}: phires is {_phi['mass']}/{_phi['width']} but the |BW|^2 "
        f"weight is built with --phi-mass {a.phi_mass} / --phi-width "
        f"{a.phi_width}; the weight identity requires the same phi")
    m12 = np.asarray(phsp["mass"])[:, 0]          # mass slot 0 = the phi pair
    den = (a.phi_mass ** 2 - m12 ** 2) ** 2 + (a.phi_mass * a.phi_width) ** 2
    w_phi = 1.0 / den
    assert np.isfinite(w_phi).all() and (w_phi > 0).all()
    print(f"  phsp weight = |BW_phi(m12)|^2 with m0={a.phi_mass}, "
          f"Gamma={a.phi_width}: [{w_phi.min():.3g}, "
          f"{w_phi.max():.3g}], median {np.median(w_phi):.3g}",
          flush=True)

    # ── validity gates: finite, and inside every interpolation grid ────────
    for tag, ev in (("data", data), ("phsp", phsp)):
        for k in ("mass", "q"):
            v = ev[k]
            assert np.isfinite(v).all(), f"{tag}.{k} has non-finite values"
    m_tab = np.asarray(kc.get("basis_table", kc.get("gamma_table")))
    m_lo = float(kc.get("basis_min", kc.get("m_min", kc.get("gamma_min"))))
    m_hi = m_lo + float(kc.get("basis_delta", kc.get("m_delta",
                              kc.get("gamma_delta")))) * (m_tab.shape[-1] - 1)
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
        # whose vertex resolves to the wrong leaf gives q = 0 for ALL events of
        # that decay, silently zeroing its waves
        med = np.median(ev["q"], axis=0)
        dead = np.flatnonzero(med <= 0)
        assert dead.size == 0, (
            f"{tag}: q column(s) {dead.tolist()} are zero for >=50 % of events -- "
            f"a decay chain collapsed")
        print(f"  {tag}: all {ev['q'].shape[1]} q columns alive "
              f"(min median {med.min():.4g})")

    for name, ev, n in (("data.npz", data, a.n_data), ("phsp.npz", phsp, a.n_phsp)):
        path = os.path.join(a.out, name)
        np.savez(path, **ev)
        print(f"wrote {path} ({n:,} events, {os.path.getsize(path) / 2**20:.1f} MiB)")
    assert np.allclose(phsp["weight"], 1.0), (
        "phsp.npz must keep UNIT weights: the phi weight ships in phsp_wphi.npz, and "
        "a config that keeps the phi in the amplitude would otherwise apply it twice")
    # the weighted copy: `load_dataset` reads the weight from INSIDE the npz
    # (`{prefix}_arr` bypasses the `{prefix}_weight` sidecar), so the phi weight
    # ships as a second file over the SAME events rather than a sidecar.
    wpath = os.path.join(a.out, "phsp_wphi.npz")
    ev_w = dict(phsp)
    ev_w["weight"] = np.asarray(w_phi, np.float64)
    np.savez(wpath, **ev_w)
    print(f"wrote {wpath}  (same events, |BW_phi|^2 weights; config_gen.yml reads it)")

    if a.no_ref_format:
        print("  ref format skipped (--no-ref-format)")
        return

    # ── the reference framework's contract, from the SAME four-momenta ────
    #
    # Mapping gates.  `build_event_data` fills mass slot 0 with the invariant
    # mass of decay 1 (`phires -> Kp1 Km1`) and slot 1 with decay 2
    # (`R -> Kp2 Km2`), so the ref pairing is correct exactly when
    #   M(Kp + Km) == mass[:, 0]   and   M(Pip + Pim) == mass[:, 1].
    # That equality is exact and is the real mapping test.  NOTE: the sample is
    # FLAT four-body phase space, so the phi-pair mass spectrum does NOT peak at
    # the phi (1.019) — the flat density p_top*p12*p34 peaks near 1.5-1.7 GeV.
    # An assertion on a peak position would therefore be a test of the sample's
    # model-weighting, not of the mapping; the exact column equality below tests
    # the mapping on a flat sample.
    for tag, ev, mom, path_name in (
            ("data", data, mom_data, "Momentum_kk.npz"),
            ("phsp", phsp, mom_phsp, "Momentum_kk_phsp.npz")):
        phi_pair = mom[:, 0] + mom[:, 1]
        r_pair = mom[:, 2] + mom[:, 3]
        tot = mom.sum(axis=1)
        m_phi_pair, m_r_pair, m_tot = _mass4(phi_pair), _mass4(r_pair), _mass4(tot)
        print(f"  {tag}: Kp+Km [{m_phi_pair.min():.4f}, {m_phi_pair.max():.4f}] "
              f"(mean {m_phi_pair.mean():.4f})  "
              f"Pip+Pim [{m_r_pair.min():.4f}, {m_r_pair.max():.4f}] "
              f"(mean {m_r_pair.mean():.4f})  "
              f"sum [{m_tot.min():.6f}, {m_tot.max():.6f}]")
        assert np.allclose(m_phi_pair, ev["mass"][:, 0], atol=1e-9), \
            f"{tag}: Kp+Km is not mass column 0 (the phi pair)"
        assert np.allclose(m_r_pair, ev["mass"][:, 1], atol=1e-9), \
            f"{tag}: Pip+Pim is not mass column 1 (the R pair)"
        assert np.allclose(m_tot, M_top, atol=1e-6), \
            f"{tag}: four-momentum sum is not the psi(3686) mass"
        assert np.all(np.abs(tot[:, 1:]).max(axis=0) < 1e-6), \
            f"{tag}: total 3-momentum is not zero (top rest frame)"
        Kp, Km, Pip, Pim = write_ref_format(
            mom, os.path.join(a.out, path_name), M_top)
        assert Kp.shape == (len(mom), 4) and Kp.dtype == np.float64
        assert np.allclose(Kp[:, 3] + Km[:, 3] + Pip[:, 3] + Pim[:, 3], M_top,
                           atol=1e-6), f"{tag}: ref E column does not sum to M_top"
        print(f"wrote {os.path.join(a.out, path_name)} "
              f"({len(mom):,} events, keys Kp/Km/Pip/Pim, (n,4) px,py,pz,E, "
              f"{os.path.getsize(os.path.join(a.out, path_name)) / 2**20:.1f} MiB)")



if __name__ == "__main__":
    main()
