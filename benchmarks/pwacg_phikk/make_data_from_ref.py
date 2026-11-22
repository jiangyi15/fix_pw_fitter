#!/usr/bin/env python3
"""Build this framework's kernel-format samples from the reference's own data.

The reference (PWACG, arXiv:2403.09225) publishes its demo sample as a GitHub
release asset — `v1.0.0`, "Data", `data.zip` (≈788 MB,
`releases/download/v1.0.0/data.zip`).  That archive contains the per-event
four-momenta the paper's benchmark uses, so the honest comparison times **their
events** through both frameworks.  This script is the bridge: it reads their
`Momentum_kk.npz`, converts it to this framework's event arrays, and writes
`data/{data,phsp}.npz` — the same files `make_data.py` produces from a
generated sample, so `bench.py` needs no change.

    python make_data_from_ref.py --zip /path/to/data.zip
    python make_data_from_ref.py --ref-dir /path/to/extracted/real_data/

## The reference's array contract (verified against their sources)

`Tensor/CacheTensor.py` reads `Momentum_kk.npz` with exactly four keys and
**columns `(px, py, pz, E)`** — spatial first, energy last: it flips the spatial
part and dots with the metric `diag(-1, -1, -1, 1)` (`Tensor/BASE.py`).

    Kp, Km    the φ pair        (Kp1, Km1 in this benchmark's `dat_order`)
    Pip, Pim  the f pair        (Kp2, Km2)

`Pip`/`Pim` are a misnomer carried over from a ππ model; in the kk demo they are
the kaons of the resonance side.  This framework's `build_event_data` wants
**(E, px, py, pz)**, which is the one transformation applied here.

## Which archive members

The paper's sample is the *real-data* set (`real_data/`, 10^4 events).  The MC
integral set (`mc_int/`, ~1.4×10^6 events in their own slicing) is the phase-space
side; when it is present it becomes `data/phsp.npz`, otherwise the data set is
copied as the phase space so `bench.py` still runs (a timing sample, clearly
labelled as such in the printed header).
"""
import argparse
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tabpwa import Fitter                     # noqa: E402

#: reference key -> this benchmark's `dat_order` entry
REF_KEYS = {"Kp": "Kp1", "Km": "Km1", "Pip": "Kp2", "Pim": "Km2"}


def load_ref_momenta(path):
    """Their `(px, py, pz, E)` arrays -> `(n, 4, 4)` in `(E, px, py, pz)`."""
    with np.load(path) as d:
        missing = sorted(set(REF_KEYS) - set(d.files))
        if missing:
            raise SystemExit(
                f"{path}: missing {missing}; the reference contract is exactly "
                f"Kp/Km/Pip/Pim, got {sorted(d.files)}")
        arrays = {k: np.asarray(d[k], np.float64) for k in REF_KEYS}
    n = {a.shape[0] for a in arrays.values()}
    if len(n) != 1:
        raise SystemExit(f"{path}: the four arrays differ in length: "
                         f"{ {k: a.shape for k, a in arrays.items()} }")
    bad = [k for k, a in arrays.items() if a.ndim != 2 or a.shape[1] != 4]
    if bad:
        raise SystemExit(f"{path}: {bad} are not (n, 4): "
                         f"{ {k: arrays[k].shape for k in bad} }")
    order = ["Kp1", "Km1", "Kp2", "Km2"]
    inv = {v: k for k, v in REF_KEYS.items()}
    out = np.empty((arrays["Kp"].shape[0], 4, 4), np.float64)
    for i, name in enumerate(order):
        px, py, pz, E = (arrays[inv[name]][:, j] for j in range(4))
        out[:, i] = np.stack([E, px, py, pz], axis=-1)
    return out


def find_in_zip(zpath, wanted):
    """First member whose basename is *wanted*, preferring a real_/mc_ prefix."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if os.path.basename(n) == wanted]
        if not names:
            return None
        pref = [n for n in names if "real" in n.lower()] or names
        return sorted(pref)[0]


def convert(ref_path, f, out_dir, tag):
    kc = f.model.build_kernel_config()
    momenta = load_ref_momenta(ref_path)
    data = f.model.build_event_data(momenta, kc)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{tag}.npz")
    np.savez(dst, **data)
    mass = np.asarray(data["mass"])
    q = np.asarray(data["q"])
    print(f"{tag:5s} {momenta.shape[0]:>9,} events -> {dst}")
    print(f"      mass [{mass.min():.4f}, {mass.max():.4f}]  "
          f"q [{q.min():.4f}, {q.max():.4f}]")
    return dst, mass, q


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="the release asset, data.zip")
    src.add_argument("--ref-dir", help="an extracted directory holding "
                                       "Momentum_kk.npz (and optionally an "
                                       "mc_int/ set)")
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--max-events", type=int, default=0,
                    help="keep only the first N events of each sample")
    args = ap.parse_args()

    tmp = None
    if args.zip:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="pwacg_ref_extract_")
        for wanted in ("Momentum_kk.npz",):
            member = find_in_zip(args.zip, wanted)
            if member is None:
                raise SystemExit(f"{args.zip}: no {wanted} inside — is this the "
                                 f"v1.0.0 'Data' asset?")
            with zipfile.ZipFile(args.zip) as z:
                z.extract(member, tmp)
            ref_data = os.path.join(tmp, member)
        # the archive may also carry an MC-integral set; use it when it does
        mc_member = find_in_zip(args.zip, "Momentum_kk_mc.npz")
        if mc_member is None:
            with zipfile.ZipFile(args.zip) as z:
                cand = [n for n in z.namelist()
                        if os.path.basename(n) == "Momentum_kk.npz" and "mc" in n.lower()]
            mc_member = sorted(cand)[0] if cand else None
        ref_phsp = ref_data          # the MC set is optional; see below
        if mc_member is not None:
            with zipfile.ZipFile(args.zip) as z:
                z.extract(mc_member, tmp)
            ref_phsp = os.path.join(tmp, mc_member)
    else:
        ref_data = os.path.join(args.ref_dir, "Momentum_kk.npz")
        cand = [os.path.join(args.ref_dir, "mc_int", "Momentum_kk.npz"),
                os.path.join(args.ref_dir, "Momentum_kk_phsp.npz"),
                os.path.join(os.path.dirname(args.ref_dir.rstrip("/")),
                             "mc_int", "Momentum_kk.npz")]
        ref_phsp = next((c for c in cand if os.path.exists(c)), ref_data)
        if not os.path.exists(ref_data):
            raise SystemExit(f"{ref_data} not found")

    f = Fitter(args.config)
    f.apply_constrains()

    # --max-events is applied by slicing the reference's arrays before conversion
    def _slice(path):
        if not args.max_events:
            return path
        with np.load(path) as d:
            out = path + f".{args.max_events}.npz"
            np.savez(out, **{k: d[k][:args.max_events] for k in d.files})
        return out

    ref_data = _slice(ref_data)
    if ref_phsp != ref_data:
        ref_phsp = _slice(ref_phsp)
    d_dst, mass, _q = convert(ref_data, f, args.out, "data")
    if ref_phsp == ref_data:
        print("   note: no MC-integral set found; the phase space is a COPY of "
              "the data sample (a timing sample, not their integral)")
        p_dst = os.path.join(args.out, "phsp.npz")
        np.savez(p_dst, **dict(np.load(d_dst)))
        print(f"phsp  (copy of data) -> {p_dst}")
    else:
        p_dst, _m, _q = convert(ref_phsp, f, args.out, "phsp")

    # The reference's amplitude carries a FIXED phi BW, so the phi is a
    # parameter-free constant in the data term and a per-event WEIGHT on the
    # normalisation.  Ship the weighted phase-space copy (`config_gen.yml` reads
    # it); `phsp.npz` keeps unit weights for the config that keeps the phi.
    m_phi = float(f.config.dic["particle"]["phires"]["mass"])
    g_phi = float(f.config.dic["particle"]["phires"]["width"])
    with np.load(p_dst) as d:
        m12 = np.asarray(d["mass"])[:, 0]
        ev_w = {k: np.asarray(d[k]) for k in d.files}
    ev_w["weight"] = 1.0 / ((m_phi ** 2 - m12 ** 2) ** 2 + (m_phi * g_phi) ** 2)
    w_dst = os.path.join(args.out, "phsp_wphi.npz")
    np.savez(w_dst, **ev_w)
    print(f"wrote {w_dst}  (same events, |BW_phi|^2 weights)")

    # The converted events must lie inside the interpolation tables, or a kernel
    # reads outside its grid and the timing measures garbage.  Same gate as
    # make_data.py (which derives the extents the way the builder emits them).
    kc = f.model.build_kernel_config()
    m_tab = np.asarray(kc.get("basis_table", kc.get("gamma_table")))
    m_lo = float(kc.get("basis_min", kc.get("m_min", kc.get("gamma_min"))))
    m_hi = m_lo + float(kc.get("basis_delta", kc.get("m_delta",
                              kc.get("gamma_delta")))) * (m_tab.shape[-1] - 1)
    q_t = np.asarray(kc["fl_table"])
    q_lo = float(kc.get("fl_min", kc.get("q_min")))
    q_hi = q_lo + float(kc.get("fl_delta", kc.get("q_delta"))) * (q_t.shape[-1] - 1)
    print(f"      tables: mass [{m_lo:.4f}, {m_hi:.4f}] q [{q_lo:.4f}, {q_hi:.4f}]")
    for tag, path in (("data", d_dst), ("phsp", p_dst)):
        with np.load(path) as d:
            mass, q = np.asarray(d["mass"]), np.asarray(d["q"])
        assert np.isfinite(mass).all() and np.isfinite(q).all(), tag
        assert m_lo <= mass.min() and mass.max() <= m_hi, (
            f"{tag}: mass [{mass.min():.4f}, {mass.max():.4f}] is outside the "
            f"table [{m_lo:.4f}, {m_hi:.4f}] — these events cannot be timed "
            f"against this config")
        assert q_lo <= q.min() and q.max() <= q_hi, (
            f"{tag}: q [{q.min():.4f}, {q.max():.4f}] outside [{q_lo:.4f}, "
            f"{q_hi:.4f}]")
    print("wrote", d_dst, "and", p_dst)


if __name__ == "__main__":
    main()
