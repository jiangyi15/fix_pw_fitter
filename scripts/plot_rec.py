"""Plot resolution-fit results at the ORIGINAL-event level.

Config convention (extra keys under ``data:``):

    data:      data_momenta_smeared.npy   (n_events x res_size, 4)  → v5_pwa
    data_rec:  data_momenta_orig.npy      (n_events, 4)  original data rows
    phsp:      mc_momenta.npy              phase space used for the weights
    phsp_rec:  (optional) smeared phase-space rows IF phsp itself carries
               the resolution copies

The DATA file is the one fed to cuda_v5_pwa, so it carries the resolution
size (original event ``e`` owns rows ``[e*s, (e+1)*s)``).  The phase space
does NOT have to carry a resolution size: by default ``phsp`` is a plain
per-event sample and the weights are its normalised amplitude directly.  If
``phsp`` IS smeared (len(phsp) is an integer multiple of len(phsp_rec)),
the per-row pdf is first group-summed back to each original phsp event.

Everything is plotted at ORIGINAL-event level ({prefix}_rec):

* weights  : fitted pdf on the phase space, P_i/norm (smeared rows are
             group-summed to their original event when present);
* variable : taken from the ORIGINAL rows of ``phsp_rec`` (falls back to
             ``phsp``) and ``data_rec`` — one entry per original event, so
             no resolution-size factor appears in the counts.

Per projection (final-state invariant masses) the data_rec histogram is
compared with the phsp histogram weighted by ``w`` (area-normalised to the
data).  Plots are written into ``{prefix}_rec/``.

Usage:
    python scripts/plot_rec.py --config config.yml \\
        --result config_fit_results.json \\
        --prefix plots --backend cuda_v4_pwa
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ampfit.config_loader import Config                       # noqa: E402
from ampfit.pwa_build import pwa_event_data_tree              # noqa: E402


def _inv_mass(p4):
    e2 = p4[..., 0] ** 2
    p2 = np.sum(p4[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.clip(e2 - p2, 0.0, None))


def _load_event_arrays(cfg, kc, byt, mom_path):
    mom = np.load(mom_path)
    d = pwa_event_data_tree(cfg, kc, byt, mom)
    return d, mom


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--result", required=True)
    ap.add_argument("--backend", default="cuda_v4_pwa")
    ap.add_argument("--prefix", default="plots",
                    help="output dir becomes {prefix}_rec/")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--skip-plot", action="store_true",
                    help="only write the weighted histograms (.npy)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config(args.config)
    keys = cfg.dic["data"]
    for k in ("data", "phsp", "data_rec"):
        if k not in keys:
            raise SystemExit(f"config data: lacks '{k}' (see script docstring)")

    kc = cfg.build_all_index()
    pws = list(cfg.full_decay.get_partial_waves())
    byt = {cfg.topo_index[ch.topo_id()]: ch for _, ch in pws}

    # smeared (resolution) data + original rows; smeared/original phase space
    _d_smear, mom_smear = _load_event_arrays(cfg, kc, byt, keys["data"])
    _p_smear, mom_p = _load_event_arrays(cfg, kc, byt, keys["phsp"])
    _d_rec, mom_data_rec = _load_event_arrays(cfg, kc, byt, keys["data_rec"])

    p_rec_path = keys.get("phsp_rec")
    phsp_is_smeared = bool(p_rec_path) and os.path.isfile(p_rec_path)
    if phsp_is_smeared:
        _p_rec, mom_phsp_rec = _load_event_arrays(cfg, kc, byt, p_rec_path)
    else:
        mom_phsp_rec = mom_p           # phsp has no resolution size

    nd_s, nd_r = mom_smear.shape[0], mom_data_rec.shape[0]
    np_s, np_r = mom_p.shape[0], mom_phsp_rec.shape[0]
    if nd_s % nd_r or nd_s < nd_r:
        raise SystemExit("data size must be data_rec size x resolution_size")
    s_data = nd_s // nd_r
    s_phsp = np_s // np_r if np_s % np_r == 0 and phsp_is_smeared else 1
    print(f"resolution copies: data {nd_s}/{nd_r} = {s_data}, "
          f"phsp {'smeared ' + str(np_s) + '/' + str(np_r) + ' = ' + str(s_phsp) if phsp_is_smeared else '(none)'}")

    # ── fitted amplitude on the phase space → weights ──
    from ampfit import Fitter
    f = Fitter(args.config, backend=args.backend)
    f.apply_constrains()
    with open(args.result) as fh:
        res = json.load(fh)
    x0 = f.values_from_dict(res)
    f.load_fixed_from_dict(res)
    params, _ = f.build_params(x0)

    hp = f.backend.load_data(_p_smear)
    try:
        norm, _g, Pp = f.backend.compute(params, hp, norm=None)
        pdf = Pp / norm
        if s_phsp > 1:
            w_orig = pdf.reshape(np_r, s_phsp).sum(axis=1)
        else:
            w_orig = pdf
    finally:
        hp.free()

    # ── projections: invariant masses of every final pair ──
    finals = list(cfg.finals)
    pairs = [(i, j) for i in range(len(finals)) for j in range(i + 1,
                                                               len(finals))]
    labels = [finals[i] + finals[j] for i, j in pairs]

    data_var = [_inv_mass(mom_data_rec[:, i] + mom_data_rec[:, j])
                for i, j in pairs]
    phsp_var = [_inv_mass(mom_phsp_rec[:, i] + mom_phsp_rec[:, j])
                for i, j in pairs]

    outdir = args.prefix + "_rec"
    os.makedirs(outdir, exist_ok=True)

    n_plot = 0
    for lab, dv, pv in zip(labels, data_var, phsp_var):
        lo = min(float(dv.min()), float(pv.min()))
        hi = max(float(dv.max()), float(pv.max()))
        edges = np.linspace(lo, hi, args.bins + 1)
        ndat, _ = np.histogram(dv, bins=edges)
        nmodel, _ = np.histogram(pv, bins=edges, weights=w_orig)
        scale = ndat.sum() / nmodel.sum() if nmodel.sum() > 0 else 1.0
        nmodel = nmodel * scale
        np.savez(os.path.join(outdir, f"hist_{lab}.npz"),
                 edges=edges, data=ndat, model=nmodel, weight=w_orig)
        if args.skip_plot:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        cx = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(cx, ndat, fmt="o", ms=3, lw=1, label="data_rec")
        ax.step(cx, nmodel, where="mid", label="model (phsp_rec weighted)")
        ax.set_xlabel(f"m({lab}) [GeV]")
        ax.set_ylabel("counts / original event")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"m_{lab}.png"), dpi=130)
        plt.close(fig)
        n_plot += 1

    print(f"wrote {n_plot} projections into {outdir}/  "
          f"(norm=Σ P over smeared phsp = {float(norm):.4g})")


if __name__ == "__main__":
    main()
