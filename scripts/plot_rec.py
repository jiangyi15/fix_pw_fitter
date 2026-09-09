"""Plot resolution-fit results: projection distribution vs model curve.

Config convention (extra keys under ``data:``):

    data_rec:  original (unexpanded) data rows     (n_data, 4)   — histogram
    phsp:      phase space used to build the model weights  (n_phsp, 4)
    phsp_rec:  (optional) phase-space rows whose VARIABLE enters the model
               histogram — when absent it is ``phsp`` itself

The smeared ``data`` file fed to cuda_v5_pwa is NOT used here: every
variable comes from the original ``data_rec`` / ``phsp_rec`` rows, one
entry per original event.

Model curve:
* each ``phsp`` row carries the fitted amplitude weight  w_i = P_i / norm
  (``norm`` = Σ w·P over the whole ``phsp``),
* when ``phsp`` carries the resolution copies (``phsp : phsp_rec = n : 1``)
  the per-row weights are summed back to their original phsp event:
  ``w = w.reshape(-1, n).sum(axis=-1)`` (``n = len(phsp)/len(phsp_rec)``);
  with ``n = 1`` the weights are used as they are,
* the model histogram is built over the ``phsp_rec`` variable with ``w``.

The model curve is drawn WITHOUT any count-ratio scaling (its total area is
1); pass ``--area-scale`` to additionally scale Σ weights to the data_rec
event count when the heights should match.

``phsp_rec`` only supplies the bin variable; if it is resolution-smeared
rows, no weights are available for them and the model side is meaningless —
use the original phase space here.

Per projection (invariant masses of every final pair) the script writes
``{prefix}_rec/m_<pair>.png`` and ``{prefix}_rec/hist_<pair>.npz``.

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


def _load_events(cfg, kc, byt, mom_path):
    mom = np.load(mom_path)
    return pwa_event_data_tree(cfg, kc, byt, mom), mom


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--result", required=True)
    ap.add_argument("--backend", default="cuda_v4_pwa")
    ap.add_argument("--prefix", default="plots",
                    help="output dir becomes {prefix}_rec/")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--area-scale", action="store_true",
                    help="scale Σ(model weights) up to the data_rec count")
    ap.add_argument("--skip-plot", action="store_true",
                    help="only write the weighted histograms (.npy)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config(args.config)
    keys = cfg.dic["data"]
    for k in ("data_rec", "phsp"):
        if k not in keys:
            raise SystemExit(f"config data: lacks '{k}' (see script docstring)")

    kc = cfg.build_all_index()
    pws = list(cfg.full_decay.get_partial_waves())
    byt = {cfg.topo_index[ch.topo_id()]: ch for _, ch in pws}

    ev_phsp, mom_p = _load_events(cfg, kc, byt, keys["phsp"])
    _ev_drec, mom_data_rec = _load_events(cfg, kc, byt, keys["data_rec"])

    p_rec_path = keys.get("phsp_rec")
    if p_rec_path and os.path.isfile(p_rec_path):
        _ev_prec, mom_phsp_rec = _load_events(cfg, kc, byt, p_rec_path)
    else:
        mom_phsp_rec = mom_p

    n_w = mom_p.shape[0]                 # weight rows (possibly smeared)
    n_v = mom_phsp_rec.shape[0]          # original variable rows
    if n_w < n_v or n_w % n_v:
        raise SystemExit(
            f"phsp rows ({keys['phsp']}: {n_w}) must be 1:1 with or an "
            f"integer multiple (resolution copies) of the phsp_rec rows "
            f"({p_rec_path or keys['phsp']}: {n_v})")
    n_res = n_w // n_v                   # 1 = no resolution copies
    n_phsp = n_v
    n_data = mom_data_rec.shape[0]

    # ── fitted amplitude on the phase space → per-row weights ──
    from ampfit import Fitter
    f = Fitter(args.config, backend=args.backend)
    f.apply_constrains()
    with open(args.result) as fh:
        res = json.load(fh)
    x0 = f.values_from_dict(res)
    f.load_fixed_from_dict(res)
    params, _ = f.build_params(x0)

    hp = f.backend.load_data(ev_phsp)
    try:
        norm, _g, Pp = f.backend.compute(params, hp, norm=None)
        w = Pp / norm                       # pdf per (smeared) phsp row
    finally:
        hp.free()
    if n_res > 1:
        w = w.reshape(n_phsp, n_res).sum(axis=-1)   # sum copies of event
    print(f"weights: {n_w} phsp rows -> {n_phsp} original events "
          f"(resolution {n_res}), norm=Σ P = {float(norm):.4g}, "
          f"Σ w = {float(w.sum()):.4g}")

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
        nmodel, _ = np.histogram(pv, bins=edges, weights=w)
        if args.area_scale and nmodel.sum() > 0:
            nmodel = nmodel * (n_data / nmodel.sum())
        np.savez(os.path.join(outdir, f"hist_{lab}.npz"),
                 edges=edges, data=ndat, model=nmodel, weight=w,
                 area_scaled=bool(args.area_scale))
        if args.skip_plot:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        cx = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(cx, ndat, fmt="o", ms=3, lw=1, label="data_rec")
        ax.step(cx, nmodel, where="mid", label="model (weighted phsp)")
        ax.set_xlabel(f"m({lab}) [GeV]")
        ax.set_ylabel("counts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"m_{lab}.png"), dpi=130)
        plt.close(fig)
        n_plot += 1

    print(f"wrote {n_plot} projections into {outdir}/")


if __name__ == "__main__":
    main()
