#!/usr/bin/env python
"""Convert the B→4π momentum files in a config yml to the fitter npz arrays.

Scans the config's ``data`` section for *momentum datasets* — any list
of ``.npy`` paths whose key is not an auxiliary array (keys ending in
``_time``, ``_tag1``, ``_eta1``, ``_bg_value``, ``_weight``, ``_arr``,
...) — and converts each one with :func:`ampfit.momenta_to_data` to an
output npz.  All paths resolve relative to the config file directory.

Dataset naming::

    data              -> data_arr             ("data/data_arrays.npz")
    phsp              -> phsp_arr             ("data/phsp_arrays.npz")
    phsp_noeff_sym    -> <no *_arr key>       ("data/phsp_noeff_sym_arrays.npz")

Per-event auxiliary arrays are picked up from the config when present
(mirroring ``examples/create_stack_data.py::read_data``)::

    frac    = tag==0 ? 0.5 : tag>0 ? 1-eta : eta     (from {ds}_tag1/_{ds}_eta1)
    time    = {ds}_time
    bkg_raw = {ds}_bg_value
    weight  = {ds}_weight

Usage::

    python scripts/convert_data_npz.py                       # all datasets
    python scripts/convert_data_npz.py --datasets data,phsp  # selected list
    python scripts/convert_data_npz.py --config Validation/config_angle.yml
    python scripts/convert_data_npz.py --datasets data \\
        --input  /media/.../data_sig.npy \\
        --output data/data_arrays.npz                        # explicit paths
"""
import argparse
import os

import numpy as np
import yaml

from ampfit.momenta_to_data import momenta_to_data

KEYS = ("mass", "q", "angles", "frac", "time", "bkg_raw", "weight")

# config keys that are auxiliary arrays, not momentum datasets
_AUX_SUFFIXES = ("_time", "_tag1", "_eta1", "_bg_value", "_weight",
                 "_l0_weight", "_arr")


def resolve(cfg_dir, val):
    """First element of *val* (a list), joined to the config dir."""
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if not val:
        return None
    return val if os.path.isabs(val) else os.path.join(cfg_dir, val)


def momenta_datasets(d):
    """Keys of the data section that are momentum datasets: lists of
    ``.npy`` paths that are not auxiliary arrays."""
    return [k for k, v in d.items()
            if isinstance(v, (list, tuple)) and v
            and isinstance(v[0], str) and v[0].endswith(".npy")
            and not k.endswith(_AUX_SUFFIXES)]


def aux_path(d, cfg_dir, name, suffix):
    """Resolve the auxiliary array ``{name}*{suffix}`` (exact key first)."""
    exact = name + suffix
    if exact in d:
        return resolve(cfg_dir, d[exact])
    for k, v in d.items():
        if (k.startswith(name) and k.endswith(suffix)
                and isinstance(v, (list, tuple)) and v):
            return resolve(cfg_dir, v)
    return None


def out_path(d, cfg_dir, name):
    """Output npz for a dataset: ``{name}_arr`` in config, else
    ``data/{name}_arrays.npz``."""
    key = name + "_arr"
    if key in d:
        return resolve(cfg_dir, d[key])
    return os.path.join(cfg_dir, "data", f"{name}_arrays.npz")


def convert(src, dst, time_src, tag_src, eta_src, bkg_src, weight_src):
    """Convert one momentum file *src* to the npz *dst*."""
    def _load(p):
        return np.load(p) if p and os.path.exists(p) else None

    mom = np.load(src)
    out = momenta_to_data(mom)
    print(f"  converting {src} ({mom.shape[0]} events)")

    tag = _load(tag_src)
    eta = _load(eta_src)
    if tag is not None and eta is not None:
        if len(tag) != mom.shape[0]:
            raise ValueError(
                f"tag ({len(tag)}) length != momenta ({mom.shape[0]})")
        out["frac"] = np.where(
            tag == 0, 0.5, np.where(tag > 0, 1 - eta, eta))
        print("        frac  from tag + eta")
    time = _load(time_src)
    if time is not None:
        out["time"] = time
        print(f"        time  from {time_src}")
    bkg = _load(bkg_src)
    if bkg is not None:
        out["bkg_raw"] = bkg
        print(f"        bkg   from {bkg_src}")
    weight = _load(weight_src)
    if weight is not None:
        out["weight"] = weight
        print(f"        weight from {weight_src}")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    np.savez(dst, **{k: out[k] for k in KEYS})
    print(f"  wrote {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_angle.yml")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated datasets to convert, e.g. "
                         "'data,phsp' (default: all momentum datasets found "
                         "in the config)")
    ap.add_argument("--input", default=None,
                    help="override the dataset's momenta path (single "
                         "dataset only)")
    ap.add_argument("--output", default=None,
                    help="override the dataset's output npz path (single "
                         "dataset only)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    cfg_dir = os.path.dirname(os.path.abspath(args.config))

    if args.datasets:
        datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        datasets = momenta_datasets(d)
    if not datasets:
        raise ValueError(f"no momentum datasets found in {args.config}")
    if (args.input or args.output) and len(datasets) != 1:
        raise ValueError("--input/--output apply only to a single dataset")
    print(f"datasets: {datasets}")

    for name in datasets:
        if args.input and name != datasets[0]:
            raise ValueError("--input/--output apply only to a single dataset")
        src = args.input if (args.input and len(datasets) == 1) \
            else resolve(cfg_dir, d.get(name))
        dst = args.output if (args.output and len(datasets) == 1) \
            else out_path(d, cfg_dir, name)
        if not src or not os.path.exists(src):
            raise FileNotFoundError(
                f"momenta for {name!r} not found at {src!r} — set --input or "
                f"fix {name} in {args.config}")
        convert(src, dst,
                aux_path(d, cfg_dir, name, "_time"),
                aux_path(d, cfg_dir, name, "_tag1"),
                aux_path(d, cfg_dir, name, "_eta1"),
                aux_path(d, cfg_dir, name, "_bg_value"),
                aux_path(d, cfg_dir, name, "_weight"))


if __name__ == "__main__":
    main()
