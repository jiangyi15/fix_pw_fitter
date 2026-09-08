"""Driver — run the 3 resolution-group scripts in one call:

    1. save_chain_data.py           momenta  -> per-event chain data .npy
    2. smear_chain_groups.py        chain data + per-event resolutions
                                    -> smeared group copies .npy
    3. chain_groups_to_momenta.py   group copies -> final group momenta .npy

The final output is ``--out``; the two intermediates are written into
``--work`` (a fresh temp dir unless given) with fixed names.

Example:
    python scripts/make_resolution_groups.py \\
        --config config.yml --chain pipeta \\
        --data ../data2/data_momenta.npy --copies 20 \\
        --sigma-mass sigma_a2p.npy \\
        --out data_momenta_groups.npy
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(name, argv):
    print(f"\n== {name} ==", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, name)] + argv)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed with {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--data", required=True,
                    help="original event momenta .npy (n, n_finals, 4)")
    ap.add_argument("--copies", type=int, default=20)
    ap.add_argument("--sigma-mass", help="(n,) or (n, n_res) per-event mass σ")
    ap.add_argument("--sigma-phi", help="(n,) or (n, nv) per-event φ σ")
    ap.add_argument("--sigma-theta", help="(n,) or (n, nv) per-event θ σ")
    ap.add_argument("--reflect-mass", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=None,
                    help="dir for the two intermediates "
                         "(default: a fresh temp dir)")
    ap.add_argument("--out", default="data_momenta_groups.npy")
    args = ap.parse_args()

    keep = args.work is not None
    work = args.work or tempfile.mkdtemp(prefix="v5groups_")
    os.makedirs(work, exist_ok=True)
    chain_npy = os.path.join(work, "chain.npy")
    groups_npy = os.path.join(work, "groups.npy")

    common = ["--config", args.config, "--chain", args.chain]

    _run("save_chain_data.py",
         common + ["--data", args.data, "--out",
                   os.path.splitext(chain_npy)[0]])
    _run("smear_chain_groups.py",
         common + ["--chain-data", chain_npy, "--copies", str(args.copies),
                   "--seed", str(args.seed),
                   "--reflect-mass", str(args.reflect_mass)]
         + (["--sigma-mass", args.sigma_mass] if args.sigma_mass else [])
         + (["--sigma-phi", args.sigma_phi] if args.sigma_phi else [])
         + (["--sigma-theta", args.sigma_theta] if args.sigma_theta else [])
         + ["--out", os.path.splitext(groups_npy)[0]])
    _run("chain_groups_to_momenta.py",
         common + ["--groups", groups_npy, "--out", args.out])

    print(f"\nfinal group momenta: {os.path.abspath(args.out)}")
    if not keep:
        print(f"intermediates in temp dir (deleted on exit? no — kept for "
              f"inspection): {work}")
    else:
        print(f"intermediates: {chain_npy}, {groups_npy}")


if __name__ == "__main__":
    main()
