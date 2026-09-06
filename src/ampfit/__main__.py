"""
``python -m ampfit <command> [args...]`` — run the ampfit scripts from the
repository (or the current module tree).

Examples::

    python -m ampfit run_fit --config config.yml --fit -l 5
    python -m ampfit plot_pwa_groups results.json --config config_pwa.yml
    python -m ampfit gen_toy_pwa --config config_pwa.yml
    python -m ampfit --list

Resolution order for *command*:
    1. a repository/scripts script  ``<root>/scripts/<command>.py``
    2. a repository top-level script ``<root>/<command>.py``   (e.g. run_fit)

The chosen script is executed with ``runpy.run_path(..., run_name="__main__")``
so its own argparse/usage/relative paths behave exactly as if invoked
directly.
"""

import argparse
import os
import runpy
import sys

try:  # repo layout: ampfit package under <repo>/src
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if not os.path.isdir(os.path.join(_ROOT, "scripts")):
        raise OSError
except OSError:  # installed layout (no repo scripts available)
    _ROOT = ""

_CURATED = {
    "run_fit": "run_fit.py",
    "plot_pwa_groups": "plot_pwa_groups.py",
    "plot_pw_groups": "plot_pw_groups.py",
    "plot_pw_ls": "plot_pw_ls.py",
    "plot_pw_resonance": "plot_pw_resonance.py",
    "gen_toy_pwa": "gen_toy_pwa.py",
    "save_chain_data": "save_chain_data.py",
}


def _resolve(name):
    """Return the script path for a command name, or None."""
    root = _ROOT
    if not root:
        return None
    rel = _CURATED.get(name, name + ".py")
    for base in (os.path.join(root, "scripts"), root):
        path = os.path.join(base, rel)
        if os.path.isfile(path):
            return path
    # direct fallback: <root>/<name>.py without forcing scripts/ first
    path = os.path.join(root, rel)
    return path if os.path.isfile(path) else None


def _available():
    if not _ROOT:
        return sorted(_CURATED)
    found = set()
    for base in (os.path.join(_ROOT, "scripts"), _ROOT):
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.endswith(".py") and not fn.startswith("_"):
                found.add(fn[:-3])
    return sorted(found)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m ampfit",
        description="Run ampfit scripts/fits from the current module tree.")
    parser.add_argument("command", nargs="?", help="script name, e.g. run_fit")
    parser.add_argument("--list", action="store_true", dest="list_",
                        help="list available commands")
    args, rest = parser.parse_known_args(argv)

    if args.list_ or args.command is None:
        avail = _available()
        print("ampfit commands:" if args.command is None
              else "available ampfit commands:")
        for name in avail:
            path = _resolve(name) or ""
            print(f"  {name:<22s} {path}")
        return 0

    path = _resolve(args.command)
    if path is None:
        avail = _available()
        print(f"ampfit: unknown command {args.command!r}. Available:",
              file=sys.stderr)
        for name in avail:
            print(f"  {name}", file=sys.stderr)
        return 2

    # make the script see the same argv as a direct invocation
    sys.argv = [path] + rest
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
