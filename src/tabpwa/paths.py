"""Global file-path remapping — one config, local and remote file trees.

A config often carries **absolute paths from the machine it was written
on** (a batch farm, a shared cluster directory).  Instead of editing the
YAML for every run, bind a source prefix to a local target; every path
resolved through :func:`resolve_path` is remapped **before** loading
(longest-prefix match, exact match allowed, everything else untouched)::

    # config.yml (written for the farm)
    data:
      data_arr: /farm/run42/data.npz
      phsp_arr: /farm/run42/phsp.npz
    particle:
      rho:
        model: width_linear_npy
        file: /farm/tables/rho_gamma.npy

    from tabpwa import paths

    # process-wide: /farm/run42 -> ./inputs, /farm/tables -> ./tables
    paths.bind_paths(["/farm/run42:./inputs", "/farm/tables:./tables"])
    fitter = Fitter("config.yml")          # loads the local files

The registry is **global and external** — the config file is never
modified (no ``path_map`` key is read from or written into it), and every
load site (config loader, particle models, scripts) sees the same map.
Effective map, lowest to highest priority:

1. the ``TABPWA_PATH_MAP`` environment variable,
2. the process-wide registry (:func:`bind_paths` / :func:`clear_paths`),
3. the innermost :func:`path_scope` context (nested, restores on exit).

Textual form is ``FROM:TO``; several pairs are supplied as a dict, a list,
one ``;``/newline-separated string, or JSON::

    paths.bind_paths("/remote/a:/local/a;/remote/b:/local/b")
    with paths.path_scope({"~/scratch": "/tmp/run1"}):   # temporary
        fitter = Fitter("config.yml")
    paths.clear_paths()

    python run_fit.py --config config.yml \\
        --path-map /farm/run42:./inputs --path-map /farm/tables:./tables
    TABPWA_PATH_MAP="/farm/run42:./inputs;/farm/tables:./tables" \\
        python run_fit.py

A pair is split on the **first** ``:`` (Linux paths rarely contain one); a
JSON object (``{FROM: TO}``) is the escape hatch for paths that do — e.g.
Windows drive letters.  Overlapping prefixes: the **longest** match wins;
duplicate exact keys: the **later** map wins.
"""
import contextlib
import contextvars
import json
import os

__all__ = [
    "parse_path_map", "path_map_from_env", "path_map_from_file",
    "merge_path_maps", "apply_path_map", "resolve_path",
    "bind_paths", "clear_paths", "path_scope", "current_path_map",
]

# Process-wide bindings (later binds win per key); ``path_scope`` layers on
# top of these in a contextvars scope (per thread/task, restored on exit).
_GLOBAL_PATH_MAP = {}
_SCOPE_PATH_MAP = contextvars.ContextVar("tabpwa_path_map_scope",
                                         default=None)


def _validate_pair(src, dst):
    """Both sides of one binding must be non-empty strings."""
    src = "" if src is None else str(src).strip()
    dst = "" if dst is None else str(dst).strip()
    if not src or not dst:
        raise ValueError(
            f"path_map entries need a non-empty source and target "
            f"(got {src!r} -> {dst!r})")
    return src, dst


def _split_pairs(text):
    """``FROM:TO`` entries separated by ``;`` or newlines → list of strings.

    Blank lines and ``#`` comments are ignored.
    """
    entries = []
    for chunk in str(text).replace("\r", "\n").split(";"):
        for line in chunk.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def _parse_pair(entry):
    """One ``"FROM:TO"`` entry → ``(from, to)`` (split on the first ``:``)."""
    text = str(entry).strip()
    if ":" not in text:
        raise ValueError(
            f"path_map entry {text!r} is not of the form FROM:TO (paths "
            f"containing ':' need the JSON object form)")
    src, dst = text.split(":", 1)
    return _validate_pair(src, dst)


def _parse_pairs(text):
    out = {}
    for entry in _split_pairs(text):
        src, dst = _parse_pair(entry)
        out[src] = dst                 # later duplicate keys win
    return out


def parse_path_map(spec):
    """Normalise a ``path_map`` specification to a plain ``{str: str}``.

    Accepted forms:

    * ``None`` / ``""`` → ``{}``;
    * a ``dict`` ``{FROM: TO}`` (copied, both sides stringified/validated;
      later duplicate keys win);
    * a single ``"FROM:TO"`` string;
    * a list/iterable of ``"FROM:TO"`` strings (later entries win);
    * one delimited string: ``"/a:/x;/b:/y"`` (``;`` or newlines between
      pairs);
    * a JSON object string (escape hatch, e.g. Windows drive letters).

    A pair is split on the **first** ``:``.  Both sides must be non-empty.
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        out = {}
        for key, val in spec.items():
            src, dst = _validate_pair(key, val)
            out[src] = dst             # later duplicate keys win
        return out
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return {}
        if text.startswith("{"):
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError(
                    "path_map JSON must be an object {FROM: TO}; got "
                    f"{type(loaded).__name__}")
            return parse_path_map(loaded)
        return _parse_pairs(text)
    if isinstance(spec, os.PathLike):
        return parse_path_map(os.fspath(spec))
    if hasattr(spec, "__iter__"):
        out = {}
        for item in spec:
            if item is None:
                continue
            out.update(parse_path_map(item))
        return out
    raise TypeError(
        f"cannot interpret path_map of type {type(spec).__name__}: {spec!r}")


# ── the registry ──────────────────────────────────────────────────

def bind_paths(spec):
    """Bind file-path prefixes **process-wide** (later binds win per key).

    *spec* takes every form :func:`parse_path_map` accepts (dict,
    ``"FROM:TO"``, a list of pairs, a ``;``/newline-separated string,
    JSON).  Returns the updated global map (a copy).  The bindings apply
    to every later :func:`resolve_path` call — config data files,
    particle-model tables, script-side loads — until :func:`clear_paths`.
    """
    _GLOBAL_PATH_MAP.update(parse_path_map(spec))
    return dict(_GLOBAL_PATH_MAP)


def clear_paths():
    """Forget all process-wide bindings (:func:`bind_paths`).

    An active :func:`path_scope` is unaffected — it restores on exit.
    """
    _GLOBAL_PATH_MAP.clear()


@contextlib.contextmanager
def path_scope(spec=None):
    """Temporarily add bindings for the duration of a ``with`` block.

    The scope layers **over** the global registry (and the environment),
    nested scopes layer over each other, and everything is restored on
    exit — e.g. one run against a different mount::

        with paths.path_scope({"/farm/run42": "/scratch/run42"}):
            fitter = Fitter("config.yml")

    Yields the effective map inside the scope.
    """
    scope = dict(_SCOPE_PATH_MAP.get() or {})
    scope.update(parse_path_map(spec))
    token = _SCOPE_PATH_MAP.set(scope)
    try:
        yield current_path_map()
    finally:
        _SCOPE_PATH_MAP.reset(token)


def current_path_map():
    """The effective map: env < global registry < innermost scope."""
    out = parse_path_map(os.environ.get("TABPWA_PATH_MAP"))
    out.update(_GLOBAL_PATH_MAP)
    out.update(_SCOPE_PATH_MAP.get() or {})
    return out


def path_map_from_env(env="TABPWA_PATH_MAP"):
    """``path_map`` from an environment variable (empty/unset → ``{}``)."""
    return parse_path_map(os.environ.get(env))


def path_map_from_file(path):
    """``path_map`` from a JSON file or a ``"FROM:TO"`` line / ``;`` file."""
    with open(path) as f:
        return parse_path_map(f.read())


def merge_path_maps(*maps):
    """Merge several maps — later maps win per (exact) key."""
    out = {}
    for m in maps:
        if not m:
            continue
        out.update(m if isinstance(m, dict) else parse_path_map(m))
    return out


def _norm(path):
    """``expanduser`` + ``expandvars`` + ``normpath`` as a ``str``."""
    if not isinstance(path, str):
        path = os.fspath(path)
    return os.path.normpath(
        os.path.expanduser(os.path.expandvars(str(path))))


def _under(path, prefix):
    """True when *path* is *prefix* itself or lies below it (path boundary).

    ``/database`` is NOT under ``/data`` — the prefix must end on a
    path-component boundary.
    """
    if path == prefix:
        return True
    root = prefix.rstrip(os.sep)
    if not root:                      # prefix is the filesystem root
        return path.startswith(os.sep)
    return path.startswith(root + os.sep)


def _map_one(path, path_map):
    srcs = sorted(path_map, key=len, reverse=True)
    p = _norm(path)
    for src in srcs:
        s = _norm(src)
        if not _under(p, s):
            continue
        dst = _norm(path_map[src])
        rel = os.path.relpath(p, s)
        return dst if rel == os.curdir else os.path.normpath(
            os.path.join(dst, rel))
    return path


def apply_path_map(path, path_map):
    """Remap *path* through *path_map* using the **longest** matching prefix.

    Prefixes (and the looked-up path) are normalised with ``expanduser`` /
    ``expandvars`` / ``normpath``; matching is component-wise (``/data``
    does not match ``/database``) and an exact match is allowed.  The
    unmatched remainder is joined onto the target:
    ``os.path.join(target, os.path.relpath(path, source))``.  No match →
    *path* unchanged.  Values that are not strings (e.g. ``None``) pass
    through untouched.
    """
    if not path_map or not isinstance(path, (str, os.PathLike)):
        return path
    return _map_one(path, path_map)


def resolve_path(path, base_dir=None, path_map=None):
    """Resolve one path: ``~``/``$VAR``, *base_dir*, then remap.

    The remapping uses, highest priority first: the *path_map* argument,
    the innermost :func:`path_scope`, the global :func:`bind_paths`
    registry, and ``TABPWA_PATH_MAP``.  Relative paths are joined onto
    *base_dir* first (so a config's own directory anchors relative
    entries), the result is normalised, then pushed through the map (see
    :func:`apply_path_map`).  Without any binding the path passes through
    unchanged (apart from ``~``/``$VAR`` expansion and normalisation).
    """
    if path is None:
        return None
    if not isinstance(path, str):
        path = os.fspath(path)
    p = os.path.expandvars(os.path.expanduser(path))
    if base_dir and not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    p = os.path.normpath(p) if p else p
    pm = current_path_map()
    if path_map:
        pm.update(parse_path_map(path_map))
    return str(apply_path_map(p, pm)) if pm else str(p)
