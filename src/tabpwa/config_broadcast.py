"""ConfigBroadcast — one template config → N configs (its ``broadcast:`` list).

A config may describe N variants of itself so that several fits can run
on one shared model / one x-space::

    data:
      dat_order: [pip, pim, eta]
      phsp_arr: shared_mc.npz
    broadcast:                     # one entry per config — a diff list
      - {data: {data_arr: run1.npz}}
      - {data: {data_arr: run2.npz}, purity: 0.93, label: run2}

Each entry is **deep-merged** onto the template: nested dicts merge,
anything else replaces, and dotted keys (``data.phsp_arr``) address
nested fields.  Relative ``.npy``/``.npz`` paths inside ``data:`` are
resolved against the config's directory, because the expanded configs are
handed to :class:`~tabpwa.fitter.Fitter` as dicts
(``RawConfig._config_path`` is empty for dicts, so relative paths would
otherwise resolve against the CWD).

The optional ``label`` key names a variant (used for logs/labels) and
never lands in the expanded config.

No ``broadcast`` key → one config, the source unchanged.

Programmatic scans inject the list into the template with :func:`scan`
(a pure pre-compute — this is not a different kind of variant):

::

    from tabpwa.config_broadcast import ConfigBroadcast, scan

    cfg = {**cfg, "broadcast": scan({"data.purity": [0.9, 0.95],
                                     "data.data_arr": ["a.npz", "b.npz"]})}
    cb = ConfigBroadcast(cfg)

    len(cb)            # 4 — the cartesian product
    cb.labels()        # ['purity=0.9&data_arr=a.npz', ...]
    for variant in cb:
        pass           # each variant is a config dict
    cb.pairs()         # [(label, config), ...]
"""
import copy
import os

_DATA_SUFFIXES = (".npy", ".npz")


def _load_source(source):
    """Config source (path or dict) → raw dict."""
    if isinstance(source, dict):
        return source
    from tabpwa.config_loader import load_config
    return load_config(source)


def scan(grid):
    """Pre-compute a ``broadcast:`` diff list from a value grid.

    Pure helper — the cartesian product over ``{dotted_key: [values]}``,
    one diff dict per combination, each carrying an auto ``label``
    (``key=value`` pieces joined by ``&``, paths shortened to their
    basename).  Inject it into the template::

        cfg["broadcast"] = scan({"data.purity": [0.9, 0.95]})

    Scalar values count as one-point lists.
    """
    import itertools
    grid = {k: ([v] if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__")
                else list(v)) for k, v in grid.items()}
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        diff = dict(zip(keys, combo))
        parts = []
        for key, val in diff.items():
            if isinstance(val, str) and val.endswith(_DATA_SUFFIXES):
                val = os.path.basename(val)
            parts.append(f"{key.split('.')[-1]}={val}")
        diff["label"] = "&".join(parts)
        out.append(diff)
    return out


class ConfigBroadcast:
    """Template + its ``broadcast:`` diff list → N config dicts.

    One concept: every variant is a diff dict applied on top of the
    template, and the list lives **in the template**::

        broadcast:
          - {data: {data_arr: run1.npz}}
          - {data: {data_arr: run2.npz}, purity: 0.93, label: run2}

    For programmatic scans, put a :func:`scan` grid in the template::

        cfg = {**cfg, "broadcast": scan({"data.purity": [0.9, 0.95]})}

    Container protocol: ``len(cb)``, ``for cfg in cb``, ``cb[i]``, plus
    ``cb.pairs()`` → ``[(label, config), …]``, ``cb.labels()`` and
    ``cb.expand()``.
    """

    scan = staticmethod(scan)

    def __init__(self, source):
        """Args:
            source: config path or raw config dict.  The variants live in
                its ``broadcast:`` list (a diff dict per config; an entry
                may carry ``label`` — metadata, not part of the config).
                Put a :func:`scan` grid there for programmatic scans::

                    cfg["broadcast"] = scan({"data.purity": [0.9, 0.95]})
        """
        if source is None:
            source = {}
        if not isinstance(source, dict):
            self.base = _load_source(source)
            self.base_dir = os.path.dirname(os.path.abspath(str(source)))
        else:
            self.base = source
            self.base_dir = ""
        self.source = source

        patched = self.base.get("broadcast")
        self._patches = []          # [(patch, label_or_None)]
        if patched is not None:
            if not patched:
                raise ValueError("config has an empty 'broadcast' list")
            if not all(isinstance(p, dict) for p in patched):
                raise ValueError(
                    "'broadcast' must be a list of dict diffs, e.g. "
                    "[{data: {...}}, {data: {...}, purity: 0.9}]")
            # ``label`` is metadata, not part of the config
            self._patches = [({k: v for k, v in p.items() if k != "label"},
                              p.get("label")) for p in patched]

    # ── introspection ─────────────────────────────────────────────

    @property
    def n_configs(self):
        return len(self._patches) if self._patches else 1

    @property
    def is_broadcast(self):
        """True when the source carries a broadcast block.

        This is the *form* test (a one-entry ``broadcast:`` counts), so it
        agrees with :meth:`has_broadcast` and with consumers that branch
        on it; ``n_configs`` carries the count.
        """
        return bool(self._patches)

    def labels(self):
        """Variant names: the entry's ``label`` or ``cfg<i>`` (``""`` if none)."""
        if not self._patches:
            return [""]
        return [lbl if lbl else f"cfg{i}"
                for i, (_p, lbl) in enumerate(self._patches)]

    def pairs(self):
        """``[(label, config_dict), ...]`` — instance view (no index prefix)."""
        return list(zip(self.labels(), self.expand()))

    # ── container protocol ────────────────────────────────────────

    def __len__(self):
        return self.n_configs

    def __getitem__(self, i):
        return self.config(i)

    def __iter__(self):
        """Iterate the variant configs (use ``labels()`` for names)."""
        return iter(self.expand())

    # ── class-level entry point ───────────────────────────────────

    @classmethod
    def has_broadcast(cls, source):
        """True when *source* yields several configs.

        A ``ConfigBroadcast`` instance → its own state; a config path/dict
        → the presence of a ``broadcast`` key (even an empty one, so the
        empty-list error still fires at build).  Consumers branch on this
        to pick the single-config path from the broadcast path.
        """
        if isinstance(source, cls):
            return source.is_broadcast
        raw = source if isinstance(source, dict) else _load_source(source)
        return raw.get("broadcast") is not None

    @classmethod
    def items(cls, source):
        """``[(label, config_dict), ...]`` for *source* — the read side.

        This is the one entry point consumers (e.g.
        :class:`~tabpwa.joint_fit.JointFitter`) iterate; *source* may be a
        path, a dict, or an already-built ``ConfigBroadcast``.  A plain
        config yields a single ``("", source)`` (the source itself, so a
        path keeps its ``_config_path``); variants yield
        ``(f"{i}:{label}", config)`` pairs.
        """
        bc = source if isinstance(source, cls) else cls(source)
        if not bc.is_broadcast:
            return [("", bc.source)]
        return [(f"{i}:{lbl}", cfg)
                for i, (lbl, cfg) in enumerate(zip(bc.labels(), bc.expand()))]

    # ── expansion ────────────────────────────────────────────────

    def expand(self):
        """All variant configs, in order (the source itself when not
        broadcasting)."""
        if not self._patches:
            return [self.base]
        out = []
        for patch, _lbl in self._patches:
            dic = copy.deepcopy(self.base)
            dic.pop("broadcast", None)
            self.merge_patch(dic, patch)
            if self.base_dir:
                data = dic.get("data")
                if isinstance(data, dict):
                    dic["data"] = {k: self.resolve_value(v, self.base_dir)
                                   for k, v in data.items()}
            out.append(dic)
        return out

    def config(self, i):
        """One variant (index order matches :meth:`labels`)."""
        return self.expand()[i]

    # ── reusable dict helpers ────────────────────────────────────

    @classmethod
    def merge_patch(cls, dic, patch):
        """Deep-merge *patch* into *dic* (dotted keys address nesting)."""
        for key, val in patch.items():
            if "." in key:
                cls.set_dotted(dic, key, val)
            elif isinstance(val, dict) and isinstance(dic.get(key), dict):
                cls.merge_patch(dic[key], val)
            else:
                dic[key] = copy.deepcopy(val)
        return dic

    @classmethod
    def set_dotted(cls, dic, dotted, value):
        """Set ``a.b.c`` in *dic*, creating nested dicts."""
        node = dic
        parts = dotted.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = copy.deepcopy(value)
        return dic

    @classmethod
    def get_dotted(cls, dic, dotted, default=None):
        node = dic
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    @classmethod
    def resolve_value(cls, val, base_dir):
        """Resolve data-file path(s) relative to the config's directory."""
        if isinstance(val, (list, tuple)):
            return type(val)(cls.resolve_value(v, base_dir) for v in val)
        if (isinstance(val, str) and base_dir
                and val.endswith(_DATA_SUFFIXES)
                and not os.path.isabs(val)):
            return os.path.join(base_dir, val)
        return val


def expand_configs(source):
    """Convenience: ``ConfigBroadcast(source).expand()``."""
    return ConfigBroadcast(source).expand()
