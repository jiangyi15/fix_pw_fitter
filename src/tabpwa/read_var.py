"""
Generic per-event variable readers for the pure-PWA kernel arrays.

Pure-PWA event data is stored as canonical arrays per topology slot:

    mass   (n, n_mass)        subsystem masses; topology ``t`` occupies the
                              column block ``[t·n_res, t·n_res + n_res)``
    angle  (n, n_rows, 2·nv)  per-vertex canonical φ-first variables;
                              ``n_rows`` = active topology slots; vertex ``v``
                              of topology ``t`` lives in
                              ``angle[:, t, v]`` (φ) and ``angle[:, t, nv+v]``
                              (θ), where ``nv`` is the chain's decay count.

Each variable kind is its own class with a ``read(event_dict) -> (n,)``
interface, so new item kinds can be added independently:

    MassVar  — invariant mass of a pair topology:      MassVar(cfg, 'pipeta')
    AngleVar — canonical angle of a topology/decay:    AngleVar(cfg, 'pipeta/pip', 'cos(beta)')
    ExprVar  — element-wise expression over variables: ExprVar('max(m_pipi, m_pipeta)', mass_vars)

``vars_from_config(cfg)`` enumerates the config ``plot:`` section into the
appropriate item classes.  Every variable also carries plot metadata
(display, unit, fixed range, bin width), so figures can be assembled from
the config without any per-model hard-coding.
"""

import numpy as np

_ANGLE_RANGES = {"alpha": (-np.pi, np.pi), "cos(beta)": (-1.0, 1.0)}
_ANGLE_UNITS = {"alpha": "", "cos(beta)": ""}


def _system_finals(cfg, topo_name):
    """Final names of a pair topology (strings of its decay-section entry)."""
    entry = cfg.dic["decay"][topo_name]
    if not isinstance(entry, list):
        entry = [entry]
    return [k for k in entry if isinstance(k, str)]


def chain_for_topo(cfg, tid):
    """A representative full-decay chain with topology slot *tid*."""
    for _ls, chain in cfg.full_decay.get_partial_waves():
        try:
            if cfg.topo_index.get(chain.topo_id()) == tid:
                return chain
        except (KeyError, TypeError):
            continue
    return None


def _angle_vertex(chain, particle):
    """Chain decay index selected by a tf-style angle path.

    Bare ``topology`` = the pair system in the top decay (vertex 0);
    ``topology/particle`` = the deepest decay whose outgoing leaves contain
    *particle* (the resonance sub-decay in a two-body chain).
    """
    if particle is None:
        return 0
    for i in range(1, len(chain.decays)):
        if particle in [o.name for o in chain.decays[i].outs]:
            return i
    for i, d in enumerate(chain.decays):
        if particle in [o.name for o in d.outs]:
            return i
    return 0


# ─────────────────────────────────────────────────────────────────────

class BaseVar:
    """Common interface of every variable item.

    Subclasses implement :meth:`read` and declare ``kind``; the base holds
    the plot metadata (``display``, ``unit``, ``range``, ``bin_width``) and
    an optional ``trans`` applied on top of the raw per-event values.
    """

    kind = None

    def __init__(self, display=None, unit="", range=None, bin_width=None,
                 trans=None):
        self.display = display if display is not None else self.name
        self.unit = unit
        self.range = range
        self.bin_width = bin_width
        self.trans = trans
        self.nbins = None          # per-variable 'bins' from the config
        self.plot_cfg = {}         # extra tf-style plot options (legend, …)

    def apply_plot_entry(self, entry, defaults=None):
        """Apply a tf-style plot entry to this variable's metadata.

        Supported keys (merged over the ``plot.config`` *defaults*):
        ``display``, ``range``, ``bins``, ``units``, ``legend``,
        ``legend_outside``, ``yscale``, ``upper_ylim``.
        """
        if not isinstance(entry, dict):
            entry = {}
        defaults = defaults or {}
        meta = dict(defaults)
        meta.update(entry)
        # 'nbins' is an accepted alias for the tf-style 'bins'
        if meta.get("bins") is None and meta.get("nbins") is not None:
            meta["bins"] = meta["nbins"]

        if "display" in meta:
            self.display = meta["display"]
        if "range" in meta and meta["range"] is not None:
            self.range = tuple(meta["range"])
        if "bins" in meta and meta["bins"] is not None:
            self.nbins = int(meta["bins"])
        if "units" in meta and meta["units"]:
            self.unit = meta["units"]
        for key in ("legend", "legend_outside", "yscale", "upper_ylim"):
            if key in meta:
                self.plot_cfg[key] = meta[key]
        return self

    @property
    def name(self):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"

    def _finalize(self, v):
        v = np.asarray(v, dtype=float)
        if self.trans is not None:
            v = self.trans(v)
        return v

    def read(self, data):
        raise NotImplementedError


class MassVar(BaseVar):
    """Invariant mass of a pair topology, read from the event ``mass``
    column block ``cfg.n_res * topo_slot``."""

    kind = "mass"

    def __init__(self, cfg, topo_name, display=None, unit="GeV",
                 range=None, bin_width=None, trans=None):
        self.cfg = cfg
        self.topo_name = topo_name
        tid = cfg.topo_index_from_name(topo_name)
        n_res = int(getattr(cfg, "n_res", 1))
        self.topo_id = tid
        self.col = n_res * tid
        super().__init__(display=display, unit=unit, range=range,
                         bin_width=bin_width, trans=trans)

    @property
    def name(self):
        return f"m_{self.topo_name}"

    def read(self, data):
        m = np.asarray(data["mass"])
        if m.ndim == 1:
            if self.col != 0:
                raise IndexError(
                    f"{self}: 1-D mass array, no column {self.col}")
            v = m
        else:
            if self.col >= m.shape[1]:
                raise IndexError(
                    f"{self}: mass column {self.col} but array has "
                    f"{m.shape[1]} columns (is topology {self.topo_name} "
                    "active in these data?)")
            v = m[:, self.col]
        return self._finalize(v)


class AngleVar(BaseVar):
    """A canonical angle of one decay of a topology chain.

    ``path`` is ``'topology'`` (pair system at the top decay) or
    ``'topology/particle'`` (deepest decay containing *particle* — the
    resonance sub-decay in a two-body chain).  ``kind`` selects the value:

        'alpha'     wrapped azimuth  φ
        'cos(beta)' cosine of the polar angle
    """

    kind = "angle"

    def __init__(self, cfg, path, kind="alpha", display=None,
                 unit=None, range=None, bin_width=None, trans=None):
        if kind not in _ANGLE_RANGES:
            raise ValueError(f"unknown angle kind {kind!r}; choose from "
                             f"{sorted(_ANGLE_RANGES)}")
        self.cfg = cfg
        self.path = path
        self.angle_kind = kind
        tokens = path.split("/")
        self.topo_name = tokens[0]
        tid = cfg.topo_index_from_name(self.topo_name)
        chain = chain_for_topo(cfg, tid)
        if chain is None:
            raise ValueError(
                f"no full-decay chain with topology {path!r} — is "
                f"{self.topo_name} wave-active in this config?")
        self.topo_id = tid
        self.nv = len(chain.decays)
        self.particle = tokens[1] if len(tokens) > 1 else None
        self.vertex = _angle_vertex(chain, self.particle)
        super().__init__(display=display,
                         unit=_ANGLE_UNITS[kind] if unit is None else unit,
                         range=_ANGLE_RANGES[kind] if range is None else range,
                         bin_width=bin_width, trans=trans)

    @property
    def name(self):
        path = self.topo_name
        if self.particle:
            path += f"/{self.particle}"
        return f"{self.angle_kind}({path})"

    def read(self, data):
        a = np.asarray(data["angle"])
        if a.ndim == 2:
            a = a.reshape(a.shape[0], -1, 1)
        if self.topo_id >= a.shape[1]:
            raise IndexError(
                f"{self}: angle row {self.topo_id} but array has "
                f"{a.shape[1]} rows (is topology {self.topo_name} "
                "active in these data?)")
        comp = (self.vertex if self.angle_kind == "alpha"
                else self.nv + self.vertex)
        v = a[:, self.topo_id, comp]
        if self.angle_kind == "alpha":
            v = (v + np.pi) % (2 * np.pi) - np.pi
        else:
            v = np.cos(v)
        return self._finalize(v)


class ExprVar(BaseVar):
    """Element-wise expression over named variables (usually ``MassVar``).

    The expression may reference ``m_<topo>`` names and the function names
    ``max/min/sqrt/abs/exp/log/sin/cos/pow`` (element-wise).
    """

    _FUNCS = {"max": np.maximum, "min": np.minimum, "sqrt": np.sqrt,
              "abs": np.abs, "exp": np.exp, "log": np.log,
              "sin": np.sin, "cos": np.cos, "pow": np.power}

    kind = "expr"

    def __init__(self, expr, vars_, display=None, unit="",
                 range=None, bin_width=None, trans=None, name=None):
        self.expr = expr
        self.vars = dict(vars_)
        self._name = name
        self._code = compile(expr, "<plot expr>", "eval")
        super().__init__(display=display, unit=unit, range=range,
                         bin_width=bin_width, trans=trans)

    @property
    def name(self):
        if self._name:
            return self._name
        return "expr(" + self.expr + ")"

    def read(self, data):
        env = {fname: fn for fname, fn in self._FUNCS.items()}
        for nm, v in self.vars.items():
            env[nm] = v.read(data)
        out = eval(self._code, {"__builtins__": {}}, env)
        return self._finalize(out)


# ── factory (backward-compatible single entry) ───────────────────────

def ReadVar(cfg, ident, display=None, unit="GeV", range=None,
            bin_width=None, trans=None):
    """Factory mapping an identifier to the matching variable class:

        * ``str`` or ``("mass", name)``     → :class:`MassVar`
        * ``("angle", path, kind)``         → :class:`AngleVar`
    """
    if isinstance(ident, str):
        ident = ("mass", ident)
    if not isinstance(ident, (tuple, list)) or len(ident) < 2:
        raise ValueError(f"ReadVar ident must be a topology name or a "
                         f"(kind, ...) tuple, got {ident!r}")
    kind0 = ident[0]
    if kind0 in ("mass", "m"):
        return MassVar(cfg, ident[1], display=display, unit=unit,
                       range=range, bin_width=bin_width, trans=trans)
    if kind0 in ("angle", "ang"):
        if len(ident) != 3:
            raise ValueError("angle ident = ('angle', path, kind)")
        return AngleVar(cfg, ident[1], ident[2], display=display,
                        range=range, bin_width=bin_width, trans=trans)
    raise ValueError(f"unknown variable kind {kind0!r}")


def vars_from_config(cfg, section="plot"):
    """Every plot variable declared in the config ``plot:`` section.

    Parses the tf_pwa-style tree::

        plot:
          mass:       {topo: {display: ...}}
          angle:      {path: {alpha: {...}, cos(beta): {...}}}
          extra_vars: {name: {expr: "max(m_pipeta, m_pimeta)"}}

    Returns a list of ``(key, BaseVar)`` ordered as written.  Items whose
    topology is not wave-active in *cfg* (no chain to resolve angles from)
    are skipped so a single-topology model can still carry the full generic
    plot section.
    """
    plot = cfg.dic.get(section) or {}
    if not isinstance(plot, dict):
        return []
    if "plot" in plot and not any(k in plot for k in
                                  ("mass", "angle", "extra_vars")):
        plot = plot["plot"]

    out = []
    mass_vars = {}

    def _meta(entry):
        return entry if isinstance(entry, dict) else {}

    defaults = (plot.get("config") if isinstance(plot.get("config"), dict)
                else {})
    if not isinstance(defaults, dict):
        defaults = {}

    for topo, entry in (plot.get("mass") or {}).items():
        v = MassVar(cfg, topo).apply_plot_entry(_meta(entry), defaults)
        mass_vars[v.name] = v
        out.append((topo, v))

    for path, group in (plot.get("angle") or {}).items():
        for kind, entry in (group or {}).items():
            try:
                v = AngleVar(cfg, path, kind).apply_plot_entry(
                    _meta(entry), defaults)
            except ValueError:
                continue          # topology not wave-active here
            out.append((f"{path} {kind}", v))

    for name, entry in (plot.get("extra_vars") or {}).items():
        expr = _meta(entry).get("expr")
        if expr is None:
            continue
        v = ExprVar(expr, mass_vars, name=name).apply_plot_entry(
            _meta(entry), defaults)
        out.append((name, v))
    return out


def _where_var(cfg, spec, env):
    """Resolve a ``where: {name: spec}`` entry of a 2Dplot panel.

    ``spec`` is ``[mass, topo]``, ``[angle, path, kind]`` or the string
    key/name of an already-declared variable.
    """
    if isinstance(spec, str):
        if spec not in env:
            raise ValueError(f"where spec {spec!r} is not a declared "
                             f"plot variable")
        return env[spec]
    kind0, rest = spec[0], list(spec[1:])
    if kind0 in ("mass", "m"):
        return MassVar(cfg, rest[0])
    if kind0 in ("angle", "ang"):
        return AngleVar(cfg, rest[0], rest[1])
    raise ValueError(f"unknown where variable kind {kind0!r}")


def plots_2d_from_config(cfg, section="plot"):
    """2D pull panels from the config ``plot.2Dplot:`` section (tf-pwa style).

        plot:
          2Dplot:
            dalitz_1:
              x: m_pipi**2                # expression over declared names
              y: m_pipeta**2
              where: {t: [angle, pipi, cos(beta)]}   # optional extra names
              add_dalitz_boundary: [pipi, pipeta]    # 2 resonance names,
              # or 4 masses [m0, m1, m2, m3]; optional — no key = no curve
              adaptive_binning: [[2, 2], [2, 2], [2, 2]]
              dalitz_boundary_style: {color: gray, alpha: 0.5}
              pull_scatter_style: {c: black, s: 1, alpha: 0.0}
              xlabel / ylabel: axis labels (default: the x/y expressions;
              a tf-pwa ``display: "A vs B"`` splits into the two labels
              whenever xlabel/ylabel are not given)

    ``add_dalitz_boundary: [a, b]`` follows tf-pwa ``get_dalitz``: *a* and
    *b* are two decays sharing one parent and exactly one shared final; the
    analytic curve bounds the region of the plotted variables under
    (m(a)², m(b)²) — i.e. x/y must be squared masses, exactly like tf-pwa.

    Returns ``[(panel_key, spec)]`` with
    ``spec = {x, y, labels, binning, boundary, boundary_style,
    scatter_style}``; ``x``/``y`` are :class:`ExprVar` readers.
    """
    plot = cfg.dic.get(section) or {}
    if not isinstance(plot, dict):
        return []
    if "plot" in plot and not any(k in plot for k in
                                  ("mass", "angle", "extra_vars", "2Dplot")):
        plot = plot["plot"]
    twod = plot.get("2Dplot") or {}
    if not isinstance(twod, dict):
        return []

    env = {}
    for key, v in vars_from_config(cfg, section):
        env.setdefault(v.name, v)
        env.setdefault(key, v)

    out = []
    for key, entry in twod.items():
        if not isinstance(entry, dict) or "x" not in entry or "y" not in entry:
            continue
        env_e = dict(env)
        for nm, spec in (entry.get("where") or {}).items():
            env_e[nm] = _where_var(cfg, spec, env)
        ex, ey = str(entry["x"]), str(entry["y"])
        vx = ExprVar(ex, env_e, name=ex)
        vy = ExprVar(ey, env_e, name=ey)

        # axis labels: xlabel/ylabel, else tf-pwa's display "X vs Y" split
        xlabel, ylabel = entry.get("xlabel"), entry.get("ylabel")
        display = entry.get("display")
        if display and "vs" in str(display):
            left, _, right = str(display).partition("vs")
            xlabel = xlabel if xlabel is not None else left.strip()
            ylabel = ylabel if ylabel is not None else right.strip()

        boundary = entry.get("add_dalitz_boundary")
        if boundary is not None:
            boundary = list(boundary)
            if len(boundary) == 2:
                if not all(isinstance(s, str) for s in boundary):
                    raise ValueError(
                        f"2Dplot {key!r}: add_dalitz_boundary with 2 entries "
                        f"must be two particle names, got {boundary}")
            elif len(boundary) == 4:
                boundary = [float(m) for m in boundary]
            else:
                raise ValueError(
                    f"2Dplot {key!r}: add_dalitz_boundary must be 2 particle "
                    f"names or 4 masses, got {boundary}")

        binning = entry.get("adaptive_binning")
        if binning is not None:
            binning = [[int(n) for n in level] for level in binning]
        # else: None — config_2d_panels derives levels from the data size

        out.append((key, {
            "x": vx, "y": vy,
            "labels": [xlabel if xlabel is not None else ex,
                       ylabel if ylabel is not None else ey],
            "binning": binning,
            "boundary": boundary,
            "boundary_style": entry.get("dalitz_boundary_style"),
            "scatter_style": entry.get("pull_scatter_style"),
            "plot_figs": entry.get("plot_figs"),
        }))
    return out
