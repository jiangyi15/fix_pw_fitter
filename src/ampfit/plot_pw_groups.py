"""
Plot partial-wave group contributions to distributions.

Groups (Rx → 3π, B → R1 R2) are shown as step histograms overlaid
on data points.  B0 and B0bar are combined.
"""

import os
import numpy as np


def discover_groups(config):
    """Return dict mapping label → merged ck indices (B0+B0bar)."""
    idx = 0
    chain_ranges = []
    for chain in config.full_decay.chains:
        n = len(chain.get_gls_combination())
        chain_ranges.append((idx, idx + n, chain))
        idx += n
    n_base = idx

    inner_set = set()
    for chain in config.full_decay.chains:
        inner_set.update(chain.inner)

    def _charge_stem(name):
        if name.startswith("a1(1260)"):
            return name
        if len(name) > 1 and name[-1] in ("p", "m"):
            return name[:-1]
        return name

    raw_3pi = {}
    raw_B = {}

    for start, end, chain in chain_ranges:
        d1 = chain.decays[1]
        d2 = chain.decays[2]
        d1_inner = any(o.name in inner_set for o in d1.outs)

        if d1_inner:
            res = _charge_stem(d1.core.name)
            raw_3pi.setdefault(res, []).extend(range(start, end))
        elif not d1_inner and not any(o.name in inner_set for o in d2.outs):
            r1, r2 = d1.core.name, d2.core.name
            key = tuple(sorted([r1, r2]))
            raw_B.setdefault(key, []).extend(range(start, end))

    def expand(base):
        ck = []
        for block in range(8):
            offset = block * n_base
            for i in base:
                ck.append(offset + i)
        return ck

    groups = {}
    for k, v in raw_3pi.items():
        groups[k] = expand(v)
    for k, v in raw_B.items():
        groups["+".join(k)] = expand(v)

    return groups


class PWGroupPlotter:
    """Compute and cache group |A|² weights, then plot variables.

    Takes pre-built groups (dict mapping label → list of ck indices).
    Use :func:`discover_groups` to build the dict from a Config.

    Usage::

        groups = discover_groups(config)
        plotter = PWGroupPlotter(fitter, fit_result, groups)
        plotter.compute()

        # Plot mass (first 6 mass columns in a grid)
        plotter.plot_var(
            lambda x: [x["mass"][:, i] for i in range(6)],
            [f"mass[{i}]" for i in range(6)],
            0.2, 5.2, 100, "mass", legend=True)
    """

    def __init__(self, fitter, fit_result, groups):
        self.fitter = fitter
        self.fit_result = fit_result
        self.groups = groups
        self.labels = sorted(groups.keys())
        self._P_total = None
        self._P_groups = None
        self._scale = None

    # ── weight computation ────────────────────────────────────────

    def compute(self):
        """Compute |A|² per event for the total model and each group.

        Call once before :meth:`plot_var`.  Results stay in memory for
        subsequent calls.
        """
        params, _, _, _ = self.fitter._build_params(self.fit_result.x)

        _, _, self._P_total = self.fitter.backend.compute(
            params, self.fitter._phsp_holder, norm=None)

        self._P_groups = []
        for label in self.labels:
            p_group = dict(params)
            p_group["ck"] = params["ck"].copy()
            mask = self.groups[label]
            for i in range(len(p_group["ck"])):
                if i not in mask:
                    p_group["ck"][i] = 0.0j
            _, _, Pg = self.fitter.backend.compute(
                p_group, self.fitter._phsp_holder, norm=None)
            self._P_groups.append(Pg)

        purity = self.fitter._purity if self.fitter._purity is not None else 1.0
        target = float(np.sum(self.fitter._data_np["weight"])) * purity
        pw = self.fitter._phsp_np["weight"]
        total_sum = float(np.sum(pw * self._P_total))
        self._scale = target / total_sum if total_sum > 0 else 0.0
        return self

    @property
    def _ready(self):
        return self._P_total is not None

    # ── accessors (for reuse outside plotting) ────────────────────

    @property
    def weights(self):
        """Return ``(P_total, P_groups, scale)`` after :meth:`compute`."""
        return self._P_total, self._P_groups, self._scale

    # ── plotting ──────────────────────────────────────────────────

    def plot_var(self, varfun, labels, lo, hi, n_bins, prefix,
                 legend=False, output="plots/", ranges=None,
                 group_labels=None, colors=None,
                 data_weight_extra=None, phsp_weight_extra=None):
        """Plot variable(s) using pre-computed weights.

        Args:
            varfun: callable(dict) → list of arrays (one per subplot).
                    Called with the fitter's ``_data_np`` and ``_phsp_np``.
            labels: subplot titles.
            lo, hi: x range (fallback when *ranges* is None).
            n_bins: number of bins.
            prefix: filename stem (saved as ``{prefix}.png``).
            legend: if True, draw legend on first subplot.
            output: output directory.
            ranges: optional list of ``(lo, hi)`` per subplot.
            group_labels, colors: override defaults.
            data_weight_extra: optional per-event weight multiplying
                ``data["weight"]`` (e.g. tag weights, cuts).
            phsp_weight_extra: optional per-event weight multiplying
                ``phsp["weight"]`` (e.g. tag weights, cuts).
        """
        if not self._ready:
            raise RuntimeError("call .compute() before .plot_var()")

        import matplotlib.pyplot as plt

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        pw = f._phsp_np["weight"] * (phsp_weight_extra if phsp_weight_extra is not None else 1.0)

        # Recompute normalisation with extra weights
        data_total = float(np.sum(dw))
        purity = f._purity if f._purity is not None else 1.0
        target = data_total * purity
        total_sum = float(np.sum(pw * self._P_total))
        scale = target / total_sum if total_sum > 0 else 0.0

        colors = colors or plt.cm.tab20(np.linspace(0, 1, len(self.labels)))
        glabels = group_labels or self.labels

        var_data = varfun(f._data_np)
        var_phsp = varfun(f._phsp_np)
        n_var = len(var_data)

        n_cols = min(3, n_var)
        n_rows = (n_var + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for i in range(n_var):
            ax = axes.flatten()[i]
            d = var_data[i]
            p = var_phsp[i]
            xlo, xhi = (ranges[i] if ranges else (lo, hi))
            bins = np.linspace(xlo, xhi, n_bins + 1)

            dy, _ = np.histogram(d, bins=bins, weights=dw)
            dw2, _ = np.histogram(d, bins=bins, weights=dw ** 2)
            ty, _ = np.histogram(p, bins=bins,
                                 weights=pw * self._P_total * scale)
            gys = [np.histogram(p, bins=bins,
                                weights=pw * Pg * scale)[0]
                   for Pg in self._P_groups]

            ax.errorbar((bins[:-1] + bins[1:]) / 2, dy, yerr=np.sqrt(dw2),
                        fmt='o', color='black', markersize=3, capsize=2,
                        label='data' if legend and i == 0 else None)
            ax.step(bins[1:], ty, where='post', color='grey', linewidth=2,
                    label='total fit' if legend and i == 0 else None)
            for gy, c, lab in zip(gys, colors, glabels):
                ax.step(bins[1:], gy, where='post', color=c, linewidth=1.5,
                        label=lab if legend and i == 0 else None)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(0, None)
            ax.set_xlabel(labels[i])
            ax.tick_params(labelsize=9)

        for i in range(n_var, n_rows * n_cols):
            axes.flatten()[i].set_visible(False)

        if legend:
            axes.flatten()[0].legend(fontsize=7, ncol=2)

        plt.tight_layout()
        path = os.path.join(output, prefix + ".png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")
