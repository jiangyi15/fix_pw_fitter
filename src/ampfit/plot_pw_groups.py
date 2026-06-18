"""
Plot partial-wave group contributions to distributions.

Groups (Rx → 3π, B → R1 R2) are shown as step histograms overlaid
on data points.  B0 and B0bar are combined.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


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

    def compute_coefficients(self, phsp_weight_extra=None):
        """Compute time-integral coefficients I, Ī, J.

        Uses the kernel with modified data (time, frac) and scalar
        parameters to extract the fundamental integrals:

        * **I**  = Σ w·|A|²   — B0 partial rate
        * **Ī**  = Σ w·|Ā|²   — B0bar partial rate
        * **J**  = Σ w·A*·Ā   — complex interference (Re(J), Im(J))

        If *phsp_weight_extra* is given, it multiplies the phsp weights
        (e.g. cut or tag weights), correctly renormalising the integrals.

        Returns:
            ``(I, Ibar, ReJ, ImJ)`` — all floats.
        """
        import numpy as np

        f = self.fitter
        # Save original phsp and reload at the end
        orig_phsp = dict(f._phsp_np)
        params, _, _, _ = f._build_params(self.fit_result.x)

        delta_m = float(params["scalar"][2])

        # ── helper: set phsp data and run compute ────────────────
        def _run(time_array, frac_array, poq_rho=1.0, pop_phi=0.0,
                 gamma=0.0, delta_gamma=0.0, A_prod=0.0):
            p = dict(params)
            p["scalar"] = [gamma, delta_gamma, delta_m, A_prod, poq_rho, pop_phi]
            phsp = dict(orig_phsp)
            phsp["time"] = np.asarray(time_array, dtype=np.float64)
            phsp["frac"] = np.asarray(frac_array, dtype=np.float64)
            if phsp_weight_extra is not None:
                phsp["weight"] = phsp["weight"] * np.asarray(phsp_weight_extra, dtype=np.float64)
            # Free old GPU batched data before reloading (prevents corruption)
            if hasattr(f.backend, 'free_phsp_batched'):
                f.backend.free_phsp_batched()
            f.set_phsp(phsp)
            Q, _, _ = f.backend.compute(p, f._phsp_holder, norm=None)
            return float(Q)

        ne = orig_phsp["mass"].shape[0]

        # 1. I = Σ w·|ap|²   (time=0, frac=1)
        I = _run(np.zeros(ne), np.ones(ne))

        # 2. Ī = Σ w·|am|²   (time=0, frac=0)
        Ibar = _run(np.zeros(ne), np.zeros(ne))

        # 3. J integrals: time = π/(2·Δm), need cross terms
        #    With this time: gp = gm = 1/√2 (for ΔΓ=0, Γ=0)
        t_cross = np.full(ne, np.pi / (2.0 * delta_m))
        sum_IIbar = I + Ibar

        # Re(J): use poq = -i → cross term gives Re(ap*·am)
        Q_Re = _run(t_cross, np.ones(ne), poq_rho=1.0, pop_phi=-np.pi/2)
        ReJ = Q_Re - (sum_IIbar / 2.0)

        # Im(J): use poq = 1 → cross term gives Im(ap*·am)
        Q_Im = _run(t_cross, np.ones(ne), poq_rho=1.0, pop_phi=0.0)
        ImJ = (sum_IIbar / 2.0) - Q_Im

        # Restore original phsp
        f.set_phsp(orig_phsp)

        return I, Ibar, ReJ, ImJ

    # ── plotting ──────────────────────────────────────────────────

    def plot_var(self, varfun, labels, lo, hi, bin_width, prefix,
                 legend=False, output="plots/", ranges=None,
                 group_labels=None, colors=None,
                 data_weight_extra=None, phsp_weight_extra=None,
                 smooth_sigma=None, unit="GeV"):
        """Plot variable(s) using pre-computed weights.

        Args:
            varfun: callable(dict) → list of arrays (one per subplot).
            labels: subplot titles.
            lo, hi: x range (fallback when *ranges* is None).
            bin_width: width of each histogram bin in x‑axis units.
            prefix: filename stem (saved as ``{prefix}.png``).
            legend: if True, draw legend on first subplot.
            output: output directory.
            ranges: optional list of ``(lo, hi)`` per subplot.
            group_labels, colors: override defaults.
            data_weight_extra, phsp_weight_extra: optional extra weights.
            smooth_sigma: sigma in x‑axis units for Gaussian smoothing.
            unit: physical unit for bin width in y-label (default "GeV").
        """
        if not self._ready:
            raise RuntimeError("call .compute() before .plot_var()")

        import matplotlib.pyplot as plt

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        pw = f._phsp_np["weight"] * (phsp_weight_extra if phsp_weight_extra is not None else 1.0)

        # Extra weights are multiplicative per-event — they do NOT change the
        # overall normalisation scale (determined by base weights in compute()).
        colors = colors or plt.cm.tab20(np.linspace(0, 1, len(self.labels)))
        glabels = group_labels or self.labels

        # Background normalisation (constant across subplots)
        bkg = f._phsp_np.get("bkg", np.zeros(len(pw)))
        bkg_norm = float(np.sum(pw * bkg))
        purity = f._purity if f._purity is not None else 1.0
        data_total = float(np.sum(dw))
        bkg_scale = data_total * (1.0 - purity) / bkg_norm if bkg_norm > 0 else 0.0

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
            n_bins = max(1, int(round((xhi - xlo) / bin_width)))
            xhi = xlo + n_bins * bin_width  # exact bin width
            bins = np.linspace(xlo, xhi, n_bins + 1)
            bin_c = (bins[:-1] + bins[1:]) / 2

            dy, _ = np.histogram(d, bins=bins, weights=dw)
            dw2, _ = np.histogram(d, bins=bins, weights=dw ** 2)

            # Signal
            sig, _ = np.histogram(p, bins=bins,
                                  weights=pw * self._P_total * self._scale)

            # Background (filled area)
            bkg_y, _ = np.histogram(p, bins=bins, weights=pw * bkg * bkg_scale)

            # Total fit = signal + background
            tot = sig + bkg_y

            # Partial waves (signal only)
            gys = [np.histogram(p, bins=bins,
                                weights=pw * Pg * self._scale)[0]
                   for Pg in self._P_groups]

            ax.errorbar(bin_c, dy, yerr=np.sqrt(dw2),
                        fmt='o', color='black', markersize=3, capsize=2,
                        label='data' if legend and i == 0 else None)
            # Background fill
            ax.fill_between(bin_c, 0, bkg_y, step='mid', alpha=0.3,
                            color='C3', edgecolor='none', linewidth=0,
                            label='bkg' if legend and i == 0 else None)
            # Total fit step (signal + bkg)
            ax.step(bins[1:], tot, where='post', color='grey', linewidth=2,
                    label='total fit' if legend and i == 0 else None)
            # Partial wave groups (signal only) — smoothed if requested
            for gy, c, lab in zip(gys, colors, glabels):
                if smooth_sigma is not None and smooth_sigma > 0:
                    gy_s = gaussian_filter1d(gy.astype(np.float64), smooth_sigma)
                    # Cubic interpolation onto finer grid for smooth curve
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(bin_c, gy_s, bc_type='natural')
                    x_fine = np.linspace(xlo, xhi, n_bins * 10)
                    ax.plot(x_fine, cs(x_fine), '-', color=c, linewidth=1.5,
                            label=lab if legend and i == 0 else None)
                else:
                    ax.plot(bin_c, gy, '-', color=c, linewidth=1.5,
                            label=lab if legend and i == 0 else None)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(0, None)
            xl = labels[i]
            if unit:
                xl = f"{xl} ({unit})"
            ax.set_xlabel(xl)
            if i == 0:
                bw = (xhi - xlo) / n_bins
                yl = f"Events / ({bw:.3f})" if not unit else f"Events / ({bw:.3f} {unit})"
                ax.set_ylabel(yl)
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

    def plot_asymmetry(self, varfun, labels, lo, hi,
                       n_bins, prefix, tag_data=None, tag_phsp=None,
                       output="plots/", ranges=None,
                       data_weight_extra=None, phsp_weight_extra=None):
        """Plot asymmetry ``(N(tag>0) - N(tag<0)) / (N(tag>0) + N(tag<0))``.

        When *tag_data* / *tag_phsp* are ``None`` (default), uses
        ``frac - 0.5`` (so ``frac == 0.5`` events are excluded from
        both sides — no tagging information).  Only the total model
        is shown (no group decomposition).

        Args:
            varfun: callable(dict) → list of arrays (one per subplot).
            labels, lo, hi, n_bins, prefix: see :meth:`plot_var`.
            tag_data: array of tag values for data (default: frac - 0.5).
            tag_phsp: array of tag values for phsp (default: frac - 0.5).
        """
        if not self._ready:
            raise RuntimeError("call .compute() before .plot_asymmetry()")

        import matplotlib.pyplot as plt

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        pw = f._phsp_np["weight"] * (phsp_weight_extra if phsp_weight_extra is not None else 1.0)

        # Default: centre frac on 0 so frac==0.5 → tag==0 (excluded from both sides)
        # frac = P(B0) in data → tag = frac - 0.5
        # so tag>0 selects B0-like (frac>0.5), tag<0 selects B0bar-like (frac<0.5)
        if tag_data is None:
            tag_data = f._data_np["frac"] - 0.5
        if tag_phsp is None:
            tag_phsp = f._phsp_np["frac"] - 0.5

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
            bin_c = (bins[:-1] + bins[1:]) / 2

            # Data split by tag
            m1 = tag_data > 0
            m2 = tag_data < 0

            N1d, _ = np.histogram(d[m1], bins=bins, weights=dw[m1])
            N2d, _ = np.histogram(d[m2], bins=bins, weights=dw[m2])
            W1d, _ = np.histogram(d[m1], bins=bins, weights=dw[m1] ** 2)
            W2d, _ = np.histogram(d[m2], bins=bins, weights=dw[m2] ** 2)

            denom = N1d + N2d
            Ad = np.divide(N1d - N2d, denom, where=denom > 0, out=np.zeros_like(denom))
            # Error propagation: var(A) = 4/(N1+N2)^4 * (N2^2*W1 + N1^2*W2)
            numer = N2d**2 * W1d + N1d**2 * W2d
            var = np.divide(4.0 * numer, denom**4, where=denom > 0,
                            out=np.zeros_like(denom))
            Ad_err = np.sqrt(var)

            # Model split by tag (total fit only)
            pm1 = tag_phsp > 0
            pm2 = tag_phsp < 0

            N1m, _ = np.histogram(p[pm1], bins=bins,
                                  weights=pw[pm1] * self._P_total[pm1] * self._scale)
            N2m, _ = np.histogram(p[pm2], bins=bins,
                                  weights=pw[pm2] * self._P_total[pm2] * self._scale)
            denom_m = N1m + N2m
            Am = np.divide(N1m - N2m, denom_m, where=denom_m > 0,
                           out=np.zeros_like(denom_m))

            ax.errorbar(bin_c, Ad, yerr=Ad_err, fmt='o',
                        color='black', markersize=3, capsize=2, label='data')
            ax.step(bins[1:], Am, where='post', color='grey', linewidth=2,
                    label='total fit')
            ax.axhline(y=0, color='grey', linestyle=':', linewidth=1)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlo, xhi)
            ax.set_xlabel(labels[i])
            ax.tick_params(labelsize=9)

        for i in range(n_var, n_rows * n_cols):
            axes.flatten()[i].set_visible(False)

        axes.flatten()[0].legend(fontsize=7)

        plt.tight_layout()
        path = os.path.join(output, prefix + ".png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

    def plot_time_asymmetry(self, t_min=0, t_max=10, n_bins=20,
                            output="plots/", prefix="time_asym", params=None,
                            data_weight_extra=None, phsp_weight_extra=None):
        """Plot time-dependent asymmetry using exact theoretical formula.

        Data: binned time asymmetry ``(N(tag>0) - N(tag<0))/(N(tag>0) + N(tag<0))``.

        Model: continuous curve from the fitted time parameters and
        the integrals I, Ī, J, scaled by dilution ``⟨|1-2·frac|⟩``.

        Args:
            t_min, t_max: time range.
            n_bins: number of time bins.
            output: output directory.
            prefix: filename stem.
            params: optional params dict (uses from fit_result if None).
            data_weight_extra: optional per-event weight for data.
            phsp_weight_extra: optional per-event weight for phsp
                (also applied in coefficient calculation).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        frac_data = f._data_np["frac"]
        tag_data = frac_data - 0.5

        # ── Compute coefficients I, Ī, J (with optional extra phsp weight) ──
        I_val, Ibar_val, ReJ, ImJ = self.compute_coefficients(
            phsp_weight_extra=phsp_weight_extra)

        # ── Time parameters from the fit ──────────────────────────
        if params is None:
            params, _, _, _ = f._build_params(self.fit_result.x)
        sc = params["scalar"]
        gamma = float(sc[0])
        delta_gamma = float(sc[1])
        delta_m = float(sc[2])
        A_prod = float(sc[3])
        r = float(sc[4])
        phi = float(sc[5])

        # ── Dilution from data (common scale) ────────────────────
        tagged = np.abs(tag_data) > 1e-10
        dilution = float(np.mean(np.abs(1.0 - 2.0 * frac_data[tagged])))

        # ── Data asymmetry in time bins ───────────────────────────
        time_data = f._data_np["time"]
        bins = np.linspace(t_min, t_max, n_bins + 1)
        bin_c = (bins[:-1] + bins[1:]) / 2

        m1 = tag_data > 0
        m2 = tag_data < 0

        N1d, _ = np.histogram(time_data[m1], bins=bins, weights=dw[m1])
        N2d, _ = np.histogram(time_data[m2], bins=bins, weights=dw[m2])
        W1d, _ = np.histogram(time_data[m1], bins=bins, weights=dw[m1] ** 2)
        W2d, _ = np.histogram(time_data[m2], bins=bins, weights=dw[m2] ** 2)

        denom = N1d + N2d
        Ad = np.divide(N1d - N2d, denom, where=denom > 0, out=np.zeros_like(denom))
        numer = N2d**2 * W1d + N1d**2 * W2d
        var = np.divide(4.0 * numer, denom**4, where=denom > 0,
                        out=np.zeros_like(denom))
        Ad_err = np.sqrt(var)

        # ── Theoretical asymmetry curve ───────────────────────────
        I = I_val
        Ibar = Ibar_val
        r2 = r**2

        t_grid = np.linspace(t_min, t_max, 200)

        def P_of_t(t, I, Ibar, ReJ, ImJ):
            cht = np.cosh(t * delta_gamma / 2)
            ct = np.cos(t * delta_m)
            sht = np.sinh(t * delta_gamma / 2)
            st = np.sin(t * delta_m)
            expt = np.exp(-t * gamma)

            C_ang = np.cos(phi) * ReJ - np.sin(phi) * ImJ
            D_ang = np.cos(phi) * ImJ + np.sin(phi) * ReJ

            P1 = (I + r2 * Ibar) * cht
            P2 = (I - r2 * Ibar) * ct
            P3 = C_ang * sht
            P4 = D_ang * st
            P = expt * (P1 + P2 - 2.0 * r * P3 - 2.0 * r * P4)

            Pbar1 = (I / r2 + Ibar) * cht
            Pbar2 = (I / r2 - Ibar) * ct
            Pbar3 = C_ang * sht
            Pbar4 = D_ang * st
            Pbar = expt * (Pbar1 - Pbar2 - 2.0 / r * Pbar3 + 2.0 / r * Pbar4)

            return P, Pbar

        P_t, Pbar_t = P_of_t(t_grid, I, Ibar, ReJ, ImJ)
        A_theory = np.divide(P_t - Pbar_t, P_t + Pbar_t,
                             where=(P_t + Pbar_t) > 0, out=np.zeros_like(P_t))
        A_model = A_theory * dilution

        # ── Plot ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(bin_c, Ad, yerr=Ad_err, fmt='o',
                    color='black', markersize=4, capsize=2, label='data')
        ax.plot(t_grid, A_model, '-', color='grey', linewidth=2,
                label=f'model (dil={dilution:.3f})')
        ax.axhline(y=0, color='grey', linestyle=':', linewidth=1)
        ax.set_xlabel('time')
        ax.set_ylabel('asymmetry')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output, prefix + ".png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

    def plot_stacked_perm(self, varfun, xlabel, lo, hi, bin_width,
                          prefix, output="plots/",
                          data_weight_extra=None, phsp_weight_extra=None,
                          smooth_sigma=None, unit="GeV"):
        """Plot stacked histogram from permutation-equivalent variables.

        All arrays returned by *varfun* are flattened together: weights
        are repeated so each event contributes to every permutation bin.
        Errors propagate via ``√Σ w²`` over the flattened set.

        Args:
            varfun: callable(dict) → list of arrays (same length).
            xlabel: x-axis label.
            lo, hi: x range.
            bin_width: width of each histogram bin in x‑axis units.
            prefix: filename stem.
            output: output directory.
            data_weight_extra, phsp_weight_extra: optional extra weights.
            smooth_sigma: sigma in x‑axis units for Gaussian smoothing.
            unit: physical unit for bin width in y-label (default "GeV").
        """
        if not self._ready:
            raise RuntimeError("call .compute() before .plot_stacked_perm()")

        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        pw = f._phsp_np["weight"] * (phsp_weight_extra if phsp_weight_extra is not None else 1.0)

        var_data = varfun(f._data_np)
        var_phsp = varfun(f._phsp_np)
        n_p = len(var_data)

        n_bins = max(1, int(round((hi - lo) / bin_width)))
        hi = lo + n_bins * bin_width  # exact bin width
        bins = np.linspace(lo, hi, n_bins + 1)
        bin_c = (bins[:-1] + bins[1:]) / 2

        # ── data histogram ────────────────────────────────────────
        d_all = np.concatenate(var_data)
        dw_all = np.tile(dw, n_p)
        dy, _ = np.histogram(d_all, bins=bins, weights=dw_all)

        # Variance: per-event, per-bin count of permutations.
        bin_idx = np.column_stack([np.digitize(arr, bins) - 1 for arr in var_data])
        indicator = np.zeros((len(dw), n_bins), dtype=np.float64)
        for p in range(n_p):
            np.add.at(indicator, (np.arange(len(dw)), bin_idx[:, p]), 1.0)
        var_bin = np.sum((dw[:, None] * indicator) ** 2, axis=0)

        # ── model histograms ──────────────────────────────────────
        p_all = np.concatenate(var_phsp)
        pw_all_sig = np.tile(pw * self._P_total * self._scale, n_p)

        bkg = f._phsp_np.get("bkg", np.zeros(len(pw)))
        bkg_norm = float(np.sum(pw * bkg))
        purity = f._purity if f._purity is not None else 1.0
        data_total = float(np.sum(dw))
        bkg_scale = data_total * (1.0 - purity) / bkg_norm if bkg_norm > 0 else 0.0
        pw_all_bkg = np.tile(pw * bkg * bkg_scale, n_p)

        glabels = self.labels
        gcolors = plt.cm.tab20(np.linspace(0, 1, len(glabels)))
        pw_all_groups = [
            np.tile(pw * Pg * self._scale, n_p) for Pg in self._P_groups
        ]

        sig, _ = np.histogram(p_all, bins=bins, weights=pw_all_sig)
        bkg_y, _ = np.histogram(p_all, bins=bins, weights=pw_all_bkg)
        tot = sig + bkg_y
        gys = [np.histogram(p_all, bins=bins, weights=gw)[0] for gw in pw_all_groups]

        # ── smoothing helper for PW groups only ───────────────────
        def _smooth_group(arr):
            if smooth_sigma is None or smooth_sigma <= 0:
                return arr, bin_c, False
            sigma_bins = smooth_sigma
            sm = gaussian_filter1d(arr.astype(np.float64), sigma_bins)
            cs = CubicSpline(bin_c, sm, bc_type='natural')
            x_fine = np.linspace(lo, hi, n_bins * 10)
            return cs(x_fine), x_fine, True

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(bin_c, dy, yerr=np.sqrt(var_bin), fmt='o',
                    color='black', markersize=3, capsize=2, label='data')
        # Background fill — no edge lines
        ax.fill_between(bin_c, 0, bkg_y, step='mid', alpha=0.3,
                        color='C3', edgecolor='none', linewidth=0, label='bkg')
        # Total fit — step
        ax.step(bins[1:], tot, where='post', color='grey', linewidth=2,
                label='total fit')
        # PW groups — smooth if requested, else step
        for gy, c, lab in zip(gys, gcolors, glabels):
            y_plot, x_plot, is_smooth = _smooth_group(gy)
            if is_smooth:
                ax.plot(x_plot, np.maximum(y_plot, 0), '-', color=c, linewidth=1.5,
                        label=lab)
            else:
                ax.step(bins[1:], gy, where='post', color=c, linewidth=1.5,
                        label=lab)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, None)
        xl = xlabel if not unit else f"{xlabel} ({unit})"
        ax.set_xlabel(xl)
        bw = (hi - lo) / n_bins
        yl = f"Events / ({bw:.3f})" if not unit else f"Events / ({bw:.3f} {unit})"
        ax.set_ylabel(yl)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output, prefix + ".png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")
