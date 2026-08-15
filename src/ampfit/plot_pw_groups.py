"""
Plot partial-wave group contributions to distributions.

Groups (Rx → 3π, B → R1 R2) are shown as step histograms overlaid
on data points.  B0 and B0bar are combined.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ── same-charge-pair variables (B → (π⁺π⁺)(π⁻π⁻)) ────────────────

def samesign_varfun(x):
    """Extract the five same-charge-pair variables from a data dict.

    Reconstructs the momenta with :func:`ampfit.momenta_to_data.
    data_to_momentum` (row-0 = the generator's ρρ parametrisation) and
    reads the B → (π⁺₁π⁺₂)(π⁻₁π⁻₂) kinematics with
    :func:`ampfit.momenta_to_data.momenta_to_data_samesign`:

        [m(π⁺π⁺), m(π⁻π⁻), cosθ₁, cosθ₂, φ]

    with the helicity cosines in [0, 1] (identical-pion ambiguity
    fixed).  Suitable as *varfun* for :meth:`PWGroupPlotter.plot_var`.
    """
    from ampfit.momenta_to_data import (data_to_momentum,
                                        momenta_to_data_samesign)
    mom = data_to_momentum({"mass": x["mass"].reshape(-1, 24, 2),
                            "q": x["q"].reshape(-1, 24, 3),
                            "angles": x["angle"].reshape(-1, 24, 3)})
    d = momenta_to_data_samesign(mom)
    return [d["m_pp"], d["m_mm"],
            d["cos_theta1"], d["cos_theta2"], d["phi"]]


SAMESIGN_LABELS = [r"$m(\pi^+\pi^+)$", r"$m(\pi^-\pi^-)$",
                   r"$\cos\theta_1$", r"$\cos\theta_2$", r"$\phi$"]
SAMESIGN_RANGES = [(0.28, 5.2), (0.28, 5.2), (0, 1), (0, 1),
                   (-np.pi, np.pi)]
SAMESIGN_NAMES = ["m_pp", "m_mm", "cos_theta1", "cos_theta2", "phi"]
SAMESIGN_BINW = [0.05, 0.05, 0.02, 0.02, 0.05]


def plot_samesign(plotter, output="plots/", fmt="png"):
    """Plot the five same-charge-pair variables as separate figures.

    Each variable (m(π⁺π⁺), m(π⁻π⁻), cosθ₁, cosθ₂, φ) is saved as its
    own ``samesign_<name>.png`` (data vs total fit + group weights).
    The kinematics are reconstructed once per dataset and cached.
    """
    cache = {}

    def cached(x):
        key = id(x)
        if key not in cache:
            cache[key] = samesign_varfun(x)
        return cache[key]

    for j, (label, rng, name, bw) in enumerate(zip(
            SAMESIGN_LABELS, SAMESIGN_RANGES, SAMESIGN_NAMES,
            SAMESIGN_BINW)):
        plotter.plot_var(
            lambda x, jj=j: [cached(x)[jj]],
            [label], rng[0], rng[1], bw, f"samesign_{name}",
            output=output, fmt=fmt, ranges=[rng], unit="",
            smooth_sigma=1.0, legend=True, show_pull=True)



def adaptive_split_bound(datas, binning, base_bound=None):
    """Recursively split 2D data into adaptive bins.

    Mirrors ``tf_pwa.adaptive_bins.AdaptiveBound``: each level of
    *binning* (a list like ``[2, 2]``) splits every current bin into
    equal-quantile sub-bins along each dimension in turn.

    Args:
        datas: ``(2, n)`` array of the two variables.
        binning: nested list, e.g. ``[[2, 2]] * 3`` → 3 levels of 2×2.
        base_bound: ``((xmin, ymin), (xmax, ymax))``; defaults to the
                    data extrema.

    Returns:
        ``(bounds, datas)`` — list of ``((xlo, ylo), (xhi, yhi))``
        bounds and the per-bin ``(2, n_i)`` data slices.
    """
    datas = np.asarray(datas, dtype=float)
    if base_bound is None:
        lo = np.min(datas, axis=-1)
        hi = np.max(datas, axis=-1) + 1e-6
        base_bound = (lo, hi)

    def single_split(data, n, bnd):
        """Split 1D data into n equal-quantile sub-bins inside bnd."""
        lo, hi = bnd
        bounds = []
        num_lb = lo
        for j in range(1, n):
            num_rb = np.percentile(data, j / n * 100) + 1e-6
            bounds.append((num_lb, num_rb))
            num_lb = num_rb
        bounds.append((num_lb, hi))
        return bounds

    bound_chain = [base_bound]
    data_chain = [datas]
    for level in binning:
        new_bound_chain = []
        new_data_chain = []
        for bnd, data in zip(bound_chain, data_chain):
            cur_bounds = [bnd]
            cur_datas = [data]
            for idx, size in enumerate(level):
                nxt_bounds = []
                nxt_datas = []
                for cb, cd in zip(cur_bounds, cur_datas):
                    subs = single_split(cd[idx], size, (cb[0][idx], cb[1][idx]))
                    for i, (lb, rb) in enumerate(subs):
                        l_bnd = list(cb[0]); r_bnd = list(cb[1])
                        l_bnd[idx] = lb; r_bnd[idx] = rb
                        mask = (cd[idx] >= lb) & (cd[idx] < rb)
                        nxt_bounds.append((tuple(l_bnd), tuple(r_bnd)))
                        nxt_datas.append(cd[:, mask])
                cur_bounds, cur_datas = nxt_bounds, nxt_datas
            new_bound_chain.extend(cur_bounds)
            new_data_chain.extend(cur_datas)
        bound_chain = new_bound_chain
        data_chain = new_data_chain
    return bound_chain, data_chain

def discover_groups(config, merge=None):
    """Return dict mapping display label → merged ck indices (B0+B0bar).

    Group keys are particle *display* names (LaTeX, from
    :meth:`Config.name_display_map`), e.g. ``$a_2(1320)^+$`` or
    ``$\\rho$ + $\\rho$`` — the same convention used by
    ``scripts/plot_pw_resonance.py``, so legends show display names
    directly via ``PWGroupPlotter.labels``.

    Args:
        merge: optional list of ``(regex, label)`` — group keys
            (internal names, *before* display conversion) matching
            *regex* are merged into *label* (their ck index lists are
            combined).  E.g. ``[("^MI0\\\\d", "MI0")]`` merges
            MI00..MI04 into a single "MI0" wave group.
    """
    import re as _re

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

    display_map = config.name_display_map()

    def _charge_stem(name):
        if name.startswith("a1(1260)"):
            return name
        if len(name) > 1 and name[-1] in ("p", "m"):
            return name[:-1]
        return name

    def _display(name):
        """Internal group key → display name.

        Exact config-name match first (``a1(1260)p`` → ``$a_1(1260)^+$``,
        so charge-conjugate waves stay separate).  Charge-merged keys
        (formed by ``_charge_stem`` merging the ``p``/``m`` variants,
        e.g. ``a2(1320)``) get the ``^{\\pm}`` superscript from the
        p-variant display.  Synthetic merged labels (e.g. ``MI0``) fall
        back to the key itself.
        """
        if name in display_map:
            return display_map[name]
        if name + "p" in display_map:
            disp = display_map[name + "p"]
            if disp.endswith("^+$") or disp.endswith("^-$"):
                return disp[:-3] + r"^{\pm}$"
            return disp
        if name + "m" in display_map:
            return display_map[name + "m"]
        return name

    def _merge_label(label):
        """Apply merge patterns to an internal group label."""
        merged = label
        if merge:
            for pat, newlabel in merge:
                if _re.match(pat, label):
                    merged = newlabel
                    break
        return merged

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
        internal = _merge_label(k)
        groups.setdefault(_display(internal), []).extend(v)
    for k, v in raw_B.items():
        internal = _merge_label("+".join(k))
        label = (" + ".join(_display(p) for p in internal.split("+"))
                 if "+" in internal else _display(internal))
        groups.setdefault(label, []).extend(v)

    # Apply expand() to each merged group
    return {k: expand(v) for k, v in groups.items()}


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
        params, _ = self.fitter.build_params(self.fit_result.x)

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

        # Store zorder for plotting: smallest |weight| → highest zorder (on top)
        pw = self.fitter._phsp_np["weight"]
        pw_abs = np.array([float(np.sum(np.abs(pw * Pg))) for Pg in self._P_groups])
        rank = np.argsort(np.argsort(pw_abs))  # 0 = smallest
        n = len(rank)
        self._zorders = [5 + (n - 1 - r) * 2 for r in rank]  # smallest → highest zorder

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
        params, _ = f.build_params(self.fit_result.x)

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
                 smooth_sigma=None, unit="GeV", show_pull=False,
                 fmt="png"):
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
            show_pull: if True, show pull (data−model)/σ below each panel.
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
        if show_pull:
            import matplotlib.gridspec as mgs
            fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
            gs = mgs.GridSpec(2 * n_rows, n_cols, figure=fig,
                              height_ratios=[3, 1] * n_rows, hspace=0)
            axes = np.empty((2 * n_rows, n_cols), dtype=object)
            for r in range(2 * n_rows):
                for c in range(n_cols):
                    axes[r, c] = fig.add_subplot(gs[r, c])
            # Share x within each column
            for r in range(1, 2 * n_rows):
                for c in range(n_cols):
                    axes[r, c].sharex(axes[0, c])
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                                     squeeze=False)

        for i in range(n_var):
            row, col = divmod(i, n_cols)
            ax = axes[2 * row, col] if show_pull else axes[row, col]
            ax_pull = axes[2 * row + 1, col] if show_pull else None
            d = var_data[i]
            p = var_phsp[i]
            xlo, xhi = (ranges[i] if ranges else (lo, hi))
            n_bins = max(1, int(round((xhi - xlo) / bin_width)))
            xhi = xlo + n_bins * bin_width
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

            # Pull = (data - model) / sqrt(var(data))
            pull = np.divide(dy - tot, np.sqrt(dw2), where=dw2 > 0, out=np.zeros_like(dy))

            ax.errorbar(bin_c, dy, yerr=np.sqrt(dw2),
                        fmt='o', color='black', markersize=3, capsize=2,
                        label='data' if legend and i == 0 else None)
            # Background fill
            ax.fill_between(bin_c, 0, bkg_y, step='mid', alpha=0.3,
                            color='C3', edgecolor='none', linewidth=0,
                            label='bkg' if legend and i == 0 else None)
            # Total fit step (signal + bkg)
            ax.step(bin_c, tot, where='mid', color='grey', linewidth=2,
                    label='total fit' if legend and i == 0 else None)
            # Partial wave groups (signal only) — binned KDE: Σ wi·G(x−xi)
            for gy, c, lab, zo in zip(gys, colors, glabels, self._zorders):
                if smooth_sigma is not None and smooth_sigma > 0:
                    x_fine = np.linspace(xlo, xhi, n_bins * 10)
                    dx = bin_c[:, None] - x_fine[None, :]
                    sigma = smooth_sigma * bin_width
                    gy_fine = np.dot(gy, np.exp(-0.5 * (dx / sigma)**2))
                    gy_fine *= bin_width / (np.sqrt(2 * np.pi) * sigma)
                    ax.plot(x_fine, gy_fine, '-', color=c, linewidth=1.5,
                            label=lab if legend and i == 0 else None, zorder=zo)
                else:
                    ax.step(bin_c, gy, where='mid', color=c, linewidth=1.5,
                            label=lab if legend and i == 0 else None, zorder=zo)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(0, None)
            if show_pull:
                ax.tick_params(labelbottom=False)
            if i == 0:
                bw = (xhi - xlo) / n_bins
                yl = f"Events / ({bw:.3f})" if not unit else f"Events / ({bw:.3f} {unit})"
                ax.set_ylabel(yl)
            ax.tick_params(labelsize=8)

            # Pull (data - model) / σ  — only when show_pull=True
            if show_pull:
                pull_den = np.clip(np.sqrt(dw2), 1.0, None)
                pull = (dy - tot) / pull_den
                pmax = max(np.nanmax(np.abs(pull)), 5)
                ax.set_title(rf"$\chi^2$/ndf = {np.sum(pull**2):.1f}/{np.sum(dw2 > 0)}",
                             fontsize=8)
                ax.tick_params(direction="in")
                ax_pull.bar(bin_c, pull, width=np.diff(bins), alpha=0.5,
                            color='grey', edgecolor='none')
                ax_pull.axhline(y=0, color='black', linewidth=0.5)
                ax_pull.axhline(y=3, color='red', linestyle=':', linewidth=0.8)
                ax_pull.axhline(y=-3, color='red', linestyle=':', linewidth=0.8)
                ax_pull.axhline(y=5, color='red', linestyle=':', linewidth=0.5)
                ax_pull.axhline(y=-5, color='red', linestyle=':', linewidth=0.5)
                ax_pull.set_ylim(-pmax, pmax)
                ax_pull.set_xlim(xlo, xhi)
                ax_pull.minorticks_on()
                ax_pull.tick_params(direction="in")
                ax_pull.set_ylabel("pull", fontsize=8)
                ax_pull.tick_params(labelsize=7)
                # X-label on bottom row
                if row == n_rows - 1 or i + n_cols >= n_var:
                    xl = labels[i]
                    if unit:
                        xl = f"{xl} ({unit})"
                    ax_pull.set_xlabel(xl, fontsize=8)
                else:
                    ax_pull.tick_params(labelbottom=False)
            else:
                xl = labels[i]
                if unit:
                    xl = f"{xl} ({unit})"
                ax.set_xlabel(xl)

        # Hide unused subplots
        n_total = n_rows * n_cols
        # Hide unused subplots
        n_grid_rows = 2 * n_rows if show_pull else n_rows
        for j in range(i + 1, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            if show_pull:
                axes[2 * r, c].set_visible(False)
                axes[2 * r + 1, c].set_visible(False)
            else:
                axes[r, c].set_visible(False)

        if show_pull:
            fig.subplots_adjust(hspace=0)

        if legend:
            fig.axes[0].legend(fontsize=7, ncol=2)

        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

    def plot_asymmetry(self, varfun, labels, lo, hi,
                       n_bins, prefix, tag_data=None, tag_phsp=None,
                       output="plots/", ranges=None,
                       data_weight_extra=None, phsp_weight_extra=None,
                       fmt="png"):
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
                                 figsize=(6 * n_cols, 4.5 * n_rows),
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
            ax.step(bin_c, Am, where='mid', color='grey', linewidth=2,
                    label='total fit')
            ax.axhline(y=0, color='grey', linestyle=':', linewidth=1)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlo, xhi)
            ax.set_xlabel(labels[i])
            ax.tick_params(labelsize=9)

        for i in range(n_var, n_rows * n_cols):
            axes.flatten()[i].set_visible(False)

        axes.flatten()[0].legend(fontsize=7, ncol=2)

        plt.tight_layout()
        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

    def plot_time_asymmetry(self, t_min=0, t_max=10, n_bins=20,
                            output="plots/", prefix="time_asym", params=None,
                            data_weight_extra=None, phsp_weight_extra=None,
                            fmt="png"):
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
            params, _ = f.build_params(self.fit_result.x)
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
        fig, ax = plt.subplots(figsize=(6, 4.5))
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

        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")


    def plot_2d(self, varfun, labels, prefix, binning=[[2, 2]] * 3,
                output="plots/", data_weight_extra=None,
                phsp_weight_extra=None, cmap="jet", plot_scatter=True,
                scatter_style={"s": 1, "c": "black"}, scatter_step=1, fmt="png", ax=None,
                x_range=None, y_range=None):
        """2D adaptive-bin pull plot: data scatter + total-fit pull grid.

        Splits the (var1, var2) plane adaptively (equal-quantile bins,
        following ``tf_pwa``'s ``plot_function_2dpull``).  Each bin is a
        rectangle colored by the pull::

            pull = (sum w_data - sum w_fit) / sqrt(sum w_fit)

        where ``w_fit = phsp_w * P_total * scale`` (+ bkg if present).
        A data scatter overlay and a colorbar are drawn.

        Args:
            varfun: callable(dict) -> ``[var1_data, var2_data]``.
            labels: ``[var1_label, var2_label]``.
            prefix: filename stem.
            binning: adaptive split levels (default ``[[2,2]]*3``).
            output: output directory.
            data_weight_extra, phsp_weight_extra: optional weights.
            cmap: colormap for the pull rectangles.
            plot_scatter: draw data scatter points on top.
            scatter_style: dict passed to ``ax.scatter``.
        """
        if not self._ready:
            raise RuntimeError("call .compute() before .plot_2d()")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patches as mpatches

        f = self.fitter
        dw = f._data_np["weight"] * (data_weight_extra if data_weight_extra is not None else 1.0)
        pw = f._phsp_np["weight"] * (phsp_weight_extra if phsp_weight_extra is not None else 1.0)

        d1, d2 = varfun(f._data_np)
        p1, p2 = varfun(f._phsp_np)

        # Data (cut zero weights)
        cut = dw != 0
        x, y = d1[cut], d2[cut]
        w = dw[cut]

        # Phsp total fit weight
        w_fit = pw * self._P_total * self._scale
        bkg = f._phsp_np.get("bkg", np.zeros(len(pw)))
        bkg_norm = float(np.sum(pw * bkg))
        if bkg_norm > 0:
            purity = f._purity if f._purity is not None else 1.0
            data_total = float(np.sum(dw))
            bkg_scale = data_total * (1.0 - purity) / bkg_norm
            w_fit = w_fit + pw * bkg * bkg_scale

        xlo0, xhi0 = x_range if x_range is not None else (np.min(p1), np.max(p1))
        ylo0, yhi0 = y_range if y_range is not None else (np.min(p2), np.max(p2))
        base_bound = ((xlo0 - 1e-6, ylo0 - 1e-6),
                      (xhi0 + 1e-6, yhi0 + 1e-6))
        bounds, _ = adaptive_split_bound(np.array([x, y]), binning, base_bound)

        pulls = []
        for bnd in bounds:
            xlo, ylo = bnd[0]
            xhi, yhi = bnd[1]
            mask_d = (x >= xlo) & (x < xhi) & (y >= ylo) & (y < yhi)
            mask_p = (p1 >= xlo) & (p1 < xhi) & (p2 >= ylo) & (p2 < yhi)
            ndata = float(np.sum(w[mask_d]))
            nmc = float(np.sum(w_fit[mask_p]))
            pulls.append((ndata - nmc) / np.sqrt(max(nmc, 1.0)))

        max_weight = max(np.max(np.abs(pulls)), 5)
        my_cmap = plt.get_cmap(cmap)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5.5))
        else:
            fig = ax.get_figure()
        if plot_scatter:
            ax.scatter(x[::scatter_step], y[::scatter_step], **scatter_style)
        for bnd, pull in zip(bounds, pulls):
            xlo, ylo = bnd[0]
            xhi, yhi = bnd[1]
            rect = mpatches.Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo, linewidth=1,
                facecolor=my_cmap(pull / max_weight / 2 + 0.5),
                edgecolor="none", zorder=-1)
            ax.add_patch(rect)

        normal = mpl.colors.Normalize(vmin=-max_weight, vmax=max_weight)
        im = mpl.cm.ScalarMappable(norm=normal, cmap=my_cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title(r"$\chi^2/Nbins={:.2f}/{}$".format(
            np.sum(np.abs(pulls) ** 2), len(bounds)))
        ax.set_xlim(xlo0, xhi0)
        ax.set_ylim(ylo0, yhi0)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if output is not None:
            path = os.path.join(output, prefix + "." + fmt)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            if ax is None:
                plt.close(fig)
            print(f"  saved {path}")


    def plot_stacked_perm(self, varfun, xlabel, lo, hi, bin_width,
                          prefix, output="plots/",
                          data_weight_extra=None, phsp_weight_extra=None,
                          smooth_sigma=None, unit="GeV", scales=None,
                          show_pull=False, legend=False,
                          fmt="png"):
        """Plot stacked histogram from permutation-equivalent variables.

        All arrays returned by *varfun* are flattened together: weights
        are repeated so each event contributes to every permutation bin.
        Errors propagate via ``√Σ w²`` over the flattened set.

        With *scales*, each permutation is multiplied by its scale factor
        before summing, e.g. ``scales=[1, -1]`` gives ``mass1 - mass2``.

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
            scales: optional list of scale factors for each permutation.
            show_pull: if True, show pull (data−model)/σ below the panel.
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
        if scales is not None:
            assert len(scales) == n_p, "scales length must match number of permutations"
        scl = np.ones(n_p) if scales is None else np.array(scales, dtype=np.float64)

        n_bins = max(1, int(round((hi - lo) / bin_width)))
        hi = lo + n_bins * bin_width
        bins = np.linspace(lo, hi, n_bins + 1)
        bin_c = (bins[:-1] + bins[1:]) / 2

        # ── data histogram ────────────────────────────────────────
        # Flatten perms, apply per-perm scale, tile weights
        d_all = np.concatenate(var_data)
        dw_all = np.concatenate([dw * s for s in scl])
        dy, _ = np.histogram(d_all, bins=bins, weights=dw_all)

        # Variance: per-event, per-bin with scales
        bin_idx = np.column_stack([np.digitize(arr, bins) - 1 for arr in var_data])
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # protect against out-of-range
        indicator = np.zeros((len(dw), n_bins), dtype=np.float64)
        for p in range(n_p):
            np.add.at(indicator, (np.arange(len(dw)), bin_idx[:, p]), scl[p])
        var_bin = np.sum((dw[:, None] * indicator) ** 2, axis=0)

        # ── model histograms (with per-perm scales) ───────────────
        p_all = np.concatenate(var_phsp)
        pw_all_sig = np.concatenate([pw * self._P_total * self._scale * s for s in scl])

        bkg = f._phsp_np.get("bkg", np.zeros(len(pw)))
        bkg_norm = float(np.sum(pw * bkg))
        purity = f._purity if f._purity is not None else 1.0
        data_total = float(np.sum(dw))
        bkg_scale = data_total * (1.0 - purity) / bkg_norm if bkg_norm > 0 else 0.0
        pw_all_bkg = np.concatenate([pw * bkg * bkg_scale * s for s in scl])

        glabels = self.labels
        gcolors = plt.cm.tab20(np.linspace(0, 1, len(glabels)))
        pw_all_groups = [
            np.concatenate([pw * Pg * self._scale * s for s in scl])
            for Pg in self._P_groups
        ]

        sig, _ = np.histogram(p_all, bins=bins, weights=pw_all_sig)
        bkg_y, _ = np.histogram(p_all, bins=bins, weights=pw_all_bkg)
        tot = sig + bkg_y
        gys = [np.histogram(p_all, bins=bins, weights=gw)[0] for gw in pw_all_groups]

        # ── smoothing helper for PW groups only ───────────────────
        def _smooth_group(arr):
            if smooth_sigma is None or smooth_sigma <= 0:
                return arr, bin_c, False
            x_fine = np.linspace(lo, hi, n_bins * 10)
            dx = bin_c[:, None] - x_fine[None, :]
            sigma = smooth_sigma * bin_width
            sm = np.dot(arr, np.exp(-0.5 * (dx / sigma)**2))
            sm *= bin_width / (np.sqrt(2 * np.pi) * sigma)
            return sm, x_fine, True

        n_ax = 2 if show_pull else 1
        if show_pull:
            fig = plt.figure(figsize=(6, 5))
            ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax_pull = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.errorbar(bin_c, dy, yerr=np.sqrt(var_bin), fmt='o',
                    color='black', markersize=3, capsize=2, label='data')
        # Background fill — no edge lines
        ax.fill_between(bin_c, 0, bkg_y, step='mid', alpha=0.3,
                        color='C3', edgecolor='none', linewidth=0, label='bkg')
        # Total fit — step
        ax.step(bin_c, tot, where='mid', color='grey', linewidth=2,
                label='total fit')
        # PW groups — smooth if requested, else step
        for gy, c, lab, zo in zip(gys, gcolors, glabels, self._zorders):
            y_plot, x_plot, is_smooth = _smooth_group(gy)
            if is_smooth:
                ax.plot(x_plot, np.maximum(y_plot, 0), '-', color=c, linewidth=1.5,
                        label=lab, zorder=zo)
            else:
                ax.step(bin_c, gy, where='mid', color=c, linewidth=1.5,
                        label=lab, zorder=zo)
        ax.set_xlim(lo, hi)
        y_bottom = None if any(s < 0 for s in scl) else 0
        ax.set_ylim(y_bottom, None)

        if show_pull:
            # Pull below
            ax.tick_params(labelbottom=False, direction="in")
            pull_den = np.clip(np.sqrt(var_bin), 1.0, None)
            pull = (dy - tot) / pull_den
            pmax = max(np.nanmax(np.abs(pull)), 5)
            ax.set_title(rf"$\chi^2$/ndf = {np.nansum(pull**2):.1f}/{np.sum(var_bin > 0)}",
                         fontsize=9)
            ax_pull.bar(bin_c, pull, width=np.diff(bins), alpha=0.5,
                        color='grey', edgecolor='none')
            ax_pull.axhline(y=0, color='black', linewidth=0.5)
            ax_pull.axhline(y=3, color='red', linestyle=':', linewidth=0.8)
            ax_pull.axhline(y=-3, color='red', linestyle=':', linewidth=0.8)
            ax_pull.axhline(y=5, color='red', linestyle=':', linewidth=0.5)
            ax_pull.axhline(y=-5, color='red', linestyle=':', linewidth=0.5)
            ax_pull.set_ylim(-pmax, pmax)
            ax_pull.set_xlim(lo, hi)
            ax_pull.minorticks_on()
            ax_pull.tick_params(direction="in")
            ax_pull.set_ylabel("pull", fontsize=9)
            ax_pull.tick_params(labelsize=8)
            xl = xlabel if not unit else f"{xlabel} ({unit})"
            ax_pull.set_xlabel(xl)
            fig.subplots_adjust(hspace=0)
        else:
            xl = xlabel if not unit else f"{xlabel} ({unit})"
            ax.set_xlabel(xl)
        bw = (hi - lo) / n_bins
        yl = f"Events / ({bw:.3f})" if not unit else f"Events / ({bw:.3f} {unit})"
        ax.set_ylabel(yl)
        if legend:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Save
        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")
