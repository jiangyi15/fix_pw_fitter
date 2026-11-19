"""Shared drawing primitives for the partial-wave plotters.

One place owns the layout/histogramming/pull logic, so every plotter just
calls these functions with a **pre-binning sample dict** (the shape
produced by ``PWGroupPlotter.var_samples`` — possibly merged over several
datasets by :class:`~tabpwa.plot_pw_stack.PWStackPlotter`):

    fig, axes, n_rows, n_cols = make_var_axes(n_var, show_pull)
    bins, centers, n_bins = bin_edges(xlo, xhi, bin_width)
    draw_var_panel(ax, samples, i, bins, ax_pull=…, first=(i == 0), …)
    save_figure(fig, output, prefix, fmt)

Sample dict keys (values are lists, one entry per sub-variable):
``x_data``/``w_data``, ``x_model``/``w_model``, ``x_bkg``/``w_bkg`` and
``w_groups`` (``{label: weights}``).
"""
import os

import numpy as np


def bin_edges(xlo, xhi, bin_width):
    """``(edges, centers, n_bins)`` with *xhi* snapped to whole bins."""
    n_bins = max(1, int(round((xhi - xlo) / bin_width)))
    xhi = xlo + n_bins * bin_width
    edges = np.linspace(xlo, xhi, n_bins + 1)
    return edges, (edges[:-1] + edges[1:]) / 2, n_bins


_LOG_DECADES = 4.0     # a log-y panel spans at most this many decades


def per_variable(value, n_var, name="value"):
    """Broadcast one axis-style setting over *n_var* sub-variables.

    ``None`` means "leave the axes alone" for every panel; a scalar (a
    string such as ``"log"`` or a number such as ``20.0``) applies to all;
    a sequence must hold exactly *n_var* entries — one per sub-variable, in
    the order the variables are drawn.
    """
    if value is None:
        return [None] * n_var
    if isinstance(value, str) or np.ndim(value) == 0:
        return [value] * n_var
    seq = list(value)
    if len(seq) != n_var:
        raise ValueError(
            f"{name} has {len(seq)} entries but this figure has "
            f"{n_var} sub-variable(s)")
    return seq


def panel_ylimits(values, upper_ylim=None, log=False):
    """``(bottom, top)`` y limits for a main panel (``top=None`` = auto).

    On a linear axis the floor is 0 (the usual histogram convention) and an
    unset top is left to matplotlib.  On a log axis the limits are always
    explicit: 0 is not a valid limit and the autoscale of an axis whose
    artists contain zero-height bins is meaningless, so the floor comes
    from the smallest POSITIVE drawn value (capped at ``_LOG_DECADES``
    below the top) and the top is padded by 5 % of that span.  *upper_ylim*
    caps the top on either scale, exactly.
    """
    top = None if upper_ylim is None else float(upper_ylim)
    if not log:
        return 0.0, top
    arrs = [np.asarray(v, dtype=float).ravel() for v in values]
    arr = np.concatenate(arrs) if arrs else np.zeros(0)
    pos = arr[np.isfinite(arr) & (arr > 0)]
    if pos.size == 0:
        return (0.1, 1.0 if top is None else top)
    hi = top if top is not None else float(pos.max())
    floor = max(float(pos.min()) * 0.5, hi * 10.0 ** (-_LOG_DECADES))
    if not floor < hi:
        floor = hi * 0.5
    if top is None:
        decades = np.log10(hi) - np.log10(floor)
        top = float(hi * 10.0 ** (0.05 * decades))
    return floor, top


def make_var_axes(n_var, show_pull=False, n_cols=3):
    """Figure + axes grid for *n_var* sub-variables.

    With ``show_pull`` each column carries a narrow pull panel below the
    main one (shared x).  Returns ``(fig, axes, n_rows, n_cols)``.
    """
    import matplotlib.pyplot as plt

    n_cols = min(n_cols, n_var)
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
        for r in range(1, 2 * n_rows):
            for c in range(n_cols):
                axes[r, c].sharex(axes[0, c])
    else:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 4.5 * n_rows),
                                 squeeze=False)
    return fig, axes, n_rows, n_cols


def panel_axes(axes, i, n_cols, show_pull):
    """The ``(main, pull)`` axes for sub-variable *i* (pull may be None)."""
    row, col = divmod(i, n_cols)
    if show_pull:
        return axes[2 * row, col], axes[2 * row + 1, col]
    return axes[row, col], None


def _log_safe(y, log_y):
    """Copy of *y* with non-positive entries masked for a log axis.

    A line vertex at 0 has no log coordinate; matplotlib would draw the
    neighbouring vertex's drop as a full-panel spike.  Masking breaks the
    curve there instead (a zero is simply off the bottom of the axis).
    """
    if not log_y:
        return y
    y = np.array(y, dtype=float)
    y[y <= 0] = np.nan
    return y


def draw_var_panel(ax, samples, i, bins, *, ax_pull=None, first=False,
                   legend=False, colors=None, glabels=None, zorders=None,
                   smooth_sigma=None, unit="GeV", mc_uncert=False,
                   bin_width=None, xlabel=None,
                   data_label="data", model_label="total fit",
                   bkg_label="bkg", yscale=None, xscale=None,
                   upper_ylim=None):
    """Histogram sub-variable *i* of *samples* and draw the standard panel.

    Draws data points (errors ``√Σw²``), the background fill, the total
    fit step and the partial-wave group curves, plus the optional pull
    panel.  Returns the per-bin pull array.

    *yscale* / *xscale* select the axis scale (``"log"``, …) and
    *upper_ylim* caps the y axis; the y limits are scale-aware, so a log
    axis gets a positive floor instead of the linear ``(0, None)``.  The
    pull panel is always linear.
    """
    import matplotlib.pyplot as plt

    d = samples["x_data"][i]
    p = samples["x_model"][i]
    dw = samples["w_data"]
    xlo, xhi = float(bins[0]), float(bins[-1])
    n_bins = len(bins) - 1
    bin_c = (bins[:-1] + bins[1:]) / 2

    dy, _ = np.histogram(d, bins=bins, weights=dw)
    dw2, _ = np.histogram(d, bins=bins, weights=dw ** 2)
    sig, _ = np.histogram(p, bins=bins, weights=samples["w_model"])
    bkg_y, _ = np.histogram(p, bins=bins, weights=samples["w_bkg"])
    tot = sig + bkg_y

    pull_var = dw2.copy()
    if mc_uncert:
        # one event carries ONE model weight: the signal and background
        # predictions are two components of the same phsp event, so their
        # statistical fluctuation is the fluctuation of their SUM (the
        # cross term is part of the variance).
        mc_w = samples["w_model"] + samples["w_bkg"]
        mc2, _ = np.histogram(p, bins=bins, weights=mc_w ** 2)
        pull_var = pull_var + mc2

    group_items = list(samples["w_groups"].items())
    gys = [np.histogram(p, bins=bins, weights=w)[0] for _lbl, w in group_items]
    labels_g = (glabels if glabels is not None
                else [lbl for lbl, _w in group_items])
    zorders = list(zorders) if zorders is not None else [5] * len(gys)
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(gys))))

    pull = np.divide(dy - tot, np.sqrt(pull_var),
                     where=pull_var > 0, out=np.zeros_like(dy))

    log_y = yscale is not None and str(yscale).lower() == "log"
    yvals = [dy, tot, bkg_y, *gys]
    # scale-aware y limits: on a log axis the fill has to start at the
    # panel floor (0 is not a valid log coordinate) and the top is explicit
    bottom, top = panel_ylimits(yvals, upper_ylim, log_y)

    if log_y:
        # an empty bin has no place on a log axis: a zero-height error bar
        # there is drawn as a full-panel spike, so skip those points (a
        # zero count is not visible on a log axis anyway)
        keep = dy > 0
        eb_x, eb_y, eb_e = bin_c[keep], dy[keep], np.sqrt(dw2)[keep]
    else:
        eb_x, eb_y, eb_e = bin_c, dy, np.sqrt(dw2)
    ax.errorbar(eb_x, eb_y, yerr=eb_e, fmt='o', color='black',
                markersize=3, capsize=2,
                label=data_label if legend and first else None)
    ax.fill_between(bin_c, bottom, bkg_y, step='mid', alpha=0.3, color='C3',
                    edgecolor='none', linewidth=0,
                    label=bkg_label if legend and first else None)
    ax.step(bin_c, _log_safe(tot, log_y), where='mid', color='grey',
            linewidth=2,
            label=model_label if legend and first else None)

    bw = (xhi - xlo) / n_bins if bin_width is None else bin_width
    for gy, c, lab, zo in zip(gys, colors, labels_g, zorders):
        if smooth_sigma is not None and smooth_sigma > 0:
            x_fine = np.linspace(xlo, xhi, n_bins * 10)
            dx = bin_c[:, None] - x_fine[None, :]
            sigma = smooth_sigma * bin_width
            gy_fine = np.dot(gy, np.exp(-0.5 * (dx / sigma) ** 2))
            gy_fine *= bin_width / (np.sqrt(2 * np.pi) * sigma)
            ax.plot(x_fine, _log_safe(gy_fine, log_y), '-', color=c,
                    linewidth=1.5,
                    label=lab if legend and first else None, zorder=zo)
        else:
            ax.step(bin_c, _log_safe(gy, log_y), where='mid', color=c,
                    linewidth=1.5,
                    label=lab if legend and first else None, zorder=zo)

    ax.grid(True, alpha=0.3)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(bottom, top)
    if ax_pull is not None:
        ax.tick_params(labelbottom=False)
    if first:
        yl = (f"Events / ({bw:.3f})" if not unit
              else f"Events / ({bw:.3f} {unit})")
        ax.set_ylabel(yl)
    ax.tick_params(labelsize=8)

    if ax_pull is not None:
        pull_den = np.clip(np.sqrt(pull_var), 1.0, None)
        pull = (dy - tot) / pull_den
        pmax = max(np.nanmax(np.abs(pull)), 5)
        ax.set_title(rf"$\chi^2$/ndf = {np.sum(pull**2):.1f}/"
                     rf"{np.sum(pull_var > 0)}"
                     + (" (data+MC)" if mc_uncert else ""), fontsize=8)
        ax.tick_params(direction="in")
        ax_pull.bar(bin_c, pull, width=np.diff(bins), alpha=0.5, color='grey',
                    edgecolor='none')
        ax_pull.axhline(y=0, color='black', linewidth=0.5)
        for y in (3, -3, 5, -5):
            ax_pull.axhline(y=y, color='red', linestyle=':',
                            linewidth=0.8 if abs(y) == 3 else 0.5)
        # the pull panel is always linear, whatever the main axis is
        ax_pull.set_ylim(-pmax, pmax)
        ax_pull.set_xlim(xlo, xhi)
        if xscale is not None:
            ax_pull.set_xscale(xscale)
        ax_pull.minorticks_on()
        ax_pull.tick_params(direction="in")
        ax_pull.set_ylabel("pull", fontsize=8)
        ax_pull.tick_params(labelsize=7)
        if xlabel is not None:
            ax_pull.set_xlabel(xlabel, fontsize=8)
    return pull


def draw_var_figure(samples, labels, lo, hi, bin_width, prefix, *,
                    ranges=None, legend=False, output="plots/", fmt="png",
                    group_labels=None, colors=None, zorders=None,
                    smooth_sigma=None, unit="GeV", show_pull=False,
                    mc_uncert=False, n_cols=3, yscale=None, xscale=None,
                    upper_ylim=None):
    """Draw the standard variable figure for one sample dict.

    This is the whole "plot one (merged) sample set" step: layout, one
    panel per sub-variable, pull panels, legend, save.  Returns the path.

    *yscale* / *xscale* / *upper_ylim* style the axes: each may be a single
    value for every sub-variable or one entry per sub-variable (in draw
    order).
    """
    n_var = len(samples["x_data"])
    ysc = per_variable(yscale, n_var, "yscale")
    xsc = per_variable(xscale, n_var, "xscale")
    uyl = per_variable(upper_ylim, n_var, "upper_ylim")
    fig, axes, n_rows, n_cols = make_var_axes(n_var, show_pull, n_cols)
    for i in range(n_var):
        ax, ax_pull = panel_axes(axes, i, n_cols, show_pull)
        xlo, xhi = (ranges[i] if ranges else (lo, hi))
        bins, _c, _n = bin_edges(xlo, xhi, bin_width)
        xl = labels[i] if not unit else f"{labels[i]} ({unit})"
        last_row = (i // n_cols == n_rows - 1) or (i + n_cols >= n_var)
        draw_var_panel(
            ax, samples, i, bins, ax_pull=ax_pull, first=(i == 0),
            legend=legend, colors=colors, glabels=group_labels,
            zorders=zorders, smooth_sigma=smooth_sigma, unit=unit,
            mc_uncert=mc_uncert, bin_width=bin_width,
            xscale=xsc[i], yscale=ysc[i], upper_ylim=uyl[i],
            xlabel=(xl if (show_pull and last_row) else None))
        if not show_pull:
            ax.set_xlabel(xl)
        elif not last_row:
            ax_pull.tick_params(labelbottom=False)

    hide_unused(axes, n_var, n_rows, n_cols, show_pull)
    if show_pull:
        fig.subplots_adjust(hspace=0)
    if legend:
        fig.axes[0].legend(fontsize=7, ncol=2)
    return save_figure(fig, output, prefix, fmt)


def draw_overlay_figure(samples_list, labels, lo, hi, bin_width, prefix, *,
                        ranges=None, legend=True, output="plots/", fmt="png",
                        unit="GeV", group_labels=None, n_cols=3,
                        yscale=None, xscale=None, upper_ylim=None):
    """Overlay several datasets on shared axes (one colour per dataset).

    Each entry draws its data points and its model curve for every
    sub-variable.  No binning needs to be shared per dataset — only
    *samples_list* may contain anything that came from the same varfun.

    *yscale* / *xscale* / *upper_ylim* style the axes: each may be a single
    value for every sub-variable or one entry per sub-variable (in draw
    order).
    """
    import matplotlib.pyplot as plt

    n_var = len(samples_list[0]["x_data"])
    ysc = per_variable(yscale, n_var, "yscale")
    xsc = per_variable(xscale, n_var, "xscale")
    uyl = per_variable(upper_ylim, n_var, "upper_ylim")
    fig, axes, n_rows, n_cols = make_var_axes(n_var, False, n_cols)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(samples_list))))

    for i in range(n_var):
        ax, _ = panel_axes(axes, i, n_cols, False)
        xlo, xhi = (ranges[i] if ranges else (lo, hi))
        bins, bin_c, n_bins = bin_edges(xlo, xhi, bin_width)
        log_y = ysc[i] is not None and str(ysc[i]).lower() == "log"
        yvals = []
        for k, (smp, lab, col) in enumerate(zip(samples_list, labels, colors)):
            kw = dict(bins=bins)
            dy, _ = np.histogram(smp["x_data"][i], weights=smp["w_data"], **kw)
            dw2, _ = np.histogram(smp["x_data"][i],
                                  weights=smp["w_data"] ** 2, **kw)
            mod, _ = np.histogram(smp["x_model"][i],
                                  weights=smp["w_model"], **kw)
            bkg, _ = np.histogram(smp["x_model"][i],
                                  weights=smp["w_bkg"], **kw)
            yvals.extend((dy, mod + bkg))
            # as in draw_var_panel: empty bins are not drawable on a log
            # axis (a zero-height error bar becomes a full-panel spike)
            keep = dy > 0 if log_y else np.ones(dy.shape, dtype=bool)
            ax.errorbar(bin_c[keep], dy[keep], yerr=np.sqrt(dw2)[keep],
                        fmt='o', color=col, markersize=3, capsize=2,
                        label=f"{lab} (data)")
            ax.step(bin_c, _log_safe(mod + bkg, log_y), where='mid', color=col,
                    linewidth=1.5, label=f"{lab} (fit)")
        ax.grid(True, alpha=0.3)
        if xsc[i] is not None:
            ax.set_xscale(xsc[i])
        if ysc[i] is not None:
            ax.set_yscale(ysc[i])
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(*panel_ylimits(yvals, uyl[i], log_y))
        xl = labels[i] if not unit else f"{labels[i]} ({unit})"
        ax.set_xlabel(xl)
        ax.set_ylabel(f"Events / ({bin_width:.3f} {unit})" if i == 0 else "")

    hide_unused(axes, n_var, n_rows, n_cols, False)
    if legend:
        fig.axes[0].legend(fontsize=7, ncol=2)
    return save_figure(fig, output, prefix, fmt)


def hide_unused(axes, n_var, n_rows, n_cols, show_pull=False):
    """Hide the axes of unused slots (and their pull panels)."""
    for j in range(n_var, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        if show_pull:
            axes[2 * r, c].set_visible(False)
            axes[2 * r + 1, c].set_visible(False)
        else:
            axes[r, c].set_visible(False)


def save_figure(fig, output, prefix, fmt="png"):
    """Write ``{output}/{prefix}.{fmt}`` and close the figure."""
    import matplotlib.pyplot as plt

    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, prefix + "." + fmt)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")
    return path
