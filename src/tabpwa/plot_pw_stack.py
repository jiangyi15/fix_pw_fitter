"""PWStackPlotter — merge the plot data of several datasets, then draw.

Every dataset keeps its own
:class:`~tabpwa.plot_pw_groups.PWGroupPlotter` (its own weights,
normalisation, purity and — if used — resolution rows); this class
**merges the plot data** (the pre-binning weighted samples from
:meth:`PWGroupPlotter.var_samples`) and bins once, so nothing forces the
datasets to share axes or binning.  Drawing itself is the shared code in
:mod:`tabpwa.plot_pw_draw`.

Typical use, right after a joint fit::

    plotter = PWStackPlotter(
        [(f, result, groups) for f in joint.fitters],
        labels=joint.labels())
    plotter.plot_var(lambda x: [x["mass"][:, 0]], ["m(pipi)"],
                     0.28, 1.5, 0.02, "mpipi", stacked=True)

with ``stacked=False`` for one-colour-per-dataset overlays (data points
plus that dataset's own model curve).

Not tied to a joint fit at all: any list of
``(fitter, x_or_result, groups)`` sharing the same variables works.
"""
import warnings

import numpy as np

from .plot_pw_draw import draw_overlay_figure, draw_var_figure


class PWStackPlotter:
    """Merge the plot data of N datasets and draw stacked/overlaid panels."""

    def __init__(self, entries, labels=None, output="plots/"):
        """Args:
            entries: ``[(fitter, x_or_result, groups), ...]`` — one per
                dataset; ``x_or_result`` is the (shared) parameter vector
                or a ``FitResult``.
            labels: per-dataset names (defaults to ``cfg0``, ``cfg1``, …).
            output: default output directory.
        """
        self.entries = list(entries)
        self.labels = list(labels) if labels is not None else \
            [f"cfg{i}" for i in range(len(self.entries))]
        if len(self.labels) != len(self.entries):
            raise ValueError("labels and entries must have the same length")
        self.output = output
        self._plotters = None

    # ── per-dataset computation ───────────────────────────────────

    def compute(self):
        """Build + compute one PWGroupPlotter per dataset (idempotent)."""
        from tabpwa.fit_result import FitResult
        from tabpwa.plot_pw_groups import PWGroupPlotter

        self._plotters = []
        for fitter, x, groups in self.entries:
            res = x if hasattr(x, "x") else FitResult.from_x(
                np.asarray(x, dtype=float).ravel())
            self._plotters.append(
                PWGroupPlotter(fitter, res, dict(groups or {})).compute())
        return self

    @property
    def plotters(self):
        if self._plotters is None:
            self.compute()
        return self._plotters

    # ── merged plot data ──────────────────────────────────────────

    def merged_samples(self, varfun, data_weight_extra=None,
                       phsp_weight_extra=None, cut_function=None):
        """Concatenate every dataset's weighted samples (pre-binning).

        Returns the same dict shape as
        :meth:`PWGroupPlotter.var_samples`, with the per-dataset arrays
        concatenated.  ``cut_function`` is the ONE callable applied to every
        dataset's data AND phsp (see :func:`tabpwa.plot_pw_groups.
        resolve_extra`); the per-sample arrays override it.  Group curves are restricted to the labels present
        in **every** dataset (a warning lists any dropped ones).
        """
        per = [pl.var_samples(varfun, data_weight_extra, phsp_weight_extra,
                              cut_function=cut_function)
               for pl in self.plotters]
        n_var = per[0]["meta"]["n_var"]

        def cat(key):
            return [np.concatenate([s[key][i] for s in per])
                    for i in range(n_var)]

        # group curves need the label in EVERY dataset — intersect over all
        # (not just dataset 0), and report anything that gets dropped
        common = [lbl for lbl in per[0]["w_groups"]
                  if all(lbl in s["w_groups"] for s in per)]
        dropped = [lbl for s in per for lbl in s["w_groups"]
                   if lbl not in common]
        dropped = sorted(set(dropped))
        if dropped:
            warnings.warn(
                f"PWStackPlotter: group(s) {dropped} are not present in "
                f"every dataset — dropping them from the stack",
                RuntimeWarning, stacklevel=2)
        groups = {lbl: np.concatenate([s["w_groups"][lbl] for s in per])
                  for lbl in common}
        return {
            "x_data": cat("x_data"), "w_data": np.concatenate(
                [s["w_data"] for s in per]),
            "x_model": cat("x_model"), "w_model": np.concatenate(
                [s["w_model"] for s in per]),
            "x_bkg": cat("x_bkg"), "w_bkg": np.concatenate(
                [s["w_bkg"] for s in per]),
            "w_groups": groups,
            "meta": {"n_var": n_var, "labels": list(self.labels)},
        }

    # ── drawing ───────────────────────────────────────────────────

    def plot_var(self, varfun, labels, lo, hi, bin_width, prefix,
                 stacked=True, ranges=None, legend=True, show_pull=False,
                 data_weight_extra=None, phsp_weight_extra=None,
                 cut_function=None,
                 mc_uncert=False, smooth_sigma=None, unit="GeV", fmt="png",
                 output=None, group_labels=None, colors=None, zorders=None,
                 yscale=None, xscale=None, upper_ylim=None):
        """Draw one variable set, stacked (default) or overlaid.

        ``stacked=True`` merges the plot data and bins once: data points,
        ``√Σw²`` errors, the summed model/background and the summed group
        curves.  ``stacked=False`` overlays the datasets (each with its
        own colour, data points + its own model curve) so no normalisation
        is implied.  Returns the saved path.

        ``yscale`` / ``xscale`` / ``upper_ylim`` style the axes; each is a
        single value for every sub-variable or one entry per sub-variable.

        ``cut_function`` is ONE callable applied to every dataset's data AND
        phsp (a cut, a tag weight, …); ``data_weight_extra`` /
        ``phsp_weight_extra`` are the per-sample arrays and override it.
        """
        output = output or self.output
        if stacked:
            smp = self.merged_samples(varfun, data_weight_extra,
                                      phsp_weight_extra, cut_function)
            return draw_var_figure(
                smp, labels, lo, hi, bin_width, prefix, ranges=ranges,
                legend=legend, output=output, fmt=fmt,
                group_labels=group_labels, colors=colors, zorders=zorders,
                smooth_sigma=smooth_sigma, unit=unit, show_pull=show_pull,
                mc_uncert=mc_uncert, yscale=yscale, xscale=xscale,
                upper_ylim=upper_ylim)
        samples = [pl.var_samples(varfun, data_weight_extra,
                                  phsp_weight_extra, cut_function)
                   for pl in self.plotters]
        return draw_overlay_figure(
            samples, self.labels, lo, hi, bin_width, prefix, ranges=ranges,
            legend=legend, output=output, fmt=fmt, unit=unit,
            yscale=yscale, xscale=xscale, upper_ylim=upper_ylim)

    def plot_common(self, stacked=True, output=None, fmt="png", **kw):
        """Mass overview (all mass columns) — stacked or overlaid."""
        output = output or self.output
        nm = self.plotters[0].fitter._data_np["mass"].shape[1] // 8
        return self.plot_var(
            lambda x: [x["mass"][:, i] for i in range(nm)],
            [f"mass[{i}]" for i in range(nm)], 0.2, 5.2, 0.05, "mass",
            stacked=stacked, output=output, fmt=fmt, smooth_sigma=1.0, **kw)

    def __repr__(self):
        return (f"PWStackPlotter(n_datasets={len(self.entries)}, "
                f"labels={self.labels})")
