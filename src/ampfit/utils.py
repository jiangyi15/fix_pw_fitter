"""Utility functions for formatting and display."""
import numpy as np


def fmt_meas(v, e, pct=False):
    """Format ``v ± e`` with decimal places determined by error thresholds.

    Error is first rounded to 3 significant figures, then the
    threshold check determines the number of decimal places:

      0.000 ≤ |e₃| < 0.355 → 2 decimal places
      0.355 ≤ |e₃| < 0.950 → 1 decimal place
      0.950 ≤ |e₃|         → 0 decimal places

    Parameters
    ----------
    v : float
        Central value.
    e : float
        Uncertainty (must be >= 0).
    pct : bool
        If True, append a LaTeX percent sign to the output.

    Returns
    -------
    str
        LaTeX-formatted string: ``$value\\pm error$``, ``$value$``
        when *e* = 0, or ``$-$`` when *v* is None.
    """
    if v is None:
        return r"$-$"

    if e == 0:
        if pct:
            return f"${v:.1f}$\\%"
        return f"${v:.2f}$"

    # Round error to 3 significant figures
    abs_e = abs(e)
    mag = int(np.floor(np.log10(abs_e)))
    scaled = abs_e * 10 ** (-mag)          # in [1.0, 10.0)
    e_3dig = round(scaled, 2) * 10 ** mag  # 3 sig figs

    if e_3dig < 0.355:
        dp = 2
    elif e_3dig < 0.950:
        dp = 1
    else:
        dp = 0

    e_r = round(e, dp)
    v_r = round(v, dp)
    pct_suf = r"\%" if pct else ""
    return f"${v_r:.{dp}f}\\pm{e_r:.{dp}f}{pct_suf}$"
