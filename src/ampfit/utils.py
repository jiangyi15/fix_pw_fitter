"""Utility functions for formatting and display."""
import numpy as np
import re


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


# ── Particle name → LaTeX display ────────────────────────────────

_PARTICLE_NAMES = {
    "a0(980)":       r"a_0(980)",
    "a0(1450)":      r"a_0(1450)",
    "a1(1260)":      r"a_1(1260)",
    "a1(1640)":      r"a_1(1640)",
    "a2(1320)":      r"a_2(1320)",
    "a2(1700)":      r"a_2(1700)",
    "b1(1235)":      r"b_1(1235)",
    "D0":            r"D^0",
    "Dbar0":         r"\bar{D}^0",
    "f0(500)":       r"f_0(500)",
    "f0(980)":       r"f_0(980)",
    "f0(1370)":      r"f_0(1370)",
    "f2(1270)":      r"f_2(1270)",
    "f2(1525)":      r"f_2'(1525)",
    "K0star(700)":   r"K_0^*(700)",
    "K0star(1430)":  r"K_0^*(1430)",
    "K1(1270)":      r"K_1(1270)",
    "K1(1400)":      r"K_1(1400)",
    "K2(1430)":      r"K_2^*(1430)",
    "NR0":           r"\text{NR}",
    "omega(782)":    r"\omega(782)",
    "phi(1020)":     r"\phi(1020)",
    "pi(1300)":      r"\pi(1300)",
    "pi(1800)":      r"\pi(1800)",
    "pi1(1600)":     r"\pi_1(1600)",
    "pi2(1670)":     r"\pi_2(1670)",
    "rho(770)":      r"\rho(770)",
    "rho(1450)":     r"\rho(1450)",
    "rhoA":          r"\rho",
    "rhoB":          r"\rho",
}


def fmt_particle(name, full=True):
    """Convert a particle config name to a LaTeX display string.

    Handles charge suffixes ``p``/``m`` → ``+``/``-``.

    Parameters
    ----------
    name : str
        Particle name from config, e.g. ``"a1(1260)p"`` or ``"f0(980)"``.
    full : bool
        If True (default), returns a LaTeX math string ``$...$``.
        If False, returns just the formatted name without delimiters.

    Returns
    -------
    str
        LaTeX display name.
    """
    # Strip charge suffix
    base = name
    charge = ""
    if base.endswith("p") and not base.endswith("rhoA") and not base.endswith("KMA"):
        base = base[:-1]
        charge = "+"
    elif base.endswith("m") and not base.endswith("rhoB") and not base.endswith("KMB"):
        base = base[:-1]
        charge = "-"
    elif base.endswith("b"):
        base = base[:-1]
        charge = r"\bar"

    # Look up base; fall back to raw
    if base in _PARTICLE_NAMES:
        display = _PARTICLE_NAMES[base]
    else:
        # Try to parse as generic name with mass: "name(MASS)"
        m = re.match(r'^([a-zA-Z_0-9]+)\((\d+)\)$', base)
        if m:
            sym = m.group(1)
            mass = m.group(2)
            # Try to convert symbol to LaTeX: pi → \pi, rho → \rho, f0 → f_0, etc.
            sym_latex = _SYMBOL_MAP.get(sym, sym)
            display = rf"{sym_latex}({mass})"
        else:
            display = base

    if charge in ("+", "-"):
        result = rf"{display}^{charge}"
    elif charge:
        result = rf"{charge}{display}"
    else:
        result = display

    if full:
        return f"${result}$"
    return result


_SYMBOL_MAP = {
    "pi":       r"\pi",
    "rho":      r"\rho",
    "omega":    r"\omega",
    "phi":      r"\phi",
    "K":        r"K",
    "B":        r"B",
    "D":        r"D",
    "J":        r"J",
    "f0":       r"f_0",
    "f2":       r"f_2",
    "a0":       r"a_0",
    "a1":       r"a_1",
    "a2":       r"a_2",
    "b1":       r"b_1",
    "pi1":      r"\pi_1",
    "pi2":      r"\pi_2",
}
