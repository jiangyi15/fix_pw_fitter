#!/usr/bin/env python3
"""Print fitted masses and widths with uncertainties.

Usage:
    python scripts/print_mass_width.py <results.json>
    python scripts/print_mass_width.py fit_output8/results.json
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/print_mass_width.py <results.json>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    v = data.get("value", data)
    e = data.get("error", {})

    # Find mass/width keys with non-zero error (free params)
    items = []
    for key in sorted(v):
        if not ("_mass" in key or "_width" in key):
            continue
        val = float(v[key])
        err = float(e.get(key, 0.0))
        # Extract particle name and type (mass/width)
        if key.endswith("_mass"):
            particle = key[:-5]
            typ = "mass "
        elif key.endswith("_width"):
            particle = key[:-6]
            typ = "width"
        else:
            continue
        # Skip charge-conjugate alias (m_xxx where p_xxx already printed)
        if particle.endswith("m_") or particle.endswith("m"):
            continue
        items.append((particle, typ, val, err))

    if not items:
        print("No free mass/width parameters found.")
        return

    print(f"{'Particle':<25} {'Type':<8} {'Value':<12} {'Error':<12}  {'Status'}")
    print("-" * 70)
    for particle, typ, val, err in items:
        status = "free" if err > 0 else "fixed"
        err_str = f"{err:.6f}" if err > 0 else "—"
        print(f"{particle:<25} {typ:<8} {val:<12.6f} {err_str:<12}  {status}")


if __name__ == "__main__":
    main()
