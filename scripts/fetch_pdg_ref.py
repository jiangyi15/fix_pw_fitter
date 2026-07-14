#!/usr/bin/env python3
"""Fetch PDG mass/width reference values for light meson resonances.

Generates CSV files (one per resonance) with best-fit mass and width
values from the PDG API, including multiple measurement methods where
available (e+e-, photoproduced, hadroproduced, individual decay modes).

Format::

    name,mass,mass_err_lo,mass_err_hi,width,width_err_lo,width_err_hi
    "NEUTRAL ONLY, e+ e-",775.3,0.2,0.2,147.4,0.8,0.8
    "MIXED CHARGES, OTHER REACTIONS",763.0,1.2,1.2,149.5,1.3,1.3
    a_2(1320) MASS,1318.2,0.6,0.6,107.0,5.0,5.0

When errors are symmetric, ``err_lo = err_hi``.
When unavailable, values are 0.


Usage::

    pip install pdg
    python scripts/fetch_pdg_ref.py <output_dir>

Requires: pdg package (pip install pdg)
"""
import os, csv, sys
from pdg import connect


# Resonances: name → list of (PDG_lookup_name, charge_label)
# PDG names use underscore convention: a_1(1260), pi_2(1670), rho(770)+
RESONANCES = {
    'rho(770)':   [('rho(770)0', 'neutral'), ('rho(770)+', 'charged')],
    'rho(1450)':  [('rho(1450)0', 'neutral'), ('rho(1450)+', 'charged')],
    'rho(1700)':  [('rho(1700)0', 'neutral'), ('rho(1700)+', 'charged')],
    'a1(1260)':   [('a_1(1260)0', 'charged')],
    'a1(1640)':   [('a_1(1640)0', 'neutral'), ('a_1(1640)+', 'charged')],
    'a2(1320)':   [('a_2(1320)0', 'neutral'), ('a_2(1320)+', 'charged')],
    'a2(1700)':   [('a_2(1700)0', 'neutral'), ('a_2(1700)+', 'charged')],
    'f2(1270)':   [('f_2(1270)', '')],
    'f0(980)':    [('f_0(980)', '')],
    'pi(1300)':   [('pi(1300)0', 'neutral'), ('pi(1300)+', 'charged')],
    'pi(1800)':   [('pi(1800)0', 'neutral'), ('pi(1800)+', 'charged')],
    'pi1(1600)':  [('pi_1(1600)0', 'neutral'), ('pi_1(1600)+', 'charged')],
    'pi2(1670)':  [('pi_2(1670)0', 'neutral'), ('pi_2(1670)+', 'charged')],
}


def sanitize(name):
    """Sanitize a resonance name for use as a filename."""
    return name.replace('(', '_').replace(')', '').replace(' ', '_')


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/fetch_pdg_ref.py <output_dir>', file=sys.stderr)
        sys.exit(1)
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    api = connect()

    n_total = 0
    for pname, specs in sorted(RESONANCES.items()):
        mass_map = {}
        width_map = {}

        for lookup_name, charge in specs:
            try:
                p = api.get_particle_by_name(lookup_name)
            except Exception as e:
                print(f'  [!] {pname} ({charge}): {e}', file=sys.stderr)
                continue

            def extract(prop, store, label_prefix):
                """Extract best value or fallback to individual measurements."""
                bs = prop.best_summary()
                if bs is None:
                    return
                # Strip particle name prefix from description
                prefix = p.name.rstrip('0+*^').strip()
                label = prop.description
                if label.startswith(prefix):
                    label = label[len(prefix):].strip()
                label = label.strip()
                if not label:
                    return

                val, err_hi, err_lo = None, None, None

                if bs.value is not None:
                    # Single best value with errors
                    err_hi = bs.error_positive or 0
                    err_lo = bs.error_negative or err_hi
                    if err_hi > 0:
                        val = bs.value
                else:
                    # Range value like "200 TO 600" → use midpoint ± half-range
                    vt = (bs.value_text or '').upper()
                    import re
                    m = re.match(r'([\d.]+)\s*(?:TO|–|-)\s*([\d.]+)', vt)
                    if m:
                        lo, hi = float(m.group(1)), float(m.group(2))
                        val = (lo + hi) / 2.0
                        err_hi = err_lo = (hi - lo) / 2.0

                if val is not None and err_hi and err_hi > 0 and label not in store:
                    store[label] = (val, err_hi, err_lo or err_hi)

            for prop in p.masses():
                extract(prop, mass_map, 'MASS')
            for prop in p.widths():
                extract(prop, width_map, 'WIDTH')

        # Combine mass/width entries with matching descriptions
        # e.g. "a_2(1320) MASS" + "a_2(1320) WIDTH" → single row
        # Also "pi_1(1600) MASS (eta pi mode)" + "pi_1(1600) WIDTH (eta pi mode)"
        def strip_mw(label):
            import re
            m = re.sub(r'\bMASS\b', '', label)
            m = re.sub(r'\bWIDTH\b', '', m)
            m = re.sub(r'\s+', ' ', m).strip()
            if m == label:
                return None
            return m  # may be empty string — that's fine, it matches MASS↔WIDTH

        # Build mass/width swap lookup: base → counterpart label
        mass_by_base = {}
        width_by_base = {}
        for label in mass_map:
            base = strip_mw(label)
            if base is not None:
                mass_by_base[base] = label
        for label in width_map:
            base = strip_mw(label)
            if base is not None:
                width_by_base[base] = label

        all_labels = set(mass_map) | set(width_map)
        if not all_labels:
            print(f'  [ ] {pname}  — no data')
            continue

        rows = []
        used = set()
        for label in sorted(all_labels):
            if label in used:
                continue
            base = strip_mw(label)
            if base is not None and base in mass_by_base and base in width_by_base:
                # Both "X MASS" and "X WIDTH" exist → combine
                m_label = mass_by_base[base]
                w_label = width_by_base[base]
                m = mass_map.get(m_label)
                w = width_map.get(w_label)
                used.add(m_label)
                used.add(w_label)
                # Use the mass label (or width) for the combined row name
                out_label = m_label if m else w_label
            else:
                m = mass_map.get(label)
                w = width_map.get(label)
                used.add(label)
                out_label = label
            if m:
                m_val, m_err_hi, m_err_lo = m
            else:
                m_val = m_err_lo = m_err_hi = 0.0
            if w:
                w_val, w_err_hi, w_err_lo = w
            else:
                w_val = w_err_lo = w_err_hi = 0.0
            m_val_s = f'{m_val:.1f}'
            m_lo_s = f'{m_err_lo:.1f}'
            m_hi_s = f'{m_err_hi:.1f}'
            w_val_s = f'{w_val:.1f}'
            w_lo_s = f'{w_err_lo:.1f}'
            w_hi_s = f'{w_err_hi:.1f}'
            # Build display name: strip MASS/WIDTH when present
            stripped = strip_mw(out_label)
            if stripped is None:
                # No MASS/WIDTH in label → keep as-is
                display = f'PDG {out_label}' if out_label else 'PDG'
            elif not stripped.strip():
                # Was entirely MASS/WIDTH → just PDG
                display = 'PDG'
            else:
                # Has meaningful content after stripping
                display = f'PDG {stripped.strip()}'
            rows.append([display, m_val_s, m_lo_s, m_hi_s,
                         w_val_s, w_lo_s, w_hi_s])

        fpath = os.path.join(outdir, f'{sanitize(pname)}.csv')
        with open(fpath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['name', 'mass', 'mass_err_lo', 'mass_err_hi',
                        'width', 'width_err_lo', 'width_err_hi'])
            w.writerows(rows)

        n_total += len(rows)
        print(f'  [✓] {pname:>12s}  ({len(rows)} entries)')

    print(f'\nWrote {n_total} entries across {len(RESONANCES)} files to {outdir}/')


if __name__ == '__main__':
    main()
