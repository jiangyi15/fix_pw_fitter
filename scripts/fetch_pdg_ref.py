#!/usr/bin/env python3
"""Fetch PDG mass/width reference values for light meson resonances.

Generates CSV files (one per resonance) with best-fit mass and width
values from the PDG API, including multiple measurement methods where
available (e+e-, photoproduced, hadroproduced, individual decay modes).

Format::

    name,mass,mass_err,width,width_err
    "rho(770) neutral NEUTRAL ONLY, e+ e-",775.3,0.2,147.4,0.8
    "rho(770) neutral MIXED CHARGES, OTHER REACTIONS",763.0,1.2,150.9,1.7

Each row always has both mass and width columns (0 when not available),
so plot_bw_cov.py can read it without errors.

Usage::

    pip install pdg
    python scripts/fetch_pdg_ref.py <output_dir>

Requires: pdg package (pip install pdg)
"""
import os, csv, sys
from pdg import connect


# Resonances: name → [(MCID, charge_label), ...]
RESONANCES = {
    'rho(770)':   [(113, 'neutral'), (213, 'charged')],
    'rho(1450)':  [(100113, 'neutral'), (100213, 'charged')],
    'rho(1700)':  [(30113, 'neutral'), (30213, 'charged')],
    'a1(1260)':   [(20113, 'charged'), (20213, 'charged')],
    'a1(1640)':   [(9020113, 'neutral'), (9020213, 'charged')],
    'a2(1320)':   [(115, 'neutral'), (215, 'charged')],
    'a2(1700)':   [(9000115, 'neutral'), (9000215, 'charged')],
    'f2(1270)':   [(225, '')],
    'f0(980)':    [(9010221, '')],
    'pi(1300)':   [(100111, 'neutral'), (100211, 'charged')],
    'pi(1800)':   [(9010111, 'neutral'), (9010211, 'charged')],
    'pi1(1600)':  [(9010113, 'neutral'), (9010213, 'charged')],
    'pi2(1670)':  [(10115, 'neutral'), (10215, 'charged')],
}


def safe_name(name):
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
        # Collect mass and width entries per unique description label
        mass_map = {}
        width_map = {}

        for mcid, charge in specs:
            try:
                p = api.get_particle_by_mcid(mcid)
            except Exception:
                continue

            base = f'{pname} {charge}'.strip() if charge else pname

            for prop in p.masses():
                bs = prop.best_summary()
                if bs is None or bs.value is None:
                    continue
                err = bs.error_positive or 0
                if err <= 0:
                    continue
                desc = prop.description.replace('MASS', '').strip()
                label = f'{base} {desc}' if desc else base
                if label not in mass_map:
                    mass_map[label] = (bs.value, err)

            for prop in p.widths():
                bs = prop.best_summary()
                if bs is None or bs.value is None:
                    continue
                err = bs.error_positive or 0
                if err <= 0:
                    continue
                desc = prop.description.replace('WIDTH', '').strip()
                label = f'{base} {desc}' if desc else base
                if label not in width_map:
                    width_map[label] = (bs.value, err)

        all_labels = set(mass_map) | set(width_map)
        if not all_labels:
            print(f'  [ ] {pname}  — no data')
            continue

        rows = []
        for label in sorted(all_labels):
            m = mass_map.get(label)
            w = width_map.get(label)
            m_val = f'{m[0]:.1f}' if m else '0'
            m_err = f'{m[1]:.1f}' if m else '0'
            w_val = f'{w[0]:.1f}' if w else '0'
            w_err = f'{w[1]:.1f}' if w else '0'
            rows.append([label, m_val, m_err, w_val, w_err])

        fpath = os.path.join(outdir, f'{safe_name(pname)}.csv')
        with open(fpath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['name', 'mass', 'mass_err', 'width', 'width_err'])
            w.writerows(rows)

        n_total += len(rows)
        print(f'  [✓] {pname:>12s}  ({len(rows)} entries)')

    print(f'\nWrote {n_total} entries across {len(RESONANCES)} files to {outdir}/')


if __name__ == '__main__':
    main()
