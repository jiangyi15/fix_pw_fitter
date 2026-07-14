#!/usr/bin/env python3
"""Fetch PDG mass/width reference values for light meson resonances.

Generates CSV files (one per resonance) with best-fit mass and width
values from the PDG API, including multiple measurement methods where
available (e+e-, photoproduced, hadroproduced, individual decay modes).

Output: ``{output_dir}/{resonance_name}.csv``

Requires: ``pip install pdg``
"""
import os, sys, csv, argparse
from pdg import connect


# resonances: name → [(MCID, charge_label), ...]
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
    """Sanitize a resonance name for use as a filename."""
    return name.replace('(', '_').replace(')', '').replace(' ', '_')


def main():
    ap = argparse.ArgumentParser(description='Fetch PDG mass/width reference CSVs')
    ap.add_argument('output_dir', help='Output directory for CSV files')
    ap.add_argument('--edition', default=None,
                    help='PDG edition (e.g. 2024, 2026). Default: latest installed.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = connect()

    n_total = 0
    for pname, specs in sorted(RESONANCES.items()):
        rows = []
        seen = set()

        for mcid, charge in specs:
            try:
                p = api.get_particle_by_mcid(mcid)
            except Exception as e:
                print(f'  [{pname}] MCID={mcid}: {e}', file=sys.stderr)
                continue

            base = f'{pname} {charge}'.strip() if charge else pname

            # --- Mass measurements ---
            for prop in p.masses():
                bs = prop.best_summary()
                if bs is None or bs.value is None:
                    continue
                err = bs.error_positive or 0
                if err <= 0:
                    continue
                desc = prop.description.replace('MASS', '').strip()
                label = f'{base} {desc}' if desc else base
                if label in seen:
                    continue
                seen.add(label)
                rows.append([label, f'{bs.value:.1f}', f'{err:.1f}', '', ''])

            # --- Width measurements ---
            for prop in p.widths():
                bs = prop.best_summary()
                if bs is None or bs.value is None:
                    continue
                err = bs.error_positive or 0
                if err <= 0:
                    continue
                desc = prop.description.replace('WIDTH', '').strip()
                label = f'{base} {desc}' if desc else base
                if label in seen:
                    continue
                seen.add(label)
                rows.append([label, '', '', f'{bs.value:.1f}', f'{err:.1f}'])

        if not rows:
            print(f'  [ ] {pname}  — no data')
            continue

        fpath = os.path.join(args.output_dir, f'{safe_name(pname)}.csv')
        with open(fpath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['name', 'mass', 'mass_err', 'width', 'width_err'])
            w.writerows(rows)
        n_total += len(rows)
        print(f'  [✓] {pname:>12s}  {fpath}  ({len(rows)} entries)')

    print(f'\nWrote {n_total} entries across {len(RESONANCES)} files to {args.output_dir}/')


if __name__ == '__main__':
    main()
