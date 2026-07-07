#!/usr/bin/env python3
"""Convert old CK gamma names ``re_01`` → ``re_0_1``, ``im_12`` → ``im_1_2``, etc.

Old format (ambiguous with multi-digit indices)::

    {name}_re_01, {name}_re_00, {name}_im_12

New format (unambiguous)::

    {name}_re_0_1, {name}_re_0_0, {name}_im_1_2

Usage::

    python scripts/convert_ck_names.py results.json [results2.json ...]
    python scripts/convert_ck_names.py results.json -o converted.json
"""

import sys, re, json

# Match e.g. ``_re_01`` or ``_im_123`` — digits after _re_ or _im_ with no underscore
PAT = re.compile(r"(_re_|_im_)(\d+)$")


def _rename(key):
    m = PAT.search(key)
    if not m:
        return key
    prefix, digits = m.group(1), m.group(2)
    # Only convert if all digits are single-character indices
    # (old format was always single-digit per index, so split every char)
    if len(digits) < 2:
        return key
    new_suffix = "_".join(digits)        # "01" → "0_1", "121" → "1_2_1"
    return key[:m.start(2)] + new_suffix  # keep _re_/_im_ prefix


def convert_json(data):
    """Rename all CK-style keys in *data* (dict or nested)."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            k2 = _rename(k)
            out[k2] = convert_json(v)
        return out
    elif isinstance(data, list):
        return [convert_json(item) for item in data]
    return data


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Convert old CK gamma names to new format")
    ap.add_argument("input", nargs="+", help="JSON file(s) to convert")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path (requires single input)")
    args = ap.parse_args()

    if args.output and len(args.input) > 1:
        print("error: -o requires a single input file", file=sys.stderr)
        sys.exit(1)

    for path in args.input:
        with open(path) as f:
            data = json.load(f)

        converted = convert_json(data)

        if args.output:
            out_path = args.output
        else:
            base, ext = path.rsplit(".", 1) if "." in path else (path, "json")
            out_path = f"{base}_converted.{ext}"

        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2)

        print(f"  {path} → {out_path}")


if __name__ == "__main__":
    main()
