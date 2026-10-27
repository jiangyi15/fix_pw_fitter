# Path remapping — one config, local and remote

Configs often carry absolute paths from the machine they were written on
(batch farm, cluster share).  Instead of editing the YAML for every site,
bind a source prefix to a local target; **every path resolved by tabpwa is
remapped before loading**.  The config is never modified — there is no
`path_map:` key, and expanded config dicts are not rewritten.

## Quick use

```python
from tabpwa import paths

paths.bind_paths("/farm/run42:/data/run42;/farm/tables:/data/tables")
fitter = Fitter("config.yml")          # loads the local files

with paths.path_scope({"/farm/run42": "/scratch/run42"}):
    fitter = Fitter("config.yml")      # temporary override
paths.clear_paths()
```

Shell equivalents:

```bash
TABPWA_PATH_MAP="/farm/run42:/data/run42;/farm/tables:/data/tables" \
    python run_fit.py --config config.yml

python run_fit.py --config config.yml \
    --path-map /farm/run42:/data/run42 \
    --path-map /farm/tables:/data/tables
# or a file of pairs / JSON:
python run_fit.py --config config.yml --path-map-file maps.txt
```

## Priority

Highest first:

1. `resolve_path(..., path_map=...)` explicit argument (library/internal),
2. the innermost `paths.path_scope(...)`,
3. the process-wide `paths.bind_paths(...)` registry,
4. the `TABPWA_PATH_MAP` environment variable.

## Format

- pair: `FROM:TO`, split on the **first** `:`; multiple pairs separated by
  `;` or newlines, or passed as a list / dict / JSON object;
- JSON (`{"FROM": "TO"}`) is the escape hatch when a path contains `:`
  (e.g. Windows drive letters);
- matching is **longest prefix**, on path-component boundaries
  (`/data` does not match `/database`); an exact match is allowed; the
  unmatched remainder is appended to the target;
- duplicate exact keys: the later map wins;
- unmatched paths pass through unchanged.

## What is remapped

All config-driven loads go through the single
`tabpwa.paths.resolve_path` helper:

- `Fitter.load_dataset` / `load_momenta_conf` — `{prefix}_arr`,
  `{prefix}`, `{prefix}_weight`, `{prefix}_bg_value` (and any other
  `data:` prefix);
- `ConfigBroadcast` — variant `data:` paths (relative entries are still
  anchored on the template config's directory first);
- particle-model files in the `particle:` table — `file`, `knots`,
  `*_file`, `*_arr`, `*_path`, or any value ending in `.npy`/`.npz`;
- `run_fit.py` binds the CLI/env map once at startup, so later loads
  (including particle tables) see it.

Relative paths keep their existing anchor: config files anchor on the
config's directory; dict configs resolve relative to the CWD (a dict has no
anchor).

## Example: farm config used locally

```yaml
# config.yml — unchanged between sites
data:
  dat_order: [pip, pim, eta]
  data_arr: /farm/run42/data.npz
  phsp_arr: /farm/run42/phsp.npz
particle:
  rho:
    model: width_linear_npy
    file: /farm/tables/rho_gamma.npy
```

```bash
# local
TABPWA_PATH_MAP="/farm/run42:$PWD/inputs;/farm/tables:$PWD/tables" \
    python run_fit.py --config config.yml --fit

# remote (paths already valid)
python run_fit.py --config config.yml --fit
```
