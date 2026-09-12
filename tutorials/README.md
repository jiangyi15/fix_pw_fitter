# Tutorial — a self-contained toy fit

This directory holds everything needed to run a complete fit with the
repository's `./fit.sh`:

| File | Content |
|------|---------|
| `config.yml` | pure-PWA config (jpsi → pipi/pipeta/pimeta, same as the tests) |
| `generate_data.py` | generates the samples below by wrapping `scripts/gen_toy_pwa.py` |
| `data_arr.npz` | toy events (kernel arrays) — created by the generator |
| `phsp_arr.npz` | phase-space events (kernel arrays) — created by the generator |
| `init_pwa.json` | constraint-driven start point for `run_fit --init` |

## 1. Generate the toy + phase-space samples

```bash
python tutorials/generate_data.py                 # defaults: 800 data / 2000 phsp
python tutorials/generate_data.py --ndata 2000 --nph 5000
```

The generator draws a flat phase-space sample with the config masses, then
draws the toy events from an independent proposal sample weighted by the
model density at a seed `ck` — so the toy follows the model and the fit has
something to converge to.

## 2. Fit it

```bash
./fit.sh                                          # uses the tutorials/ defaults
# or explicitly:
./fit.sh tutorials/config.yml tutorials/data_arr.npz tutorials/phsp_arr.npz \
         tutorials/fit_output tutorials/init_pwa.json 200
```

`fit.sh` positional arguments are:

```
./fit.sh <config> <data.npz> <phsp.npz> <output-prefix> [init.json] [maxiter]
```

Environment: `BACKEND=<name>` overrides the default `numpy_pwa` (e.g.
`BACKEND=cuda_v4_pwa ./fit.sh`).  Results are written to
`<output-prefix>/results.json` and plots to `<output-prefix>/plots/`.

## 3. Inspect

* `tutorials/fit_output/results.json` — fitted parameter values, NLL, status;
* `tutorials/fit_output/results_error_matrix.npy` — covariance (when the
  Hessian is available);
* `tutorials/fit_output/plots/` — fit projections.

For a resolution (group-log) fit, build smeared data with the
`make_resolution_groups.py` pipeline and run the `cuda_v5_pwa` backend with
`resolution_size` equal to the copy count.
