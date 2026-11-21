# Building CTPWA for the §10 comparison

The PyPI wheel is `cp312` only and links ROOT, so it was built from source in
this box's environment (Python 3.13, torch 2.14, CUDA 13.4), CUDA backend only,
`sm_120` only.  The revision these numbers used is
**`bad50bd3fd38968643f49018d96a475e6f872559` (2026-11-20)**; pin it, because
their tree changes.

```bash
git clone --depth 1 https://github.com/BHXiang/ctpwa.git
cd ctpwa
git fetch --depth 1 origin bad50bd3fd38968643f49018d96a475e6f872559   # the revision
git checkout --detach bad50bd3fd38968643f49018d96a475e6f872559       # these numbers used
sed -i 's/^    desired_sm = \[70, 75, 80, 86, 89, 90, 100, 120\]/    desired_sm = [120]/' setup.py
grep -c 'desired_sm = \[120\]' setup.py     # must print 1: the anchor above is
                                            # 4 SPACES, an 8-space anchor (as an
                                            # earlier version of this file had)
                                            # matches nothing and silently keeps the
                                            # 8-arch list
sed -i 's/-std=c++17/-std=c++20/g' setup.py              # torch >= 2.14 needs C++20
sed -i 's/^import os$/import os\nimport pybind11/' setup.py   # not vendored in torch
sed -i 's#os.path.join(project_dir, "include"),#os.path.join(project_dir, "include"),\n            pybind11.get_include(),#' setup.py

export CONDA_PREFIX=<env with torch, pybind11, yaml-cpp, CUDA headers>
export CUDA_HOME=$CONDA_PREFIX ROOTSYS=<env with ROOT>
export CTPWA_BACKEND=cuda
python setup.py build_ext --inplace          # -> ctpwa/_cuda.so
PYTHONPATH=$PWD python -c "import ctpwa; print(ctpwa.backend())"   # -> cuda
```

Requirements found the hard way: `pybind11` (torch >= 2.14 no longer vendors the
headers), `yaml-cpp`, and `CUDA_HOME` (their `setup.py` joins it
unconditionally).  ROOT only needs headers + libs; a `pyroot` conda env is
enough (`ROOTSYS`, `root-config` on `PATH`).

Then:

```bash
python gen_data.py --config config.yml --n-data 1000000 --n-phsp 5000000
python bench2.py config.yml            # fixed column  (all masses/widths fixed)
python bench2.py config_free.yml       # floating column (ten resonance params)

`bench2.py`'s "forward+grad" repeats the SAME theta, so it is their value-keyed
fast path, ~16 % below a real iteration at 38 free.  Only `vary.py`'s
"vary theta" column is the per-iteration cost a fit pays; `bench2.py` and
`vary.py`'s "same theta" agree to 0.2 %.

# the free column depends on whether theta actually MOVES -- vary.py measures
# both, on the same sample (fixed theta repeats the previous parameters, which
# their code serves faster than a real fit can):
python vary.py config_free.yml         # 10 free: 23.22 -> 26.61 ms at 0.6x10^6
python vary.py config_free4.yml        #  4 free:  7.63 ->  8.35
python vary.py config_free20.yml       # 20 free: 46.07 -> 53.33
python vary.py config_freeall.yml      # 38 free: 79.74 -> 92.69
python vary.py config.yml              # fixed:    1.09 ->  1.08  (control)
```

The four configs differ only in `free: [0, 1]` (mass+width) on a subset of the
resonances: `config_free.yml` frees the five of §10 (phi1680, X1750, omega2220,
K3st1780, K4st2045 — ten real parameters), `config_free4.yml` only the first two,
`config_free20.yml` frees ten resonances (a different set, kept because it is the
20-parameter point the paper's own table invites comparison with), and
`config_freeall.yml` every non-`ONE` resonance (19 → 38 parameters).
`config_free20.yml` frees phi1680, X1750, omega2220, rho3_2250, rho5_2350,
Kst1680, Kst2380, K2st1980, K4st2045, K5st2380.

`config.yml` sets `polarization: ["1","-1"]` on the J/psi, matching this
framework's `spins: [-1, 1]`; their default is **3** polarizations, which
inflates the fixed column by ~1.5x.

Their amplitude table is `34 x n_pol x 16 B` per event — 1088 B/event at
`n_pol = 2`, i.e. **5.44 GB** at 5x10^6.  The **11.96 GB** quoted in §10/§11 is
not this product: it is their precheck's OWN output for the 2-polarization
6x10^6 run (the same table plus its upcast double copy and workspaces), which is
why the two numbers differ for the same thing.  MEASURED peak device use on this
card (`nvidia-smi` reports 12227 MiB total = 11.94 GiB, ~11.5 GiB free), sampled
at 5 Hz through the whole process so the torch caching allocator is included:

| configuration | 0.6x10^6 | 1.2x10^6 | 2.4x10^6 |
|---|---|---|---|
| all masses/widths fixed (`config.yml`) | — | 6.13 GB | 9.61 GB |
| all free, 38 parameters (`config_freeall.yml`) | 4.38 GB | 7.52 GB | **OOM** |

so the marginal cost is ~3.0 kB/event fixed and ~5.2 kB/event free, and a
5x10^6-event sample would need ~18-20 GB — it would not run here at all, free
or fixed.  The free column's wall is the 38-parameter gradient path, not the
sample: the same 2.4x10^6 sample runs all-fixed in 3.31 ms.

## CPU backend

Also builds here (no GPU needed), which is what the per-wave amplitude check
uses — it is ~10x faster than the CUDA build for small samples
(init 15.7 s vs 147 s; 15.3 s vs 147 s per `getIntensity` call at 2000 events).

```bash
# The one trap: this conda env's CPPFLAGS carries the CUDA include dirs
#     -I $PREFIX/targets/x86_64-linux/include -I .../cccl
# distutils prepends CPPFLAGS, so the real CUDA/cuComplex/CUB headers win over
# `include/cpu_shims/` and the build fails with ~2300 errors (threadIdx,
# float2 redefinition, cub internals).  Clear it and let the shims shadow:
export ROOTSYS=<env with ROOT 6.26+, e.g. a conda pyroot env>
export PATH=$ROOTSYS/bin:$PATH          # root-config must be on PATH
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15   # a compiler that injects no CUDA -I
export CPPFLAGS= CFLAGS=
export CXXFLAGS="-O3 -fPIC -I$PWD/include/cpu_shims"   # shim FIRST
export CTPWA_BACKEND=cpu MAX_JOBS=8
python setup.py build_ext --inplace                    # -> ctpwa/_cpu.so

CTPWA_DEVICE=cpu PYTHONPATH=$PWD python -c "import ctpwa; print(ctpwa.backend())"
```

The PyPI wheel ships the same `_cpu.so` prebuilt, but it is `cp312` and links
ROOT, so on this 3.13 env the source build above is the simpler route (the
wheel needs its own 3.12 env plus `libCore.so.*` on `LD_LIBRARY_PATH`).

## Re-measured 2026-11-22 (fresh build, this box)

Rebuilt from the pinned revision exactly as above, `ctpwa.backend()` -> `cuda`,
`-gencode=arch=compute_120,code=sm_120` only, zero compile errors (85 unused-
variable warnings).  The build is validated against a number recorded here
rather than assumed: `vary.py config_free.yml` at 0.6x10^6 gives **23.23 /
26.63 ms** against the 23.22 / 26.61 **above** (+0.04 % same-theta, +0.075 %
vary-theta).

`config_freeall.yml` (38 free resonance parameters, `nvec` 34, `ntheta` 38),
`vary.py`, min of 4, same sample ratio 1:5:

| total events | data / phsp | init | same theta | vary theta |
|---|---|---|---|---|
| 0.3x10^6 | 50 k / 250 k | 140.1 s | 40.58 ms | **47.86 ms** |
| 0.6x10^6 | 100 k / 500 k | 146.9 s | 79.84 ms | **92.55 ms** |
| 1.2x10^6 | 200 k / 1 M | 157.8 s | 158.15 ms | **182.23 ms** |
| 2.4x10^6 | 400 k / 2 M | (init ok) | - | **OOM** |

The 0.6x10^6 point reproduces the 79.74 -> 92.69 above to +0.13 % (same theta,
79.84 vs 79.74) and -0.15 % (vary theta, 92.55 vs 92.69); 1.2x10^6 is new.
Scaling is linear in events; `init` is nearly flat (140 -> 158 s over
0.3 -> 1.2x10^6), i.e. a phase-space precompute plus a large fixed cost.
