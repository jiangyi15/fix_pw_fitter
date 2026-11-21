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
sed -i 's/^        desired_sm = \[70, 75, 80, 86, 89, 90, 100, 120\]/        desired_sm = [120]/' setup.py
sed -i 's/-std=c++17/-std=c++20/g' setup.py              # torch >= 2.14 needs C++20
sed -i 's/^import os$/import os\nimport pybind11/' setup.py   # not vendored in torch
sed -i 's#os.path.join(project_dir, "include"),#os.path.join(project_dir, "include"),\n            pybind11.get_include(),#' setup.py

export CONDA_PREFIX=<env with torch, pybind11, yaml-cpp, CUDA headers>
export CUDA_HOME=$CONDA_PREFIX ROSYS=<env with ROOT>
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

Their amplitude table is `34 x n_pol x 16 B` per event, so the precheck scales
with `n_pol` and with the event count; at 5x10^6 events with 2 polarizations it
is 11.96 GB against the 12.14 GB this card reports.

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
