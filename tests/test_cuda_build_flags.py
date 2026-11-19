"""Per-arch nvcc flags and the PTX switch (``tabpwa.cuda.build``).

The build is arch-keyed already; these gates pin the *flag* half:

* ``TABPWA_NVCC_FLAGS``      — every kernel build;
* ``TABPWA_NVCC_FLAGS_SM120`` (or ``_SM_120``) — only when that SM is
  targeted, so a V100 recipe next to a consumer one does not invalidate the
  other's binaries;
* ``TABPWA_CUDA_PTX=1``      — emit ``code=[sm_X,compute_X]`` so one binary
  survives an SM bump by JIT;
* all of it enters the build key, so a flag change REBUILDS.

No compilation happens here: the flag lists and the key are pure functions.
"""
import numpy as np
import pytest

import tabpwa.cuda.build as build


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """No stray env from the runner, and no cached flag prefix."""
    for key in list(build.os.environ):
        if key.startswith("TABPWA_NVCC_FLAGS") or key == "TABPWA_CUDA_PTX":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(build, "_flags_cache", None)
    monkeypatch.setattr(build, "_flags_key", None)
    yield


@pytest.mark.parametrize("sig, sms", [
    ("auto:12.0", ["sm_120"]),
    ("auto:8.6", ["sm_86"]),
    ("arch:sm_70,sm_86", ["sm_70", "sm_86"]),
    ("arch:compute_70,sm_86", ["sm_70", "sm_86"]),
    ("auto:unknown", []),
])
def test_target_sms_parses_every_signature(monkeypatch, sig, sms):
    monkeypatch.setattr(build, "_arch_signature", lambda: sig)
    assert build._target_sms() == sms


def test_extra_flags_generic_and_per_arch(monkeypatch):
    monkeypatch.setattr(build, "_arch_signature", lambda: "auto:12.0")
    assert build._extra_flags() == []
    monkeypatch.setenv("TABPWA_NVCC_FLAGS", "-Xptxas -O3")
    assert build._extra_flags() == ["-Xptxas", "-O3"]
    # both spellings of the per-arch name are accepted
    monkeypatch.setenv("TABPWA_NVCC_FLAGS_SM120", "-maxrregcount=48")
    got = build._extra_flags()
    assert "-Xptxas" in got and "-maxrregcount=48" in got
    monkeypatch.delenv("TABPWA_NVCC_FLAGS_SM120")
    monkeypatch.setenv("TABPWA_NVCC_FLAGS_SM_120", "-dlcm=cg")
    assert "-dlcm=cg" in build._extra_flags()


def test_extra_flags_ignore_a_non_targeted_arch(monkeypatch):
    """A V100 recipe must not leak into an sm_120 build (or the key)."""
    monkeypatch.setattr(build, "_arch_signature", lambda: "auto:12.0")
    monkeypatch.setenv("TABPWA_NVCC_FLAGS_SM70", "-maxrregcount=40")
    assert build._extra_flags() == []


def test_ptx_switch_changes_the_gencode(monkeypatch):
    monkeypatch.setattr(build, "_override_arch", "sm_70")
    nvcc = build.find_nvcc()
    sass = build._arch_flags(nvcc)
    assert sass == ["-gencode", "arch=compute_70,code=sm_70"]
    monkeypatch.setenv("TABPWA_CUDA_PTX", "1")
    both = build._arch_flags(nvcc)
    assert both == ["-gencode", "arch=compute_70,code=[sm_70,compute_70]"]
    # a value that is not truthy leaves the native build alone
    monkeypatch.setenv("TABPWA_CUDA_PTX", "0")
    assert build._arch_flags(nvcc) == sass


def test_build_key_tracks_flags_and_ptx(monkeypatch):
    monkeypatch.setattr(build, "_arch_signature", lambda: "auto:12.0")
    nvcc = build.find_nvcc()
    src = build.VARIANTS[0][0]
    base = build._build_key(src, "", nvcc)
    monkeypatch.setenv("TABPWA_NVCC_FLAGS_SM120", "-maxrregcount=48")
    flagged = build._build_key(src, "", nvcc)
    assert flagged != base, "a per-arch flag must rebuild"
    monkeypatch.delenv("TABPWA_NVCC_FLAGS_SM120")
    monkeypatch.setenv("TABPWA_CUDA_PTX", "1")
    assert build._build_key(src, "", nvcc) != base, "PTX must rebuild"
    monkeypatch.delenv("TABPWA_CUDA_PTX")
    assert build._build_key(src, "", nvcc) == base


def test_flags_reach_the_nvcc_prefix(monkeypatch):
    monkeypatch.setattr(build, "_arch_signature", lambda: "auto:12.0")
    monkeypatch.setattr(build, "_override_arch", None)
    monkeypatch.setenv("TABPWA_NVCC_FLAGS", "-Xptxas -O3")
    flags = build._compiler_flags()
    assert flags[:3] == [build.find_nvcc(), "-shared", "-Xcompiler"]
    assert flags[-2:] == ["-Xptxas", "-O3"]
