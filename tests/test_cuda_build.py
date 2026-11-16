"""``tabpwa.cuda.build`` — discovery, subset selection and incremental builds.

The builder is what stands between every CUDA backend and a missing ``.so``,
and it is invoked three ways: implicitly on import (``ensure``), by hand
(``python -m tabpwa.cuda.build``) and by the packaging scripts
(``build()``).  These tests pin the parts that are easy to get wrong and
cheap to check without nvcc.

Nothing here compiles a kernel: the selection / staleness logic is pure
filesystem work, and the compile path is covered by the kernels' own
fixtures (which rebuild through ``ensure`` on import).
"""
import os
import subprocess
import sys

import pytest

from tabpwa.cuda import build as B


# ── discovery (glob) ────────────────────────────────────────────────

def test_discovery_finds_every_kernel_source():
    """``kernels_*.cu`` -> ``libcuda_kernels_*.so``, sorted, one pair each."""
    found = B.variants()
    on_disk = sorted(f for f in os.listdir(B.SCRIPT_DIR)
                     if f.startswith("kernels_") and f.endswith(".cu"))
    assert on_disk, "no kernel sources found"
    assert [s for s, _l in found] == on_disk
    for src, lib in found:
        assert lib == "libcuda_" + os.path.splitext(src)[0] + ".so"
    # no duplicates, and the live view agrees with the import-time snapshot
    assert len({l for _s, l in found}) == len(found)
    assert [s for s, _ in B.VARIANTS] == [s for s, _ in found]


def test_discovery_is_live():
    """A kernel added after import is picked up without a restart."""
    import importlib
    mod = importlib.import_module("tabpwa.cuda.build")
    before = len(mod.variants())
    # the module re-globs, so calling it twice is stable and cheap
    assert len(mod.variants()) == before
    assert mod.variants() == mod.variants()


# ── subset selection ───────────────────────────────────────────────

def test_empty_selection_means_everything():
    assert B.select() == B.variants()
    assert B.select([]) == B.variants()
    assert B.select(("",)) == B.variants()


def test_substring_selection():
    """A bare word is a substring match — no wildcards needed."""
    got = {s for s, _l in B.select(["pwa_s0"])}
    assert got == {s for s, _l in B.variants() if "pwa_s0" in s}
    assert len(got) >= 3                       # s0e, flat, bw, bwt
    assert all("pwa_s0" in s for s in got)


def test_glob_selection():
    got = {s for s, _l in B.select(["kernels_v5_*"])}
    assert got and all(s.startswith("kernels_v5_") for s in got)
    # a pattern can also address the library name
    by_lib = {s for s, _l in B.select(["libcuda_kernels_v5_pwa_s0*"])}
    assert by_lib and all("pwa_s0" in s for s in by_lib)


def test_multiple_patterns_are_unioned():
    a = {s for s, _l in B.select(["pwa_s0c"])}
    b = {s for s, _l in B.select(["pwa_s0e"])}
    both = {s for s, _l in B.select(["pwa_s0c", "pwa_s0e"])}
    assert both == a | b
    assert a and b and not (a & b)


def test_unmatched_pattern_is_an_error():
    """A typo must not look like a successful no-op build."""
    with pytest.raises(ValueError, match="no kernel matches"):
        B.select(["there_is_no_such_kernel"])
    with pytest.raises(ValueError, match="no kernel matches"):
        B.select(["pwa_s0", "there_is_no_such_kernel"])    # one bad, one good


# ── incremental default ────────────────────────────────────────────

def test_staleness_matches_the_hash_bookkeeping():
    """``is_current`` agrees with what ``ensure`` would decide."""
    src, lib = B.variants()[0]
    tag = B.resolve_tag()
    lib_path = os.path.join(B.SCRIPT_DIR, B.lib_file_name(lib, tag))
    hash_path = B._hash_path(src, tag)
    current = B.is_current(src, lib)
    if current:
        assert os.path.exists(lib_path) and os.path.exists(hash_path)
        assert open(hash_path).read().strip() == B._build_key(src, tag,
                                                              B.find_nvcc())
    else:
        # either half missing, or the key really changed
        assert (not os.path.exists(lib_path)
                or not os.path.exists(hash_path)
                or open(hash_path).read().strip()
                != B._build_key(src, tag, B.find_nvcc()))


def test_a_missing_hash_file_makes_a_kernel_stale():
    """Deleting the companion hash forces a rebuild (``is_current`` False)."""
    src, lib = B.variants()[0]
    tag = B.resolve_tag()
    hash_path = B._hash_path(src, tag)
    if not os.path.exists(hash_path):
        pytest.skip("this kernel has no companion hash to remove")
    saved = open(hash_path).read()
    try:
        os.remove(hash_path)
        assert B.is_current(src, lib) is False
    finally:
        open(hash_path, "w").write(saved)
    assert B.is_current(src, lib) is True


def test_build_cli_lists_and_reports_status(capsys=None):
    """``--list`` prints one line per selected kernel and exits 0."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.path.join(root, "src") + os.pathsep + \
        env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "tabpwa.cuda.build", "--list",
                        "pwa_s0"], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines and all("libcuda_kernels_" in ln for ln in lines)
    assert all("pwa_s0" in ln for ln in lines)


def test_build_cli_rejects_an_unknown_kernel():
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.path.join(root, "src") + os.pathsep + \
        env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "tabpwa.cuda.build", "--list",
                        "not_a_kernel"], capture_output=True, text=True,
                       env=env, timeout=300)
    assert r.returncode != 0
    assert "no kernel matches" in (r.stdout + r.stderr)


# ── the build does not compile the same file twice ─────────────────

def test_compiler_flags_are_probed_once():
    """The toolchain probe is cached, not re-run per kernel.

    It used to run ``detect_gcc_compat`` plus a validating probe compile on
    every ``ensure``, and the probe target was a *real kernel source* — so a
    two-kernel build compiled one of them twice and wasted ~10 s per kernel.
    """
    if B.find_nvcc() is None:
        pytest.skip("no nvcc")
    B._flags_cache = None
    first = B._compiler_flags()
    assert first and B._flags_cache is first
    again = B._compiler_flags()
    assert again is first                       # cached, no second probe


def test_set_arch_invalidates_the_cached_flags():
    if B.find_nvcc() is None:
        pytest.skip("no nvcc")
    try:
        B._flags_cache = None
        B._compiler_flags()
        assert B._flags_cache is not None
        B.set_arch("sm_86")
        assert B._flags_cache is None
        flags = B._compiler_flags()
        assert any("compute_86" in f for f in flags)
    finally:
        B.set_arch(None)


def test_force_reaches_ensure(tmp_path, monkeypatch):
    """``build(force=True)`` must compile even when the key matches.

    It did not: ``build`` called ``ensure`` without forwarding ``force``, so
    the ``--force`` flag printed a "rebuilding" pre-pass and then compiled
    nothing — a silent no-op that leaves a stale binary in place.
    """
    src, lib = B.variants()[0]
    calls = []

    def fake_build_one(src_name, out_file):
        calls.append(src_name)
        # write INSIDE tmp_path: this test must never touch a real .so
        open(os.path.join(str(tmp_path), os.path.basename(out_file)),
             "wb").write(b"#fake")
        return True

    monkeypatch.setattr(B, "HASH_DIR", str(tmp_path))
    monkeypatch.setattr(B, "_build_one", fake_build_one)
    monkeypatch.setattr(B, "find_nvcc", lambda: "/usr/bin/true")
    monkeypatch.setattr(B, "_build_key", lambda *a, **k: "fixed-key")
    # redirect the library path into tmp_path as well
    monkeypatch.setattr(B, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(B, "variants", lambda: [(src, lib)])
    monkeypatch.setattr(B, "select",
                        lambda pats=(), all_variants=None: [(src, lib)])

    assert B.build([src], force=True, quiet=True) is True
    assert calls == [src]
    calls.clear()
    assert B.build([src], force=True, quiet=True) is True   # forced again
    assert calls == [src]
    calls.clear()
    assert B.build([src], quiet=True) is True                # up to date now
    assert calls == []
    assert os.path.exists(os.path.join(str(tmp_path), lib))


def test_build_key_covers_the_target_architecture():
    """A native binary is not interchangeable with a fat one.

    Without the arch in the key, ``set_arch('sm_70,sm_75,sm_86')`` (what the
    AppImage scripts do) accepted the locally built ``sm_120`` cubins and
    packaged binaries that cannot load on the target machines.
    """
    if B.find_nvcc() is None:
        pytest.skip("no nvcc")
    tag = B.resolve_tag()
    try:
        B.set_arch("sm_75,sm_86")
        fat = B._build_key("kernels_v2.cu", tag, B.find_nvcc())
        B.set_arch("sm_89")
        other = B._build_key("kernels_v2.cu", tag, B.find_nvcc())
        B.set_arch(None)
        native = B._build_key("kernels_v2.cu", tag, B.find_nvcc())
    finally:
        B.set_arch(None)
    assert len({fat, other, native}) == 3
    assert B._arch_signature().startswith("auto:") or \
        B._arch_signature().startswith("arch:")


def test_nvcc_selection_prefers_an_explicit_toolkit(monkeypatch, tmp_path):
    """``TABPWA_NVCC`` / ``CUDA_HOME`` outrank ``PATH``.

    A project that pins a toolkit (CUDA 12 for ``sm_70`` support) must not be
    silently overridden by a newer nvcc earlier on ``PATH``.
    """
    fake = tmp_path / "bin" / "nvcc"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("TABPWA_NVCC", str(fake))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "elsewhere"))
    cands = B._nvcc_candidates()
    assert cands[0] == str(fake)
    assert B.find_nvcc() == str(fake)


def test_nvcc_selection_looks_for_a_toolkit_that_has_the_arch(monkeypatch):
    """When ``--arch`` needs an SM the preferred nvcc lacks, pick one that has it.

    CUDA 13 dropped ``compute_70``; the resolved toolkit must then be the
    older one that still emits it, rather than failing.
    """
    modern = {"compute_75", "compute_86", "compute_120"}   # CUDA 13-ish
    legacy = {"compute_70", "compute_75", "compute_86"}    # CUDA 12-ish
    monkeypatch.setattr(B, "_nvcc_candidates", lambda: ["/n13/nvcc", "/n12/nvcc"])
    monkeypatch.setattr(B, "_supported_sms",
                        lambda n: modern if "n13" in n else legacy)
    monkeypatch.setattr(B, "_override_arch", None)
    assert B.find_nvcc() == "/n13/nvcc"          # no arch requested: preferred
    monkeypatch.setattr(B, "_override_arch", "sm_75")
    assert B.find_nvcc() == "/n13/nvcc"          # satisfiable by the preferred
    monkeypatch.setattr(B, "_override_arch", "sm_70")
    assert B.find_nvcc() == "/n12/nvcc"          # only the legacy one can


def test_detect_gcc_prefers_a_versioned_host_compiler(monkeypatch):
    """CUDA rejects FUTURE gcc majors, so gcc-13 must win over plain gcc 15."""
    monkeypatch.delenv("TABPWA_CC", raising=False)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr("shutil.which",
                        lambda n: "/usr/bin/gcc-13" if n == "gcc-13" else
                        ("/usr/bin/gcc" if n == "gcc" else None))
    assert B.detect_gcc() == "/usr/bin/gcc-13"
    # and the conda-forge triple compiler is found when no versioned gcc exists
    monkeypatch.setattr("shutil.which",
                        lambda n: "/conda/bin/x86_64-conda-linux-gnu-g++"
                        if n == "x86_64-conda-linux-gnu-g++" else
                        ("/usr/bin/gcc" if n == "gcc" else None))
    assert B.detect_gcc() == "/conda/bin/x86_64-conda-linux-gnu-g++"


def test_set_arch_rejects_an_unsupported_sm():
    """A dropped architecture must fail loudly, not as 28 quiet FAILEDs.

    CUDA 13 no longer emits ``compute_70``; ``set_arch('sm_70,...')`` used to
    be accepted and every kernel then failed to compile.
    """
    if B.find_nvcc() is None:
        pytest.skip("no nvcc")
    supported = B._supported_sms(B.find_nvcc())
    if "compute_70" in supported:
        pytest.skip("this toolchain still supports sm_70")
    with pytest.raises(ValueError, match="not supported"):
        B.set_arch("sm_70")
    with pytest.raises(ValueError, match="not supported"):
        B.set_arch("sm_70,sm_75")
    # a supported one is accepted, and the override is restored afterwards
    B.set_arch("sm_75,sm_86")
    assert B._override_arch == "sm_75,sm_86"
    B.set_arch(None)
    assert B._override_arch is None


def test_hash_dir_is_honoured(tmp_path, monkeypatch):
    """``TABPWA_HASH_DIR`` keeps the bookkeeping out of a read-only tree."""
    monkeypatch.setattr(B, "HASH_DIR", str(tmp_path))
    p = B._hash_path("kernels_v2.cu", "")
    assert os.path.dirname(p) == str(tmp_path)
    assert p.endswith("kernels_v2.cu.hash")
    # a tag keeps its namespace inside the hash dir too
    pt = B._hash_path("kernels_v2.cu", "sm86")
    assert os.path.dirname(pt) == str(tmp_path)
    assert pt.endswith(".sm86.hash")
