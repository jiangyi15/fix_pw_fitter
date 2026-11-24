"""The package's declarations match its imports and its entry point.

An import added without a declared dependency fails here, a dependency only an
OPTIONAL backend needs must not sit in the base install, and the installed
console entry point must report the version.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import tabpwa

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tabpwa"

#: top-level import name -> the distribution that provides it
_DIST = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sympy": "sympy",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
    "cffi": "cffi",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(body):
    return {_norm(re.split(r"[<>=!~\[; ]", n.strip())[0])
            for n in re.findall(r'"([^"]+)"', body)}


def _tables():
    """(base dependencies, {extra: dependencies}) from pyproject."""
    text = (ROOT / "pyproject.toml").read_text()
    base = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert base is not None, "pyproject has no [project] dependencies list"
    extras_text = re.search(r"^\[project\.optional-dependencies\]$(.*?)(?=^\[)",
                            text, re.S | re.M)
    assert extras_text is not None, "pyproject has no optional-dependencies"
    extras = {name: _names(body) for name, body
              in re.findall(r"^([A-Za-z0-9_-]+) = \[(.*?)\]",
                            extras_text.group(1), re.S | re.M)}
    return _names(base.group(1)), extras


def _third_party_imports():
    mods = set()
    for p in SRC.rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
    return {m for m in mods
            if m not in sys.stdlib_module_names and m != "tabpwa"}


def _env():
    path = str(ROOT / "src")
    if os.environ.get("PYTHONPATH"):
        path += os.pathsep + os.environ["PYTHONPATH"]
    return dict(os.environ, PYTHONPATH=path)


def test_every_third_party_import_is_declared():
    base, extras = _tables()
    declared = set(base)
    for deps in extras.values():
        declared |= deps
    imports = _third_party_imports()
    unmapped = imports - set(_DIST)
    assert not unmapped, (
        f"src imports {sorted(unmapped)} with no distribution mapping — add it "
        f"to _DIST here and to pyproject.toml")
    missing = {_DIST[m] for m in imports} - declared
    assert not missing, (
        f"imported but declared nowhere in pyproject.toml: {sorted(missing)}")


def test_no_extra_repeats_a_base_dependency():
    """An extra is for what the base install does NOT already bring.

    The `cuda` extra repeated `cffi` (a base dependency, needed by the CPU
    kernels too) and was removed.
    """
    base, extras = _tables()
    assert "cffi" in base, "the CPU kernels load their .so through cffi"
    for name, deps in extras.items():
        assert not (deps & base), (name, sorted(deps & base))


def test_the_package_ships_kernel_sources_not_binaries():
    """A built kernel cannot travel: the tag is ``py3-none-any``.

    The binary is keyed by the SOURCE plus the target arch and nvcc identity, so
    another machine cannot validate it, and a ``.hash`` whose ``.so`` is absent
    is never consulted.  Sources must ship (the key hashes them); prebuilt
    binaries are the AppImage's job.
    """
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^\[tool\.setuptools\.package-data\]$(.*?)(?=^\[|\Z)",
                      text, re.S | re.M)
    assert block is not None
    patterns = re.findall(r'"([^"]+)"', block.group(1))
    assert any(p.endswith(".cu") for p in patterns), patterns
    assert any(p.endswith(".c") for p in patterns), patterns
    assert not [p for p in patterns if p.endswith((".so", ".hash"))], patterns
    assert list(SRC.glob("cuda/*.cu")) and list(SRC.glob("cpu/*.c"))


def test_optional_backends_are_not_in_the_base_install():
    base, extras = _tables()
    assert "onnx" not in base and "onnxruntime" not in base
    assert {"onnx", "onnxruntime"} <= extras["onnx"]
    assert "onnxruntime-gpu" in extras["onnx-gpu"]


def test_the_package_imports_without_the_optional_backends():
    code = ("import sys; sys.modules['onnx'] = None; "
            "sys.modules['onnxruntime'] = None; "
            "import tabpwa; print(tabpwa.__version__)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == tabpwa.__version__


def test_the_onnx_backend_refuses_with_the_extra_named():
    code = ("import sys; sys.modules['onnxruntime'] = None\n"
            "from tabpwa.backends.onnx_backend import _ONNXBackendBase\n"
            "try:\n"
            "    _ONNXBackendBase()\n"
            "except RuntimeError as e:\n"
            "    print(e)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert "tabpwa[onnx]" in out.stdout


def test_the_console_entry_point_reports_the_version(capsys):
    text = (ROOT / "pyproject.toml").read_text()
    assert "[project.scripts]" in text
    assert 'tabpwa = "tabpwa.__main__:main"' in text

    from tabpwa.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"tabpwa {tabpwa.__version__}"
