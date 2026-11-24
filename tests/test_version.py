"""The version has ONE source, and every surface reports it.

``src/tabpwa/_version.py`` is the source; ``pyproject.toml`` reads it; the
package and both CLIs report it; ``CHANGELOG.md`` must carry the current
version's heading.  A drift in any of those fails here.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import tabpwa

ROOT = Path(__file__).resolve().parent.parent
_PEP440 = re.compile(r"^\d+\.\d+\.\d+(?:[abc]|rc)?\d*(?:\.post\d+)?$")


def test_the_version_has_exactly_one_source():
    text = (ROOT / "pyproject.toml").read_text()
    # pyproject reads the attribute out of the package, it does not restate it
    assert 'version = {attr = "tabpwa._version.__version__"}' in text
    project = re.search(r"^\[project\]$(.*?)(?=^\[)", text, re.S | re.M)
    assert project is not None
    assert not re.search(r"^\s*version\s*=", project.group(1), re.M), \
        "pyproject [project] must not hardcode a version — it lives in _version.py"


def test_the_version_is_pep440_shaped():
    assert _PEP440.match(tabpwa.__version__), tabpwa.__version__


def test_the_changelog_records_the_current_version():
    text = (ROOT / "CHANGELOG.md").read_text()
    assert "## [Unreleased]" in text
    assert f"## [{tabpwa.__version__}]" in text, \
        "CHANGELOG.md needs a dated heading for the current version"


def test_both_clis_report_the_version():
    expected = f"tabpwa {tabpwa.__version__}"
    # work from a BARE checkout too: the subprocesses must find src/tabpwa even
    # when the package is not installed into this interpreter
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") +
               (os.pathsep + os.environ["PYTHONPATH"]
                if os.environ.get("PYTHONPATH") else ""))
    for argv in ([sys.executable, "-m", "tabpwa", "--version"],
                 [sys.executable, "run_fit.py", "--version"]):
        out = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                             timeout=300, env=env)
        assert out.returncode == 0, (argv, out.stderr[-500:])
        assert out.stdout.strip() == expected, (argv, out.stdout, out.stderr)
