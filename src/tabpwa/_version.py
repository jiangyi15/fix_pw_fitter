"""The package version — the ONE source.

``pyproject.toml`` reads this attribute (``[tool.setuptools.dynamic]``), so the
number is written here and nowhere else; ``tabpwa.__version__``,
``python -m tabpwa --version`` and ``run_fit.py --version`` all report it, and
``tests/test_version.py`` fails if a surface drifts.

Released versions are recorded in ``CHANGELOG.md`` and tagged ``vX.Y.Z``; the
scheme is Semantic Versioning (0.x: a minor bump may break the API).
"""

__version__ = "0.1.0"
