"""Collector for the registry-driven fixed-coefficient (mask) suite.

The suite itself lives in :mod:`_flat_masked` (imported, not duplicated); this
module exists so pytest collects it.  The list of backends is DERIVED there
from ``backends_for_model("pwa_gen", tag="fixed_coeff")`` — a backend
registered later with the tag is covered without touching either file.
"""
from _flat_masked import *  # noqa: F401,F403
