"""Collector for the registry-driven flat ``pwa_gen`` compute suite.

The suite itself lives in :mod:`_flat_compute` (imported, not duplicated); this
module exists so pytest collects it.  The list of backends is DERIVED there
from ``backends_for_model("pwa_gen")`` — a backend registered later is covered
without touching either file.
"""
from _flat_compute import *  # noqa: F401,F403
