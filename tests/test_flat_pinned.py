"""Collector for the shared PINNED-coefficient flat ``pwa_gen`` suite.

The suite itself lives in :mod:`_flat_pinned` (imported, not duplicated); this
module exists so pytest collects it.  The backend list is DERIVED there from
``backends_for_model("pwa_gen", tag="coeff_pinned")`` and parametrised over a
per-backend table, so a pinned kernel registered later is covered without
touching either file — and one registered without a table entry fails loudly.
"""
from _flat_pinned import *  # noqa: F401,F403
