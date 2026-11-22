"""Collector for the ONE case-1 isolation suite.

The table and the test live in :mod:`_flat_isolation` (they are data plus a
property, not a family's own claim), and it is parametrised over every kernel
that HAS a copy relationship.  It exists as a module so pytest collects it; the
completeness assert in :mod:`_flat_isolation` is what keeps every other flat
leaf explicitly accounted for.
"""
from _flat_isolation import *  # noqa: F401,F403
