"""Pure split tests for ShardBackend.split_offsets (no workers spawned)."""

import numpy as np
import pytest

from ampfit.backends.shard_backend import split_offsets


def test_equal_weights_no_align():
    b = split_offsets(300, [1.0, 1.0])
    assert b == [0, 150, 300]
    assert len(b) == 3            # n_workers + 1


def test_weighted_no_align():
    b = split_offsets(100, [1.0, 3.0])
    assert b[0] == 0 and b[-1] == 100
    assert b[1] == 25              # 1/4 of 100


def test_align_boundaries_are_multiples():
    ne, nw, a = 300, 2, 20
    b = split_offsets(ne, [1.0] * nw, align=a)
    assert b[0] == 0 and b[-1] == ne
    # every interior boundary is a multiple of align
    for v in b[1:-1]:
        assert v % a == 0
    # chunk lengths are multiples of align (except the tail partial)
    for i in range(nw - 1):
        assert (b[i + 1] - b[i]) % a == 0


def test_align_last_tail_partial():
    ne, nw, a = 300, 2, 13
    b = split_offsets(ne, [1.0] * nw, align=a)
    assert b[0] == 0 and b[-1] == ne
    for v in b[1:-1]:
        assert v % a == 0
    assert (b[1] - b[0]) % a == 0
    # total of full aligned chunks + tail == ne
    assert (b[1] - b[0]) + (ne - b[1]) == ne


def test_align_uneven_weights():
    b = split_offsets(1000, [1.0, 3.0], align=50)
    assert b[0] == 0 and b[-1] == 1000
    assert b[1] % 50 == 0
    assert 200 <= b[1] <= 300       # near 1/4 of 1000, on the 50-grid


def test_align_too_few_rows_raises():
    with pytest.raises(ValueError):
        split_offsets(30, [1.0, 1.0, 1.0], align=20)   # 1 block for 3 workers


def test_matches_old_default():
    """align=None reproduces the pre-align int((frac*ne)) boundaries."""
    rs = np.random.RandomState(0)
    for _ in range(20):
        ne = int(rs.randint(10, 500))
        w = rs.rand(int(rs.randint(1, 5))) + 0.1
        b = split_offsets(ne, w)
        frac = np.cumsum(w / w.sum())[:-1]
        old = np.maximum.accumulate((frac * ne).astype(np.intp)).clip(0, ne)
        assert b == [0] + [int(x) for x in old] + [ne]
        assert len(b) == len(w) + 1
