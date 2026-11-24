"""Framed messages over a binary pipe — the `mpi_service` driver ↔ worker channel.

One message is an 8-byte little-endian length followed by a `.npz` payload: a
JSON ``__meta__`` member plus one array per key.  Structured arguments (the
kernel config) travel as a JSON SKELETON with the arrays kept out of it; tuples
are tagged so they come back as tuples.

This is the same envelope the remote-protocol design specifies
(``docs/remote_protocol_design.md`` §5), reduced to what a local pipe needs —
the driver and the worker share this module, and no pickle is involved on
either side.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np

_LEN = struct.Struct("<Q")
_META = "__meta__"
#: an upper bound on one frame: a sample is GiB, but not 16 GiB, and a
#: desynchronised channel must fail instead of allocating from a garbage length
_MAX_FRAME = 1 << 34


# ── structured arguments ────────────────────────────────────────────────
def split_tree(obj, prefix="a"):
    """``(skeleton, arrays)`` — arrays replaced by ``{prefix}__N`` references.

    Dicts, lists, tuples, scalars and ``None`` round-trip through
    :func:`join_tree`; tuples are tagged (JSON would flatten them).
    """
    arrays = {}

    def walk(o):
        if isinstance(o, np.ndarray):
            key = f"{prefix}__{len(arrays)}"
            arrays[key] = np.ascontiguousarray(o)
            return {"__array__": key}
        if isinstance(o, dict):
            return {str(k): walk(v) for k, v in o.items()}
        if isinstance(o, tuple):
            return {"__tuple__": [walk(v) for v in o]}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise TypeError(f"ipc: {type(o).__name__} cannot be framed")

    return walk(obj), arrays


def join_tree(skeleton, arrays):
    """Inverse of :func:`split_tree`."""
    if isinstance(skeleton, dict):
        if "__array__" in skeleton:
            return arrays[skeleton["__array__"]]
        if "__tuple__" in skeleton:
            return tuple(join_tree(v, arrays) for v in skeleton["__tuple__"])
        return {k: join_tree(v, arrays) for k, v in skeleton.items()}
    if isinstance(skeleton, list):
        return [join_tree(v, arrays) for v in skeleton]
    return skeleton


# ── framing ─────────────────────────────────────────────────────────────
def pack(meta, arrays=None):
    """Serialise one message (``meta`` must be JSON-ready)."""
    bio = io.BytesIO()
    payload = {}
    for key, value in (arrays or {}).items():
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise TypeError(f"ipc: array {key!r} has object dtype")
        payload[str(key)] = arr
    np.savez_compressed(bio, **{_META: np.array(json.dumps(meta, allow_nan=False))}, **payload)
    body = bio.getvalue()
    return _LEN.pack(len(body)) + body


def unpack(blob):
    """``(meta, arrays)`` from one message body."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        if _META not in z.files:
            raise ValueError("ipc: message has no __meta__")
        meta = json.loads(str(z[_META]))
        arrays = {k: z[k] for k in z.files if k != _META}
    return meta, arrays


def read_msg(stream):
    """One message from a binary stream, or ``None`` on a clean EOF."""
    head = stream.read(_LEN.size)
    if head == b"":
        return None
    if len(head) < _LEN.size:
        raise EOFError("ipc: truncated length prefix")
    (n,) = _LEN.unpack(head)
    if n > _MAX_FRAME:
        # a desynchronised channel (a partial body read after a timeout) shows
        # up here as a garbage length; without a cap it asks for 2^64 bytes
        raise EOFError(f"ipc: frame length {n} exceeds the {_MAX_FRAME} cap "
                       f"(the channel is desynchronised)")
    chunks = []
    got = 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            raise EOFError("ipc: truncated message")
        chunks.append(chunk)
        got += len(chunk)
    return unpack(b"".join(chunks))


def write_msg(stream, meta, arrays=None):
    """Write one message and flush (the peer is waiting for it)."""
    stream.write(pack(meta, arrays))
    stream.flush()