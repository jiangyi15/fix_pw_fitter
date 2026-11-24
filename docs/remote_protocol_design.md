# Remote compute protocol — DESIGN (draft standard, v1)

**Kind: DESIGN (draft standard).**  There is no implementation in this tree.
Nothing here is frozen and no release ships it.  A prototype
was built during the design review and is kept ASIDE (stashed) — it is NOT in
this tree — and §16 says what it measured; the text below is the contract, and
it is written so that a server can be implemented WITHOUT tabpwa.

Motivation: a fit touches a large sample once (load) and then exchanges only
parameters and `(Q, grads)` per iteration.  A remote backend therefore needs a
transport whose cost is *one upload per sample* and a few KB per iteration, and
the compute side must be implementable on a GPU host by a small program that
does not carry the whole analysis framework.

Normative keywords: **MUST**, **MUST NOT**, **SHOULD**, **MAY**.

---

## 1. Scope

Covered:

* uploading an event sample once, addressed by its content;
* per-iteration compute of a likelihood value and its parameter gradients;
* fan-out of one sample across several servers;
* a server-side cache that survives a restart.

Not covered (v1): TLS, authentication beyond one shared bearer token, service
discovery, streaming/chunked bodies, multi-tenant quotas, compression
negotiation, distributed consensus.

**This is the REMOTE, heterogeneous deployment.  For N GPUs in one host (or a
homogeneous cluster, launched with `mpiexec`, no server program) see the
companion draft `docs/mpi_backend.md` — the two are alternatives, not layers.**

## 2. Terminology

| term | meaning |
|---|---|
| **client** | the caller; owns the parameters and the optimisation loop |
| **server** | holds a sample and computes on it |
| **sample** | the event buffer: named numeric arrays (`mass`, `q`, `angle`, `weight`, `bkg`, …), plus the kernel configuration that says what the columns mean |
| **kernel configuration** | a nested structure (arrays + scalars) produced by the model builder; the server needs it to interpret the sample |
| **handle** | server-side reference to one sample, identified by its **content key** |
| **params** | the parameter dict for one evaluation (`ck`, `m0`, `g0`, `scalar`, `coeff`, …) |
| **result** | `Q` (a scalar), `grads` (same block names as params), optional per-event `P` |

## 3. Versioning

* Every request's meta carries `"protocol": <int>`.
* The current value is **1**.  It MUST be bumped for ANY change to the framing,
  the canonical encoding, the content-key algorithm, or the meaning of a field.
* A server MUST reject a request whose `protocol` is not its own (400).  It is a
  per-request check, not only a `/health` check: a proxy may strip health.
* Additive meta fields that an old server may ignore do NOT need a bump (see
  §13); a field whose absence changes the meaning of a request does.

## 4. Transport

* HTTP/1.1, keep-alive.  One request = one message; no pipelining (a client MAY
  keep one connection per server per thread).
* Request and response bodies are binary envelopes (§5).  `Content-Length` MUST
  be set; `Transfer-Encoding: chunked` is not used in v1.
* The server SHOULD cap the body (`max_body`, advertised in `/health`); over the
  cap it MUST answer 413 without reading the body.
* The server SHOULD apply a socket timeout (order of minutes) so a stalled
  upload cannot pin a worker thread.
* Only `http://` in v1.  Clients MUST NOT silently fall back to another
  transport or to local compute.

Endpoints:

| method | path | body | 200 answer |
|---|---|---|---|
| GET | `/health` | — | JSON (§7.1) |
| POST | `/load` | envelope | envelope (§7.2) |
| POST | `/compute` | envelope | envelope (§7.3) |
| POST | `/free` | envelope | envelope (§7.4) |

`200` means success and the answer is the endpoint's own type.  Every non-200
answer is `application/json` `{"error": "<human message>"}` (see §11).

## 5. Message envelope (v1 encoding: one npz)

One message is a **zip archive in NumPy `.npz` form**:

* exactly one member named `__meta__` is a 0-d array of a JSON string (UTF-8);
* every other member is a numeric array (`.npy`), referenced from the JSON.

Rationale: `np.savez`/`np.load` is already present wherever the client runs, the
archive is binary-safe and self-describing, and a non-Python server can read a
zip of `.npy` files with standard libraries.  The envelope is the *only* thing
tied to NumPy; §13 lists a frames-based alternative for v2.

### 5.1 Skeleton references

Structured arguments (params, the sample, the kernel configuration) travel as a
JSON **skeleton** with the arrays kept out of it:

```json
{"__array__": "arr3"}                       // the member "arr3"
{"__tuple__": [ ... ]}                      // a tuple (JSON has no tuples)
{"a": 1.0, "b": null, "c": [true, "x"]}     // plain JSON
```

Rules:

* an array's member name MUST be unique inside the message; the skeleton MUST
  NOT reference a missing member;
* tuples MUST be tagged (otherwise `(1,2)` silently becomes `[1,2]`);
* JSON numbers are IEEE doubles; finite-only in meta.  NaN/±inf belong in
  arrays (a parameter that is non-finite is a `500` from the compute, not a
  framing error);
* dictionaries are unordered; their keys MUST be strings.

## 6. Content key

The key identifies a *handle*.  It MUST be computed by the client before an
upload (to ask "do you have it?") and RE-computed by the server from the upload
(a server MUST refuse a mismatch, 400: otherwise a buggy client can poison
another client's entry).

```
key = hex( sha256( "tabpwa-remote|1|" |
                   kernel_digest | "|" | model | "|" | backend_id | "|" |
                   sample_digest ) )
```

with the **canonical encoding** of §6.1 applied to `kernel_config` and to the
sample dict.  `model` is the amplitude-model name (or `null`), `backend_id` the
server-side backend identifier the client is addressing (the two together decide
the code that consumes the arrays).  `kernel_digest` is the canonical digest of
the kernel configuration.

### 6.1 Canonical encoding (normative)

Feed a SHA-256 object with, recursively:

| value | bytes |
|---|---|
| ndarray | `"A" + path + "\|"`, `str(dtype) + "\|"`, `repr(shape) + "\|"`, then the C-contiguous little-endian bytes |
| dict | `"D\|"`, then entries in ascending `str(key)` order, each with its own path |
| list / tuple | `"L\|"`, then elements in order with path `prefix[i]` |
| float | `"F" + path + "\|" + repr(value)` (shortest round-trip form) |
| bool / int | `"B"…` / `"I"…` + `repr` |
| null | `"N" + path` |
| anything else | `"S" + path + "\|" + repr(value)` |

Requirements:

* arrays MUST be numeric/bool, not object; the canonical form converts to
  C-contiguous **little-endian** before hashing, so a big-endian client and a
  little-endian server agree;
* `repr(float)` MUST be the shortest round-trip representation (Python 3, or an
  equivalent in another language); locale-dependent formatting MUST NOT be used;
* the algorithm is part of `protocol`: changing any rule invalidates every
  cache, hence a version bump.

## 7. Endpoints

### 7.1 `GET /health`

`application/json`:

```json
{"ok": true, "protocol": 1, "backend": "cuda_v4_pwa", "server": "…name…",
 "tabpwa": "0.1.0", "handles": 2, "cache_dir": "/scratch/…",
 "max_body": 8589934592, "features": ["npz"], "stats": { … }}
```

* `backend` is the identifier a client must request; the client MUST refuse to
  proceed if the server's backend differs from the one its spec asked for;
* `features` lists optional capabilities (§13); unknown entries are ignored;
* `server`/`tabpwa` are informational; clients MUST NOT gate on them.

### 7.2 `POST /load`

Two forms, distinguished by the presence of arrays:

**Probe** (no arrays) — "do you already have this key?"

```json
{"op": "load", "protocol": 1, "key": "<64 hex>"}
```

**Upload** — the sample itself:

```json
{"op": "load", "protocol": 1, "key": "<64 hex>", "model": "pwa",
 "backend": "cuda_v4_pwa",
 "payload": {"data": <skeleton>, "kc": <skeleton>}}
```

Answers:

```json
{"status": "hit",    "n": 240}      // already present (memory or disk)
{"status": "stored", "n": 240}      // upload accepted and cached
{"status": "miss"}                  // probe only: client SHOULD upload
```

A server MUST verify `key` against the upload (§6).  A re-upload of a known key
MUST answer `hit` (idempotent) and MUST NOT change the stored data.
`n` is the sample's event count; the client MUST NOT trust it over its own.

### 7.3 `POST /compute`

```json
{"op": "compute", "protocol": 1, "key": "<64 hex>",
 "params": <skeleton>, "norm": 1.234e-7 | null, "return_p": false}
```

Answer:

```json
{"status": "ok", "Q": -1234.5, "grads": <skeleton>, "none": ["g0"]}
```

plus, when `return_p` is true, an array member `"P"` with one value per event (in
the sample's order).  `none` lists gradient blocks whose value is null/missing.

Semantics (the normative compute contract):

* per event with weight `w`, intensity `P_e`, `norm` `N`, background column `b`:
  the server returns `Q = Σ_e -w_e·log(P_e/N + b_e)` and `dQ/dθ` for every
  parameter, for the sample it holds;
* `grads["norm"]` MUST be present when `norm` is non-null and MUST be the
  derivative of the SERVER's own objective w.r.t. `N` (a grouped/log-sum
  objective supplies its own; a client MUST NOT derive it from `P`);
* `return_p=false` promises that `P` is not needed; servers SHOULD skip building
  it, and clients MUST NOT depend on its presence;
* results for one `(server build, sample, params, norm, return_p)` MUST be
  deterministic.  Cross-implementation parity is NOT required bitwise; §10
  states the tolerance a pool relies on.

### 7.4 `POST /free`

```json
{"op": "free", "protocol": 1, "key": "<64 hex>"}
```

Releases the server's LIVE handle.  The on-disk cache is retained (a later
probe answers `hit`).  Idempotent; an unknown key answers `{"dropped": false}`.

## 8. Multi-server distribution (client-side)

The client shards ONE sample into `k` contiguous slices (one per server) and
uploads each slice with its own key:

```
slice_i  = rows [b_i, b_{i+1})   of every per-event array
key_i    = key(slice_i, kernel_config, model, backend)
```

Requirements:

* slices MUST be contiguous and in sample order, so a returned `P` can be
  concatenated back without an index map;
* `Q`, every `grads` block and `grads["norm"]` are SUMS over disjoint events, so
  the client may add the shard answers; `P` is concatenated;
* all servers of one pool MUST run the same `backend` (and identical kernel
  configuration), because the slices are pieces of one likelihood;
* servers MUST NOT be asked for overlapping events: a duplicated slice would
  double-count in the sum;
* the shard boundaries are part of `key_i`, so changing `k` changes the keys:
  one fresh upload per pool size, never per iteration.

## 9. Caching and identity

* **Memory**: handles, bounded by the server (`max_handles`), evicted LRU.  An
  eviction MUST NOT delete the disk entry.
* **Disk**: the exact upload, `<key>.npz`, plus a sidecar recording the server's
  **backend identity** (name + constructor arguments).
* The client-side key cannot see server-side constructor arguments (e.g. a
  grouped-objective resolution).  The server therefore MUST record its identity
  beside each entry and MUST treat an entry built with a different identity as a
  miss (and replace it on the next upload).
* A restart MUST serve probes from disk without requiring a client upload.

## 10. Precision and parity

* The protocol guarantees *structural* equality: the same block names, shapes
  and dtypes; `norm` present iff requested; `P` present iff requested.
* A pool sums shard values on the client, which changes association.  For the
  reference kernels a pool of `k` servers agrees with a single server to
  ~`1e-14` relative (measured: `max|Δgrad| = 5.9e-14` at `k=2`); conformance
  tests SHOULD use `rtol = 1e-12`.
* A single server running the same kernel as the client's local one is
  bit-identical (measured `ΔQ = 0`, `Δgrad = 0`); this is a property of the
  kernel, not of the protocol.

## 11. Errors, retries, failure semantics

| status | meaning | body |
|---|---|---|
| 200 | success | endpoint's type |
| 400 | protocol mismatch, key mismatch, missing/malformed meta field (the field is named), unknown `op` | `{"error": …}` |
| 401 | missing/wrong bearer token | |
| 404 | unknown path | |
| 413 | body over `max_body` (not read) | |
| 500 | compute/store failure; the message names the exception | |

* The client MUST NOT fall back silently: a failed `compute` MUST abort the
  objective evaluation (a fit must never continue on a locally computed value).
* Retry SHOULD be limited to connection-level failures and to requests that are
  idempotent (`/load`, `/free`, `compute` for fixed params/key).  A retried
  `/load` MUST answer `hit`/`stored` without duplicating storage.
* A pool member's failure MUST be surfaced; the client MUST NOT "rebalance" the
  lost slice onto the others implicitly (that would silently change the sample).

## 12. Security

* Bind to loopback by default; a public bind is the operator's decision.
* An optional shared bearer token (`Authorization: Bearer <token>`); compared on
  every request.  v1 has no TLS: use a trusted network or a tunnel.
* The server executes no client-supplied code; the kernel configuration selects
  a registered backend and the parameters are data.  A server SHOULD validate
  shapes/dtypes from the kernel configuration rather than trusting them.
* `max_body` and `max_handles` are the resource bounds; v1 has no quota per
  client.

## 13. Evolution

* Additive meta fields MAY be ignored by a server; a client MUST tolerate extra
  fields in answers.
* New endpoints/features are advertised in `/health.features`; a client MUST NOT
  use a feature it did not see advertised.
* The envelope is versioned as a whole (`protocol`); a v2 frames-based encoding
  (a JSON header + length-prefixed array blocks, streamable, no zip) is the
  candidate replacement for samples beyond a few GB, where the current one-blob
  framing costs ≈ the sample size in peak memory on both sides.

## 14. Open questions for the standard

1. **Encoding**: keep `.npz` (easy for Python clients, awkward for C++/Go), or
   define the frames encoding in v1?  A non-tabpwa server is the stated goal, so
   this is the main decision.
2. **Compression**: per-message compression (npz) vs `Content-Encoding`; which
   side chooses, and how is it negotiated?  Today compression is a client flag.
3. **Streaming**: upload a multi-GB sample without materialising it twice
   (length-prefixed blocks, or a resumable upload with an offset).
4. **Pool semantics**: is client-side sharding the standard, or should a server
   accept a "shard of" reference so servers fetch/peer from one another?
5. **Failure in a pool**: fail the iteration (proposed) or retry the slice on
   another server that holds the same data (needs replication)?
6. **Identity/versioning**: what exactly must a server's `backend identity`
   include (kernel source hash? compiled flags? library version?) so a stale
   entry is never reused across a rebuild?
7. **Auth**: token scope (per server, per sample), TLS requirement, and whether
   `/health` is public.
8. **Observability**: request ids, per-request timing in `/health`, a metrics
   endpoint.
9. **Eviction**: should `/free` be allowed to delete the disk entry (an
   `evict: true` flag), and what is the cache's on-disk budget?
10. **Multiple models per server**: today one process = one backend; is
    multiplexing wanted?

## 15. Considered alternatives

| option | why not for v1 |
|---|---|
| **gRPC / protobuf** | typed + streaming, but BOTH sides need a protobuf toolchain and generated stubs; the bulk payload is still opaque `bytes`, so the schema buys little over one archive. |
| **Arrow Flight** | purpose-built for bulk array transfer with metadata, and the strongest long-term fit; but it drags Arrow C++/pyarrow into a server that should stay small. |
| **ZeroMQ** | fast and simple, but no status codes, no `curl`/proxy/health-check story — the error model would be reinvented. |
| **HTTP + JSON + base64** | trivially implementable, but +33 % size and CPU on multi-GB arrays. |
| **HTTP + msgpack** | compact, but a dependency on both sides and a type-mapping problem for arrays. |
| **Shared filesystem + lock** | simplest inside one cluster (no network at all, no server), but no scheduling/health/remote-GPU story; sensible as a DEPLOYMENT option, not as the standard. |

The draft picks HTTP/1.1 + `.npz` because the client is numpy, a server needs
only zip + `.npy` (a few hundred lines in C/C++/Go), and the four endpoints are
the whole surface.  If a non-Python server becomes the primary consumer, revisit
Arrow Flight — that is the one alternative that would justify replacing the
envelope rather than extending it.

## 16. Reference prototype (non-normative)

A prototype of this draft — the envelope + `data_key`, the client with sharding,
a reference server, and its gates — was built and measured during the design
review.  It is deliberately NOT part of the tree: the standard must be decided
on its own terms, and the prototype is kept aside (stashed) rather than shipped.
The measurements quoted in the review (upload-once, params-only per iteration,
pool parity ~1e-14, the transport checks) came from it and can be re-derived
from this document.
