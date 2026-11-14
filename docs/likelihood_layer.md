# Likelihood layer — design (registered step types, general constructor)

Status: **proposed, not implemented**.  The objective is compiled from
**specs** into a `Pipeline` of `Step` objects.

Two registries, both flat dicts filled by decorators (the house pattern of
`register_model`, `register_amplitude_model`, `register_backend`,
`register_barrier`):

* `STEP_TYPES` — `@register_step("norm")` … type name -> `Step` subclass;
* `NLL_BUILDERS` — `@register_nll("standard")` … objective name ->
  builder(fitter) -> `[spec, ...]`.

An objective is therefore *composed*, not subclassed (§8).  All step types
share **one general constructor**

```python
Step(spec: dict, context: dict)
```

and one **step registry**, so a pipeline is a list of plain specs that a
common builder resolves and instantiates.  Each step declares `needs` (the
dataset **name** it reads) and `outputs` (the value names it publishes); it
stores no data.  The caller keeps a `datasets` mapping of `Dataset` records
(§7.2) and the driver passes `datasets[step.needs]` at call time.

Steps have a two-phase protocol: `setup(datasets)` runs once per dataset
(re)load and may fill `Dataset.cache`; `__call__(params, data, values)` runs
per iteration and must be pure (§7.2).

`values` (name → scalar) and `grads` (name → `∂value/∂params`) are shared
across steps, so a value computed once (e.g. the normalisation) is available
to every later step.

Signatures below are provisional — the open items in §11 are deferred until
implementation starts.

## 1. Problem

The objective is hard-coded in two functions:

- `Fitter.get_nll_raw` (`fitter.py:733`): one phsp handle → `norm`; one data
  handle → `Q(params, data, norm)`; then the norm chain is inlined
  (`total = g + grads["norm"] * norm_grads`) plus `_log_purity_const`.
- `Fitter.get_nll` (`fitter.py:872`): `build_params` → `get_nll_raw` →
  `_kernel_builder.backward` → `Prior`s → `_flat_gradient`.

Consequences: exactly one data + one phsp handle per `Fitter`, one backend,
the loop is always *norm, then data*, and `JointFitter` combines whole
Fitters.  A different objective (shared norm across datasets, extended /
yield terms, external measurement constraints) has nowhere to live.

## 2. Non-goals

- No **objective** class hierarchy (no `NLL` base class, no strategy objects).
  `Step` subclasses are small formula units built from specs; the objective is
  still a list composed by the driver.
- No capability metadata on backends, with one exception: the `gram` norm
  step must know whether a backend can pre-integrate (`build_gram`).  Either
  that is a backend *protocol* (the method already exists on
  `integrated*`/`cuda*_gram`) or a one-name capability check; see §7.3 and
  §11.
- No shared context object at run time: a step declares `needs`; the caller
  owns the `datasets` mapping.  An entry is a **`Dataset` record**
  (`handle`, `np`, `cache` — §7.2), so a step never sees two entries for the
  same data.  (`context` below is **build-time** configuration only:
  backend, constants, free parameter names.)
- No value graph / autodiff engine: `values`/`grads` are plain dicts threaded
  through a plain loop.
- No kernel or backend changes: `compute(params, handle, norm=None,
  return_p=True) -> (Q, grads, P)` and `grads["norm"]` stay exact.  The
  observed count `Σw` needs the numpy arrays, not the handle; they live in
  `Dataset.np`, so no backend method is needed.
- No change to `ConstraintManager` (fixed/same/scale/bounds/blind are
  parameter transforms, not likelihood terms).
- No per-event arrays as pipeline values (deferred; `P` stays a `return_p`
  output).

## 3. One constructor, one registry

```python
STEP_TYPES = {}                        # type name -> Step subclass


def register_step(name):
    def _f(cls):
        STEP_TYPES[name] = cls
        cls.type = name
        return cls
    return _f


class Step:
    """Base: one general constructor, data declared by name only."""

    type = None
    needs = None                       # default dataset key (class-level)

    def __init__(self, spec, context):
        self.spec = dict(spec)         # type-specific keys (may be empty)
        self.context = context         # build-time: backend, constants
        self.needs = self.spec.get("needs", type(self).needs)
        self.name = self.spec.get("name")
        self.outputs = tuple(self.spec.get("outputs", ()))

    def setup(self, datasets):
        """Build-time hook: once per dataset (re)load.  May fill
        ``Dataset.cache``; must not read the fit parameters."""

    def __call__(self, params, data, values):
        """Per iteration, pure.  data = datasets[needs], or None."""
        raise NotImplementedError


def build_step(spec, context):
    spec = dict(spec)
    try:
        typ = spec.pop("type")
    except KeyError:
        raise ValueError(f"step spec needs a 'type': {spec!r}") from None
    cls = STEP_TYPES.get(typ)
    if cls is None:
        raise ValueError(f"unknown step type {typ!r}; "
                         f"available: {sorted(STEP_TYPES)}")
    return cls(spec, context)
```

Concrete types use the same constructor; each reads its own keys from `spec`
and its build-time dependencies from `context`:

```python
@register_step("norm")
class NormStep(Step):
    needs = "phsp"

    def __init__(self, spec, context):
        super().__init__(spec, context)
        self.backend = context["backend"]
        self.outputs = (self.name or "norm",)

    def __call__(self, params, handle, values):
        norm, g = self.backend.compute(params, handle, norm=None,
                                       return_p=False)
        return {self.outputs[0]: norm}, {self.outputs[0]: {"params": g}}


@register_step("data")
class DataStep(Step):
    needs = "data"

    def __init__(self, spec, context):
        super().__init__(spec, context)
        self.backend = context["backend"]
        self.purity_const = spec.get("purity_const",
                                     context.get("purity_const", 0.0))
        self.outputs = (self.name or "nll_data",)

    def __call__(self, params, handle, values):
        q, g = self.backend.compute(params, handle, norm=values["norm"],
                                    return_p=False)
        cot = g.pop("norm")                    # ∂q/∂norm (a cotangent)
        return {self.outputs[0]: q + self.purity_const}, \
               {self.outputs[0]: {"params": g, "norm": cot}}


@register_step("sumw")
class SumWStep(Step):
    """Observed count, read from the numpy dataset (not the handle)."""

    needs = "data_np"

    def __init__(self, spec, context):
        super().__init__(spec, context)
        self.from_data = spec.get("from", "data")
        self.outputs = (f"@{self.from_data}:sumw", f"@{self.from_data}:n")

    def __call__(self, params, data_np, values):    # data_np is the numpy dict
        w = data_np.get("weight")
        n = int(len(data_np["mass"]))
        sumw = float(np.sum(w)) if w is not None else float(n)
        out_w, out_n = self.outputs
        return ({out_w: sumw, out_n: n},
                {out_w: {"params": {}}, out_n: {"params": {}}})


@register_step("sum")
class SumStep(Step):
    needs = None

    def __init__(self, spec, context):
        super().__init__(spec, context)
        self.names = tuple(spec["names"])
        self.outputs = (spec.get("out", "nll"),)

    def __call__(self, params, data, values):   # data is None
        out = self.outputs[0]
        return {out: sum(values[n] for n in self.names)}, \
               {out: {n: 1.0 for n in self.names}}
```

- `spec` carries the step-type configuration (`needs`, `name`, `names`,
  `purity_const`, …); `context` carries the build-time dependencies
  (`backend`, `purity`, `sumw`, …).  One signature for every type.
- Adding a step type = a new subclass + `@register_step("name")`; nothing
  else changes.
- The registry stores any callable `factory(spec, context) -> step`; a
  `Step` subclass is the common case, and `register_step` may also wrap a
  plain function (`FnStep`), in which case `build_step` calls it with the
  same two arguments.

`values`/`grads` rules:

- `cur_values`: `{name: float}` — one entry per output (multi-output is just
  several entries).
- `cur_grads`: `{name: {source: derivative}}` — for every output, the
  derivative w.r.t. each **source**:
  - `source == "params"` → `{param_name: ∂value/∂param}` (kernel frame);
  - any other source → a scalar cotangent `∂value/∂source_value`.

Reserved names:

- `"params"` — source key holding a value's direct `∂v/∂params`;
- `"nll"` — **the objective**, produced only by the final reducer; terms
  publish unique names (`norm`, `nll_data`, `nll_poisson`,
  `nll_penalty_<tag>`), so no term reads or aliases the objective;
- `"@<dataset>:sumw"`, `"@<dataset>:n"` — **dataset metadata published by
  `SumWStep`**, which reads the numpy entry (`needs="data_np"`).  Constants
  with no parameter gradient (`{"params": {}}`); they exist so later steps
  (e.g. the Poisson term) can read the observed count without any hand-passed
  value.

### Driver

```python
def run(steps, params, datasets):
    values, grads = {}, {}

    for step in steps:
        if step.needs is None:
            handle = None
        elif step.needs not in datasets:
            raise KeyError(f"step {step.outputs} needs dataset "
                           f"{step.needs!r}; available: {sorted(datasets)}")
        else:
            handle = datasets[step.needs]

        cur_values, cur_grads = step(params, handle, values)
        values.update(cur_values)

        for name, sources in cur_grads.items():
            if name in grads:
                raise ValueError(f"value {name!r} published twice")
            g = {k: np.asarray(v).copy()
                 for k, v in sources.get("params", {}).items()}
            for src, cot in sources.items():
                if src == "params":
                    continue
                for k, v in grads[src].items():       # ∂src/∂params
                    g[k] = g.get(k, 0.0) + cot * v    # ∂name/∂src · ∂src/∂params
            grads[name] = g

    if "nll" not in values:
        raise ValueError("pipeline produced no objective 'nll'")
    return values, grads
```

Per output:

```
grads[name] = ∂name/∂params + Σ_src (∂name/∂src) · grads[src]
```

**Recursion.**  Every published value's gradient is already complete (it was
chained when published), so later steps only multiply by their cotangent.
Nothing is applied twice and no step author performs a chain multiplication.

## 4. Default pipeline as specs

A pipeline is a list of specs; `build_pipeline` resolves each with the
registry:

```python
STANDARD_SPEC = [
    {"type": "norm", "needs": "phsp"},
    {"type": "data", "needs": "data"},
    {"type": "sum", "names": ["nll_data"]},
]

context = {"backend": backend, "purity_const": purity_const}
pipeline = [build_step(s, context) for s in STANDARD_SPEC]

# at evaluation time:
#   run(pipeline, params, {"phsp": phsp_handle, "data": data_handle})
```

`Fitter.get_nll_raw(params)` calls `run(self.pipeline, params,
self.datasets)` and returns `values["nll"], grads["nll"]` — the same
`backend.compute` calls, the same `g + dq_dnorm · ∂norm/∂params`, the same
purity constant.

## 5. One norm, many data

Dataset names are part of the spec, so one pipeline describes several
datasets:

```python
SPEC = [
    {"type": "norm", "needs": "phsp1", "name": "norm_1"},
    {"type": "norm", "needs": "phsp2", "name": "norm_2"},
    {"type": "sum", "names": ["norm_1", "norm_2"], "out": "norm"},
    {"type": "data", "needs": "data1", "name": "nll_run1"},
    {"type": "data", "needs": "data2", "name": "nll_run2"},
    {"type": "sum", "names": ["nll_run1", "nll_run2"]},
]

pipeline = [build_step(s, context) for s in SPEC]
run(pipeline, params, {"phsp1": phsp_a, "phsp2": phsp_b,
                       "data1": run1, "data2": run2})
```

- each norm step is an independent summand with its own `∂/∂params`;
- the reducer publishes `norm` with unit cotangents;
- both data steps read the same `norm` and publish distinct names;
- the final reducer publishes the objective `"nll"`.

The same pipeline object can be re-run against a different `datasets` mapping
without rebuilding: only the handles behind the names change.

## 6. What the named steps mean

| type | value | formula (current kernel convention) |
|---|---|---|
| `"norm"` | `norm` | `Σ_phsp w_phsp·P(Φ)` — the pdf denominator (`Σw_phsp = 1`) |
| `"data"` | `nll_data` | `-Σ_e w_e·log(P_e/norm + bkg_e)` + purity constant (shape term) |
| `"poisson"` | `nll_poisson` | `N_exp - S·log N_exp`, `S = Σ_e w_e`, `N_exp = norm/p` |
| `"penalty"` | `nll_penalty_<tag>` | an external constraint expressible in kernel-frame params |
| `"sum"` | `nll` (objective) | `Σ` of the term names it is given |
| yield | — | **not needed by default**: the yield is `N_exp = norm/p`; only for models where `N_exp` is an independent function |

Notes:

- The per-event `Q` from the data step is already the full normalized shape
  term; the extended likelihood only *adds* the Poisson term.
- `norm` is the expected signal count under the phsp normalization, so
  `N_exp = norm/p` is the expected total; this removes the shape-only scale
  degeneracy of `ck`.
- The data step's purity constant is `-S·log p`; the extended term adds
  `-S·log N_exp = -S·log norm + S·log p`, so the two `S·log p` pieces cancel
  — no double counting and no extra constant.
- **`S` comes from the numpy dataset entry.**  The handle is opaque (a CUDA
  handle carries no numpy weights), so the observed count is a **separate
  dataset**: `SumWStep` declares `needs="data_np"` and reads the numpy
  arrays the fitter registers (`datasets["data_np"] = self._data_np`),
  publishing `"@<from>:sumw"` / `"@<from>:n"` with an empty
  `{"params": {}}` gradient; `PoissonStep` reads
  `values[f"@{self.from_data}:sumw"]`.  **No spec field, no hand
  calculation, no driver pre-seeding, no backend change, and the pipeline
  stays static** (the mapping is re-bound when data changes).
  With no `weight` key the default is `np.ones(n)` ⇒ `S = n_events`; with
  several datasets each `SumWStep` publishes its own key
  (`"@run1:sumw"`, `"@run2:sumw"`, …).  **Signed weights** (sPlot-style)
  are not compatible with the Poisson term: `log N_exp` is undefined for
  `Σw ≤ 0`; either omit the term or use an explicitly documented effective
  count (`Σ|w|`) — a step-level choice, not a hidden kernel detail.
- A constraint on a **resolved** name (e.g. `rho770_mass`) stays on the
  `Prior` path (merged after `_kernel_builder.backward`); `"penalty"` is only
  for kernel-frame constraints.

```python
@register_step("poisson")
class PoissonStep(Step):
    needs = None

    def __init__(self, spec, context):
        super().__init__(spec, context)
        self.from_data = spec.get("from", "data")   # dataset whose count to use
        self.purity = spec.get("purity", context["purity"])
        self.outputs = (self.name or "nll_poisson",)

    def __call__(self, params, data, values):       # data is None
        out = self.outputs[0]
        sumw = values[f"@{self.from_data}:sumw"]     # published by DataStep
        n_exp = values["norm"] / self.purity
        q = n_exp - sumw * np.log(n_exp)
        return {out: q}, \
               {out: {"params": {},
                      "norm": 1.0 / self.purity - sumw / values["norm"]}}


EXTENDED_SPEC = [
    {"type": "norm", "needs": "phsp"},
    {"type": "data", "needs": "data"},
    {"type": "sumw", "needs": "data_np", "from": "data"},
    {"type": "poisson", "from": "data"},
    {"type": "penalty", "mu": ..., "sigma": ...},
    {"type": "sum", "names": ["nll_data", "nll_poisson", "nll_penalty"]},
]
```

Adding a term = a new spec entry with a unique `name`, plus its name in the
reducer's list.

## 7. Fitter integration

- Build-time context: `fitter.step_context = {"backend": self.backend,
  "purity_const": self._log_purity_const}` (constructors only).
- `Fitter.__init__`: `self.pipeline = build_pipeline(self)` from the
  registered builder.  The pipeline holds no data but does own its
  `needs` set and its validation (§7.1).
- `set_phsp` / `set_data` register **one `Dataset` record** per dataset and
  then refresh the step caches:
  ```python
  self.datasets["phsp"] = Dataset("phsp", phsp_holder, phsp_np, meta,
                                  {})
  self.datasets["data"] = Dataset("data", holder, data_np, meta, {})
  self.pipeline.setup(self.datasets)      # after constraints exist (§7.2)
  ```
  No pipeline rebuild: the steps resolve names at call time.
- **Preparation is ordered — `set_phsp` before `set_data`** (see §7.4), and it
  is the *only* ordering the Fitter imposes.
- `Fitter.get_nll_raw(params, resolved=None)` → `self.pipeline.run(params,
  self.datasets)`, returning `values["nll"], grads["nll"]`.
- `Fitter(cfg, nll="standard")` / `Fitter(cfg, pipeline_spec=...)` constructor
  arguments (preferred) plus attribute overrides; the chosen `nll_name` is
  stored in the saved fit metadata.
- `JointFitter`: unchanged in mode 1 (sum of sub-Fitter `get_nll`); mode 2
  shares one spec list and a `datasets` mapping covering every config.
- Priors stay where they are.

### 7.1 The compiled form: a `Pipeline` object

`register_nll` builders return **specs** (plain data); the Fitter needs a
compiled object, because three things cannot be answered by a bare list:

```python
class Pipeline:                 # final; NOT a base class for objectives
    name: str                   # registered builder name
    steps: tuple[Step, ...]
    needs: frozenset[str]       # dataset names the steps read
    nll_source: str             # the single step publishing "nll"

    def setup(self, datasets): ...        # build-time caches (§7.2)
    def to_spec(self) -> list[dict]       # fit metadata / reproduction
    def run(self, params, datasets): ...  # the plain loop, unchanged
```

- **`needs` is a pipeline property.**  A norm step requires `"phsp"`, a
  `SumW` step the numpy form, a yield-only objective neither — the Fitter
  must know this *before* the first call to decide whether the expensive
  `set_phsp` pass is needed at all.  From a bare list it would have to
  re-scan specs and special-case step types.
- **Validation happens once, at construction**: exactly one producer of
  `"nll"`, no duplicate value names, every `needs` resolvable, every
  gradient source either `"params"` or an earlier value.
- **Metadata**: `name` + `to_spec()` go into the fit record, so a resumed or
  re-run fit reconstructs the same objective.

Non-negotiable: the class holds **no objective-specific branches** (no
`if name == …`), keeps no data, and is not subclassed per objective — a new
objective is a new `@register_nll` builder.

### 7.2 `Dataset` records and step caches

One record per dataset, not two entries:

```python
@dataclass
class Dataset:
    name: str
    handle: object      # backend handle — passed through UNCHANGED
    np: dict | None     # numpy form (weights, bkg, per-event arrays)
    meta: dict          # dataset-intrinsic summaries (n, sumw, N_b, ...)
    cache: dict         # per-step memo, namespaced by step
```

Three dicts with three different owners:

| dict | written by | contents |
|---|---|---|
| `np` | the Fitter, at registration | the per-event arrays the kernels see (`weight`, `bkg`, `mass`, …) |
| `meta` | the Fitter, at registration | dataset-intrinsic **scalars** (`n`, `sumw`, `sumw_raw`, `N_b`) |
| `cache` | steps, in `setup` | derived data a step memoises (Gram, masks, group maps) |

`meta` is shared and public (any step may read it); `cache` is step-private by
namespace.  Neither may hold anything parameter-dependent.

The former `"data_np"` / `"phsp_np"` entries disappear: a step that needs the
numpy form reads `ds.np`.

**Why the dataset owns the cache.**  A step-attribute cache silently survives
`set_data()` unless the step hand-rolls handle-identity checks; a backend
cache can only hold what the backend knows about.  Attaching to the dataset
gives the correct lifetime for free — `set_data`/`set_phsp` replace the
record, so the cache goes with it, and every step (and every pipeline built
on it) shares it.

**Caching is structurally iteration-invariant.**  The two-phase protocol
(§3) makes it impossible for a step to cache a parameter-dependent quantity:
`setup` has no parameters, `__call__` has no cache write.

Rules:

- keys are namespaced per step (`ds.cache[step.namespace][key]`); two
  objectives cannot collide and a joint fit cannot cross-contaminate;
- `Dataset.handle` is **not** wrapped or proxied — the backend's own caches
  (`amp`, `us`, Gram) already own the heavy data on the device;
- never cache `P`, `norm`, `dQ/dA` or anything else depending on the fit
  parameters.

| quantity | owner |
|---|---|
| amplitude cache, γ-interp cache, Gram matrices built by the backend | backend handle |
| Σw, n, masks, efficiency maps, event→group maps, binned references | `Dataset.cache` via `setup` |
| `P`, `norm`, `dQ/dA`, `values`/`grads` | recomputed every call |

### 7.3 The norm step — retiring the `integrated` backends

The phase-space norm is a computation over one dataset, reused every
iteration: a step with a dataset cache, not a backend.  Today `integrated` /
`integrated_pwa` are backends wrapping a backend (`base=`), which forces a
per-model registration and hides an objective-level precondition inside a
backend name.

```python
@register_step("norm")
class NormStep(Step):
    needs = "phsp"
    # {"type": "norm", "needs": "phsp", "method": "forward"}
    # {"type": "norm", "needs": "phsp", "method": "gram"}

    def setup(self, datasets):
        if self.method != "gram":
            return
        # the Gram pre-integrates m0/g0 -> they MUST be fixed
        bad = set(self.context["free_names"]) & (m0_names | g0_names)
        if bad:
            raise ValueError(
                "norm method 'gram' pre-integrates m0/g0 into the wave Gram "
                f"matrix, so they must be fixed; free: {sorted(bad)}. "
                "Use method='forward' (or fix them).")
        be = self.context["backend"]
        self._require(be, "build_gram")     # fails listing capable backends
        datasets["phsp"].cache[self.ns]["D"] = be.build_gram(
            datasets["phsp"])

    def __call__(self, params, data, values):
        if self.method == "gram":
            D = data.cache[self.ns]["D"]
            # values["norm"] = ck D ck*;  grads["norm"] = {"params": {"ck": D@ck},
            #                                              "@norm": -1}
```

Consequences:

- `method="gram"` is legal **only with m0/g0 fixed**.  Fixed parameters are
  not part of the parameter space, so `∂norm/∂m0 = 0` is a fact rather than a
  policy — there is no `freeze` key, no `rebuild` policy and no staleness
  window.  `D` is parameter-independent and lives as long as the dataset.
- the precondition is **asserted at setup**, which the current backends never
  do.
- **defect this removes**: `IntegratedBackend.compute` /
  `IntegratedPWABackend.compute` zero `grads["m0"]` / `grads["g0"]` on the
  *delegated data-NLL path* too, so `backend="integrated"` /
  `"integrated_pwa"` with **free** m0/g0 silently returns a zero m0/g0
  gradient instead of the base backend's correct one — a wrong fit, no
  warning.  With the step form the masking disappears: nothing to mask when
  the precondition holds, a clear error when it does not.
- migration: `integrated` / `integrated_pwa` stay as deprecation aliases
  that build `[{"type": "norm", "method": "gram"}, …]` **and** verify the
  precondition, so existing configs either keep working correctly or fail
  loudly.  The `base=` nesting disappears: the data-NLL backend is simply
  `Fitter(backend=…)`.
- the Gram build needs a backend capability, so `build_step`'s `context`
  carries `backend` and the step validates it at setup
  (`build_gram` / `capability("gram")`), failing with the list of backends
  that provide it.
- tests that pin the current bug: `gram` (and `integrated*`) with a **free**
  m0/g0 must raise, and the data-NLL m0/g0 gradient must not be zeroed while
  they are free.

### 7.4 Dataset preparation is ordered: `phsp` before `data`

`set_phsp` and `set_data` are **not symmetric**.  The background is given as a
*yield* (`bkg_raw`) and becomes a per-event density only after division by the
phsp-averaged background:

```python
# 1. the normalisation sample
phsp["weight"] /= Σ w                     # Σw = 1
N_b = Σ (phsp_weight · bkg_raw_phsp)      # RAW integral, before any scaling

# 2. ONE scale, derived from the phsp sample ...
s = (1 - p) / (p * N_b)                   # p = purity

# 3. ... applied to EVERY dataset that carries a background
bkg(X) = bkg_raw(X) * s                   # X in {data, phsp}
#   => Σ w_phsp * bkg_phsp_scaled = (1-p)/p   (background-to-signal yield ratio)

# 4. the constant is a function of the same (s, N_b), not an independent formula
log_purity_const(X) = -log(p) * Σ w_X  ==  Σ w_X * log(1 + s * N_b)
```

so the dependency graph is

```
                    purity (config)
                          │
   phsp ──► N_b ──► s ─────┼──► bkg(data)  ──► data NLL
     │                    ├──► bkg(phsp)  ──► plots / extended terms
     │                    └──► log_purity_const per dataset
     └──────────────────────────────────────► norm step
```

`N_b` is the raw phsp integral; `s` is the only scale; every background
carrying dataset and every constant are downstream of the same `s`.

`set_data` without a phsp record must fail with a clear error (today it does:
`RuntimeError("set_data() requires set_phsp() first ...")` — but the
`set_data` docstring still claims purity correction is *skipped* instead,
treating `p = 1`; the code and the docstring disagree and the docstring is
wrong).

Two consequences of writing it as one unit:

- **the scale is a model-level quantity, not a data-level one**: it must be
  written into *every* sample that carries a background (data **and** phsp),
  otherwise the background shape used by a plot or by an extended term is
  normalised differently from the one the NLL used.  The current code only
  scales the data sample and lets `plot_pw_groups` re-derive
  `Σ w_phsp·bkg_raw` — the same number twice, which is the drift risk;
- **`log_purity_const` cannot be prepared separately** from `s`: it is
  `Σw·log(1 + s·N_b)`, so a step that writes the arrays and a different step
  that writes the constant could disagree.  One derivation, several outputs
  (arrays per dataset, one constant per dataset, `meta["N_b"]`,
  `meta["bkg_scale"]`).

**The invariant that makes caching legal here:** preparation may consume
**other datasets** and build-time context (purity), but never the fit
parameters.  `N_b` depends only on the phsp arrays, so `data["bkg"]` and
`log_purity_const` are constants during the fit and can live in `Dataset.np`
/ `Dataset.meta`.  If a future background model defines its normalisation
from the parameter-dependent norm instead, it can no longer be prepared —
it becomes a step value recomputed every iteration, and `DataStep`'s
`dQ/dnorm` gains the corresponding term.

Two more preparation details that must be explicit, because they currently
happen silently inside `set_phsp`:

- phsp weights are **normalised to Σw = 1** before the norm integration; the
  raw sum is kept as `meta["sumw_raw"]` so a plot or a check can undo it;
- `purity_const` is a *value*, not a step: it is a constant offset with an
  empty gradient, published/meta-carried so the objective does not hand-pass
  it.

Where the preparation lives: **the Fitter's dataset layer, not a pipeline
step**.  It is a property of the data (every objective needs the same `bkg`
scaling and weight normalisation), and no objective-specific choice enters
it — the pipeline only declares `needs` and reads `ds.np` / `ds.meta`.
A preparation *step* would force every objective to repeat it and would make
`Pipeline` order-sensitive.

### 7.5 Making the two setters independent: a lazy `prepare()` phase

The dependency in §7.4 is real, but it should not be an **ordering rule on
the user** — it is a build-time derivation that can simply happen later.
Three ways to get there:

| option | what changes | cost |
|---|---|---|
| **A. lazy `prepare()`** (recommended) | setters only record arrays + `meta`; `bkg`, `purity_const` and the backend handles are produced on demand, once both datasets exist | `set_data`/`set_phsp` no longer build handles eagerly; a `_data_holder` property triggers preparation so existing readers keep working |
| B. pass the scale into the kernels | `compute(..., bkg_scale=...)` | every backend, wrapper and kernel signature — massive, no physics gain |
| C. redefine `bkg_raw` as phsp-normalised (`N_b = 1`) | the sample-derived `N_b` disappears and the setters become truly independent | semantic change to existing configs; needs a yield/density flag + migration |

`set_phsp` and `set_data` in **either order**, then:

```python
set_data(raw)  -> Dataset("data", handle=None, np=raw,  meta={...})
set_phsp(raw)  -> Dataset("phsp", handle=None, np=w/Σw, meta={"N_b": ...})

prepare()      -> data.np["bkg"] = bkg_raw * (1-p)/p/N_b
                  data.meta["log_purity_const"] = -log(p) * Σw
                  build both backend handles
                  pipeline.setup(datasets)

run(params)    -> asserts prepared
```

`prepare()` is idempotent, is invoked automatically by the first
`get_nll`/plot/handle access, and is invalidated by either setter (which
drops the old handles and the derived arrays).  The same `bkg` array is
produced as today, so the numbers are unchanged: this is a lifecycle
refactor, not a physics change.

**Why B is rejected — measured cost.**  Threading the scale down to the
kernels touches, in this tree: 24 CUDA kernels that read `bkg`, 3 CPU
kernels, 43 `load_data` implementations, 48 backend registrations (1
universal + 47 model-scoped), and the 2 ONNX backends (which bake `bkg` into
the graph at build time, so the scale would need a graph input or a
re-trace) — roughly 80-100 edit points, each with a bit-exactness risk, for a
multiply performed once per fit.

Cheapest equivalent: keep the multiply on the host and let the backend see a
finished `b` per event, which is what today's code does — the only thing worth
changing is **when** the host computes it (§7.5).  That is ~1 file
(`fitter.py`) and no kernel, backend, wrapper or ONNX change.

**Why C is the right contract long-term:** with `bkg_raw` normalised over
phsp, `N_b = 1`, `bkg = bkg_raw·(1-p)/p`, and the datasets never interact.
The sample-derived `N_b` exists only to reinterpret a background *yield*.
Keep it until a config flag can distinguish "yield" from "density"; then the
ordering disappears entirely rather than being deferred.

For `JointFitter`: each sub-Fitter prepares its own pair; a shared phsp
`Dataset` record (§11) shares `meta["N_b"]`, which is what a purity scan
wants.

### 7.6 One generic dataset API — no roles, just specs  ·  **DEFERRED**

> **Status: deferred.**  Nothing in §7.5 needs this.  It pays off only when a
> *third* dataset (second sample, sideband, auxiliary measurement) actually
> appears; until then the two setters plus a lazy `prepare()` are enough.
> Keep it as the target shape, not as Phase A work.


`set_data` / `set_phsp` are two hard-coded instances of the same thing:
a dataset plus a build-time derivation.  A third dataset (second sample,
sideband, control region, external measurement) has no home, and §5's
"one norm, many data" had to be expressed as a joint fit.

**Rejected: a `role` taxonomy** (`role="phsp"|"data"|"aux"` + a role
registry).  It swaps two methods for a closed catalogue that itself needs:

- a second registry and a lifecycle (roles declare each other's `needs`);
- a user-supplied label that is silently wrong when mis-typed;
- per-role parameters — the moment two samples want different `purity`, the
  "role = behaviour" premise collapses and the label becomes a weak pointer
  to a configured behaviour.

That is a new axis, not a simplification.

**The uniform fix uses only what §3 and §7.1 already define**: the step
registry and a spec list.  A dataset is a name and a raw dict; its
derivations are specs.

```python
fitter.set_dataset("data",  raw)
fitter.set_dataset("phsp",  raw)
fitter.set_dataset("data2", raw)          # a second sample: nothing new
```

`Pipeline` owns **two spec lists**, built by the same `build_step`:

```yaml
prepare:                                    # build-time derivations, in order
  - {type: norm_weights, dataset: phsp}     #   Σw = 1
  - {type: purity,      datasets: [data, phsp], purity: 0.914,
     needs: [phsp]}                         #   N_b -> s -> bkg per dataset
                                            #   + @<ds>:log_purity_const
steps:                                      # per-iteration objective
  - {type: norm, needs: phsp}
  - {type: data, needs: data}
  - {type: sum, names: [nll_data]}
```

Rules:

- a `prepare` entry is an **ordinary step that only implements `setup`**
  (`phase = "prepare"` class attribute; `build_pipeline` rejects a prepare
  entry without it).  It may rewrite `ds.raw` — that is the entire point —
  and publish constants into `ds.meta`;
- a prepare step may write **several datasets** (`datasets: [...]`): the
  purity scale is one such derivation (scale the background in data *and*
  phsp, and publish the matching constant in both), which is why the scale
  and the constant belong to a single entry rather than two;
- `needs` on a prepare step declares the inter-dataset edge (`bkg_scale`
  needs `phsp`); the §7.4 ordering becomes **data in the spec**, executed in
  list order — not a rule the user must remember;
- backend handles are built **after** `prepare`, from `ds.raw`.  That is
  precisely why today's code had to order `set_phsp` before `set_data`: it
  wanted the scaled `bkg` before calling `load_data`;
- `set_data` / `set_phsp` survive as **sugar** that supplies the *default*
  prepare specs for the dataset named `"data"` / `"phsp"`.  Existing callers
  and configs are unchanged, and the defaults are visible in `to_spec()`
  instead of being hidden in the Fitter;
- a new dataset is a name plus an optional prepare list — no setter, no role,
  no new registry: adding a *derivation* is `@register_step`, the same
  registry the NLL uses.

What it buys:

| use case | today | spec-based datasets |
|---|---|---|
| second data sample | `JointFitter` with two Fitters | `set_dataset("data2", raw)` + a second `{type: data, needs: data2}` and one shared norm, with `bkg_scale` twice (one prep entry per sample) |
| sideband / control region | not expressible | a name + the prep list it needs |
| external measurement | not expressible | an `aux`-style name; its prep list may be empty |
| systematic / weight variation | rebuild the Fitter | another dataset name |

Consequences that fall out for free:

- **per-dataset purity** is just the `purity` key on that sample's
  `bkg_scale` entry — no config-format split and no "default for the dataset
  named data" rule;
- **`log_purity_const` is per dataset** (`@<dataset>:log_purity_const`, empty
  gradient), summed by the reducer; two samples contribute two constants
  instead of overwriting one attribute;
- `_data_np` / `_phsp_np` / `_data_holder` become views on
  `datasets[name]`, so existing readers keep working;
- `Fitter(config)`, `load_fit` and `JointFitter` are untouched.

Cost: one class attribute (`phase`) and one more spec list.  No new
registry, no taxonomy, and every new behaviour is a `@register_step`.

### 7.7 Cost model (measured, not estimated)

400 data + 400 phsp events, `cpu_pwa_s0c`, 18 free parameters, idle box:

| component | per iteration |
|---|---|
| norm pass (kernel) | 35.0 us |
| data pass (kernel) | 35.4 us |
| current `Fitter.get_nll` | **186.7 us** |
| -> of which Python (params, constraints, transforms, priors) | ~117 us (62 %) |
| 5-step driver skeleton + dict churn | 0.6 us |
| one 500-element gradient-chain add | 0.8 us |

So the whole step/pipeline layer is **~1-3 us per iteration**:

| shape | compute per iteration | pipeline overhead |
|---|---|---|
| 400 events, 6 waves | 187 us | 0.5-1.6 % |
| 65k events, 198 waves | ~14 ms | ~0.02 % |
| degenerate tiny objective (~50 us) | 50 us | 2-6 % |

Build-time only: `prepare()` scales `bkg` over `ne` floats (~0.2 ms at
262k events) once per dataset load; `Dataset` access is a dict lookup plus an
attribute (~0.1 us).

Rules that keep it at that level (otherwise the overhead is Python, not the
math):

- reuse the `values` / `grads` dicts across calls (clear in place, do not
  allocate per iteration);
- write the gradient chain in place (`np.add(..., out=...)`), no per-step
  temporaries on parameter-sized vectors;
- keep the step count small — the driver is a loop over 3-6 steps, not a
  graph engine;
- the `gram` norm step must call the backend capability when the Gram lives
  on the device; it must not reimplement the matvec in numpy.  (For
  `integrated_pwa` the Gram is already numpy: `np.einsum` to build `D`,
  `ck.conj() @ (D @ ck)` per iteration, so moving it into a step is
  speed-neutral there.)

Two facts this pins down:

1. the per-iteration cost of the *existing* code is dominated by Python in
   `get_nll` at small sizes — 62 % of a 187 us call — so the likelihood layer
   is not where a speed problem would come from;
2. preparation is a one-off: it must never be called from `run()`.

## 8. Registries (steps and pipelines)

Two plain dicts, no class hierarchy for the objective:

```python
# both populated by decorators (§3), never by a literal dict
STEP_TYPES   = {}                    # type name      -> Step subclass / factory
NLL_BUILDERS = {}                    # objective name -> builder(fitter) -> [spec, ...]


def register_nll(name):
    def _f(builder):
        NLL_BUILDERS[name] = builder
        return builder
    return _f


@register_nll("standard")
def standard_spec(fitter):
    return [
        {"type": "norm", "needs": "phsp"},
        {"type": "data", "needs": "data"},
        {"type": "sum", "names": ["nll_data"]},
    ]


def build_pipeline(fitter, name=None):
    name = name or getattr(fitter, "nll_name", None) or "standard"
    builder = NLL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unknown nll {name!r}; "
                         f"available: {sorted(NLL_BUILDERS)}")
    return [build_step(s, fitter.step_context) for s in builder(fitter)]
```

Rules:

- resolution order: explicit `pipeline_spec=...` → `Fitter(nll="name")` →
  `config["nll"]` → `"standard"`;
- unknown names fail with the registered lists (same style as
  `backends_for_model`), never a silent fallback;
- the resolved **name is stored on the fitter** and written into
  `save_params`/checkpoint metadata, so a resumed fit reconstructs the same
  objective;
- **specs are plain data**, so a config file can carry them verbatim
  (`nll: [{type: norm, needs: phsp}, {type: data, needs: data},
  {type: sum, names: [nll_data]}]`) — the deferred YAML composition is now a
  direct consequence of the registry, not extra machinery;
- **no per-model defaults**: the objective does not depend on the amplitude
  model — so, unlike `register_backend`, `register_nll` takes no `model=`
  argument and the namespace is flat;
- **name collisions warn**: re-registering an existing objective logs a
  `Warning` and the latest wins (the `JointFitter` convention).  A silent
  override would change the objective of every config that names it;
- **an empty builder is an error**: a builder returning `[]` raises at
  `build_pipeline`, otherwise the objective silently becomes a constant;
- **exactly one `"nll"` producer** is asserted when the `Pipeline` is
  constructed (§7.1) — the registry cannot enforce it, a builder can publish
  `"nll"` from any step type.

## 9. Migration

- **Phase A** — `tabpwa/likelihood.py` with `Step` (+ `setup`),
  `register_step`, `STEP_TYPES`, `build_step`, `NormStep`, `DataStep`,
  `SumStep`, `Pipeline` (§7.1), the `NLL_BUILDERS` registry,
  `build_pipeline`, `Dataset` (§7.2) and `Fitter.get_nll_raw` delegating.
  Bit-exactness test vs the current implementation on
  `tests/config_pwa.yml`, `tests/config_amp.yml`, `tests/config_angle.yml`
  (NLL + full gradient), `numpy_pwa` + one CUDA backend.  No backend/kernel
  edits.
- **Phase B** — multi-dataset specs (§5), optional `JointFitter` mode 2,
  shared `Dataset` records across Fitters.
- **Phase C** — `PoissonStep`, `PenaltyStep`, `nll:` config plumbing
  (specs already serialisable).
- **Phase D** — `NormStep` with `method="gram"` (§7.3), `integrated` /
  `integrated_pwa` reduced to deprecation aliases, and the m0/g0
  precondition asserted.  This is the phase that removes the silent
  gradient-zeroing defect.
- **Phase E** (deferred) — per-event values, if composed observables need
  `dQ/dP`.

## 10. Tests

1. Bit-exact NLL/grad vs the pre-change path (three configs, random `x`,
   `numpy_pwa` + one CUDA backend).
2. `build_step` registry: `{"type": "data", ...}` returns a `DataStep`;
   unknown type raises with the registered list; a spec without `type` raises.
3. Steps called directly with a handle (no Fitter): `DataStep` returns
   `{"nll_data": {"params": g, "norm": dq_dnorm}}` with no norm contribution
   left in `g`.
4. **Declared needs**: `needs`/`outputs` come from the spec; a missing key in
   `datasets` raises with the available keys; two steps naming the same
   dataset both receive the same handle; `needs=None` receives `None`.
5. **Static pipeline**: the same `[Step, ...]` evaluated against two different
   `datasets` mappings gives the two datasets' results (no rebuild).
6. **Driver chain**: 3-step pipeline where step 2 publishes a value and step 3
   consumes it — result equals `direct + cot·grads[value]`, checked against a
   hand computation and finite differences of the composed scalar.
7. **Depth 2**: a value whose derivative flows through another value.
8. **Reducer / unique names**: duplicate names raise; no term publishes
   `"nll"`; the reducer publishes it with unit cotangents.
9. `standard` at non-zero purity (`tests/config_amp.yml`, `purity: 0.914`).
10. **Extended term**: Poisson value and `∂Q/∂norm = 1/p - S/norm` vs finite
    differences at `p < 1`; with `p = 1` the pipeline equals
    `Q_data + norm - S·log norm`.
11. **`sumw` from the dataset's numpy form**: `SumWStep` declares
    `needs="data"` and reads `ds.np`, publishing `@data:sumw` /
    `@data:n`; `PoissonStep` reads `@data:sumw`; `S` equals
    `np.sum(weight)` and, without a `weight` key, `n_events`; no spec field
    and no backend change; signed weights (`Σw ≤ 0`) raise at `log N_exp`
    instead of returning NaN.
12. Shared-norm spec (§5) equals the sum of two single-dataset pipelines.
13. `Fitter` integration: pipeline built from specs; `Dataset` records
    re-bound without a rebuild; step caches refreshed by `set_data` /
    `set_phsp`; fixed-slot zeroing unchanged.
14. Pipeline registry: resolution order, unknown name error, `nll_name`
    recorded in saved metadata; a literal `pipeline_spec` accepted.
15. Error paths: backend without `grads["norm"]`; a source not in `grads`; a
    published value named `"params"`; `poisson` naming an unknown dataset.
16. `Pipeline` object: `needs` is the union of the steps' datasets (a
    norm-only pipeline requires `phsp`, a yield-only one nothing); zero or
    two `"nll"` producers raise; duplicate value names raise; an empty
    builder raises; `to_spec()` round-trips to the same pipeline; `name`
    lands in the fit metadata.
17. `Dataset.cache`: a value written in `setup` is reused across iterations
    and recomputed when the dataset is replaced; per-step namespaces do not
    collide; `__call__` cannot write the cache (protocol test).
18. **Preparation order**: `set_data` before `set_phsp` raises with the
    `N_b` message; after both, `data["bkg"]` equals
    `bkg_raw·(1-p)/p/N_b` (bit-identical to the current implementation) and
    `meta["log_purity_const"] == -log(p)·Σw`; phsp weights in the record sum
    to 1 and `meta["sumw_raw"]` keeps the original; `bkg` is unchanged by a
    parameter change (it is prepared, not computed per iteration).
19. **Setter order is irrelevant**: `set_data` then `set_phsp` and the reverse
    give identical `data["bkg"]`, `purity_const`, NLL and gradients;
    `prepare()` is idempotent; a later setter invalidates it; reading
    `_data_holder` before `prepare()` triggers it lazily.
20. **Generic datasets**: a third dataset is registered with an empty prepare
    list, read by a step and ignored by the others; two data samples over one
    phsp reproduce the `JointFitter` shared-norm result exactly; per-dataset
    purity via the `bkg_scale` spec key; a prepare step whose `needs` is
    missing raises with the available names; a `prepare` entry without
    `phase = "prepare"` is rejected; `set_data`/`set_phsp` remain equivalent
    sugar for the default prepare specs.
21. `norm method="gram"`: builds once per phsp dataset; `m0`/`g0` **free**
    raises at `setup` (and the same for `integrated*`); with them fixed the
    norm matches the full forward pass to fp64 round-off and the data-NLL
    m0/g0 gradient is *not* zeroed.  These two assertions fail against the
    current `integrated*` backends — they pin the defect.

## 11. Open questions / deferred decisions

Originally "not decided yet"; the items below are now split into what this
revision settled and what is genuinely still open.

Settled since the first draft (decisions, not questions):

- **spec keys are flat** with a `type` discriminator — the spec list is the
  public YAML surface (`nll: [...]`), so nested payloads are only allowed if
  a step's configuration grows large;
- **`Dataset` record** replaces `"data_np"` / `"phsp_np"` (§7.2);
- **`Fitter` owns the `datasets` mapping**; `run`/`setup` receive it (§7.1);
- **duplicate value names are an error**, as is an empty builder or a
  pipeline without exactly one `"nll"` producer (§8).

Still open:

- **Context contents.**  `backend`, `purity`, `sumw`, `purity_const`,
  `free_names` — and whether new steps may consume arbitrary extra keys.
- **Class vs spec defaults.**  `needs` may come from the class attribute or
  the spec (current: spec overrides); whether `outputs` should also have a
  class default.
- **Reserved source name.**  `"params"` vs a nested
  `{"params": …, "inputs": {…}}` split.
- **Capability API on backends.**  §7.3 needs "does this backend provide
  `build_gram`"; the doc's non-goal says "no capability metadata on
  backends".  Either drop that non-goal for this one capability
  (`hasattr`/`capability("gram")` + a list of capable backends) or declare
  Gram support a backend *protocol* (`compute_gram` already exists).
- **Sharing a `Dataset.cache` across Fitters** (joint fits, scans): opt-in
  by passing the same `Dataset` record, or always per-Fitter?  (A shared
  phsp record would also share `meta["N_b"]`, which is what a scan over
  purity wants.)
- **Preparation hooks.**  §7.5 makes `prepare()` the explicit phase; the
  remaining question is whether a custom background model (e.g. one whose
  normalisation is parameter-dependent) may register its own preparation, or
  must become a step value instead.
- **Whether §7.6 is worth doing at all.**  It is deferred; the trigger is a
  concrete third dataset.  Until then the cheaper path is `set_data` /
  `set_phsp` + a lazy `prepare()`, and the purity derivation stays two lines
  in the Fitter.
- **`prepare` as a separate list vs a `phase` field.**  §7.6 keeps two lists
  (`prepare`, `steps`) sharing one registry and one builder.  Merging them
  into one list with a `phase` key is equivalent and slightly more uniform,
  at the cost of interleaving build-time and per-iteration entries in the
  serialised objective.
- **`bkg` yield vs density.**  Option C of §7.5 — a config flag selecting
  "`bkg_raw` is a yield normalised by the phsp sample" (today) vs "is already
  a density" (`N_b = 1`) — would remove the inter-dataset dependency
  altogether.  Which becomes the default on the next config-format break?
- Module name: `likelihood.py` vs `objective.py`.
- Should `DataStep` pass `return_p=False` explicitly (as today) or leave it to
  the backend default.
- `run` return: `(values, grads)` vs `(values["nll"], grads["nll"])`.
