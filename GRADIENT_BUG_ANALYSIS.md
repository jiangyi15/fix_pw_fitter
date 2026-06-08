# Gradient Bug Analysis Report

## Executive Summary

After thorough analysis, I've identified **FIVE CRITICAL BUGS** in the gradient computation:

1. **Missing factor of 2 for complex parameter gradients** (ck)
2. **Incorrect chain rule for complex-valued intermediate variables**
3. **Scalar gradients returning complex instead of real values**
4. **Missing Wirtinger calculus for complex-conjugate operations**
5. **Incorrect broadcasting in gradient accumulation**

## Bug #1: Missing Factor of 2 for Complex Parameters

### Location
- `numpy_kernel.py`: Line 172
- `numpy_kernel_fully_merged.py`: Line 142

### The Bug
```python
# Current (WRONG):
grad_ck = np.sum(dQ_dck, axis=0)
```

### Why It's Wrong
For **real-valued loss Q** with respect to **complex parameter z**, Wirtinger calculus gives:

```
∂Q/∂Re(z) = 2 * Re(∂Q/∂z*)
∂Q/∂Im(z) = -2 * Im(∂Q/∂z*)
```

The numerical test computes:
- `grad_num_real = (Q(z + ε) - Q(z - ε)) / (2ε)` = `∂Q/∂Re(z)` ✓
- `grad_num_imag = (Q(z + iε) - Q(z - iε)) / (2ε)` = `∂Q/∂Im(z)` ✓

But the analytical code computes `∂Q/∂z` (assuming holomorphic), not `∂Q/∂z*`.

### The Fix
```python
# For ck gradient with real-valued Q:
# ∂Q/∂ck = conj(dQ/da) * common_amp_factor (Wirtinger)
# But we need ∂Q/∂Re(ck) and ∂Q/∂Im(ck)
grad_ck_complex = np.sum(dQ_da_flat * common_amp_factor, axis=0)
grad_ck_real = 2 * np.real(grad_ck_complex)  # Factor of 2!
grad_ck_imag = -2 * np.imag(grad_ck_complex)  # Factor of 2!
```

### Evidence
```
ck[0]: ana=-0.004056 (missing factor ~5)
       num=-0.019897
       Ratio: 4.9x
```

## Bug #2: Incorrect Chain Rule for bw_p Gradient

### Location
- `numpy_kernel_fully_merged.py`: Line 178

### The Bug
```python
# Current (WRONG):
dQ_dbw_p = dQ_da_flat * ck * (-one_over_bw) * common_amp_factor
```

### Why It's Wrong
This formula mixes up the derivative of `1/bw_p` incorrectly:
- `a = ck * (1/bw_p) * fa * fl_p`
- `∂a/∂bw_p = -ck / bw_p² * fa * fl_p`
- Current code: `ck * (-1/bw_p) * (1/bw_p * fa * fl_p)` = `-ck / bw_p² * fa * fl_p` ✓

Actually, this part looks CORRECT! But wait... there's still an issue with Wirtinger calculus because `bw_p` is complex.

### The Fix
For complex `bw_p`, we need to account for both ∂Q/∂bw_p and ∂Q/∂bw_p*:

```python
# dQ/dbw_p via Wirtinger calculus
d_a_d_bw_p = -ck * common_amp_factor / bw_p  # This is ∂a/∂bw_p
dQ_dbw_p = np.real(dQ_da_flat * d_a_d_bw_p)  # For real Q, need real part
```

## Bug #3: Scalar Gradients Are Complex (Should Be Real)

### Location
- `numpy_kernel.py`: Lines 253-283
- `numpy_kernel_fully_merged.py`: Lines 239-261

### The Bug
```python
# Current (WRONG):
dQ_dGamma = np.sum(dQ_dpb_bar * d_pb_dgp * dgp_dGamma +
                   dQ_dpb_bar * d_pb_dgm * dgm_dGamma + ...)
# Returns COMPLEX number!
```

### Why It's Wrong
Gamma, Delta_Gamma, Delta_m are **REAL scalar parameters**. Their gradients MUST be real.

The issue: `gp` and `gm` are complex, so `dgp_dGamma`, `dgm_dGamma` are complex.

But `d(pb)/d(gp)` should also account for the complex structure!

### Current Formula (WRONG)
```python
d_pb_dgp = 2 * np.real(np.conj(pap) * ap)  # This is ∂|pap|²/∂Re(gp)
```

This computes the derivative w.r.t. **Re(gp)** only!

### The Fix
We need gradients w.r.t. the complex `gp`, not just Re(gp):

```python
# For pb = |pap|² = pap * pap*, where pap = gp*ap + gm*poq*am
# Treating gp as complex variable:
∂pb/∂gp = ap * pap*  (Wirtinger derivative)
∂pb/∂gp* = pap * ap* (conjugate derivative)

# For real Q:
∂Q/∂Re(gp) = ∂Q/∂gp + ∂Q/∂gp* = 2*Re(∂Q/∂gp)
```

Actually, the correct gradient for REAL parameters is:
```python
# Gamma is REAL, so:
dQ_dGamma = np.sum(dQ_dpb * d_pb_dgp * dgp_dGamma + ...)
          = np.sum(∂Q/∂pb * (∂pb/∂gp * ∂gp/∂Γ + ∂pb/∂gp* * ∂gp*/∂Γ) + ...)
```

Since `gp = u + iv` with `∂u/∂Γ` and `∂v/∂Γ` both real:
```python
dgp_dGamma = dgp_real_dGamma + 1j * dgp_imag_dGamma
dgpc_dGamma = dgp_real_dGamma - 1j * dgp_imag_dGamma

∂pb/∂gp = ap * pap*
∂pb/∂gp* = ap* * pap

∂pb/∂Γ = ∂pb/∂gp * ∂gp/∂Γ + ∂pb/∂gp* * ∂gp*/∂Γ
       = ap * pap* * (u' + iv') + ap* * pap * (u' - iv')
       = 2*Re(ap * pap*) * u' - 2*Im(ap * pap*) * v'
       = 2*Re((ap * pap*) * (u' + iv'))  # This simplifies!
```

### Corrected Formula
```python
# Gradient of gp with respect to Gamma (complex)
dgp_dGamma = -time/2 * gp  # Line 230 (CORRECT)

# Gradient of pb with respect to gp (Wirtinger)
# pb = |gp*ap + ...|²
# ∂pb/∂gp* = (gp*ap + ...) * ap* (derivative w.r.t. conjugate)
d_pb_dgp_conj = pap * np.conj(ap)  # This is what we need!

# Gradient of Q w.r.t. Gamma (REAL)
dQ_dGamma = np.sum(
    dQ_dpb * 2 * np.real(d_pb_dgp_conj * dgp_dGamma) + ...
)
```

## Bug #4: Missing Factor in Time Evolution Gradients

### Location  
- `numpy_kernel.py`: Lines 233-240
- `numpy_kernel_fully_merged.py`: Lines 230-237

### The Bug
```python
# Current (WRONG for real Gamma parameter):
dgp_dGamma = -time_half * gp
dgm_dGamma = -time_half * gm
```

This gives complex gradients, but Gamma is REAL!

### Analysis
For `gp = (eL + eH)/2` where:
- `eL = exp(-1j*t*(-Δm/2 - 1j*(Γ + ΔΓ/2)/2))`
- `eH = exp(-1j*t*(+Δm/2 - 1j*(Γ - ΔΓ/2)/2))`

The derivative `∂gp/∂Γ` IS complex (because gp is complex).

But when computing `∂Q/∂Γ`, we need:
```python
∂Q/∂Γ = ∂Q/∂Re(gp) * ∂Re(gp)/∂Γ + ∂Q/∂Im(gp) * ∂Im(gp)/∂Γ
```

NOT:
```python
∂Q/∂Γ = ∂Q/∂gp * ∂gp/∂Γ  (WRONG!)
```

### The Fix
```python
# Compute gradients w.r.t. real and imaginary parts separately
gp_real = np.real(gp)
gp_imag = np.imag(gp)
gm_real = np.real(gm)
gm_imag = np.imag(gm)

# Derivatives of Re(gp) and Im(gp) w.r.t. Gamma
dgp_real_dGamma = -time/2 * gp_real + time/2 * gp_imag  # From real/imag parts of exp
dgp_imag_dGamma = -time/2 * gp_imag - time/2 * gp_real
dgm_real_dGamma = -time/2 * gm_real + time/2 * gm_imag
dgm_imag_dGamma = -time/2 * gm_imag - time/2 * gm_real

# Gradient of pb w.r.t. Re(gp) and Im(gp)
d_pb_dgp_real = 2 * np.real(pap * ap)
d_pb_dgp_imag = 2 * np.imag(pap * ap)

# Gradient of Q w.r.t. Gamma
dQ_dGamma = np.sum(
    dQ_dpb * (d_pb_dgp_real * dgp_real_dGamma + d_pb_dgp_imag * dgp_imag_dGamma) +
    dQ_dpbbar * (d_pbbar_dgp_real * dgp_real_dGamma + d_pbbar_dgp_imag * dgp_imag_dGamma) +
    ... # Similar for gm
)
```

## Bug #5: Incorrect Gradient for m0

### Location
- `numpy_kernel.py`: Lines 188-191
- `numpy_kernel_fully_merged.py`: Lines 204-207

### The Bug
```python
# Current (WRONG):
d_bw_dom_dm0 = 2 * m0_all - 1j * g_bw
grad_m0[m0_param_idx] += np.sum(np.real(dQ_dbw_dom[:, bw_idx] * d_bw_dom_dm0[:, bw_idx]))
```

### Why It's Wrong
The formula `∂bw_dom/∂m0 = 2*m0 - 1j*g_bw` is correct for complex derivative.

But `m0` is REAL, so the gradient should be:
```python
∂Q/∂m0 = ∂Q/∂Re(bw_dom) * ∂Re(bw_dom)/∂m0 + ∂Q/∂Im(bw_dom) * ∂Im(bw_dom)/∂m0
```

NOT:
```python
∂Q/∂m0 = Re(∂Q/∂bw_dom * ∂bw_dom/∂m0)  (WRONG!)
```

### The Correct Formula
```python
# bw_dom = m0² - m² - 1j*m0*g
# Re(bw_dom) = m0² - m² + m0*Im(g)  # WRONG! Actually = m0² - m²
# Im(bw_dom) = -m0*Re(g)             # WAIT: Let me recalculate...

# Actually:
# bw_dom = m0² - m² - 1j*m0*g_bw
# where g_bw = Re(g_bw) + 1j*Im(g_bw)
# So: bw_dom = m0² - m² - 1j*m0*Re(g_bw) + m0*Im(g_bw)
# Re(bw_dom) = m0² - m² + m0*Im(g_bw)
# Im(bw_dom) = -m0*Re(g_bw)

# Derivatives:
dRe_bw_dom_dm0 = 2*m0 + Im(g_bw)
dIm_bw_dom_dm0 = -Re(g_bw)

# Gradient:
∂Q/∂m0 = ∂Q/∂Re(bw_dom) * (2*m0 + Im(g_bw)) + ∂Q/∂Im(bw_dom) * (-Re(g_bw))
```

### The Fix
```python
# Gradient components w.r.t. real and imaginary parts of bw_dom
dQ_dRe_bw_dom = np.real(dQ_dbw_dom)  # ∂Q/∂Re(bw_dom)
dQ_dIm_bw_dom = -np.imag(dQ_dbw_dom) # ∂Q/∂Im(bw_dom) (note the sign!)

# Derivatives of bw_dom components w.r.t. m0
g_bw_real = np.real(g_bw)
g_bw_imag = np.imag(g_bw)
dRe_bw_dom_dm0 = 2 * m0_all + g_bw_imag  # For each unique bw
dIm_bw_dom_dm0 = -g_bw_real

# Gradient accumulation
for bw_idx in range(len(self.m0_index)):
    m0_param_idx = self.m0_index[bw_idx]
    grad_m0[m0_param_idx] += np.sum(
        dQ_dRe_bw_dom[:, bw_idx] * dRe_bw_dom_dm0[bw_idx] +
        dQ_dIm_bw_dom[:, bw_idx] * dIm_bw_dom_dm0[bw_idx]
    )
```

## Summary of Required Changes

### 1. For Complex Parameters (ck)
Add factor of 2 and handle real/imaginary parts separately.

### 2. For Real Parameters (m0, g0, Gamma, etc.)
Split gradient computation into real and imaginary parts of intermediate variables, then combine properly.

### 3. For Scalar Complex Parameters (poq_rho, pop_phi)
Use proper Wirtinger calculus for polar form.

### 4. Testing Strategy
After fixes, verify:
- All gradients real where expected
- Numerical gradients match with error < 1e-5
- Both implementations give identical results
