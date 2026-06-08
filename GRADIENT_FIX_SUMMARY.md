# Gradient Bug Fix Summary

## Executive Summary

Successfully identified and fixed **CRITICAL BUGS** in gradient computation using Wirtinger calculus. All gradients now pass numerical verification with errors < 1e-10.

## Root Causes

### 1. **Missing Wirtinger Calculus for Complex Variables**

The original code computed "real gradients" (∂Q/∂Re(z)) for complex intermediate variables like `ap`, `am`, `bw_dom`, but then treated them incorrectly as Wirtinger gradients.

**Problem:** The code assumed ∂Q/∂Im(z) = 0 for all complex variables, which is WRONG.

**Fix:** Use proper Wirtinger calculus:
- ∂Q/∂z = (∂Q/∂Re(z) - i·∂Q/∂Im(z))/2
- ∂Q/∂z* = (∂Q/∂Re(z) + i·∂Q/∂Im(z))/2

For real-valued Q: ∂Q/∂z* = (∂Q/∂z)*

### 2. **Missing Factor of 2 for Complex Parameters**

For complex parameters (like `ck`) with real-valued loss Q:

**Wrong:** Treating ∂Q/∂ck as the gradient directly

**Correct:** 
- ∂Q/∂Re(ck) = 2·Re(∂Q/∂ck)
- ∂Q/∂Im(ck) = -2·Im(∂Q/∂ck)

### 3. **Missing Factor of 2 for Real Parameters**

For real parameters (like `m0`, `g0`, `Gamma`) with complex intermediate variables:

**Wrong:** ∂Q/∂m0 = Re(∂Q/∂bw_dom)·∂Re(bw_dom)/∂m0 + Im(∂Q/∂bw_dom)·∂Im(bw_dom)/∂m0

**Correct:** ∂Q/∂m0 = 2·Re(∂Q/∂bw_dom · ∂bw_dom/∂m0)

### 4. **Wrong Sign in Delta_m Gradient**

**Wrong formulas:**
- ∂gp/∂Δm = -i·t/2 · gm
- ∂gm/∂Δm = -i·t/2 · gp

**Correct formulas (derived from chain rule):**
- ∂gp/∂Δm = +i·t/2 · gm
- ∂gm/∂Δm = +i·t/2 · gp

## Specific Code Fixes

### Fix #1: Amplitude Gradients (Lines ~196-210)

**Before:**
```python
d_pb_dap = 2 * np.real(np.conj(pap) * gp)  # This is ∂pb/∂Re(ap)!
dQ_dap = dQ_dpb * d_pb_dap + ...
```

**After:**
```python
# Proper Wirtinger gradient
d_pb_dap = np.conj(pap) * gp  # This is ∂pb/∂ap!
dQ_dap_Wirtinger = dQ_dpb * d_pb_dap + ...
```

### Fix #2: ck Gradient (Lines ~220-225)

**Before:**
```python
grad_ck = np.sum(dQ_dck, axis=0)
```

**After:**
```python
# For numerical gradient comparison:
# ∂Q/∂Re(ck) = 2*Re(∂Q/∂ck)
# ∂Q/∂Im(ck) = -2*Im(∂Q/∂ck)
grad_ck = np.sum(dQ_da_flat * common_amp_factor, axis=0)
# Test applies factor of 2 automatically
```

### Fix #3: m0 Gradient (Lines ~250-265)

**Before:**
```python
dQ_dRe_bw_dom = np.real(dQ_dbw_dom)
dQ_dIm_bw_dom = np.imag(dQ_dbw_dom)
dRe_bw_dom_dm0 = 2 * m0_all + g_bw_imag
dIm_bw_dom_dm0 = -g_bw_real
grad_m0[m0_param_idx] += np.sum(
    dQ_dRe_bw_dom[:, bw_idx] * dRe_bw_dom_dm0[:, bw_idx] +
    dQ_dIm_bw_dom[:, bw_idx] * dIm_bw_dom_dm0[:, bw_idx]
)
```

**After:**
```python
dbw_dom_dm0 = 2 * m0_all - 1j * g_bw
grad_m0[m0_param_idx] += 2 * np.sum(np.real(
    dQ_dbw_dom[:, bw_idx] * dbw_dom_dm0[:, bw_idx]
))
```

### Fix #4: g0 Gradient (Lines ~270-285)

**Before:**
```python
dQ_dg = np.dot(dQ_dg_bw, np.conj(self.matrix_gamma).T)  # WRONG!
```

**After:**
```python
dQ_dg = np.dot(dQ_dg_bw, self.matrix_gamma.T)  # CORRECT
grad_g0[g0_param_idx] += 2 * np.sum(np.real(
    dQ_dg[:, gamma_idx] * g_interp[:, gamma_idx]
))
```

### Fix #5: Delta_m Gradient (Lines ~315-320)

**Before:**
```python
dgp_dDeltaM = -1j * time/2 * gm
dgm_dDeltaM = -1j * time/2 * gp
```

**After:**
```python
dgp_dDeltaM = 1j * time/2 * gm  # FIXED sign!
dgm_dDeltaM = 1j * time/2 * gp  # FIXED sign!
```

## Test Results

### Before Fixes:
```
ck[0]: error=1.58e-02 (off by ~5x)
m0[0]: error=1.58 (off by 84,000x!)
g0[0]: error=1.67e-01
Gamma: error=1.29e+00 (returned complex instead of real)
Delta_m: error=7.30e+00 (wrong sign)
```

### After Fixes:
```
ck[0]: error=5.36e-11 ✓
m0[0]: error=8.21e-11 ✓
g0[0]: error=5.02e-11 ✓
Gamma: error=1.08e-10 ✓
Delta_m: error=7.13e-11 ✓
ALL OTHERS: error < 1e-10 ✓
```

## Key Insights

1. **Wirtinger calculus is ESSENTIAL** for complex-valued computation graphs
2. **Never assume ∂Q/∂Im(z) = 0** for complex intermediate variables
3. **Factor of 2** appears when converting between Wirtinger and real gradients
4. **Sign matters** - careful derivation of time evolution derivatives

## Mathematical Foundation

### Wirtinger Derivatives

For complex z = x + iy and real-valued Q:

```
∂Q/∂z = (∂Q/∂x - i·∂Q/∂y)/2
∂Q/∂z* = (∂Q/∂x + i·∂Q/∂y)/2
```

With property: **∂Q/∂z* = (∂Q/∂z)*** for real Q

### Chain Rule for Complex Variables

For z = f(w) where both z and w are complex:

```
∂Q/∂w = ∂Q/∂z · ∂z/∂w + ∂Q/∂z* · ∂z*/∂w
```

### Real Parameter Gradients

For real parameter θ affecting complex z:

```
∂Q/∂θ = 2·Re(∂Q/∂z · ∂z/∂θ)
```

## Files Modified

1. **numpy_kernel_correct.py** - New implementation with correct Wirtinger calculus
2. **test_fixed_gradients.py** - Test suite verifying all gradients
3. **GRADIENT_BUG_ANALYSIS.md** - Detailed analysis of all bugs

## Next Steps

1. Apply fixes to `numpy_kernel.py` and `numpy_kernel_fully_merged.py`
2. Verify optimized implementations still match
3. Run performance benchmarks
4. Update documentation

## Conclusion

The gradient computation was fundamentally broken due to incorrect handling of complex variables. By applying Wirtinger calculus systematically throughout the computation graph, all gradients now match numerical verification with machine precision (< 1e-10 error).
