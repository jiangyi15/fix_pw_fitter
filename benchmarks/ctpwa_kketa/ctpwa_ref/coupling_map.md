# Coupling map: CTPWA ↔ this framework

Their coupling vector and ours describe the same amplitudes up to a **real,
per-wave constant**:

```
A_w^theirs(e) = alpha_w · A_w^ours(e)

    c_ours[j(w)] = alpha_w · c_theirs[w]

    alpha_w = (-1)^(L_w+1) · sqrt( |A_w^theirs|^2 / |A_w^ours|^2 )
```

The `(-1)^(L+1)` is a parity convention in their tensor/barrier normalisation: it is
**measured**, not fitted (signs are negative exactly for the L = 2 and L = 4
waves: K2*(1430), K2*(1980), K4*(2045) in both charges).

## Closed form (resonant waves)

`alpha_w` is not tabulated physics — it is the product of the two vertices'
Blatt-Weisskopf normalisations.  With the symbols the codes define — this
framework's `B'_L(q, q0, d)` (`barrier_ratio`, the normalised form factor, `P_L`
its barrier polynomial) and CTPWA's `Bf(L, q, q0, d)` (the same object, its
polynomial called `N_L`) — the factor is this framework's `B'_L` taken at
`q = q0_ref` against each vertex's physical `q0`:

```
alpha_w = C · (-1)^(L+1) · B'_L(q0_ref, q0_prod, d) · B'_L(q0_ref, q0_dec, d)

    B'_L(q, q0, d) = sqrt( P_L(z0) / P_L(z) ),   z = (q d)^2,  z0 = (q0 d)^2
    =>  B'_L(q0_ref, q0_v, d) = sqrt( P_L((d q0_v)^2) / P_L((d q0_ref)^2) )

C = 100/(4 pi) = 7.957747    d = 3.0    q0_ref = 1 GeV    L = J (both vertices)

(the constant is pi-related: C*pi = 25.0000000164, i.e. 100/(4pi) to 7e-10; it is
SPIN-INDEPENDENT -- identical for L = 1, 3 and 5 -- so it is not a (2J+1) factor
but the angular normalisation 1/sqrt(4pi) per vertex, i.e. the difference between
their spherical harmonics and this framework's cos/sin monomials.  Being global
it cancels in every ratio -- fit fractions, asymmetries -- and is absorbed by
the overall coupling normalisation.)

q0_prod = p_cm(M_J/psi; m_R, m_eta)     [K+K- sector]
          p_cm(M_J/psi; m_K, m_R)       [K eta sector]
q0_dec  = p_cm(m_R; m_K, m_K)           [K+K-]
          p_cm(m_R; m_K, m_eta)         [K eta]
```

Check: **all 27 BWR waves** give `alpha_w / pred = +/- 7.957747` with a spread
below 1e-6 relative, the sign being `(-1)^(L+1)`.  `C` is a global normalisation (identical
for every wave), so it is unobservable and can be absorbed into the coupling.

The **7 non-resonant (`ONE`) waves are the exception**: their measured `alpha`
(4.62, 5.75, 7.98, ...) is not reproduced by this form under the `q0 = 1/d`
convention their `oneRefQ0` suggests, so those seven constants remain measured
rather than derived.

## Verification

2 000 events from their own generator, both frameworks on the identical
momenta:

| check | result |
|---|---|
| one-hot per wave, all 34 | Pearson correlation 1.000000; ratio constant to median 9.4e-11 |
| random complex vector (all 34 couplings, covers every interference term) | median ratio 0.9999999987, robust scatter 1.98e-10, corr 1−4.6e-15 |
| residual | at this framework's table-interpolation floor (~1e-10) |

## The map

`w` is their index (`getAmplitudeNames()` order: grouped by J within each
topology, non-resonant terms inside their J group), `ours` is this framework's
`get_ck_map()` index.  The charge blocks are **crossed**: their 14–23 are the
K⁻η resonances (`Jpsi→Kp+R, R→Km+eta`), ours 14–23 are the K⁺η ones.

| w | ours | alpha | their wave | our wave |
|---|---|---|---|---|
| 0 | 0 | +3.4776 | eta phi1020 | phi1020.eta / phi1020→Kp.Km |
| 1 | 1 | +5.1863 | eta omega1420 | omega1420.eta / omega1420→Kp.Km |
| 2 | 2 | +5.7204 | eta phi1680 | phi1680.eta / phi1680→Kp.Km |
| 3 | 3 | +5.7729 | eta X1750 | X1750.eta / X1750→Kp.Km |
| 4 | 4 | +5.0539 | eta omega2220 | omega2220.eta / omega2220→Kp.Km |
| 5 | 11 | +4.6187 | eta NR1_KK | NR1_KK.eta / NR1_KK→Kp.Km |
| 6 | 5 | +4.5152 | eta omega3_1670 | omega3_1670.eta / omega3_1670→Kp.Km |
| 7 | 6 | +4.2942 | eta phi3_1850 | phi3_1850.eta / phi3_1850→Kp.Km |
| 8 | 7 | +4.4893 | eta rho3_1690 | rho3_1690.eta / rho3_1690→Kp.Km |
| 9 | 8 | +4.1272 | eta rho3_1990 | rho3_1990.eta / rho3_1990→Kp.Km |
| 10 | 9 | +3.8644 | eta rho3_2250 | rho3_2250.eta / rho3_2250→Kp.Km |
| 11 | 12 | +5.7522 | eta NR3_KK | NR3_KK.eta / NR3_KK→Kp.Km |
| 12 | 10 | +5.5253 | eta rho5_2350 | rho5_2350.eta / rho5_2350→Kp.Km |
| 13 | 13 | +7.9763 | eta NR5_KK | NR5_KK.eta / NR5_KK→Kp.Km |
| 14 | 24 | +5.0548 | Kp Kst1410 | Kp.K1st1410m / K1st1410m→Km.eta |
| 15 | 25 | +5.7134 | Kp Kst1680 | Kp.K1st1680m / K1st1680m→Km.eta |
| 16 | 26 | +4.5443 | Kp Kst2380 | Kp.K1st2380m / K1st2380m→Km.eta |
| 17 | 32 | +4.6560 | Kp NR1_Keta | Kp.NR1_Kmeta / NR1_Kmeta→Km.eta |
| 18 | 27 | **−4.2114** | Kp K2st1430 | Kp.K2st1430m / K2st1430m→Km.eta |
| 19 | 28 | **−4.4921** | Kp K2st1980 | Kp.K2st1980m / K2st1980m→Km.eta |
| 20 | 29 | +4.4096 | Kp K3st1780 | Kp.K3st1780m / K3st1780m→Km.eta |
| 21 | 33 | +5.8737 | Kp NR3_Keta | Kp.NR3_Kmeta / NR3_Kmeta→Km.eta |
| 22 | 30 | **−4.7017** | Kp K4st2045 | Kp.K4st2045m / K4st2045m→Km.eta |
| 23 | 31 | +5.5865 | Kp K5st2380 | Kp.K5st2380m / K5st2380m→Km.eta |
| 24 | 14 | +5.0548 | Km Kst1410 | Km.K1st1410p / K1st1410p→Kp.eta |
| 25 | 15 | +5.7134 | Km Kst1680 | Km.K1st1680p / K1st1680p→Kp.eta |
| 26 | 16 | +4.5443 | Km Kst2380 | Km.K1st2380p / K1st2380p→Kp.eta |
| 27 | 22 | +4.6560 | Km NR1_Keta | Km.NR1_Kpeta / NR1_Kpeta→Kp.eta |
| 28 | 17 | **−4.2114** | Km K2st1430 | Km.K2st1430p / K2st1430p→Kp.eta |
| 29 | 18 | **−4.4921** | Km K2st1980 | Km.K2st1980p / K2st1980p→Kp.eta |
| 30 | 19 | +4.4096 | Km K3st1780 | Km.K3st1780p / K3st1780p→Kp.eta |
| 31 | 23 | +5.8737 | Km NR3_Keta | Km.NR3_Kpeta / NR3_Kpeta→Kp.eta |
| 32 | 20 | **−4.7017** | Km K4st2045 | Km.K4st2045p / K4st2045p→Kp.eta |
| 33 | 21 | +5.5865 | Km K5st2380 | Km.K5st2380p / K5st2380p→Kp.eta |

Per-charge blocks are identical in magnitude (|alpha| = 5.0548, 5.7134, 4.5443, …), as
charge symmetry requires: their `w` and `w+10` (14↔24, …) carry the same
`|alpha|`.

## Reproducing (procedure; the scan harness is not shipped)

The map is produced by three passes — this needs both frameworks, which cannot
coexist in one process (their backends are mutually exclusive and ours is the
`tabpwa` install), so:

1. **theirs** (`CTPWA_DEVICE=cpu`, their env): for each `w` in `0..33`, call
   `ana.getIntensity(momenta, p)` with `p[w] = 1` and all else 0 — **`p[w]`, not
   `p[2w]`**, their vector is `[Re_p, Im_p, theta]` (`Analysis.cu:752`); then
   with `p[0] = p[w] = 1` for the interference term.  Use
   `polarization: ["1","-1"]` (their default is 3) and ~2000 events; ~15 s per
   call on the CPU backend.
2. **ours**: `model.build_event_data(momenta, kc)` on the same events, then
   `compute(params, handle, norm=None)` with `ck[w] = 1` and with
   `ck[0] = ck[j] = 1`, for `j = 0..33`.
3. **merge** (pure numpy): match waves by Pearson correlation of the one-hot
   profiles (`|A_w|^2`, mean-subtracted — a dot product of max-normalised
   spiky profiles is *not* a similarity measure), take
   `alpha^2 = median(A_th/A_ou)`, and get the sign from
   `cross_th/cross_ou = alpha_0*alpha_w` with `cross = I(0,w) - I(0) - I(w)`.
   Validation: one random complex coupling vector, translated with
   `c_ours[j] = alpha_w c_theirs[w]`, must reproduce their intensity.

Two traps that cost time here: the split parameter layout above, and the
default 3 polarizations (which inflates the fixed column by ~1.5x and changes
nothing else).
