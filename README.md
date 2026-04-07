# Fixed Partial Waves Fitter

This package is used to fit partial waves analysis with fixed shape partial waves.

For the fixed partial waves, the amplitude is only related to the coupling constant. It greatly simplifies the problem of partial wave analysis.

## Formula

For the fixed partial waves, all dynamic amplitudes are cached as $F_{ijk}$, where $i$ is for the events, $j$ for the projections, $k$ for the partial waves or components. The coupling parameters are $c_k$, which are complex numbers.

The total amplitude is $A_{ij} = \sum_{k} c_{k} F_{ijk}$, and the probability density is $S_{i} = \sum_{j} | A_{ij} |^2$.

The total probability density is the combination of signal and background. The background is also a fixed shape as $B_{i}$. The ratio is another fixed constant $purity$. The total probability density is $P_i = S_{i}/ N_s * purity + B_{i}/ N_b * (1-purity)$.
$N_s$ and $N_b$ are the sum of MC samples $N_{s} = \sum_{i'} \omega_{i'} S_{i'}$, $N_{b} = \sum_{i'} \omega_{i'} B_{i'}$, where $i'$ is the index of events of MC, and $\omega_{i'}$ is the weight of MC.

To reduce the calculation, the MC value is reordered as
$$
N_{s} = \sum_{i'} \sum_{j} \omega_{i'} \sum_{k}\sum_{k'} c_{k} F_{i'jk} c_{k'}^{\*} F_{i'jk'}^{\*} =
\sum_{k}\sum_{k'} c_{k} c_{k'}^{\*} M_{kk'}
$$, where $M_{kk'}= \sum_{i'} \sum_{j}  \omega_{i'} F_{i'jk} F_{i'jk'}^{*}$.

In the fit, we minimize $-\ln L = - \sum w_i \ln P_{i}$. $w_i$ is the weight of data.

## Gradients

To reduce the calculation of gradients, we use complex number gradients.

$\partial |\sum_{k} c_k F_{ijk}|^2/\partial c_k = \sum_{k'} F_{ijk} F_{ijk}^{\*} c_{k'}^{\*} =  F_{ijk} A_{ij}^{\*}$. Using the complex conjugate relation we can directly get $\partial (-\ln L)/\partial c_k^{\*} = (\partial (-\ln L)/\partial c_k)^{\*}$. This reduces many computations. The other parts are similar and can be evaluated using chain rules. Using common sub-expressions, we can reduce much more computations.

