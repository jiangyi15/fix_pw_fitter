Partial wave analysis fitter when shape is fixed.
=================================================

The package is to solve the problem as

$$\vartheta =\arg\min (- \ln L) = \arg\min\[-\sum_{j} w_j \ln \frac{|\sum_i c_i(\vartheta) f_{ij} |^2}{\sum_k |\sum_i c_i(\vartheta) f_{ik} |^2} \].$$

$f_{ij} = f_i(x_j)$ is the fixed shape. $c_i(\vartheta)$ is the complex number coupling of $f_{i}(x)$.
It could be some combination of real fit parameters $\vartheta$.

The analytic gradient is simple based on the complex number.
Here we define some simple intermediate variable, and using Einstein summation convention,

$$A_j = c_i F_{ij}.$$

$$P_j = |A_j|^2 = A_j A_j^\star.$$

$$M_{ik} = F_{ij} F_{kj}^\star.$$

$$N = c_i M_{ik} c_k^\star.$$

We can get the gradient is

$$\frac{\partial P_j }{\partial c_i} = F_{ij}A_j^\star.$$

$$\frac{\partial \ln P_j }{\partial c_i} = \frac{F_{ij}A_j^\star}{P_j} = \frac{F_{ij}}{A_j}.$$

$$\frac{\partial N }{\partial c_i} = M_{ik} c_{k}^\star.$$

$$\frac{\partial \ln N }{\partial c_i} = \frac{M_{ik} c_{k}^\star}{N}.$$

then the total gradient of $c_i$ is

$$\frac{\partial \ln L}{\partial c_i} = w_j \frac{\partial \ln P_j }{\partial c_i} - w_j \frac{\partial \ln N }{\partial c_i}.$$

Base on the complex grdients relation $\frac{\partial \ln L}{\partial c_i^\star}=(\frac{\partial \ln L }{\partial c_i})^\star$, we can convert it to real value using

$$\frac{\partial \ln L}{\partial x_i}=\frac{\partial \ln L }{\partial c_i}\frac{c_i }{\partial x_i} + \frac{\partial \ln L }{\partial c_i^\star}\frac{\partial c_i^\star}{\partial x_i} = 2Re(\frac{\partial \ln L }{\partial c_i}). $$

$$\frac{\partial \ln L}{\partial y_i} = \frac{\partial \ln L }{\partial c_i}\frac{c_i }{\partial y_i} + \frac{\partial \ln L }{\partial c_i^\star}\frac{\partial c_i^\star}{\partial y_i} = -2Im(\frac{\partial \ln L }{\partial c_i}). $$

and polar version $\frac{\partial \ln L}{\partial \rho_i} = 2Re (\frac{\partial \ln L }{\partial c_i} \exp(i\phi)). $
$\frac{\partial \ln L}{\partial \phi_i} = -2\rho_i Im(\frac{\partial \ln L }{\partial c_i} \exp(i\phi)). $
