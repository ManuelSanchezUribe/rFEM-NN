# Case Study 3: Two-Material Transmission

### Problem Description

This case considers a heterogeneous one-dimensional self-adjoint elliptic boundary value problem over the domain $[0,1]$, with Dirichlet boundary conditions at both endpoints. The manufactured solution is given by:

$$
u(x) = \frac{\sin(2\pi x)}{\sigma(x)},
$$

where the piecewise constant coefficient $\sigma(x)$ is defined as:

$$
\sigma(x) = 
\begin{cases}
\sigma_1, & \text{if } x < 0.5, \\
\sigma_2, & \text{if } x \geq 0.5.
\end{cases}
$$

The resulting forcing term is independent of $\sigma$ and takes the form:

$$
f(x) = -u''(x) = 4\pi^2 \sin(2\pi x).
$$

---

### Non-Parametric Case

In the non-parametric setting, the material coefficients are fixed as $\sigma_1 = 1$ and $\sigma_2 = 10$. The $r$-adaptive mesh incorporates the fixed node at $x = 0.5$, corresponding to the material interface, as required by the mesh sorting constraints.

---

### Parametric Case

To explore the problem's sensitivity with respect to material contrast, the parametric study fixes $\sigma_1 = 1$ and treats $\sigma_2 = \sigma$ as the varying parameter. This reduces the general two-material configuration $(\sigma_1, \sigma_2)$ to a single-parameter problem by rescaling the solution.

The parameter $\sigma_2$ is sampled over a logarithmic scale:

- $\sigma_2 \in [10^{-4}, 10^4]$, with 1000 values taken as $\sigma_2 = 10^{\beta_j}$ for uniformly spaced $\beta_j \in [-4, 4]$.
- The first, second, penultimate, and last values are explicitly included in the training set.
- The fixed node at $x = 0.5$ is enforced, leaving $N - 2$ nodes free to adapt for a mesh with $N$ elements.

Before training the neural network using the `example.ipynb` notebook, ensure that you run the `generate_dataset.py` script to generate the required dataset.

---
