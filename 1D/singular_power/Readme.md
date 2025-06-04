# Case Study 2: Singular Power Solutions

### Problem Description

This test case addresses a one-dimensional self-adjoint elliptic boundary value problem on the domain $[0,1]$, with Dirichlet boundary conditions at $x=0$ and Neumann conditions at $x=1$. The exact manufactured solution is given by a power function with a singularity at the left boundary:

$$
u(x) = x^{\alpha},
$$

which leads to the forcing term:

$$
f(x) = -u''(x) = \alpha(1 - \alpha)x^{\alpha - 2}.
$$

This setting is designed to evaluate the effectiveness of the $r$-adaptive method in handling singularities in the solution near the domain boundary.

---

### Non-Parametric Case

In the non-parametric configuration, the parameter is fixed as $\alpha = 0.7$, resulting in a solution with a strong singularity at $x = 0$ where the derivative becomes unbounded.

---

### Parametric Case

To assess generalization over a range of singularities, a parametric study is performed by sampling $\alpha$ over the interval $[0.51, 5]$ using a base-10 logarithmic scale:

- 200 values of $\alpha$ are generated.
- The first, second, penultimate, and last values are explicitly included in the training set.

Before training the neural network using the `example.ipynb` notebook, ensure that you run the `generate_dataset.py` script to generate the required dataset.

---
