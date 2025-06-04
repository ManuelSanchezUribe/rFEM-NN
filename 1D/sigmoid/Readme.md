# Case Study 1: Arctangent Sigmoid Function

### Problem Description

This test case considers a one-dimensional self-adjoint elliptic boundary value problem on the domain $[0,1]$, with homogeneous Dirichlet boundary conditions at $x=0$ and Neumann conditions at $(x=1)$. The exact solution is manufactured as a sigmoid-type function featuring a sharp internal gradient:

$$u(x) = \arctan(\alpha(x - s)) + \arctan(\alpha s),$$
which leads to the forcing term:

$$f(x) = -u''(x) = \frac{2\alpha^3(x - s)}{(1 + \alpha^2(x - s)^2)^2}$$

The problem is designed to test the ability of the $r$-adaptive method to resolve high-gradient regions by adapting the mesh accordingly.

---

### Non-Parametric Case

In the non-parametric setting, the parameters are fixed as $\alpha = 10$ and $s = 0.5$. The goal is to evaluate the performance of the $r$-adaptive method compared to a uniform mesh, particularly in terms of error convergence and mesh distribution.

---

### Parametric Case

To generalize the model and test its robustness across varying solution features, a parametric study is conducted:

- $\alpha \in [1, 50]$, sampled using an inverse base-2 logarithmic scale:
    - $  \alpha = (\alpha_{\text{max}} + 1) - \text{np.logspace}(\log_2(1), \log_2(50), 100, \text{base}=2)  $
- $s \in [0.2, 0.8]$, sampled linearly with 100 points.

The dataset consists of all possible combinations of the sampled $\alpha$ and $s$ values. Additionally, the corner points of the parameter space $(\alpha, s) = (1,0.2),\ (1,0.8),\ (50,0.2),\ (50,0.8)$ are explicitly included in the training set. The dataset is then split into 70% for training and 30% for testing.

Before training the neural network using the `example.ipynb` notebook, ensure that you run the `generate_dataset.py` script to generate the required dataset.

