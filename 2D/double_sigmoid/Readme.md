# 2D double sigmoid

### Problem Description

This test case considers a two-dimensional self-adjoint elliptic boundary value problem on the domain $(0,1)^2$, with homogeneous Dirichlet boundary conditions at the left and bottom boundaries and Neumann conditions at the right and upper boundaries. The exact solution is manufactured as a sigmoid-type function featuring a sharp internal gradient:

$$u(x, y) = u_1(x)u_2(y),\quad u_j(t) = \arctan(\alpha(t-s_j)) + \arctan(\alpha s_j)\quad \text{for }j=1,2,$$
which leads to the forcing term:

$$f(x,y) = u_1''(x)u_2(y)+u_1(x)u_2''(y),\quad u_j''(t) = \frac{2\alpha^3(t-s_j)}{(1+(\alpha(t-s_j))^2)^2}\quad\text{for }j=1,2$$

The problem is designed to test the ability of the $r$-adaptive method to resolve high-gradient regions by adapting the mesh accordingly.

---

### Non-Parametric Case

In the non-parametric setting, the parameters are fixed as $\alpha = 10$ and $s_1 = s_2 = 0.5$. The goal is to evaluate the performance of the $r$-adaptive method compared to a uniform mesh, particularly in terms of error convergence and mesh distribution.

---

### Parametric Case

To generalize the model and test its robustness across varying solution features, a parametric study is conducted:

- $\alpha \in [1, 20]$, sampled using an inverse base-2 logarithmic scale:
    - $  \alpha = (\alpha_{\text{max}} + 1) - \text{np.logspace}(\log_2(1), \log_2(20), 20, \text{base}=2)  $
- $s_1, s_2 \in [0.1, 0.9]$, sampled linearly with 10 points.

The dataset consists of all possible combinations of the sampled $\alpha$, $s_1$ and $s_2$ values. Additionally, the corner points of the parameter space $(\alpha, s_1, s_2) = (1,0.1,0.1),\ (1,0.1,0.9),\ (1,0.9,0.1),\ (1,0.9,0.9),\ (20,0.1,0.1),\ (20,0.1,0.9),\ (20,0.9,0.1),\ (20,0.9,0.9)$ are explicitly included in the training set. The dataset is then split into 70% for training and 30% for testing.

Before training the neural network using the `example.ipynb` notebook, ensure that you run the `generate_dataset.ipynb` script to generate the required dataset.
