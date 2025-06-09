# L-shape domain 

### Problem Description

This test case considers a two-dimensional self-adjoint elliptic boundary value problem on the domain $(0,1)^2\setminus ([0.5,1]\times [0,0.5])$, with homogeneous Dirichlet boundary conditions on the entire boundary and a multi-material domain.
The forcing is $f(x,y) = 1$ and the exact solution is not known analytically. We fixed nodes with $x=0.5$ and $y=0.5$ coodinates.

---

### Non-Parametric Case

In the non-parametric setting, we solve $\Delta u=1$. The goal is to evaluate the performance of the $r$-adaptive method compared to a uniform mesh, particularly in terms of error convergence and mesh distribution.

---

### Parametric Case

To generalize the model and test its robustness across varying solution features, a parametric study is conducted, now we solve $\Delta (\sigma u) = 1$, with 

$$ \sigma(x,y) = \begin{cases}
\sigma_0=1 \quad (x,y)\in (0,0.5)\times (0.5,1),\\
\sigma_1 \quad (x,y)\in (0,0.5)^2,\\
\sigma_2 \quad (x,y)\in (0.5,1)^2,
\end{cases} $$
and 
- $\sigma_1,\sigma_2 \in [0.1, 10]$, sampled using base-10 logarithmic scale:
    - $  \sigma_{k,j} = 10^{\beta_j} \quad \text{where }\beta_j \text{ are equispaced in }[-1,1]  $

The dataset consists of all possible combinations of the sampled $\sigma_1$ and $\sigma_2$ values. Additionally, the corner points of the parameter space $(\sigma_1, \sigma_2) = (0.1,0.1),\ (0.1,10),\ (10,0.1),\ (10,10)$ are explicitly included in the training set. The dataset is then split into 70% for training and 30% for testing.

Before training the neural network using the `example.ipynb` notebook, ensure that you run the `generate_dataset.ipynb` script to generate the required dataset.
