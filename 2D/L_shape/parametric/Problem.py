import jax.numpy as jnp

class Problem:
    def __init__(self, sigma):
        """
        Initializes the 2D elliptic problem.

        Parameters:
        - f (callable): Right-hand side function f(x).
        - g0 (float): Dirichlet boundary condition at x=0.
        - g1 (float): Dirichlet boundary condition at x=1.
        - sigma (callable): Coefficient function sigma(x).
        - u (callable, optional): Solution function (if known, default is None).
        """
        self.f     = lambda x, y: 1
        self.sigma = jnp.array(sigma)
        self.sigma= jnp.insert(self.sigma, 1, 1)
        self.sigma= jnp.insert(self.sigma, 1, 1)
        self.x_sigma = 0.5
        self.y_sigma = 0.5
        



