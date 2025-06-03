import jax.numpy as jnp
class Problem:
    def __init__(self, alpha, s):
        """
        Initializes the 1D elliptic problem.

        Parameters:
        - a (float): Left endpoint of the domain.
        - b (float): Right endpoint of the domain.
        - s (float): Parameter for the problem, affecting the boundary conditions.
        - alpha (float): Parameter for the problem, affecting the boundary conditions.
        - g0 (float): Dirichlet boundary condition at x=a.
        - g1 (float): Neumann boundary condition at x=b.
        """
        self.a     = 0.0  # Left endpoint of the domain
        self.b     = 1.0  # Right endpoint of the domain
        self.s     = s
        self.alpha = alpha
        self.g0    = jnp.arctan(self.alpha*(self.a - self.s)) + jnp.arctan(self.alpha*self.s) # Dirichlet at x=a
        self.g1    = self.alpha / (1 + (self.alpha*(self.b-self.s))**2)  # Neumann at x=b
