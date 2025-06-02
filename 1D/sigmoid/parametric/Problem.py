import jax.numpy as jnp
class Problem:
    def __init__(self, alpha, s):
        """
        Initializes the 1D elliptic problem.

        Parameters:
        - f (callable): Right-hand side function f(x).
        - g0 (float): Dirichlet boundary condition at x=0.
        - g1 (float): Dirichlet boundary condition at x=1.
        - sigma (callable): Coefficient function sigma(x).
        - u (callable, optional): Solution function (if known, default is None).
        """
        self.a     = 0.0  # Left endpoint of the domain
        self.b     = 1.0  # Right endpoint of the domain
        self.s     = s
        self.alpha = alpha
        self.u     = lambda x: jnp.arctan(self.alpha*(x - self.s)) + jnp.arctan(self.alpha*self.s) # Solution
        self.f     = lambda x: 2 * self.alpha**3 * (x - self.s) / ( (1 + (self.alpha * (x - self.s))**2)**2 )
        self.g0    = jnp.arctan(self.alpha*(self.a - self.s)) + jnp.arctan(self.alpha*self.s) # Dirichlet at x=0
        self.g1    = self.alpha / (1 + (self.alpha*(self.b-self.s))**2)  # Neumann at x=1
        self.sigma = lambda x: 1
