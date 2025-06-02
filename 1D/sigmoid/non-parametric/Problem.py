import jax.numpy as jnp
class Problem:
    def __init__(self):
        self.a     = 0.0  # Left endpoint of the domain
        self.b     = 1.0  # Right endpoint of the domain
        self.s     = 0.5
        self.alpha = 10
        self.g0    = jnp.arctan(self.alpha*(self.a - self.s)) + jnp.arctan(self.alpha*self.s) # Dirichlet at x=a
        self.g1    = self.alpha / (1 + (self.alpha*(self.b-self.s))**2)  # Neumann at x=b
