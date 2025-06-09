import jax.numpy as jnp

class Problem:
    def __init__(self, params):
        """
        Initializes the 2D elliptic problem.

        Parameters:
        - f (callable): Right-hand side function f(x).
        - g0 (float): Dirichlet boundary condition at x=0.
        - g1 (float): Dirichlet boundary condition at x=1.
        - sigma (callable): Coefficient function sigma(x).
        - u (callable, optional): Solution function (if known, default is None).
        """
        self.a     = 0.0  # Left endpoint of the domain
        self.b     = 1.0  # Right endpoint of the domain
        self.sigma = 1
        self.alphax = params[0]
        self.alphay = self.alphax
        self.sx     = params[1]
        self.sy     = params[2]
        self.f      = lambda x, y: (jnp.atan(self.alphay*(y - self.sy)) + jnp.atan(self.alphay*self.sy))*2*self.alphax**2*(self.alphax*(x-self.sx))/(1 + (self.alphax*(x - self.sx))**2)**2 \
                      + (jnp.atan(self.alphax*(x - self.sx)) + jnp.atan(self.alphax*self.sx))*2*self.alphay**2*(self.alphay*(y-self.sy))/(1 + (self.alphay*(y - self.sy))**2)**2
        self.gu     = lambda x, y: (jnp.atan(self.alphax*(x-self.sx)) + jnp.atan(self.alphax*self.sx)) * self.alphay / (1 + (self.alphay*(y-self.sy))**2)
        self.gr     = lambda x, y: (jnp.atan(self.alphay*(y-self.sy)) + jnp.atan(self.alphay*self.sy)) * self.alphax / (1 + (self.alphax*(x-self.sx))**2) 