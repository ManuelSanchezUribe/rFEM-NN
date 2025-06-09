#########################################################################################################
# An r-adaptive finite element method using neural networks for parametric self-adjoint elliptic problems
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ángel J. Omella,
#         David Pardo, Manuel A. Sánchez, Ignacio Tapia, Carlos Uriarte
#########################################################################################################


import jax.numpy as jnp
class Problem:
    def __init__(self, u=None):
        """
        Initializes the 2D elliptic problem.

        Parameters:
        - s (float): Parameter for the problem, affecting the boundary conditions.
        - alpha (float): Parameter for the problem, affecting the boundary conditions.
        - gu (float): Dirichlet boundary condition at the upper boundary.
        - gr (float): Neumann boundary condition at the right boundary.
        """
        self.alpha = 10
        self.s     = 0.05
        self.f     = lambda x, y: (jnp.atan(self.alpha*(y - self.s)) + jnp.atan(self.alpha*self.s))*2*self.alpha**2*(self.alpha*(x-self.s))/(1 + (self.alpha*(x - self.s))**2)**2 \
                    + (jnp.atan(self.alpha*(x - self.s)) + jnp.atan(self.alpha*self.s))*2*self.alpha**2*(self.alpha*(y-self.s))/(1 + (self.alpha*(y - self.s))**2)**2
        self.gu    = lambda x, y: (jnp.atan(self.alpha*(x-self.s)) + jnp.atan(self.alpha*self.s)) * self.alpha / (1 + (self.alpha*(y-self.s))**2)
        self.gr    = lambda x, y: (jnp.atan(self.alpha*(y-self.s)) + jnp.atan(self.alpha*self.s)) * self.alpha / (1 + (self.alpha*(x-self.s))**2)
        



