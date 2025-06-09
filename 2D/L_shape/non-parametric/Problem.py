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
        - f (callable): Right-hand side function f(x).
        - g0 (float): Dirichlet boundary condition at x=0.
        - g1 (float): Dirichlet boundary condition at x=1.
        - sigma (callable): Coefficient function sigma(x).
        - u (callable, optional): Solution function (if known, default is None).
        """
        self.f     = lambda x, y: 1
        self.gN    = lambda x, y: 0
        self.gS    = lambda x, y: 0
        self.gE    = lambda x, y: 0
        self.gW    = lambda x, y: 0
        self.sigma = [1, 1, 1, 1]
        self.x_sigma = 0.5
        self.y_sigma = 0.5
        



