import jax.numpy as jnp
import numpy.polynomial.legendre as leg

def gaussian_quadrature(n):
    nodes, weights = leg.leggauss(n)
    return nodes, weights

class Integration_info:
    def __init__(self):
        number_of_points = 2
        nodes_int, weights_int = gaussian_quadrature(number_of_points)

        phi0 = lambda x, y: 0.25*(1-x)*(1-y)
        phi1 = lambda x, y: 0.25*(1+x)*(1-y)
        phi2 = lambda x, y: 0.25*(1+x)*(1+y)
        phi3 = lambda x, y: 0.25*(1-x)*(1+y)

        X, Y = jnp.meshgrid(nodes_int, nodes_int, indexing="ij")

        values_phi0_ = phi0(X,Y)
        values_phi1_ = phi1(X,Y)
        values_phi2_ = phi2(X,Y)
        values_phi3_ = phi3(X,Y)

        wx, wy = jnp.meshgrid(weights_int, weights_int, indexing="ij")

        self.values_phi0 = values_phi0_
        self.values_phi1 = values_phi1_
        self.values_phi2 = values_phi2_
        self.values_phi3 = values_phi3_
        self.wx = wx
        self.wy = wy
        self.nodes = nodes_int
        self.weights = weights_int