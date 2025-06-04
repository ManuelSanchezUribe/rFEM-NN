#########################################################################################################
# An r-adaptive finite element method using neural networks for parametric self-adjoint elliptic problems
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ángel J. Omella,
#         David Pardo, Manuel A. Sánchez, Ignacio Tapia, Carlos Uriarte
#########################################################################################################

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
from functools import partial
import sys
import os
from Problem import Problem

sys.path.append(os.path.abspath('../..'))
from CSR_functions import create_COO, to_csr

@jax.jit
def element_stiffness(coords):
    ''' Computes the exact element stiffness matrix.'''
    x1, x2 = coords
    h      = x2 - x1
    midpt  = (x2 + x1) / 2
    problem_test = Problem()
    sigma = problem_test.sigma
    k     = 1e32
    x_sigma = problem_test.x_sigma
    sigma_c = sigma[1] * jax.nn.sigmoid(k * (midpt - x_sigma)) + sigma[0] * (1 - jax.nn.sigmoid(k * (midpt - x_sigma)))
    return sigma_c * jnp.array([[1, -1], [-1, 1]]) / h

@jax.jit
def element_load_exact(coords):
    '''Analytical integration routine to compute element load vector.'''
    x1, x2 = coords

    h = x2 - x1
    # 1. integrate f(x)*lambda1(x), x=x1..x2; lambda1(x) = (x2-x)/(x2-x1)
    b_1 = (-2*jnp.pi*x2/h)*(jnp.cos(2*jnp.pi*x2) - jnp.cos(2*jnp.pi*x1) )
    b_1 += (2*jnp.pi/h)*( x2*jnp.cos(2*jnp.pi*x2) - x1*jnp.cos(2*jnp.pi*x1) )
    b_1 += (-1/h)*( jnp.sin(2*jnp.pi*x2) - jnp.sin(2*jnp.pi*x1) )

    # 2. integrate f(x)*lambda2(x), x=x1..x2; lambda2(x) = (x1-x)/(x2-x1)
    b_2 = (-2*jnp.pi/h)*( x2*jnp.cos(2*jnp.pi*x2) - x1*jnp.cos(2*jnp.pi*x1) )
    b_2 += (1/h)*( jnp.sin(2*jnp.pi*x2) - jnp.sin(2*jnp.pi*x1) )
    b_2 += (2*jnp.pi*x1/h)*(jnp.cos(2*jnp.pi*x2) - jnp.cos(2*jnp.pi*x1) )
    
    return jnp.array([b_1, b_2])


@partial(jax.jit, static_argnames=['n_elements', 'n_nodes'])
def assemble_CSR(n_elements, node_coords, element_length, n_nodes):
    '''Assembles the global stiffness matrix (in CSR format) and load vector for a 1D finite element problem.'''
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords        = node_coords[element_nodes]
    h_values      = element_length
    
    fe_values = jax.vmap(element_load_exact)(coords)
    ke_values = jax.vmap(element_stiffness)(coords)

    # Create COO matrix
    COO = create_COO(element_nodes, ke_values)

    # Convert COO to CSR
    K = to_csr(COO,n_nodes-2, n_nodes)

    F = jnp.zeros(n_nodes)
    F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return K, F


# Apply boundary conditions
@jax.jit
def apply_boundary_conditions(K, F):
    '''Applies the boundary conditions to the global stiffness matrix and load vector.'''
    problem_test = Problem()
    bc_g0        = problem_test.g0
    bc_g1        = problem_test.g1

    F = F.at[0].set(bc_g0)
    F = F.at[-1].set(F[-1] + 2*jnp.pi)

    return K, F

@jax.jit
def build_system(nodes):
    '''Builds the finite element system for the given nodes.'''
    n_nodes = nodes.shape[0]
    n_elements = n_nodes - 1
    element_length = nodes[1:] - nodes[:-1]

    K, F = assemble_CSR(n_elements, nodes, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)

    return K, F

# Solve the system
@jax.jit
def solve(nodes):
    '''Solves the finite element system for the given nodes.'''
    K, F = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return nodes, u

# Obtain the loss function
@jax.jit
def solve_and_loss(nodes):
    '''Solves the finite element system for the given nodes and computes the Ritz energy functional.'''
    K, F = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss


