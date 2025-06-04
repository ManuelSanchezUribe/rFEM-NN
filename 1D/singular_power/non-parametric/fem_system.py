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
def element_stiffness(h):
    ''' Computes the exact element stiffness matrix.'''
    return jnp.array([[1, -1], [-1, 1]]) / h

@jax.jit
def element_load_exact(coords):
    '''Analytical integration routine to compute element load vector.'''
    x1, x2       = coords
    problem_test = Problem()
    alpha        = problem_test.alpha

    x1, x2 = coords
    h = x2 - x1

    def branch_x1_zero(_):
        b_1 = (-alpha * x2 / h) * (x2 ** (alpha - 1) - 0.0)
        b_1 += (-(1 - alpha) / h) * (x2 ** alpha - 0.0)

        b_2 = ((1 - alpha) / h) * (x2 ** alpha - 0.0)
        b_2 += (alpha * x1 / h) * (x2 ** (alpha - 1) - 0.0)
        return jnp.array([b_1, b_2], dtype=jnp.float64)

    def branch_else(_):
        b_1 = (-alpha * x2 / h) * (x2 ** (alpha - 1) - x1 ** (alpha - 1))
        b_1 += (-(1 - alpha) / h) * (x2 ** alpha - x1 ** alpha)

        b_2 = ((1 - alpha) / h) * (x2 ** alpha - x1 ** alpha)
        b_2 += (alpha * x1 / h) * (x2 ** (alpha - 1) - x1 ** (alpha - 1))
        return jnp.array([b_1, b_2], dtype=jnp.float64)

    return jax.lax.cond(x1 == 0.0, branch_x1_zero, branch_else, operand=None)


@partial(jax.jit, static_argnames=['n_elements', 'n_nodes'])
def assemble_CSR(n_elements, node_coords, element_length, n_nodes):
    '''Assembles the global stiffness matrix (in CSR format) and load vector for a 1D finite element problem.'''
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords        = node_coords[element_nodes]
    h_values      = element_length
    
    fe_values = jax.vmap(element_load_exact)(coords)
    ke_values = jax.vmap(element_stiffness)(h_values)

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
    F = F.at[-1].set(F[-1]+0.7)

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


