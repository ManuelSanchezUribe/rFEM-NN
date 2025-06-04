#########################################################################################################
# An r-adaptive finite element method using neural networks for parametric self-adjoint elliptic problems
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ángel J. Omella,
#         David Pardo, Manuel A. Sánchez, Ignacio Tapia, Carlos Uriarte
#########################################################################################################

import jax
import jax.numpy as jnp
from jax import jit
import jax.experimental.sparse as sparse
from functools import partial
import sys
import os
import numpy as np
import scipy
from Problem import Problem

# Add the path of the external folder
sys.path.append(os.path.abspath('../..'))
from CSR_functions import create_COO, to_Bcsr

@jit
def element_stiffness(h):
    '''Computes the exact element stiffness matrix.'''
    return jnp.array([[1, -1], [-1, 1]]) / h

@jit
def element_load_exact(coords, alpha):
    '''Analytical integration routine to compute element load vector.'''
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
def assemble_CSR(n_elements, node_coords, element_length, n_nodes, alpha):
    '''Assembles the global stiffness matrix (in CSR format) and load vector for a 1D finite element problem.'''
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords        = node_coords[element_nodes]
    h_values      = element_length

    def element_load_trick(coord):
        return element_load_exact(coord, alpha)
    
    fe_values = jax.vmap(element_load_trick)(coords)
    ke_values = jax.vmap(element_stiffness)(h_values)

    # Create COO matrix
    COO = create_COO(element_nodes, ke_values)

    # Convert COO to CSR
    K = to_Bcsr(COO,n_nodes-2, n_nodes)

    F = jnp.zeros(n_nodes)
    F = F.at[element_nodes[:, 0]].add(fe_values[:, 0])
    F = F.at[element_nodes[:, 1]].add(fe_values[:, 1])

    return K, F

# Apply boundary conditions
@jit
def apply_boundary_conditions(K, F, alpha):
    problem_test = Problem(alpha)
    bc_g0        = problem_test.g0
    bc_g1        = problem_test.g1
    alpha        = problem_test.alpha

    F = F.at[0].set(bc_g0)
    F = F.at[-1].set(F[-1] + jnp.squeeze(alpha))

    return K, F

@jit
def build_system(nodes, alpha):
    '''Builds the finite element system for the given nodes.'''
    n_nodes     = nodes.shape[0]
    n_elements  = n_nodes - 1
    element_length = nodes[1:] - nodes[:-1]

    K, F = assemble_CSR(n_elements, nodes, element_length, n_nodes, alpha)
    K, F = apply_boundary_conditions(K, F, alpha)

    return K, F


# Solve the system
@jit
def solve(nodes, params):
    '''Solves the finite element system for the given nodes.'''
    K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return nodes, u

# Solve the system
@jit
def solve_and_loss(nodes, params):
    '''Solves the finite element system for the given nodes and computes the Ritz energy functional.'''
    K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss


########################################################################################
# SCIPY VERSION OF SOLVER
########################################################################################

@jax.custom_vjp
def solve_using_scipy(Kdata, Kindices, Kindptr, Kshape, F):
    '''Solves the finite element system using SciPy's sparse solver.'''
    batch_size = Kdata.shape[0]  # Assume that the batch is in the first axis
    
    def solve_single(i):
        K_data    = np.asarray(Kdata[i])
        K_indices = np.asarray(Kindices[i])
        K_indptr  = np.asarray(Kindptr[i])
        scipy_csr = scipy.sparse.csr_matrix((K_data, K_indices, K_indptr),  shape=Kshape[1:])
        scipy_vector = np.asarray(F[i])
        return scipy.sparse.linalg.spsolve(scipy_csr, scipy_vector)

    # Applies function along batch axis using NumPy
    u_batch = np.stack([solve_single(i) for i in range(batch_size)], axis=0)

    return jnp.asarray(u_batch)  # Convert back to JAX array

def solve_and_loss_scipy(nodes, params):
    '''Solves the finite element system for the given nodes using SciPy and computes the Ritz energy functional.'''
    K_batch, F_batch = build_system_vmap(nodes, params)  
    u_batch = solve_using_scipy(K_batch.data, K_batch.indices, K_batch.indptr, K_batch.shape, F_batch)

    # Compute Ku using sparse dot product
    Ku = jax.experimental.sparse.bcsr_dot_general(K_batch, u_batch, dimension_numbers=(([2], [1]), ([0], [0]))) 

    # Compute the loss for each batch
    loss_batch = 0.5 * jnp.sum(u_batch * Ku, axis=1) - jnp.sum(F_batch * u_batch, axis=1)
    
    return loss_batch

# Forward and backward pass for the custom VJP function
def solve_fwd(Kdata, Kindices, Kindptr, Kshape, F):
    '''Forward pass for the custom VJP function.'''
    u = solve_using_scipy(Kdata, Kindices, Kindptr, Kshape, F)
    return u, (Kdata, Kindices, Kindptr, Kshape, F)

def solve_using_scipy_bwd(residual, g):
    '''Backward pass for the custom VJP function.'''
    Kdata, Kindices, Kindptr, Kshape, F = residual
    return (jnp.zeros_like(Kdata), jnp.zeros_like(Kindices), jnp.zeros_like(Kindptr), None, jnp.zeros_like(F))


# Assign personalized VJP 
solve_using_scipy.defvjp(solve_fwd, solve_using_scipy_bwd)

# Create vectorized build function
build_system_vmap = jax.vmap(build_system, in_axes=(0, 0))
