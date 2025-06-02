########################################################################################
# Workshop: Coding for PDEs with Neural Networks
# Date: 2025-24-01
# Author: Danilo Aballay, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
########################################################################################

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
    return jnp.array([[1, -1], [-1, 1]]) / h

@jax.jit
def element_load_exact(coords):
    x1, x2 = coords
    problem_test = Problem()
    alpha = problem_test.alpha
    s     = problem_test.s

    h = x2 - x1
    # 1. integrate f(x)*lambda1(x), x=x1..x2; lambda1(x) = (x2-x)/(x2-x1)
    u1 = alpha*(x1-s); u2 = alpha*(x2-s)
    b_1 = (-1.0/h)*( (jnp.arctan(u2) - u2/(u2**2+1)) - (jnp.arctan(u1) - u1/(u1**2+1)))
    b_1 += (-alpha/h)*(x2-s)*( 1/(u2**2+1) - (1/(u1**2+1)))

    # 2. integrate f(x)*lambda2(x), x=x1..x2; lambda2(x) = (x1-x)/(x2-x1)
    b_2 = (1.0/h)*( (jnp.arctan(u2) - u2/(u2**2+1)) - (jnp.arctan(u1) - u1/(u1**2+1)))
    b_2 += (-alpha/h)*(s-x1)*( 1/(u2**2+1) - (1/(u1**2+1)))
    
    return jnp.array([b_1, b_2])

@partial(jax.jit, static_argnames=['n_elements', 'n_nodes'])
def assemble_CSR(n_elements, node_coords, element_length, n_nodes):
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
    problem_test = Problem()
    bc_g0        = problem_test.g0
    bc_g1        = problem_test.g1

    F = F.at[0].set(bc_g0)
    F = F.at[-1].add(bc_g1)

    return K, F

@jax.jit
def build_system(nodes):
    n_nodes = nodes.shape[0]
    n_elements = n_nodes - 1
    element_length = nodes[1:] - nodes[:-1]

    K, F = assemble_CSR(n_elements, nodes, element_length, n_nodes)
    K, F = apply_boundary_conditions(K, F)

    return K, F

# Solve the system
@jax.jit
def solve(nodes):
    K, F = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return nodes, u

# Obtain the loss function
@jax.jit
def solve_and_loss(nodes):
    K, F = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss


