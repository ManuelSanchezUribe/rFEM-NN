###################################################################################################
# An r-adaptive finite element method using Neural Networks for parametric self-adjoint
# elliptic problems.
# Date: 2025-02-06
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ignacio Tapia y Manuel Sánchez
###################################################################################################


# Falta añadir descripciones
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
    return jnp.array([[1, -1], [-1, 1]]) / h

@jit
def element_load_exact(coords, alpha, s):
    x1, x2 = coords
    problem_test = Problem(alpha, s)
    f = problem_test.f

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
def assemble_CSR(n_elements, node_coords, element_length, n_nodes, alpha, s):
    element_nodes = jnp.stack((jnp.arange(0, n_elements), jnp.arange(1, n_elements+1)), axis=1) 
    coords        = node_coords[element_nodes]
    h_values      = element_length
    lenval        = 3*(n_nodes-2) + 4
    values        = jnp.zeros((lenval))

    def element_load_trick(coord):
        return element_load_exact(coord, alpha, s)
    
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
def apply_boundary_conditions(K, F, alpha, s):
    problem_test = Problem(alpha, s)
    bc_g0        = problem_test.g0
    bc_g1        = problem_test.g1

    F = F.at[0].set(bc_g0)
    F = F.at[-1].add(bc_g1)

    return K, F

@jit
def build_system(nodes, params):
    alpha, s = params.flatten()
    n_nodes = nodes.shape[0]
    n_elements = n_nodes - 1
    element_length = nodes[1:] - nodes[:-1]

    K, F = assemble_CSR(n_elements, nodes, element_length, n_nodes, alpha, s)
    K, F = apply_boundary_conditions(K, F, alpha, s)

    return K, F


# Solve the system
@jit
def solve(nodes, params):
    K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return nodes, u

# Solve the system
@jit
def solve_and_loss(nodes, params):
    K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss


########################################################################################
# SCIPY VERSION OF SOLVER
########################################################################################

@jax.custom_vjp
def solve_using_scipy(Kdata, Kindices, Kindptr, Kshape, F):
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
    K_batch, F_batch = build_system_vmap(nodes, params)  
    u_batch = solve_using_scipy(K_batch.data, K_batch.indices, K_batch.indptr, K_batch.shape, F_batch)

    # Compute Ku using sparse dot product
    Ku = jax.experimental.sparse.bcsr_dot_general(K_batch, u_batch, dimension_numbers=(([2], [1]), ([0], [0]))) 

    # Compute the loss for each batch
    loss_batch = 0.5 * jnp.sum(u_batch * Ku, axis=1) - jnp.sum(F_batch * u_batch, axis=1)
    
    return loss_batch

# Forward and backward pass for the custom VJP function
def solve_fwd(Kdata, Kindices, Kindptr, Kshape, F):
    u = solve_using_scipy(Kdata, Kindices, Kindptr, Kshape, F)
    return u, (Kdata, Kindices, Kindptr, Kshape, F)

def solve_using_scipy_bwd(residual, g):
    Kdata, Kindices, Kindptr, Kshape, F = residual
    return (jnp.zeros_like(Kdata), jnp.zeros_like(Kindices), jnp.zeros_like(Kindptr), None, jnp.zeros_like(F))


# Assign personalized VJP 
solve_using_scipy.defvjp(solve_fwd, solve_using_scipy_bwd)

# Create vectorized build function
build_system_vmap = jax.vmap(build_system, in_axes=(0, 0))
