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
from jax.ops import segment_sum
import sys
import os
import numpy as np
import scipy
from Problem import Problem

# Add the path of the external folder
sys.path.append(os.path.abspath('../..'))
from CSR_functions import create_COO, to_csr, to_Bcsr

sys.path.append(os.path.abspath('..'))
from integration_info import Integration_info
integration_info = Integration_info()
values_phi0_ = integration_info.values_phi0
values_phi1_ = integration_info.values_phi1
values_phi2_ = integration_info.values_phi2
values_phi3_ = integration_info.values_phi3
wx = integration_info.wx
wy = integration_info.wy
nodes_int = integration_info.nodes
weights_int = integration_info.weights

@partial(jit, static_argnums=(0, 1))
def generate_mesh(nx, ny, x, y):
    x_coords, y_coords = jnp.meshgrid(x, y, indexing='xy')
    coords = jnp.column_stack((x_coords.flatten(), y_coords.flatten()))
    i, j = jnp.meshgrid(jnp.arange(nx-1), jnp.arange(ny-1), indexing='xy')
    ind = j * nx + i
    zero = ind
    one = zero + 1
    three = zero + nx
    two = three + 1
    elements = jnp.stack([zero.flatten(), one.flatten(), two.flatten(), three.flatten()], axis=-1)

    return coords, elements


@jit
def element_stiffness(coords, dir):
    '''Computes the exact element stiffness matrix.'''
    x1, y1, x2, y2 = coords
    hx, hy = x2 - x1, y2 - y1

    problem_test = Problem([1, 1, 1])
    sigma = problem_test.sigma

    array_x = jnp.array([[2, -2, -1, 1, 0],
                        [-2, 2, 1, -1, 0],
                        [-1, 1, 2, -2, 0],
                        [1, -1, -2, 2, 0],
                        [0, 0, 0, 0, 0]]) * hy / 6 / hx
    array_y = jnp.array([[2, 1, -1, -2, 0],
                        [1, 2, -2, -1, 0],
                        [-1, -2, 2, 1, 0],
                        [-2, -1, 1, 2, 0],
                        [0, 0, 0, 0, 0]]) * hx / 6 / hy
    
    array_x = array_x.at[dir,:].set(0)
    array_x = array_x.at[:,dir].set(0)
    array_x = array_x.at[dir,dir].set(1)
    array_y = array_y.at[dir,:].set(0)
    array_y = array_y.at[:,dir].set(0)
    array_y = array_y.at[dir,dir].set(1)
    
    return sigma * (array_x[0:4,0:4] + array_y[0:4,0:4])

@jit
def element_load(coords, params):
    '''Numerical integration routine to compute element load vector.'''
    x1, y1, x2, y2   = coords
    hx, hy = x2 - x1, y2 - y1
    transf_nodes_x = 0.5 * hx * nodes_int + 0.5 * (x1 + x2)
    transf_nodes_y = 0.5 * hy * nodes_int + 0.5 * (y1 + y2)

    tx, ty = jnp.meshgrid(transf_nodes_x, transf_nodes_y, indexing="ij")
    jacobian = hx * hy * 0.25

    problem_test = Problem(params)
    f            = problem_test.f

    f_vals = f(tx, ty)
    sum0 = jnp.sum(f_vals * values_phi0_ * wx * wy * jacobian)
    sum1 = jnp.sum(f_vals * values_phi1_ * wx * wy * jacobian)
    sum2 = jnp.sum(f_vals * values_phi2_ * wx * wy * jacobian)
    sum3 = jnp.sum(f_vals * values_phi3_ * wx * wy * jacobian)

    return jnp.array([sum0, sum1, sum2, sum3])


@partial(jax.jit, static_argnames=['nx', 'ny', 'n_nodes'])
def assemble_CSR(elements, node_coords, nx, ny, n_nodes, dir_elements, params):
    '''Assembles the global stiffness matrix (in CSR format) and load vector for a 2D finite element problem.'''
    coords_corners = jnp.concatenate([node_coords[elements[:,0]], node_coords[elements[:,2]]], axis=1)
    
    def element_load_trick(coord):
        return element_load(coord, params)
    
    fe_values = jax.vmap(element_load_trick)(coords_corners)
    ke_values = jax.vmap(element_stiffness, in_axes=(0,0))(coords_corners, dir_elements)

    # Create COO matrix
    COO = create_COO(elements, ke_values, n_nodes)

    to_remove = 16 + 2*(nx-2)*6 + 2*(ny-2)*6 + 9*(nx-2)*(ny-2) 
    A_val_length = 16*elements.shape[0]

    # Convert COO to CSR
    K = to_Bcsr(COO,A_val_length-to_remove, n_nodes)

    F = jnp.zeros(n_nodes)
    F = F.at[elements[:, 0]].add(fe_values[:, 0])
    F = F.at[elements[:, 1]].add(fe_values[:, 1])
    F = F.at[elements[:, 2]].add(fe_values[:, 2])
    F = F.at[elements[:, 3]].add(fe_values[:, 3])
    return K, F

# Apply boundary conditions
@jit
def apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, node_coords, params):
    '''Applies the boundary conditions to the global stiffness matrix and load vector.'''
    problem_test = Problem(params)
    
    def compute_neumann_contribution(i, g):
        e = neumann_edges[i, 0].astype(jnp.int64)
        n1, n2 = neumann_edges[i, 1:3].astype(jnp.int64)
        x1, y1 = node_coords[elements[e, n1], :]
        x2, y2 = node_coords[elements[e, n2], :]
        norm = 0.5 * jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        transformed_x = x1 + 0.5 * (nodes_int + 1) * (x2 - x1)
        transformed_y = y1 + 0.5 * (nodes_int + 1) * (y2 - y1)
        sum0 = jnp.sum(norm * weights_int * g(transformed_x, transformed_y) * (0.5 * (1 - nodes_int)))
        sum1 = jnp.sum(norm * weights_int * g(transformed_x, transformed_y) * (0.5 * (1 + nodes_int)))
        return elements[e, n1], sum0, elements[e, n2], sum1
    
    neumann_indices_1 = jnp.arange(neumann_edges.shape[0] // 2)
    neumann_indices_2 = jnp.arange(neumann_edges.shape[0] // 2, neumann_edges.shape[0])
    
    apply_neumann_1 = jax.vmap(lambda i: compute_neumann_contribution(i, problem_test.gr))(neumann_indices_1)
    apply_neumann_2 = jax.vmap(lambda i: compute_neumann_contribution(i, problem_test.gu))(neumann_indices_2)
    
    # Apply contributions to F
    F = F.at[apply_neumann_1[0]].add(apply_neumann_1[1])
    F = F.at[apply_neumann_1[2]].add(apply_neumann_1[3])
    F = F.at[apply_neumann_2[0]].add(apply_neumann_2[1])
    F = F.at[apply_neumann_2[2]].add(apply_neumann_2[3])
    
    # Apply Dirichlet conditions
    F = F.at[dirichlet_nodes].set(0)
    
    return K, F


@jit
def build_system(nodes, params):
    '''Builds the finite element system for the given nodes.'''
    nx = nodes.shape[0]//2
    ny = nx
    x = nodes[:nx]
    y = nodes[nx:]
    node_coords, elements = generate_mesh(nx, ny, x, y)
    dirichlet_nodes = jnp.concatenate([jnp.arange(nx),nx*jnp.arange(1,ny)])#,nx*jnp.arange(1,ny)+nx-1, nx*(ny-1)+jnp.arange(1,nx-1)])
    dir_elements = 4*jnp.ones(elements.shape, dtype=jnp.int64)

    # vertex
    dir_elements = dir_elements.at[0,:3].set(jnp.array([0, 1, 3]))
    dir_elements = dir_elements.at[nx-2, 0:2].set(jnp.array([0, 1]))
    dir_elements = dir_elements.at[(nx-1)*(ny-2), 0:2].set(jnp.array([0, 3]))
    # dir_elements = dir_elements.at[(nx-1)*(ny-2)+(nx-2), 0:3].set(jnp.array([1, 2, 3]))
    # edges
    dir_elements = dir_elements.at[1:(nx-2), 0:2].set(jnp.array([0, 1]))
    # dir_elements = dir_elements.at[(nx-1)*(ny-2)+1:(nx-1)*(ny-2)+(nx-2), 0:2].set(jnp.array([2, 3]))
    dir_elements = dir_elements.at[(nx-1):(nx-1)*(ny-2):(nx-1), 0:2].set(jnp.array([0, 3]))
    # dir_elements = dir_elements.at[(nx-1)+(nx-2):(nx-1)*(ny-2)+(nx-2):(nx-1), 0:2].set(jnp.array([1, 2]))

    ind1 = jnp.arange(ny-1, dtype=jnp.int64) * (nx-1) + (nx-2)
    ind2 = (ny-2) * (nx-1) + jnp.arange(nx-1, dtype=jnp.int64)
    local1 = jnp.append(jnp.ones(ny-1), 2 * jnp.ones(nx-1))
    local2 = jnp.append(2 * jnp.ones(ny-1), 3 * jnp.ones(nx-1))
    side = jnp.append(jnp.zeros(ny-1), jnp.ones((nx-1)))
    ind = jnp.append(ind1, ind2)
    neumann_edges = jnp.reshape(jnp.concatenate([ind, local1, local2, side], axis=0), ((nx-1) + (ny-1), 4), order='F')
   
    
    K, F = assemble_CSR(elements, node_coords, nx, ny, nx*ny, dir_elements, params)
    K, F = apply_boundary_conditions(K, F, dirichlet_nodes, neumann_edges, elements, node_coords, params)

    return node_coords, K, F


# Solve the system
@jit
def solve(nodes, params):
    '''Solves the finite element system for the given nodes.'''
    node_coords, K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return node_coords, u

# Solve the system
@jit
def solve_and_loss(nodes, params):
    '''Solves the finite element system for the given nodes and computes the Ritz energy functional.'''
    node_coords, K, F = build_system(nodes, params)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss


########################################################################################
# SCIPY VERSION OF SOLVER
########################################################################################

@jax.custom_vjp
def solve_using_scipy(Kdata, Kindices, Kindptr, Kshape, F):
    '''Solves the finite element system using SciPy's sparse solver.'''
    batch_size = Kdata.shape[0] # Assume that the batch is in the first axis
    
    def solve_single(i):
        K_data    = np.asarray(Kdata[i])
        K_indices = np.asarray(Kindices[i])
        K_indptr  = np.asarray(Kindptr[i])
        scipy_csr = scipy.sparse.csr_matrix((K_data, K_indices, K_indptr),  shape=Kshape[1:])
        scipy_vector = np.asarray(F[i])
        return scipy.sparse.linalg.spsolve(scipy_csr, scipy_vector)

    u_batch = np.stack([solve_single(i) for i in range(batch_size)], axis=0)

    return jnp.asarray(u_batch)  

@jit
def compute_loss(K_batch, u_batch, F_batch):
    '''Computes the loss for a batch.'''
    Ku = jax.experimental.sparse.bcsr_dot_general(K_batch, u_batch, dimension_numbers=(([2], [1]), ([0], [0])))

   
    loss_batch = 0.5 * jnp.sum(u_batch * Ku, axis=1) - jnp.sum(F_batch * u_batch, axis=1)
    return loss_batch


def solve_and_loss_scipy(nodes, alpha):
    '''Solves the finite element system for the given nodes using SciPy and computes the Ritz energy functional.'''
    node_coords, K_batch, F_batch = build_system_vmap(nodes, alpha)  
    u_batch = solve_using_scipy(K_batch.data, K_batch.indices, K_batch.indptr, K_batch.shape, F_batch)

    loss_batch = compute_loss(K_batch, u_batch, F_batch)
    
    return loss_batch

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
build_system_vmap = jax.vmap(build_system, in_axes=(None, 0))
