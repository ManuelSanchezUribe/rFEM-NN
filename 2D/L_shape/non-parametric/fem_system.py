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
from Problem import Problem

sys.path.append(os.path.abspath('../..'))
from CSR_functions import create_COO, to_csr

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

def generate_mesh(nx, ny, x, y):
    ''' Generates a mesh grid for the finite element method with quad elements.'''
    x_coords, y_coords = jnp.meshgrid(x, y, indexing='xy')
    coords = jnp.column_stack((x_coords.flatten(), y_coords.flatten()))
    i, j = jnp.meshgrid(jnp.arange(nx-1), jnp.arange(ny-1), indexing='xy')
    ind = j * nx + i
    zero = ind; one = zero + 1
    three = zero + nx; two = three + 1
    elements = jnp.stack([zero.flatten(), one.flatten(), two.flatten(), three.flatten()], axis=-1)

    return coords, elements

@jit
def element_stiffness(coords, dir, dir_val):
    ''' Computes the exact element stiffness matrix with Dirichlet boundary conditions.'''
    # see build_system for the definition of dir and dir_val
    x1, y1, x2, y2 = coords
    hx, hy = x2 - x1, y2 - y1

    midptx = (x2 + x1) / 2
    midpty = (y2 + y1) / 2
    problem_test = Problem()
    sigma = problem_test.sigma 
    x_sigma = problem_test.x_sigma
    y_sigma = problem_test.y_sigma
    # Obtain material properties with sign functions to identify current quadrant
    sigma_c = (sigma[3] * 0.5*(jnp.sign(midpty-y_sigma)+1) + sigma[2] * 0.5*(jnp.sign(-midpty+y_sigma)+1)) * 0.5*(jnp.sign(midptx-x_sigma)+1) 
    sigma_c += (sigma[1] * 0.5*(jnp.sign(midpty-y_sigma)+1) + sigma[0] * 0.5*(jnp.sign(-midpty+y_sigma)+1)) * 0.5*(jnp.sign(-midptx+x_sigma)+1) 
    
    # Compute the stiffness matrix, adding an auxiliary row and column
    array_x = jnp.array([[2, -2, -1, 1, 0],
                        [-2, 2, 1, -1, 0],
                        [-1, 1, 2, -2, 0],
                        [1, -1, -2, 2, 0],
                        [0, 0, 0, 0, 0]]) * hy / 6 / hx
    array_x += jnp.array([[2, 1, -1, -2, 0],
                        [1, 2, -2, -1, 0],
                        [-1, -2, 2, 1, 0],
                        [-2, -1, 1, 2, 0],
                        [0, 0, 0, 0, 0]]) * hx / 6 / hy
    
    array_x *= sigma_c

    # Obtain local nodes that are not Dirichlet and compute the correction
    nodir = jnp.setdiff1d(jnp.array([0, 1, 2, 3]), dir, size=4, fill_value=4)
    dir_cumsum = array_x[nodir,:][:,dir] @ dir_val
    dir_correction = jnp.zeros(5, dtype=jnp.float64)
    dir_correction = dir_correction.at[nodir].set(dir_cumsum)

    # Apply Dirichlet boundary conditions, setting the corresponding rows and columns to zero
    # Nodes that are not Dirichlet only modify the auxiliary row and column
    array_x = array_x.at[dir,:].set(0)
    array_x = array_x.at[:,dir].set(0)
    array_x = array_x.at[dir,dir].set(1)
    
    # Return stiffness matrix and correction vector without auxiliary row and column
    return array_x[0:4,0:4], dir_correction[0:4]

@jit
def element_load(coords):
    '''Quadrature integration routine to compute element load vector.'''
    x1, y1, x2, y2   = coords
    hx, hy = x2 - x1, y2 - y1
    transf_nodes_x = 0.5 * hx * nodes_int + 0.5 * (x1 + x2)
    transf_nodes_y = 0.5 * hy * nodes_int + 0.5 * (y1 + y2)
    
    tx, ty = jnp.meshgrid(transf_nodes_x, transf_nodes_y, indexing="ij")
    jacobian = hx * hy * 0.25

    problem_test = Problem()
    f            = problem_test.f

    f_vals = f(tx, ty)
    sum0 = jnp.sum(f_vals * values_phi0_ * wx * wy * jacobian)
    sum1 = jnp.sum(f_vals * values_phi1_ * wx * wy * jacobian)
    sum2 = jnp.sum(f_vals * values_phi2_ * wx * wy * jacobian)
    sum3 = jnp.sum(f_vals * values_phi3_ * wx * wy * jacobian)

    return jnp.array([sum0, sum1, sum2, sum3])


@partial(jax.jit, static_argnames=['nx', 'ny', 'n_nodes'])
def assemble_CSR(elements, node_coords, nx, ny, n_nodes, dir_elements, dir_values):
    '''Assembles the global stiffness matrix (in CSR format) and load vector for a 2D finite element problem.'''
    coords_corners = jnp.concatenate([node_coords[elements[:,0]], node_coords[elements[:,2]]], axis=1)
    
    fe_values = jax.vmap(element_load)(coords_corners)
    ke_values, dir_cumsum_values = jax.vmap(element_stiffness, in_axes=(0, 0, 0))(coords_corners, dir_elements, dir_values)

    # Create COO matrix
    COO = create_COO(elements, ke_values, n_nodes)

    to_remove = 16 + 2*(nx-2)*6 + 2*(ny-2)*6 + 9*(nx-2)*(ny-2) 
    A_val_length = 16*elements.shape[0]

    # Convert COO to CSR
    K = to_csr(COO,A_val_length-to_remove, n_nodes)

    F = jnp.zeros(n_nodes)
    F = F.at[elements[:, 0]].add(fe_values[:, 0])
    F = F.at[elements[:, 1]].add(fe_values[:, 1])
    F = F.at[elements[:, 2]].add(fe_values[:, 2])
    F = F.at[elements[:, 3]].add(fe_values[:, 3])
    F = F.at[elements[:, 0]].add(-dir_cumsum_values[:, 0])
    F = F.at[elements[:, 1]].add(-dir_cumsum_values[:, 1])
    F = F.at[elements[:, 2]].add(-dir_cumsum_values[:, 2])
    F = F.at[elements[:, 3]].add(-dir_cumsum_values[:, 3])
    return K, F

@jit
def apply_boundary_conditions(K, F, dirichlet_nodes):
    '''Applies the boundary conditions to the global stiffness matrix and load vector.'''
    F = jnp.concatenate([F, jnp.array([0], dtype=jnp.float64)])
    F = F.at[dirichlet_nodes].set(0)

    return K, F[:-1]

@jit
def is_zero_quadrant(x, y):
    return jnp.sign(1-(1+jnp.sign(-(x-0.5)))//2).astype(int) * jnp.sign(1-(1+jnp.sign(y-0.5))//2).astype(int)

@jit
def element_dirichlet(element, dirichlet_nodes):
    return jnp.where(jnp.isin(element, dirichlet_nodes), size=4, fill_value=4)[0]

@jit
def build_system(nodes):
    '''Builds the finite element system for the given nodes.'''
    
    nx = nodes.shape[0]//2
    ny = nx
    x = nodes[:nx]
    y = nodes[nx:]

    node_coords, elements = generate_mesh(nx, ny, x, y)

    # Create a mask for Dirichlet nodes
    mask_dirichlet = jax.vmap(is_zero_quadrant, in_axes=(0,0))(node_coords[:,0], node_coords[:,1])
    mask_dirichlet = mask_dirichlet.at[0:nx].set(1)
    mask_dirichlet = mask_dirichlet.at[nx*(ny-1)+0:nx*(ny-1)+nx].set(1)
    mask_dirichlet = mask_dirichlet.at[0:nx*(ny-1)+1:nx].set(1)
    mask_dirichlet = mask_dirichlet.at[nx-1:nx*ny:nx].set(1)

    # Identify Dirichlet nodes, the ones that are not are set to nx*ny
    dirichlet_nodes = jnp.where(mask_dirichlet == 1, size=nx*ny, fill_value=nx*ny)[0]

    def element_dirichlet_trick(element):
        return element_dirichlet(element, dirichlet_nodes)
    
    # Obtain local Dirichlet nodes for each element, the ones that are not are set to 4
    dir_elements = jax.vmap(element_dirichlet_trick)(elements)

    dir_values = jnp.zeros(elements.shape, dtype=jnp.float64)

    K, F = assemble_CSR(elements, node_coords, nx, ny, nx*ny, dir_elements, dir_values)
    K, F = apply_boundary_conditions(K, F, dirichlet_nodes)

    return node_coords, K, F, dirichlet_nodes


# Solve the system
@jit
def solve(nodes):
    node_coords, K, F, dirichlet_nodes = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    return node_coords, u

# Obtain the loss function
@jit
def solve_and_loss(nodes):
    node_coords, K, F, dirichlet_nodes = build_system(nodes)
    u  = sparse.linalg.spsolve(K.data, K.indices, K.indptr, F, reorder = 0)
    loss = 0.5 * u @ (K @ u) - F @ u
    return loss