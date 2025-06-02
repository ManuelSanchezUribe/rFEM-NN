import jax.numpy as jnp
from jax import jit
import jax.experimental.sparse as sparse
from functools import partial
from jax.ops import segment_sum


@jit
def create_COO(elements, ke_values):
    NE       = elements.shape[0]
    dof_mat  = jnp.tile(elements[:, None, :], (1, 2, 1))
    dof_rows = dof_mat.reshape(NE, -1, order='C')
    dof_cols = dof_mat.reshape(NE, -1, order='F')

    rows = dof_rows.reshape(-1)
    cols = dof_cols.reshape(-1)
    ke_values_flatten = ke_values.reshape(-1)
    ke_values_flatten = ke_values_flatten.at[1:3].set(0)
    ke_values_flatten = ke_values_flatten.at[0].set(1)

    return sparse.COO((ke_values_flatten, rows, cols), shape=(NE+1, NE+1))

@partial(jit, static_argnames=['n_removes', 'n_init_rows'])
def to_csr(COO, n_removes, n_init_rows):
    row     = COO.row
    col     = COO.col
    data    = COO.data
    max_col = col.max() + 1  
    keys    = row * max_col + col

    sort_indices = jnp.argsort(keys)
    sorted_keys  = keys[sort_indices]
    sorted_data  = data[sort_indices]
    sorted_row   = row[sort_indices]
    sorted_col   = col[sort_indices]

    unique_mask     = jnp.diff(jnp.concatenate([jnp.array([-1]), sorted_keys])) != 0
    unique_indices  = jnp.nonzero(unique_mask, size = sorted_keys.shape[0]-n_removes)[0]

    inverse_indices = jnp.cumsum(unique_mask) - 1

    data_summed = segment_sum(sorted_data, inverse_indices, num_segments=len(unique_indices))

    final_row = sorted_row[unique_indices]
    final_col = sorted_col[unique_indices]
    change_mask = jnp.concatenate([jnp.array([True]), final_row[1:] != final_row[:-1]])

    indices_filas = jnp.nonzero(change_mask, size=final_row.size, fill_value=0)[0]

    indices_filas = jnp.append(indices_filas[0:n_init_rows], len(final_col))

    return sparse.CSR((data_summed, final_col, indices_filas), shape=COO.shape)

@partial(jit, static_argnames=['n_removes', 'n_init_rows'])
def to_Bcsr(COO, n_removes, n_init_rows):
    row     = COO.row
    col     = COO.col
    data    = COO.data
    max_col = col.max() + 1  
    keys    = row * max_col + col

    sort_indices = jnp.argsort(keys)
    sorted_keys  = keys[sort_indices]
    sorted_data  = data[sort_indices]
    sorted_row   = row[sort_indices]
    sorted_col   = col[sort_indices]

    unique_mask     = jnp.diff(jnp.concatenate([jnp.array([-1]), sorted_keys])) != 0
    unique_indices  = jnp.nonzero(unique_mask, size = sorted_keys.shape[0]-n_removes)[0]

    inverse_indices = jnp.cumsum(unique_mask) - 1

    data_summed = segment_sum(sorted_data, inverse_indices, num_segments=len(unique_indices))

    final_row = sorted_row[unique_indices]
    final_col = sorted_col[unique_indices]
    change_mask = jnp.concatenate([jnp.array([True]), final_row[1:] != final_row[:-1]])

    indices_filas = jnp.nonzero(change_mask, size=final_row.size, fill_value=0)[0]

    indices_filas = jnp.append(indices_filas[0:n_init_rows], len(final_col))

    return sparse.BCSR((data_summed, final_col, indices_filas), shape=COO.shape)