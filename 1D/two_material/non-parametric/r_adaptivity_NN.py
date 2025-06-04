#########################################################################################################
# An r-adaptive finite element method using neural networks for parametric self-adjoint elliptic problems
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ángel J. Omella,
#         David Pardo, Manuel A. Sánchez, Ignacio Tapia, Carlos Uriarte
#########################################################################################################

import jax
import jax.numpy as jnp
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

from fem_system import solve_and_loss

# Set the random seed
np.random.seed(1234)
keras.utils.set_random_seed(1234)

dtype='float64' # double precision set to default in the SCR functions
jax.config.update("jax_enable_x64", True)
keras.backend.set_floatx(dtype)

# =============================================================================
#
#          Neural Network
#
# =============================================================================

class special_layer(keras.layers.Layer):
    def __init__(self, n_nodes, dimension, w_interior_initial_values = None, **kwargs):
        super().__init__(**kwargs)

        self.mobile_interior_vertices = self.add_weight(shape = (dimension, n_nodes), initializer = 'ones')
        
        if w_interior_initial_values is not None:
            self.mobile_interior_vertices.assign(w_interior_initial_values)
        else:
            self.mobile_interior_vertices.assign(jnp.ones((dimension, n_nodes)) / n_nodes)

    def call(self, inputs):
        distances = jax.nn.softmax(jnp.array(self.mobile_interior_vertices))
        nodes     = jnp.cumsum(distances)
        nodes_with_fixed_point = jnp.insert(nodes, 0, 0.5)
        nodes_with_fixed_point = jnp.insert(nodes_with_fixed_point, 0, 0)
        return nodes_with_fixed_point.sort()

def make_model(n_nodes, dimension=1, w_interior_initial_values=None):
    L = special_layer(n_nodes, dimension, w_interior_initial_values)
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)
    output = L(xvals)
    model = keras.Model(inputs=xvals, outputs=output, name='model')
    return model

class loss(keras.layers.Layer):
    def __init__(self,model,**kwargs):

        """
        Initializes the PINNS loss layer with provided parameters.

        Args:
            model (keras.Model): The neural network model for the approximate
                                    solution.
            n_pts (int): Number of integration points.
            f (function): Source - RHS of the PDE

            kwargs: Additional keyword arguments.
        """
        super(loss, self).__init__()

        self.model = model

    def call(self, inputs):

        """
        Computes the collocation - PINNs loss.

        Args:
            inputs: The input data (dummy).

        Returns:
            keras.Tensor: The loss value.
        """

        theta = self.model(jnp.array([1]))
        loss = solve_and_loss(theta)
        return loss

def make_loss_model(model):
    """
    Constructs a loss model for PINNs.

    Args:
        model (keras.Model): The neural network model for the approximate solution.
        n_pts (int): Number of integration points.

    Returns:
        keras.Model: A model with the collocation-based loss function.
    """
    xvals = keras.layers.Input(shape=(1,), name='x_input',dtype=dtype)

    # Compute the loss using the provided neural network and
    # integration parameters
    output = loss(model)(xvals)
    # Create a Keras model for the loss
    loss_model = keras.Model(inputs=xvals, outputs=output)

    return loss_model

def tricky_loss(y_pred, y_true):
    """
    A placeholder loss function that can be replaced as needed.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        float: The loss value.
    """
    # This is a placeholder loss function that can be substituted with a
    # custom loss if required.
    return y_true

# Adatative learning rate
def lr_schedule(epoch, lr):
    if epoch % 5000 == 0 and epoch != 0:
        return lr / 10
    return lr

  
  