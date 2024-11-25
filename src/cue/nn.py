import jax.numpy as jnp
from flax import linen as nn
from jax import jit, grad
import optax
import pickle


class Speculator(nn.Module):
    """
    SPECULATOR model implemented in JAX/Flax.
    """
    n_parameters: int
    n_pcas: int
    pca_transform_matrix: jnp.ndarray
    parameters_shift: jnp.ndarray
    parameters_scale: jnp.ndarray
    pca_shift: jnp.ndarray
    pca_scale: jnp.ndarray
    log_spectrum_shift: jnp.ndarray
    log_spectrum_scale: jnp.ndarray
    hidden_units: list

    def setup(self):
        """
        Set up the network layers based on the architecture.
        """
        self.layers = [
            nn.Dense(units, kernel_init=nn.initializers.kaiming_normal())
            for units in self.hidden_units
        ]
        self.output_layer = nn.Dense(self.n_pcas, kernel_init=nn.initializers.kaiming_normal())

    def __call__(self, parameters):
        """
        Forward pass through the network to predict PCA coefficients.
        """
        x = (parameters - self.parameters_shift) / self.parameters_scale
        for layer in self.layers:
            x = nn.relu(layer(x))
        pca_coefficients = self.output_layer(x)
        return pca_coefficients * self.pca_scale + self.pca_shift

    def log_spectrum(self, parameters):
        """
        Predict the log spectrum given input parameters.
        """
        pca_coefficients = self(parameters)
        spectrum = jnp.dot(pca_coefficients, self.pca_transform_matrix)
        return spectrum * self.log_spectrum_scale + self.log_spectrum_shift

    @staticmethod
    def compute_loss(predicted_spectrum, true_spectrum, noise_floor):
        """
        Compute the root-mean-square error with a noise floor.
        """
        return jnp.sqrt(
            jnp.mean(
                (jnp.exp(predicted_spectrum) - true_spectrum) ** 2 / noise_floor**2
            )
        )


def create_speculator(
    n_parameters,
    n_pcas,
    pca_transform_matrix,
    parameters_shift,
    parameters_scale,
    pca_shift,
    pca_scale,
    log_spectrum_shift,
    log_spectrum_scale,
    hidden_units=[50, 50],
):
    """
    Helper function to initialize the Speculator with the given configuration.
    """
    speculator = Speculator(
        n_parameters=n_parameters,
        n_pcas=n_pcas,
        pca_transform_matrix=jnp.array(pca_transform_matrix),
        parameters_shift=jnp.array(parameters_shift),
        parameters_scale=jnp.array(parameters_scale),
        pca_shift=jnp.array(pca_shift),
        pca_scale=jnp.array(pca_scale),
        log_spectrum_shift=jnp.array(log_spectrum_shift),
        log_spectrum_scale=jnp.array(log_spectrum_scale),
        hidden_units=hidden_units,
    )
    return speculator


class SpeculatorTrainer:
    """
    Training wrapper for the Speculator model.
    """

    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)

    @jit
    def train_step(self, params, true_spectrum, noise_floor, input_params):
        """
        Perform one training step on the model.
        """
        def loss_fn(params):
            speculator = self.model.bind({"params": params})
            predicted_spectrum = speculator.log_spectrum(input_params)
            return self.model.compute_loss(predicted_spectrum, true_spectrum, noise_floor)

        grads = grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def train(self, params, dataset, noise_floor, num_epochs=100, batch_size=32):
        """
        Train the model over the specified number of epochs.
        """
        for epoch in range(num_epochs):
            for batch in dataset.batch(batch_size):
                params, self.opt_state = self.train_step(
                    params, batch["spectrum"], noise_floor, batch["parameters"]
                )
        return params


def save_speculator(speculator, filename):
    """
    Save the speculator model to a file.
    """
    with open(filename, "wb") as f:
        pickle.dump(speculator, f)


def load_speculator(filename):
    """
    Load a saved speculator model from a file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
