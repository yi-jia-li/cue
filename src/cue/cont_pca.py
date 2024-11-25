import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from scipy.linalg import svd


class SpectrumPCA:
    """
    SPECULATOR PCA compression class rewritten in JAX.
    """

    def __init__(self, n_wavelengths, n_pcas, n_batches, spectrum_filenames,
                 parameter_filenames, parameter_index=[2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16], logCO=True, parameter_selection=None):
        """
        Constructor.
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param n_pcas: number of PCA components
        :param spectrum_filenames: list of .npy filenames for log spectra (each one an [n_samples, n_wavelengths] array)
        :param parameter_filenames: list of .npy filenames for parameters (each one an [n_samples, n_parameters] array)
        """

        # Input parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.spectrum_filenames = spectrum_filenames
        self.n_parameters = len(parameter_index)
        self.parameter_index = parameter_index
        self.parameter_filenames = parameter_filenames
        self.n_batches = n_batches

        # PCA properties
        self.parameter_selection = parameter_selection
        self.logCONO = logCO
        self.n_files = len(self.parameter_filenames) // self.n_batches

        # Initialization
        self.pca_transform_matrix = None

    def train_pca(self):
        # Initialize shift and scale
        self.log_spectrum_shift = jnp.zeros(self.n_wavelengths)
        self.log_spectrum_scale = jnp.ones(self.n_wavelengths)
        self.parameter_shift = jnp.zeros(self.n_parameters)
        self.parameter_scale = jnp.ones(self.n_parameters)

        all_data = []

        # Loop over batches to preprocess spectra and parameters
        for i in range(self.n_batches):
            log_spectra = jnp.vstack([
                jnp.load(self.spectrum_filenames[j]) for j in range(self.n_files * i, self.n_files * (i + 1))
            ])
            log_spectra = jnp.log10(jnp.where(log_spectra == 0, 1e-37, log_spectra)[:, 122:])

            # Compute shifts and scales
            self.log_spectrum_shift += jnp.mean(log_spectra, axis=0) / self.n_batches
            self.log_spectrum_scale += jnp.std(log_spectra, axis=0) / self.n_batches
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift) / self.log_spectrum_scale
            all_data.append(normalized_log_spectra)

        all_data = jnp.vstack(all_data)
        # PCA decomposition using SVD
        u, s, vh = svd(all_data, full_matrices=False)
        self.pca_transform_matrix = vh[:self.n_pcas]

    def transform_and_stack_training_data(self, filename, retain=False):
        training_pca = []
        training_parameters = []

        for i in range(self.n_batches):
            log_spectra = jnp.vstack([
                jnp.load(self.spectrum_filenames[j]) for j in range(self.n_files * i, self.n_files * (i + 1))
            ])
            log_spectra = jnp.log10(jnp.where(log_spectra == 0, 1e-37, log_spectra)[:, 122:])
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift) / self.log_spectrum_scale
            training_pca.append(jnp.dot(normalized_log_spectra, self.pca_transform_matrix.T))

            parameters = jnp.vstack([
                np.loadtxt(self.parameter_filenames[j])[:, self.parameter_index] for j in range(self.n_files * i, self.n_files * (i + 1))
            ])
            if self.logCONO:
                parameters = parameters.at[:, -2:].set(jnp.log10(parameters[:, -2:]))
            training_parameters.append(parameters)

        training_pca = jnp.vstack(training_pca)
        training_parameters = jnp.vstack(training_parameters)

        # Save PCA and parameters
        np.save(filename + '_pca.npy', training_pca)
        np.save(filename + '_parameters.npy', training_parameters)

        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters

    def validate_pca_basis(self, spectrum_filename, parameter_filename=None):
        log_spectra = jnp.vstack([jnp.load(i) for i in spectrum_filename])
        log_spectra = jnp.log10(jnp.where(log_spectra == 0, 1e-37, log_spectra)[:, 122:])
        normalized_log_spectra = (log_spectra - self.log_spectrum_shift) / self.log_spectrum_scale
        log_spectra_pca = jnp.dot(normalized_log_spectra, self.pca_transform_matrix.T)
        log_spectra_in_basis = jnp.dot(log_spectra_pca, self.pca_transform_matrix) * self.log_spectrum_scale + self.log_spectrum_shift

        return log_spectra, log_spectra_pca, log_spectra_in_basis
