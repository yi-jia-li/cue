import jax.numpy as jnp
from jax.numpy.linalg import svd
from jax import jit


class JAXPCA:
    """
    JAX-compatible PCA implementation using Singular Value Decomposition (SVD).
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        """
        Fit PCA using SVD on input data X.
        """
        self.mean_ = jnp.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, Vt = svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

    def transform(self, X):
        """
        Project the data into PCA space.
        """
        X_centered = X - self.mean_
        return jnp.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_pca):
        """
        Reconstruct data from PCA space.
        """
        return jnp.dot(X_pca, self.components_) + self.mean_


class SpectrumPCA:
    """
    Generalized PCA compression class for JAX.
    """

    def __init__(self, n_wavelengths, n_pcas, n_batches, spectrum_filenames, 
                 parameter_filenames, parameter_index=[2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16], 
                 wav_selection=None, parameter_selection=None):
        """
        Constructor.
        """
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.spectrum_filenames = spectrum_filenames
        self.parameter_filenames = parameter_filenames
        self.parameter_index = parameter_index
        self.wav_selection = wav_selection
        self.parameter_selection = parameter_selection
        self.n_batches = n_batches
        self.n_files = len(parameter_filenames) // n_batches

        # Initialize JAXPCA
        self.PCA = JAXPCA(n_components=n_pcas)

    def _load_and_preprocess_data(self, i):
        """
        Helper function to load and preprocess data.
        """
        log_spectra = jnp.vstack([
            jnp.load(self.spectrum_filenames[j]) for j in range(self.n_files * i, self.n_files * (i + 1))
        ])
        log_spectra = jnp.where(log_spectra == 0, 1e-10, log_spectra)
        log_spectra = jnp.log10(log_spectra[:, self.wav_selection])

        parameters = jnp.vstack([
            jnp.loadtxt(self.parameter_filenames[j])[:, self.parameter_index]
            for j in range(self.n_files * i, self.n_files * (i + 1))
        ])

        if self.parameter_selection:
            selection = self.parameter_selection(parameters)
            log_spectra = log_spectra[selection]
            parameters = parameters[selection]

        return log_spectra, parameters

    def train_pca(self):
        """
        Train PCA incrementally.
        """
        log_spectrum_shift = jnp.zeros(self.n_wavelengths)
        log_spectrum_scale = jnp.ones(self.n_wavelengths)
        parameter_shift = jnp.zeros(len(self.parameter_index))
        parameter_scale = jnp.ones(len(self.parameter_index))

        for i in range(self.n_batches):
            log_spectra, parameters = self._load_and_preprocess_data(i)

            # Update shifts and scales
            log_spectrum_shift += jnp.mean(log_spectra, axis=0) / self.n_batches
            log_spectrum_scale += jnp.std(log_spectra, axis=0) / self.n_batches
            parameter_shift += jnp.mean(parameters, axis=0) / self.n_batches
            parameter_scale += jnp.std(parameters, axis=0) / self.n_batches

            # Normalize and fit PCA
            normalized_log_spectra = (log_spectra - log_spectrum_shift) / log_spectrum_scale
            self.PCA.fit(normalized_log_spectra)

        self.log_spectrum_shift = log_spectrum_shift
        self.log_spectrum_scale = log_spectrum_scale
        self.parameter_shift = parameter_shift
        self.parameter_scale = parameter_scale

    def transform_and_stack_training_data(self, filename, retain=False):
        """
        Transform and stack training data.
        """
        training_pca = []
        training_parameters = []

        for i in range(self.n_batches):
            log_spectra, parameters = self._load_and_preprocess_data(i)

            normalized_log_spectra = (log_spectra - self.log_spectrum_shift) / self.log_spectrum_scale
            transformed_pca = self.PCA.transform(normalized_log_spectra)
            training_pca.append(transformed_pca)
            training_parameters.append(parameters)

        training_pca = jnp.vstack(training_pca)
        training_parameters = jnp.vstack(training_parameters)

        # Save data
        jnp.save(filename + '_pca.npy', training_pca)
        jnp.save(filename + '_parameters.npy', training_parameters)

        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters

    def validate_pca_basis(self, spectrum_filename, parameter_filename=None):
        """
        Validate PCA basis.
        """
        log_spectra = jnp.vstack([jnp.load(f) for f in spectrum_filename])
        log_spectra = jnp.where(log_spectra == 0, 1e-10, log_spectra)
        log_spectra = jnp.log10(log_spectra[:, self.wav_selection])

        normalized_log_spectra = (log_spectra - self.log_spectrum_shift) / self.log_spectrum_scale
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = self.PCA.inverse_transform(log_spectra_pca)

        return log_spectra, log_spectra_pca, log_spectra_in_basis
