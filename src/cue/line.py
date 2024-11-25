import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import svd
from flax import linen as nn
from scipy.interpolate import CubicSpline
from .line_pca import SpectrumPCA
from .utils import nn_wavelength, nn_name, logQ


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


class SpeculatorNN(nn.Module):
    """
    A simple feedforward neural network using Flax.
    """
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class Predict:
    """
    Nebular Line Emission Prediction using JAX.
    """

    def __init__(self, pca_basis, nn, theta=None, gammas=None, log_L_ratios=None, log_QH=None,
                 n_H=None, log_OH_ratio=None, log_NO_ratio=None, log_CO_ratio=None,
                 wavelength=nn_wavelength, line_ind=None):
        """
        Constructor.
        """
        self.pca_basis = pca_basis
        self.nn = nn
        self.n_segments = len(nn)
        self.wavelength = jnp.array(wavelength)
        self.line_ind = line_ind or jnp.arange(len(wavelength))

        if theta is None:
            if jnp.size(log_QH) == 1:
                self.n_sample = 1
                self.theta = jnp.hstack([
                    gammas, log_L_ratios, log_QH, n_H,
                    log_OH_ratio, log_NO_ratio, log_CO_ratio
                ]).reshape((1, 12))
            else:
                self.n_sample = len(log_QH)
                self.theta = self._build_theta(
                    gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio
                )
        else:
            self.theta = jnp.array(theta)
            self.n_sample = len(self.theta)

    def _build_theta(self, gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio):
        gammas = jnp.array(gammas)
        log_L_ratios = jnp.array(log_L_ratios)
        log_QH = jnp.reshape(log_QH, (len(log_QH), 1))
        n_H = jnp.reshape(n_H, (len(n_H), 1))
        log_OH_ratio = jnp.reshape(log_OH_ratio, (len(log_OH_ratio), 1))
        log_NO_ratio = jnp.reshape(log_NO_ratio, (len(log_NO_ratio), 1))
        log_CO_ratio = jnp.reshape(log_CO_ratio, (len(log_CO_ratio), 1))
        return jnp.hstack([gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio])

    def nn_predict(self):
        """
        Predict line spectra using the neural network and PCA.
        """
        wavind_sorted = jnp.argsort(self.wavelength)
        fit_spectra = []

        if self.n_segments == 1:
            fit_spectra = self._predict_single_segment(wavind_sorted)
        else:
            fit_spectra = self._predict_multiple_segments(wavind_sorted)

        self.wavelength = self.wavelength[wavind_sorted]
        return self.wavelength, 10**fit_spectra

    def _predict_single_segment(self, wavind_sorted):
        nn_output = self.nn.log_spectrum_(self.theta)
        pca_output = self.pca_basis.inverse_transform(nn_output) * self.nn.log_spectrum_scale_ + self.nn.log_spectrum_shift_
        fit_spectra = jnp.squeeze(pca_output)
        fit_spectra = fit_spectra[wavind_sorted] if self.n_sample == 1 else fit_spectra[:, wavind_sorted]
        return fit_spectra[self.line_ind]

    def _predict_multiple_segments(self, wavind_sorted):
        this_spec = []
        for j in range(self.n_segments):
            nn_output = self.nn[j].log_spectrum_(self.theta)
            pca_output = self.pca_basis[j].inverse_transform(nn_output) * self.nn[j].log_spectrum_scale_ + self.nn[j].log_spectrum_shift_
            this_spec.append(pca_output)
        combined = jnp.hstack(this_spec)
        return jnp.squeeze(combined[:, wavind_sorted][:, self.line_ind])


def get_line(par, pca_basis, nn):
    """
    A wrapper of nebular line emulator for SED fitting.
    """
    neb_line = Predict(
        pca_basis=pca_basis, nn=nn,
        gammas=[par['ionspec_index1'], par['ionspec_index2'], par['ionspec_index3'], par['ionspec_index4']],
        log_L_ratios=[par['ionspec_logLratio1'], par['ionspec_logLratio2'], par['ionspec_logLratio3']],
        log_QH=logQ(par['gas_logu'], lognH=par['gas_logn']),
        n_H=10**par['gas_logn'],
        log_OH_ratio=par['gas_logz'],
        log_NO_ratio=par['gas_logno'],
        log_CO_ratio=par['gas_logco']
    ).nn_predict()

    line_spec = neb_line[1] / 3.839E33 / 10**logQ(par['gas_logu'], lognH=par['gas_logn']) * 10**par['log_qion']
    return {"normalized nebular line continuum": line_spec}
