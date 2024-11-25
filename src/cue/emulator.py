import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import svd
from flax import linen as nn
from scipy.interpolate import CubicSpline
from .utils import (cont_lam, nn_wavelength, line_old, logQ)


class JAXPCA:
    """
    JAX-compatible PCA implementation using Singular Value Decomposition.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        """
        Fit the PCA model on input data X.
        """
        self.mean_ = jnp.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, Vt = svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

    def transform(self, X):
        """
        Transform the data into the PCA space.
        """
        X_centered = X - self.mean_
        return jnp.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_pca):
        """
        Reconstruct the original data from the PCA space.
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


class Emulator:
    """
    Nebular Continuum and Line Emission Prediction in JAX.
    """

    def __init__(self, cont_pca_basis, cont_nn, line_pca_basis, line_nn,
                 theta=None,
                 ionspec_index1=19.7, ionspec_index2=5.3, ionspec_index3=1.6, ionspec_index4=0.6,
                 ionspec_logLratio1=3.9, ionspec_logLratio2=0.01, ionspec_logLratio3=0.2,
                 gas_logu=-2.5, gas_logn=2, gas_logz=0., gas_logno=0., gas_logco=0.,
                 log_qion=49.1):
        """
        Constructor.
        """
        self.cont_pca_basis = cont_pca_basis
        self.cont_nn = cont_nn
        self.line_pca_basis = line_pca_basis
        self.line_nn = line_nn
        self.cont_n_segments = len(cont_nn)
        self.line_n_segments = len(line_nn)
        self.cont_wavelength = jnp.array(cont_lam[122:])
        self.line_wavelength = jnp.array(nn_wavelength)
        self.line_ind = line_old
        self.log_qion = log_qion

        if theta is None:
            self.use_theta_arr = False
            if jnp.size(ionspec_index1) == 1:
                self.n_sample = 1
                self.ionspec_index1 = ionspec_index1
                self.ionspec_index2 = ionspec_index2
                self.ionspec_index3 = ionspec_index3
                self.ionspec_index4 = ionspec_index4
                self.ionspec_logLratio1 = ionspec_logLratio1
                self.ionspec_logLratio2 = ionspec_logLratio2
                self.ionspec_logLratio3 = ionspec_logLratio3
                self.gas_logu = gas_logu
                self.gas_logq = logQ(gas_logu, lognH=gas_logn)
                self.gas_logn = gas_logn
                self.gas_logz = gas_logz
                self.gas_logno = gas_logno
                self.gas_logco = gas_logco
            else:
                self.n_sample = len(ionspec_index1)
                self._reshape_inputs(ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4,
                                     ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3,
                                     gas_logu, gas_logn, gas_logz, gas_logno, gas_logco)
        else:
            self.theta = jnp.array(theta)
            self.use_theta_arr = True
            self.n_sample = len(self.theta)
            self.gas_logq = logQ(self.theta[:, -5:], lognH=self.theta[:, -4:])

    def _reshape_inputs(self, *inputs):
        reshaped = [jnp.reshape(x, (len(x), 1)) for x in inputs]
        (self.ionspec_index1, self.ionspec_index2, self.ionspec_index3, self.ionspec_index4,
         self.ionspec_logLratio1, self.ionspec_logLratio2, self.ionspec_logLratio3,
         self.gas_logu, self.gas_logn, self.gas_logz, self.gas_logno, self.gas_logco) = reshaped
        self.gas_logq = logQ(self.gas_logu, lognH=self.gas_logn)

    def update(self, **kwargs):
        for arg, value in kwargs.items():
            if arg in self.__dict__ and value is not None:
                setattr(self, arg, value)
        if not self.use_theta_arr:
            self._reshape_inputs(self.ionspec_index1, self.ionspec_index2, self.ionspec_index3,
                                 self.ionspec_index4, self.ionspec_logLratio1, self.ionspec_logLratio2,
                                 self.ionspec_logLratio3, self.gas_logu, self.gas_logn, self.gas_logz,
                                 self.gas_logno, self.gas_logco)
            self.theta = jnp.hstack([self.ionspec_index1, self.ionspec_index2, self.ionspec_index3,
                                     self.ionspec_index4, self.ionspec_logLratio1, self.ionspec_logLratio2,
                                     self.ionspec_logLratio3, self.gas_logq, 10**self.gas_logn,
                                     self.gas_logz, self.gas_logno, self.gas_logco])

    def predict_cont(self, wave, **kwargs):
        self.update(**kwargs)
        wavind_sorted = jnp.argsort(self.cont_wavelength)
        fit_spectra = self._predict_nn(self.cont_nn, self.cont_pca_basis, wavind_sorted)
        self.output_cont_wavelength = self.cont_wavelength[wavind_sorted]
        neb_cont_cs = CubicSpline(self.cont_wavelength, fit_spectra, extrapolate=True)
        return neb_cont_cs(wave)

    def predict_lines(self, **kwargs):
        self.update(**kwargs)
        wavind_sorted = jnp.argsort(self.line_wavelength)
        fit_spectra = self._predict_nn(self.line_nn, self.line_pca_basis, wavind_sorted)
        self.output_line_wavelength = self.line_wavelength[wavind_sorted]
        return fit_spectra

    def _predict_nn(self, nn, pca_basis, wavind_sorted):
        transformed = pca_basis.inverse_transform(nn(self.theta)) * nn.log_spectrum_scale_ + nn.log_spectrum_shift_
        return 10**(transformed - self.gas_logq + self.log_qion - jnp.log10(3.839E33))[:, wavind_sorted]
