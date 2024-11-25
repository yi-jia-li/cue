import jax.numpy as jnp
from jax import vmap, jit
from scipy.interpolate import CubicSpline
from .cont_pca import SpectrumPCA
from .utils import cont_lam, logQ


class Predict:
    """
    Nebular Continuum Emission Prediction in JAX.
    :param theta: nebular parameters of n samples, (n, 12) matrix 
    :param gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio: 12 input parameters, vectors
    :param wavelength: wavelengths corresponding to the neural net output luminosities
    :output sorted wavelength, and L_nu in erg/Hz sorted by wavelength
    """

    def __init__(self, pca_basis=None, nn=None, theta=None, gammas=None, log_L_ratios=None, log_QH=None,
                 n_H=None, log_OH_ratio=None, log_NO_ratio=None, log_CO_ratio=None, 
                 wavelength=cont_lam[122:]):
        """
        Constructor.
        """
        self.pca_basis = pca_basis
        self.nn = nn
        self.n_segments = len(nn) if nn is not None else 1
        self.wavelength = jnp.array(wavelength)
        
        if theta is None:
            if log_QH is not None and jnp.size(log_QH) == 1:
                self.n_sample = 1
                self.theta = jnp.hstack([gammas, log_L_ratios, log_QH, n_H, 
                                         log_OH_ratio, log_NO_ratio, log_CO_ratio]).reshape((1, 12))
            else:
                self.n_sample = len(log_QH) if log_QH is not None else 0
                self.gammas = jnp.array(gammas) if gammas is not None else None
                self.log_L_ratios = jnp.array(log_L_ratios) if log_L_ratios is not None else None
                self.log_QH = jnp.reshape(log_QH, (len(log_QH), 1)) if log_QH is not None else None
                self.n_H = jnp.reshape(n_H, (len(n_H), 1)) if n_H is not None else None
                self.log_OH_ratio = jnp.reshape(log_OH_ratio, (len(log_OH_ratio), 1)) if log_OH_ratio is not None else None
                self.log_NO_ratio = jnp.reshape(log_NO_ratio, (len(log_NO_ratio), 1)) if log_NO_ratio is not None else None
                self.log_CO_ratio = jnp.reshape(log_CO_ratio, (len(log_CO_ratio), 1)) if log_CO_ratio is not None else None
                self.theta = jnp.hstack([self.gammas, self.log_L_ratios, self.log_QH, self.n_H, 
                                         self.log_OH_ratio, self.log_NO_ratio, self.log_CO_ratio])
        else:
            self.theta = jnp.array(theta)
            self.n_sample = len(self.theta)

    def nn_predict(self):
        """
        Perform the prediction using the neural network and PCA basis.
        """
        wavind_sorted = jnp.argsort(self.wavelength)
        fit_spectra = []

        if self.n_segments == 1:
            fit_spectra = self.pca_basis.PCA.inverse_transform(
                self.nn.log_spectrum_(self.theta)
            ) * self.nn.log_spectrum_scale_ + self.nn.log_spectrum_shift_

            fit_spectra = jnp.squeeze(fit_spectra)
            if self.n_sample == 1:
                fit_spectra = fit_spectra[wavind_sorted]
            else:
                fit_spectra = fit_spectra[:, wavind_sorted]

            self.nn_spectra = fit_spectra
        else:
            this_spec = []
            for j in range(self.n_segments):
                spec = self.pca_basis[j].PCA.inverse_transform(
                    self.nn[j].log_spectrum_(self.theta)
                ) * self.nn[j].log_spectrum_scale_ + self.nn[j].log_spectrum_shift_
                this_spec.append(spec)
            fit_spectra = jnp.hstack(this_spec)[:, wavind_sorted]
            self.nn_spectra = jnp.squeeze(jnp.array(fit_spectra))

        self.wavelength = self.wavelength[wavind_sorted]
        return self.wavelength, 10**self.nn_spectra


def get_cont(par):
    """
    A wrapper of nebular continuum emulator for SED fitting.
    """
    neb_cont = Predict(
        gammas=[par['ionspec_index1'], par['ionspec_index2'], 
                par['ionspec_index3'], par['ionspec_index4']],
        log_L_ratios=[par['ionspec_logLratio1'], par['ionspec_logLratio2'],
                      par['ionspec_logLratio3']],
        log_QH=logQ(par['gas_logu'], lognH=par['gas_logn']),
        n_H=10**par['gas_logn'],
        log_OH_ratio=par['gas_logz'],
        log_NO_ratio=par['gas_logno'],
        log_CO_ratio=par['gas_logco']
    ).nn_predict()

    cont_spec = neb_cont[1] / 3.839E33 / 10**logQ(par['gas_logu'], lognH=par['gas_logn']) * 10**par['log_qion']
    neb_cont_cs = CubicSpline(neb_cont[0], cont_spec, extrapolate=True)

    return {"normalized nebular continuum": cont_spec, "interpolator": neb_cont_cs}
