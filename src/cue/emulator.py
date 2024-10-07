import numpy as np
#import glob
import tensorflow as tf
#import pickle
import dill as pickle
from .line_pca import SpectrumPCA
from .nn import Speculator
from .utils import (cont_lam, nn_wavelength, nn_name, line_old, logQ, logU)
from pkg_resources import resource_filename, resource_listdir

from .line_pca import SpectrumPCA
for this_name in nn_name:
    with open(resource_filename("cue", "data/pca_line_new_"+this_name+".pkl"), 'rb') as f:
        globals()["PCABasis_"+this_name] = pickle.load(f)
    globals()["speculator_"+this_name] = Speculator(restore=True, restore_filename=resource_filename("cue", "data/speculator_line_new_"+this_name))

line_PCABasis = [globals()["PCABasis_"+this_name] for this_name in nn_name]
line_speculator = [globals()["speculator_"+this_name] for this_name in nn_name]

from .cont_pca import SpectrumPCA
with open(resource_filename("cue", "data/pca_cont_new.pkl"), 'rb') as f:
    cont_PCABasis = pickle.load(f)
cont_speculator = Speculator(restore = True,
                             restore_filename = resource_filename("cue", "data/speculator_cont_new"))

class Emulator():
    """
    Nebular Continuum and Line Emission Prediction
    :param theta: nebular parameters of n samples, (n, 12) matrix
    :param use_stellar_ionizing:
        If true, fit the csp and to get the ionizing spectrum parameters, else read from the model
    :param ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4, ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3:
        ionizing parameters, follow the range
        ionspec_index1: [1, 42], ionspec_index2: [-0.3, 30],
        ionspec_index3: [-1, 14], ionspec_index4: [-1.7, 8],
        ionspec_logLratio1: [-1., 10.1], ionspec_logLratio2: [-0.5, 1.9], ionspec_logLratio3: [-0.4, 2.2]
    :param gas_logu, gas_logn, gas_logz, gas_logno, gas_logco:
        nebular parameters, follow the range
        gas_logu: [-4, -1], gas_logn: [1, 4], gas_logz: [-2.2, 0.5],
        gas_logno: [-1, log10(5.4)], gas_logco: [-1, log10(5.4)]
    :param log_qion: ionizing QH as the normalization factor of the emulator output, sum of Qs from four power laws if use_stellar_ionizing==True,
    :param wave: wavelengths for the output nebular continuum, AA
    :output normalized nebular continuum in Lsun/Hz, and normalized line luminosities in Lsun
    """

    def __init__(self, cont_pca_basis = cont_PCABasis, cont_nn = cont_speculator,
                 line_pca_basis = line_PCABasis, line_nn = line_speculator,
                 theta=None,
                 ionspec_index1=19.7, ionspec_index2=5.3, ionspec_index3=1.6, ionspec_index4=0.6,
                 ionspec_logLratio1=3.9, ionspec_logLratio2=0.01, ionspec_logLratio3=0.2,
                 gas_logu=-2.5, gas_logn=2, gas_logz=0., gas_logno=0., gas_logco=0.,
                 log_qion=49.1,
                 #wav_selection = None,
                 #parameter_selection = None
                 **kwargs
                ):
        """
        Constructor.
        """
        # input parameters
        self.cont_pca_basis = cont_pca_basis
        self.cont_nn = cont_nn
        self.line_pca_basis = line_pca_basis
        self.line_nn = line_nn
        self.cont_n_segments = np.size(cont_nn)
        self.line_n_segments = np.size(line_nn)
        self.cont_wavelength = cont_lam[122:]
        self.line_wavelength = nn_wavelength
        self.line_ind = line_old
        self.log_qion = log_qion
        if theta is None:
            self.use_theta_arr = False
            if (np.size(ionspec_index1)==1):
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
                self.ionspec_index1 = np.reshape(ionspec_index1, (len(ionspec_index1), 1))
                self.ionspec_index2 = np.reshape(ionspec_index2, (len(ionspec_index2), 1))
                self.ionspec_index3 = np.reshape(ionspec_index3, (len(ionspec_index3), 1))
                self.ionspec_index4 = np.reshape(ionspec_index4, (len(ionspec_index4), 1))
                self.ionspec_logLratio1 = np.reshape(ionspec_logLratio1, (len(ionspec_logLratio1), 1))
                self.ionspec_logLratio2 = np.reshape(ionspec_logLratio2, (len(ionspec_logLratio2), 1))
                self.ionspec_logLratio3 = np.reshape(ionspec_logLratio3, (len(ionspec_logLratio3), 1))
                self.gas_logu = np.reshape(gas_logu, (len(gas_logu), 1))
                self.gas_logn = np.reshape(gas_logn, (len(gas_logn), 1))
                self.gas_logq = logQ(self.gas_logu, lognH=self.gas_logn)
                self.gas_logz = np.reshape(gas_logz, (len(gas_logz), 1))
                self.gas_logno = np.reshape(gas_logno, (len(gas_logno), 1))
                self.gas_logco = np.reshape(gas_logco, (len(gas_logco), 1))
        #self.params = {'ionspec_index1': self.theta[:,0], 'ionspec_index2': self.theta[:,1],
        #               'ionspec_index3': self.theta[:,2], 'ionspec_index4': self.theta[:,3],
        #               'ionspec_logLratio1': self.theta[:,4], 'ionspec_logLratio2': self.theta[:,5],
        #               'ionspec_logLratio3': self.theta[:,6], 'gas_logq': self.theta[:,7],
        #               'gas_logn': self.theta[:,8], 'gas_logz': self.theta[:,9],
        #               'gas_logno': self.theta[:,10], 'gas_logco': self.theta[:,11],
        #               'log_qion': self.log_qion}
        else:
            self.theta = np.array(theta)
            self.use_theta_arr = True
            self.n_sample = len(self.theta)
            self.gas_logq = logQ(theta[:,-5:], lognH=theta[:,-4:])
            #self.theta[:,-2:] = 10**self.theta[:,-2:]
        #raise ValueError('NEBULAR PARAMETER ERROR: input {0} parameters but required 12'.format(len(theta[0]))

    def update(self, **kwargs):
        for arg, value in kwargs.items():
            if arg in self.__dict__ and value is not None:
                setattr(self, arg, value)
        if not self.use_theta_arr:
            if (np.size(self.ionspec_index1)==1):
                self.gas_logq = logQ(self.gas_logu, lognH=self.gas_logn)
                self.theta = np.hstack([self.ionspec_index1, self.ionspec_index2, self.ionspec_index3, self.ionspec_index4,
                                        self.ionspec_logLratio1, self.ionspec_logLratio2, self.ionspec_logLratio3,
                                        self.gas_logq, 10**self.gas_logn, self.gas_logz, self.gas_logno,
                                        self.gas_logco]).reshape((1, 12))
            else:
                self.ionspec_index1 = np.reshape(self.ionspec_index1, (len(self.ionspec_index1), 1))
                self.ionspec_index2 = np.reshape(self.ionspec_index2, (len(self.ionspec_index2), 1))
                self.ionspec_index3 = np.reshape(self.ionspec_index3, (len(self.ionspec_index3), 1))
                self.ionspec_index4 = np.reshape(self.ionspec_index4, (len(self.ionspec_index4), 1))
                self.ionspec_logLratio1 = np.reshape(self.ionspec_logLratio1, (len(self.ionspec_logLratio1), 1))
                self.ionspec_logLratio2 = np.reshape(self.ionspec_logLratio2, (len(self.ionspec_logLratio2), 1))
                self.ionspec_logLratio3 = np.reshape(self.ionspec_logLratio3, (len(self.ionspec_logLratio3), 1))
                self.gas_logu = np.reshape(self.gas_logu, (len(self.gas_logu), 1))
                self.gas_logn = np.reshape(self.gas_logn, (len(self.gas_logn), 1))
                self.gas_logq = logQ(self.gas_logu, lognH=self.gas_logn)
                self.gas_logz = np.reshape(self.gas_logz, (len(self.gas_logz), 1))
                self.gas_logno = np.reshape(self.gas_logno, (len(self.gas_logno), 1))
                self.gas_logco = np.reshape(self.gas_logco, (len(self.gas_logco), 1))
                self.theta = np.hstack([self.ionspec_index1, self.ionspec_index2, self.ionspec_index3, self.ionspec_index4,
                                        self.ionspec_logLratio1, self.ionspec_logLratio2, self.ionspec_logLratio3,
                                        self.gas_logq, 10**self.gas_logn,
                                        self.gas_logz, self.gas_logno, self.gas_logco])

    def predict_cont(self, wave, **kwargs):
        self.update(**kwargs)
        wavind_sorted = np.argsort(self.cont_wavelength)
        fit_spectra = list()
        if self.cont_n_segments == 1:
            fit_spectra = self.cont_pca_basis.PCA.inverse_transform(self.cont_nn.log_spectrum_(self.theta)) * self.cont_nn.log_spectrum_scale_ + self.cont_nn.log_spectrum_shift_
            if self.n_sample == 1:
                fit_spectra = np.squeeze(fit_spectra)[wavind_sorted]
            else:
                fit_spectra = np.squeeze(fit_spectra)[:,wavind_sorted]
            self.cont_nn_spectra = fit_spectra
        else:
            this_spec = list()
            for j in range(self.cont_n_segments):
                this_spec.append(self.cont_pca_basis[j].PCA.inverse_transform(self.cont_nn[j].log_spectrum_(self.theta)) * self.cont_nn[j].log_spectrum_scale_ + self.cont_nn[j].log_spectrum_shift_)
            fit_spectra.append(np.hstack(this_spec)[:,wavind_sorted])
            self.cont_nn_spectra = np.squeeze(np.array(fit_spectra))
        #self.cont_nn_spectra = 10**self.cont_nn_spectra/3.839E33/10**self.gas_logq*10**self.log_qion # convert to the unit in FSPS
	self.cont_nn_spectra = 10**(self.cont_nn_spectra - self.gas_logq + self.log_qion - np.log10(3.839E33))
        self.output_cont_wavelength = self.cont_wavelength[wavind_sorted]
        from scipy.interpolate import CubicSpline
        neb_cont_cs = CubicSpline(self.cont_wavelength, self.cont_nn_spectra, extrapolate=True) # interpolate onto the fsps wavelengths
        return neb_cont_cs(wave)

    def predict_lines(self, **kwargs):
        self.update(**kwargs)
        wavind_sorted = np.argsort(self.line_wavelength)
        fit_spectra = list()
        if self.line_n_segments == 1:
            fit_spectra = self.line_pca_basis.PCA.inverse_transform(self.line_nn.log_spectrum_(self.theta)) * self.line_nn.log_spectrum_scale_ + self.line_nn.log_spectrum_shift_
            if self.n_sample == 1:
                fit_spectra = np.squeeze(fit_spectra)[wavind_sorted]
            else:
                fit_spectra = np.squeeze(fit_spectra)[:,wavind_sorted]
            self.line_nn_spectra = fit_spectra[self.line_ind]
        else:
            this_spec = list()
            for j in range(self.line_n_segments):
                this_spec.append(self.line_pca_basis[j].PCA.inverse_transform(self.line_nn[j].log_spectrum_(self.theta)) * self.line_nn[j].log_spectrum_scale_ + self.line_nn[j].log_spectrum_shift_)
            fit_spectra.append(np.hstack(this_spec)[:,wavind_sorted][:,self.line_ind])
            self.line_nn_spectra = np.squeeze(np.array(fit_spectra))
        self.output_line_wavelength = self.line_wavelength[wavind_sorted]
        #self.line_nn_spectra = 10**self.line_nn_spectra/3.839E33/10**self.gas_logq*10**self.log_qion # convert to the unit in FSPS
	self.line_nn_spectra = 10**(self.line_nn_spectra - self.gas_logq + self.log_qion - np.log10(3.839E33))
        return self.line_nn_spectra
