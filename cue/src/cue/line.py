### temperal line prediction function
import numpy as np
#import glob
import tensorflow as tf
import tqdm
import pickle
#import dill as pickle
from .line_pca import SpectrumPCA
from .nn import Speculator
from .utils import (unsorted_line_lam, line_lam)
import __main__
__main__.SpectrumPCA = SpectrumPCA

    
### read the fit PCAs and NN
try:
    from pkg_resources import resource_filename, resource_listdir
except(ImportError):
    pass
with open(resource_filename("cue", "data/line_pca_2m_ele_1.pkl"), 'rb') as f:
    PCABasis_H = pickle.load(f)
with open(resource_filename("cue", "data/line_pca_2m_ele_2.pkl"), 'rb') as f:
    PCABasis_He = pickle.load(f)
with open(resource_filename("cue", "data/line_pca_2m_ele_3.pkl"), 'rb') as f:
    PCABasis_C = pickle.load(f)
with open(resource_filename("cue", "data/line_pca_2m_ele_4.pkl"), 'rb') as f:
    PCABasis_N = pickle.load(f)
with open(resource_filename("cue", "data/line_pca_2m_ele_5.pkl"), 'rb') as f:
    PCABasis_O = pickle.load(f)
with open(resource_filename("cue", "data/line_pca_2m_ele_6.pkl"), 'rb') as f:
    PCABasis_other = pickle.load(f)
speculator_H = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_1"))
speculator_He = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_2"))
speculator_C = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_3"))
speculator_N = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_4"))
speculator_O = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_5"))
speculator_other = Speculator(restore=True, 
                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_6"))

line_PCABasis = [PCABasis_H, PCABasis_He, PCABasis_C, PCABasis_N, PCABasis_O, PCABasis_other]
line_speculator = [speculator_H, speculator_He, speculator_C, speculator_N, speculator_O, speculator_other]

wav_selection = np.concatenate([np.searchsorted(line_lam, unsorted_line_lam[np.arange(0,38)]),
                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(38,46)]),
                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(46,59)]),
                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(59,67)]),
                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(67,85)]),
                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(85,128)])])

class predict():
    """
    Nebular Line Emission Prediction
    :param theta: nebular parameters of n samples, (n, 12) matrix 
    :param gammas, log_norm_ratios, logQH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio: 12 input parameters, vectors
    :param wavelength: wavelengths corresponding to the neural net output luminosities
    :output sorted wavelength, and L_nu in erg/Hz sorted by wavelength
    """
    
    def __init__(self, pca_basis=line_PCABasis, nn=line_speculator, theta=None, gammas=None, log_norm_ratios=None, logQH=None, 
                 n_H=None, log_OH_ratio=None, log_NO_ratio=None, log_CO_ratio=None, 
                 wavelength=line_lam[wav_selection],  
                 #wav_selection = None,
                 #parameter_selection = None
                ):
        """
        Constructor.
        """        
        # input parameters
        self.pca_basis = pca_basis
        self.nn = nn
        self.n_segments = np.size(nn)
        self.wavelength = np.array(wavelength)
        if theta is None:
            if (np.size(logQH)==1):
                self.n_sample = 1
                self.theta = np.hstack([gammas, log_norm_ratios, logQH, n_H, 
                                        log_OH_ratio, 10**log_NO_ratio, 10**log_CO_ratio]).reshape((1, 12))
            else:
                self.n_sample = len(logQH)
                self.gammas = np.array(gammas)
                self.log_norm_ratios = np.array(log_norm_ratios)
                self.logQH = np.reshape(logQH, (len(logQH), 1))
                self.n_H = np.reshape(n_H, (len(n_H), 1))
                self.log_OH_ratio = np.reshape(log_OH_ratio, (len(log_OH_ratio), 1))
                self.log_NO_ratio = np.reshape(log_NO_ratio, (len(log_NO_ratio), 1))
                self.log_CO_ratio = np.reshape(log_CO_ratio, (len(log_CO_ratio), 1))
                self.theta = np.hstack([self.gammas, self.log_norm_ratios, self.logQH, self.n_H, 
                                        self.log_OH_ratio, 10**self.log_NO_ratio, 10**self.log_CO_ratio])
        else:
            self.theta = np.array(theta)
            self.n_sample = len(self.theta)
            self.theta[:,-2:] = 10**self.theta[:,-2:]
        #raise ValueError('NEBULAR PARAMETER ERROR: input {0} parameters but required 12'.format(len(theta[0]))
    
    def nn_predict(self):

        # shift and scale
        wavind_sorted = np.argsort(self.wavelength)
        fit_spectra = list()
        if self.n_segments == 1:
            fit_spectra = self.pca_basis.PCA.inverse_transform(self.nn.log_spectrum_(self.theta)) * self.nn.log_spectrum_scale_ + self.nn.log_spectrum_shift_
            if self.n_sample == 1:
                fit_spectra = np.squeeze(fit_spectra)[wavind_sorted]
            else:
                fit_spectra = np.squeeze(fit_spectra)[:,wavind_sorted]
            self.nn_spectra = fit_spectra
        else:
            this_spec = list()
            for j in range(self.n_segments):
                this_spec.append(self.pca_basis[j].PCA.inverse_transform(self.nn[j].log_spectrum_(self.theta)) * self.nn[j].log_spectrum_scale_ + self.nn[j].log_spectrum_shift_)
            fit_spectra.append(np.hstack(this_spec)[:,wavind_sorted])
            self.nn_spectra = np.squeeze(np.array(fit_spectra))
        self.wavelength = self.wavelength[wavind_sorted]
        return self.wavelength, 10**self.nn_spectra    


### Example
#test_ind = np.random.choice(len(parfiles), 1)

### read the true nebular spectra and decomposite it into PCAs
#true_spectra, pca, spectra_in_pca_basis = PCABasis_H.validate_pca_basis(spectrum_filename = linefiles[test_ind])

### generate the fit spectra from the parameters
#parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
#theta = np.loadtxt(parfiles[test_ind])[:,parameter_index]
#fit_spectra = PCABasis_H.PCA.inverse_transform(speculator_H.log_spectrum_(theta)) * \
#speculator_H.log_spectrum_scale_ +\ speculator_H.log_spectrum_shift_
