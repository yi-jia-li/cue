### temperal line prediction function
import numpy as np
#import glob
import tensorflow as tf
import tqdm
import dill as pickle
from . import cont_pca
from .cont_pca import SpectrumPCA
from .nn import Speculator
from .utils import cont_lam
#import __main__
#__main__.SpectrumPCA = SpectrumPCA

### read the fit PCAs and NN
try:
    from pkg_resources import resource_filename, resource_listdir
except(ImportError):
    pass
#import runpy
#runpy._run_module_as_main("cont_pca.SpectrumPCA")
#runpy.run_path(resource_filename("cue","cont_pca.py"), {}, "__main__")
with open(resource_filename("cue", "data/cont_pca_50comp_wavcut.pkl"), 'rb') as f:
    cont_PCABasis = pickle.load(f)
cont_speculator = Speculator(restore = True, 
                             restore_filename = resource_filename("cue", "data/speculator_cont_50comp_wavcut"))

#par = pd.DataFrame(data={'num':par[:,0], 'index1':par[:,2], 'index2':par[:,3], 'index3':par[:,4], 'index4':par[:,5], 
#                         'delta_logL1':par[:,6], 'delta_logL2':par[:,7], 'delta_logL3':par[:,8],
#                         'logU':par[:,9], 'Rinner':par[:,10], 'logQ':par[:,11], 'nH':par[:,12],
#                         'efrac':par[:,13], 'gas_logZ':par[:,14], 'NO':par[:,15], 'CO':par[:,16]
#                         })

class predict():
    """
    Nebular Continuum Emission Prediction
    :param theta: nebular parameters of n samples, (n, 12) matrix 
    :param gammas, log_L_ratios, logQH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio: 12 input parameters, vectors
    :param wavelength: wavelengths corresponding to the neural net output luminosities
    :output sorted wavelength, and L_nu in erg/Hz sorted by wavelength
    """
    
    def __init__(self, pca_basis=cont_PCABasis, nn=cont_speculator, theta=None, gammas=None, log_L_ratios=None, logQH=None, 
                 n_H=None, log_OH_ratio=None, log_NO_ratio=None, log_CO_ratio=None, 
                 wavelength=cont_lam[122:],  
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
                self.theta = np.hstack([gammas, log_L_ratios, logQH, n_H, 
                                        log_OH_ratio, 10**log_NO_ratio, 10**log_CO_ratio]).reshape((1, 12))
            else:
                self.n_sample = len(logQH)
                self.gammas = np.array(gammas)
                self.log_L_ratios = np.array(log_L_ratios)
                self.logQH = np.reshape(logQH, (len(logQH), 1))
                self.n_H = np.reshape(n_H, (len(n_H), 1))
                self.log_OH_ratio = np.reshape(log_OH_ratio, (len(log_OH_ratio), 1))
                self.log_NO_ratio = np.reshape(log_NO_ratio, (len(log_NO_ratio), 1))
                self.log_CO_ratio = np.reshape(log_CO_ratio, (len(log_CO_ratio), 1))
                self.theta = np.hstack([self.gammas, self.log_L_ratios, self.logQH, self.n_H, 
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
    
    
#testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis.validate_pca_basis(spectrum_filename = contfiles[i])
#log_spectra = np.load(contfiles[i])
#log_spectra[log_spectra == 0] = 1e-37
#log_spectra = np.log10(log_spectra[:,122:])
#testing_pca.append(PCABasis.PCA.transform((log_spectra - PCABasis.log_spectrum_shift)/PCABasis.log_spectrum_scale))
