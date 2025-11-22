## temperal line prediction function
import numpy as np
import tensorflow as tf
import tqdm
import dill as pickle
from .line_pca import SpectrumPCA
from .nn import Speculator
from .utils import (nn_wavelength, nn_name, line_old, logQ)
#import __main__
#__main__.SpectrumPCA = SpectrumPCA
 
try:
    from importlib.resources import files
except(ImportError):
    pass

### read the fit PCAs and NN, nn grouped by both elements and ionization potentials
for this_name in nn_name: 
    with open(files("cue").joinpath("data/pca_line_new_"+this_name+".pkl"), 'rb') as f:
        globals()["PCABasis_"+this_name] = pickle.load(f)
        globals()["speculator_"+this_name] = Speculator(restore = True, 
                                                        restore_filename = str(files("cue").joinpath(\
                                                                                                     "data/speculator_line_new_"+this_name))
                                                       )
line_PCABasis = [globals()["PCABasis_"+this_name] for this_name in nn_name]
line_speculator = [globals()["speculator_"+this_name] for this_name in nn_name]
                                                    
#with open(resource_filename("cue", "data/pca_line_2m_ele_H.pkl"), 'rb') as f:
#    PCABasis_H = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_He1.pkl"), 'rb') as f:
#    PCABasis_He1 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_He2.pkl"), 'rb') as f:
#    PCABasis_He2 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_C1.pkl"), 'rb') as f:
#    PCABasis_C1 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_C2C3.pkl"), 'rb') as f:
#    PCABasis_C2C3 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_N.pkl"), 'rb') as f:
#    PCABasis_N = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_O1.pkl"), 'rb') as f:
#    PCABasis_O1 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_O2.pkl"), 'rb') as f:
#    PCABasis_O2 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_O3.pkl"), 'rb') as f:
#    PCABasis_O3 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_other_ionE_1.pkl"), 'rb') as f:
#    PCABasis_other_1 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_other_ionE_2.pkl"), 'rb') as f:
#    PCABasis_other_2 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_S4.pkl"), 'rb') as f:
#    PCABasis_S4 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_Ar4.pkl"), 'rb') as f:
#    PCABasis_Ar4 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_Ne3.pkl"), 'rb') as f:
#    PCABasis_Ne3 = pickle.load(f)
#with open(resource_filename("cue", "data/pca_line_2m_ele_Ne4.pkl"), 'rb') as f:
#    PCABasis_Ne4 = pickle.load(f)

#speculator_H = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_H"))
#speculator_He1 = Speculator(restore=True, 
#                            restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_He1"))
#speculator_He2 = Speculator(restore=True, 
#                            restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_He2"))
#speculator_C1 = Speculator(restore=True, 
#                           restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_C1"))
#speculator_C2C3 = Speculator(restore=True, 
#                             restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_C2C3"))
#speculator_N = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_N"))
#speculator_O1 = Speculator(restore=True, 
#                           restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_O1"))
#speculator_O2 = Speculator(restore=True, 
#                           restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_O2"))
#speculator_O3 = Speculator(restore=True, 
#                           restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_O3"))
#speculator_other_1 = Speculator(restore=True, 
#                                restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_other_ionE_1"))
#speculator_other_2 = Speculator(restore=True, 
#                                restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_other_ionE_2"))
#speculator_S4 = Speculator(restore=True, 
#                           restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_S4"))
#speculator_Ar4 = Speculator(restore=True, 
#                            restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_Ar4"))
#speculator_Ne3 = Speculator(restore=True, 
#                            restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_Ne3"))
#speculator_Ne4 = Speculator(restore=True, 
#                            restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_Ne4"))

#nn_wavelength = [np.arange(0,38),
#                 np.arange(38,45),
#                 np.array([45]),
#                 np.arange(46,51),
#                 np.arange(51,59),
#                 np.arange(59,67),
#                 np.arange(67,72),
#                 np.arange(72,77),
#                 np.arange(77,85),
#                 np.array([93,94,95,96,97,98,99,100,108,116,117,120,121,122,125,126,127]),
#                 np.array([85,101,102,103,104,105,106,109,110,111,112,113,114,118,119,123,124]),
#                 np.array([107]),
#                 np.array([115]),
#                 np.array([86,87,88,89,90,91]),
#                 np.array([92])
#                ] 
#wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.concatenate(nn_wavelength)])

#line_PCABasis = [PCABasis_H, PCABasis_He1, PCABasis_He2, PCABasis_C1, PCABasis_C2C3, PCABasis_N,
#                 PCABasis_O1, PCABasis_O2, PCABasis_O3, PCABasis_other_1, PCABasis_other_2,
#                 PCABasis_S4, PCABasis_Ar4, PCABasis_Ne3, PCABasis_Ne4]
#line_speculator = [speculator_H, speculator_He1, speculator_He2, speculator_C1, speculator_C2C3, speculator_N,
#                   speculator_O1, speculator_O2, speculator_O3, speculator_other_1, speculator_other_2,
#                   speculator_S4, speculator_Ar4, speculator_Ne3, speculator_Ne4]

#with open(resource_filename("cue", "data/line_pca_2m_ele_1.pkl"), 'rb') as f:
#    PCABasis_H = pickle.load(f)
#with open(resource_filename("cue", "data/line_pca_2m_ele_2.pkl"), 'rb') as f:
#    PCABasis_He = pickle.load(f)
#with open(resource_filename("cue", "data/line_pca_2m_ele_3.pkl"), 'rb') as f:
#    PCABasis_C = pickle.load(f)
#with open(resource_filename("cue", "data/line_pca_2m_ele_4.pkl"), 'rb') as f:
#    PCABasis_N = pickle.load(f)
#with open(resource_filename("cue", "data/line_pca_2m_ele_5.pkl"), 'rb') as f:
#    PCABasis_O = pickle.load(f)
#with open(resource_filename("cue", "data/line_pca_2m_ele_6.pkl"), 'rb') as f:
#    PCABasis_other = pickle.load(f)
#speculator_H = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_1"))
#speculator_He = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_2"))
#speculator_C = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_3"))
#speculator_N = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_4"))
#speculator_O = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_5"))
#speculator_other = Speculator(restore=True, 
#                          restore_filename=resource_filename("cue", "data/speculator_line_2m_ele_6"))

#line_PCABasis = [PCABasis_H, PCABasis_He, PCABasis_C, PCABasis_N, PCABasis_O, PCABasis_other]
#line_speculator = [speculator_H, speculator_He, speculator_C, speculator_N, speculator_O, speculator_other]

#wav_selection = np.concatenate([np.searchsorted(line_lam, unsorted_line_lam[np.arange(0,38)]),
#                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(38,46)]),
#                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(46,59)]),
#                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(59,67)]),
#                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(67,85)]),
#                                np.searchsorted(line_lam, unsorted_line_lam[np.arange(85,128)])])

class predict():
    """
    Nebular Line Emission Prediction
    :param theta: nebular parameters of n samples, (n, 12) matrix 
    :param gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio: 12 input parameters, vectors
    :param wavelength: wavelengths corresponding to the neural net output luminosities, not required
    :param line_ind: index of the lines in the sorted wavelength array, the default return only the lines in cloudyfsps
    :output sorted wavelength, and L_nu in erg/Hz sorted by wavelength
    """
    
    def __init__(self, pca_basis=line_PCABasis, nn=line_speculator, theta=None, gammas=None, log_L_ratios=None, log_QH=None, 
                 n_H=None, log_OH_ratio=None, log_NO_ratio=None, log_CO_ratio=None, 
                 wavelength=nn_wavelength, line_ind=line_old,  #line_lam[wav_selection]
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
        self.line_ind = line_ind
        if theta is None:
            if (np.size(log_QH)==1):
                self.n_sample = 1
                self.theta = np.hstack([gammas, log_L_ratios, log_QH, n_H, 
                                        log_OH_ratio, log_NO_ratio, log_CO_ratio]).reshape((1, 12)) #10**log_NO_ratio, 10**log_CO_ratio
            else:
                self.n_sample = len(log_QH)
                self.gammas = np.array(gammas)
                self.log_L_ratios = np.array(log_L_ratios)
                self.log_QH = np.reshape(log_QH, (len(log_QH), 1))
                self.n_H = np.reshape(n_H, (len(n_H), 1))
                self.log_OH_ratio = np.reshape(log_OH_ratio, (len(log_OH_ratio), 1))
                self.log_NO_ratio = np.reshape(log_NO_ratio, (len(log_NO_ratio), 1))
                self.log_CO_ratio = np.reshape(log_CO_ratio, (len(log_CO_ratio), 1))
                self.theta = np.hstack([self.gammas, self.log_L_ratios, self.log_QH, self.n_H, 
                                        self.log_OH_ratio, self.log_NO_ratio, self.log_CO_ratio]) #10**self.log_NO_ratio, 10**self.log_CO_ratio
        else:
            self.theta = np.array(theta)
            self.n_sample = len(self.theta)
            #self.theta[:,-2:] = 10**self.theta[:,-2:]
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
            self.nn_spectra = fit_spectra[self.line_ind]
        else:
            this_spec = list()
            for j in range(self.n_segments):
                this_spec.append(self.pca_basis[j].PCA.inverse_transform(self.nn[j].log_spectrum_(self.theta)) * self.nn[j].log_spectrum_scale_ + self.nn[j].log_spectrum_shift_)
            fit_spectra.append(np.hstack(this_spec)[:,wavind_sorted][:,self.line_ind])
            self.nn_spectra = np.squeeze(np.array(fit_spectra))
        self.wavelength = self.wavelength[wavind_sorted]
        return self.wavelength, 10**self.nn_spectra 

def get_line(par):
    """
    A wrapper of nebular line emulator for SED fitting.
    """
    neb_line = line_predict(gammas=[par['ionspec_index1'], par['ionspec_index2'], 
                                    par['ionspec_index3'], par['ionspec_index4']],
                            log_L_ratios=[par['ionspec_logLratio1'], par['ionspec_logLratio2'],
                                          par['ionspec_logLratio3']],
                            log_QH=logQ(par['gas_logu'], lognH=par['gas_logn']),
                            n_H=10**par['gas_logn'],
                            log_OH_ratio=par['gas_logz'],
                            log_NO_ratio=par['gas_logno'],
                            log_CO_ratio=par['gas_logco'],
                           ).nn_predict()
    line_spec = neb_line[1]/3.839E33/10**logQ(par['gas_logu'], lognH=par['gas_logn'])*10**par['gas_logqion'] # convert to the unit in FSPS
    return {"normalized nebular lineinuum": line_spec}


### Example
#test_ind = np.random.choice(len(parfiles), 1)

### read the true nebular spectra and decomposite it into PCAs
#true_spectra, pca, spectra_in_pca_basis = PCABasis_H.validate_pca_basis(spectrum_filename = linefiles[test_ind])

### generate the fit spectra from the parameters
#parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
#theta = np.loadtxt(parfiles[test_ind])[:,parameter_index]
#fit_spectra = PCABasis_H.PCA.inverse_transform(speculator_H.log_spectrum_(theta)) * \
#speculator_H.log_spectrum_scale_ +\ speculator_H.log_spectrum_shift_

