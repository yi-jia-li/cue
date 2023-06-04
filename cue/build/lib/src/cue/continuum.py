### temperal line prediction function
import numpy as np
import glob
import tensorflow as tf
import tqdm
import pickle

from .pca import SpectrumPCA as SpectrumPCA
from .nn import Speculator

with open("/gpfs/group/jql6565/default/yli/nebular/cont_pca_50comp_wavcut.pkl", 'rb') as f:
    PCABasis = pickle.load(f)
speculator = Speculator(restore = True, 
                        restore_filename = '/gpfs/group/jql6565/default/yli/nebular/speculator_cont_50comp_wavcut')

### read test data
parfiles = np.array(glob.glob("/gpfs/group/jql6565/default/yli/nebular/train/PowerLaw*.pars"))
parfiles = parfiles[np.argsort(np.array([i.split("/")[-1].split('-')[0][8:] for i in parfiles], dtype=int))]
contfiles = np.array(glob.glob("/gpfs/group/jql6565/default/yli/nebular/train/PowerLaw*cont.npy"))
contfiles = contfiles[np.argsort(np.array([i.split("/")[-1].split('-')[0][8:] for i in contfiles], dtype=int))]

testing_parameters = list()
fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
#log_spectrum_shift = np.zeros(1963)
#log_spectrum_scale = np.zeros(1963)
for i in range(2600)[2400:]:
    #log_spectra = np.load(contfiles[i])
    #log_spectra[log_spectra == 0] = 1e-37
    #log_spectra = np.log10(log_spectra[:,122:])
    #testing_spectra.append(log_spectra)
    ##log_spectrum_scale += np.std(log_spectra, axis=0)/2000
    #testing_pca.append(PCABasis.PCA.transform((log_spectra - PCABasis.log_spectrum_shift)/PCABasis.log_spectrum_scale))
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    testing_parameters.append(theta)
    fit_spectra.append(PCABasis.PCA.inverse_transform(speculator.log_spectrum_(theta)) * speculator.log_spectrum_scale_ + speculator.log_spectrum_shift_)
#testing_spectra = np.concatenate(testing_spectra)
#testing_pca = np.concatenate(testing_pca)
testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis.validate_pca_basis(spectrum_filename = contfiles[2400:])
spectraPCA_quantile = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                  [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebcont_quantile = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile = np.quantile(10**testing_spectra*2.9979e18/fsps_lam[122:], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
chi = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low, onesigma_low, medchi, onesigma_up, twosigma_up = np.quantile(chi, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)