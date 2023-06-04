### temperal line prediction function
import numpy as np
import glob
import tensorflow as tf
import tqdm
import pickle

### read the fit PCAs and NN
### read PCA and NN
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_1.pkl", 'rb') as f:
    PCABasis_1 = pickle.load(f)
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_2.pkl", 'rb') as f:
    PCABasis_2 = pickle.load(f)
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_3.pkl", 'rb') as f:
    PCABasis_3 = pickle.load(f)
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_4.pkl", 'rb') as f:
    PCABasis_4 = pickle.load(f)
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_5.pkl", 'rb') as f:
    PCABasis_5 = pickle.load(f)
with open("/gpfs/group/jql6565/default/yli/nebular/line_pca_2m_ele_6.pkl", 'rb') as f:
    PCABasis_6 = pickle.load(f)
speculator_1 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_1")
speculator_2 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_2")
speculator_3 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_3")
speculator_4 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_4")
speculator_5 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_5")
speculator_6 = Speculator(restore=True, 
                          restore_filename="/gpfs/group/jql6565/default/yli/nebular/speculator_line_2m_ele_6")


### read test data
#parfiles = np.array(glob.glob("/gpfs/group/jql6565/default/yli/nebular/train/PowerLaw*.pars"))
#parfiles = parfiles[np.argsort(np.array([i.split("/")[-1].split('-')[0][8:] for i in parfiles], dtype=int))]
#linefiles = np.array(glob.glob("/gpfs/group/jql6565/default/yli/nebular/train/PowerLaw*line.npy"))
#linefiles = linefiles[np.argsort(np.array([i.split("/")[-1].split('-')[0][8:] for i in linefiles], dtype=int))]

wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(0,38)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_1.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_1 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

testing_parameters = list()
fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    testing_parameters.append(theta)
    fit_spectra.append(PCABasis_1.PCA.inverse_transform(speculator_1.log_spectrum_(theta)) * speculator_1.log_spectrum_scale_ + speculator_1.log_spectrum_shift_)
testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_1 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_1 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_1 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_1, onesigma_low_1, medchi_1, onesigma_up_1, twosigma_up_1 = np.quantile(chi_1, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)


wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(38,46)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_2.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_2 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    fit_spectra.append(PCABasis_2.PCA.inverse_transform(speculator_2.log_spectrum_(theta)) * speculator_2.log_spectrum_scale_ + speculator_2.log_spectrum_shift_)
#testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_2 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_2 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_2 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_2, onesigma_low_2, medchi_2, onesigma_up_2, twosigma_up_2 = np.quantile(chi_2, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)


wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(46,59)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_3.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_3 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    fit_spectra.append(PCABasis_3.PCA.inverse_transform(speculator_3.log_spectrum_(theta)) * speculator_3.log_spectrum_scale_ + speculator_3.log_spectrum_shift_)
#testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_3 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_3 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_3 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_3, onesigma_low_3, medchi_3, onesigma_up_3, twosigma_up_3 = np.quantile(chi_3, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)


wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(59,67)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_4.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_4 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    fit_spectra.append(PCABasis_4.PCA.inverse_transform(speculator_4.log_spectrum_(theta)) * speculator_4.log_spectrum_scale_ + speculator_4.log_spectrum_shift_)
#testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_4 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_4 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_4 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_4, onesigma_low_4, medchi_4, onesigma_up_4, twosigma_up_4 = np.quantile(chi_4, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)


wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(67,85)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_5.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_5 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    fit_spectra.append(PCABasis_5.PCA.inverse_transform(speculator_5.log_spectrum_(theta)) * speculator_5.log_spectrum_scale_ + speculator_5.log_spectrum_shift_)
#testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_5 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_5 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_5 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_5, onesigma_low_5, medchi_5, onesigma_up_5, twosigma_up_5 = np.quantile(chi_5, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)

wav_selection = np.searchsorted(line_lam, unsorted_line_lam[np.arange(85,128)])
testing_spectra, testing_pca, testing_spectra_in_pca_basis = PCABasis_6.validate_pca_basis(spectrum_filename = linefiles[4000:4200])
spectraPCA_quantile_6 = np.quantile((10**testing_spectra_in_pca_basis-10**testing_spectra)/10**testing_spectra, 
                                    [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

fit_spectra = list()
parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
for i in range(4200)[4000:]:
    theta = np.loadtxt(parfiles[i])[:,parameter_index]
    fit_spectra.append(PCABasis_6.PCA.inverse_transform(speculator_6.log_spectrum_(theta)) * speculator_6.log_spectrum_scale_ + speculator_6.log_spectrum_shift_)
#testing_parameters = np.concatenate(testing_parameters)
fit_spectra = np.concatenate(fit_spectra)

nebline_quantile_6 = np.quantile(10**testing_spectra, [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)
nebνLν_quantile_6 = np.quantile(10**testing_spectra*2.9979e18/line_lam[wav_selection], 
                              [0.0455/2, 0.16, 0.5, 0.84, 1-0.0455/2], axis=0)

chi_6 = (10**fit_spectra-10**testing_spectra)/10**testing_spectra
twosigma_low_6, onesigma_low_6, medchi_6, onesigma_up_6, twosigma_up_6 = np.quantile(chi_6, [0.0455/2, 0.16, 0.5, 
                                                                                 0.84, 1-0.0455/2], axis=0)


chi = np.hstack([chi_1, chi_2, chi_3, chi_4, chi_5, chi_6])
nebνLν_quantile = np.hstack([nebνLν_quantile_1, nebνLν_quantile_2, nebνLν_quantile_3, nebνLν_quantile_4, nebνLν_quantile_5, nebνLν_quantile_6])
spectraPCA_quantile = np.hstack([spectraPCA_quantile_1, spectraPCA_quantile_2, spectraPCA_quantile_3, spectraPCA_quantile_4, spectraPCA_quantile_5, spectraPCA_quantile_6])
medchi = np.hstack([medchi_1, medchi_2, medchi_3, medchi_4, medchi_5, medchi_6])
onesigma_low = np.hstack([onesigma_low_1, onesigma_low_2, onesigma_low_3, onesigma_low_4, onesigma_low_5, onesigma_low_6])
onesigma_up = np.hstack([onesigma_up_1, onesigma_up_2, onesigma_up_3, onesigma_up_4, onesigma_up_5, onesigma_up_6])
twosigma_low = np.hstack([twosigma_low_1, twosigma_low_2, twosigma_low_3, twosigma_low_4, twosigma_low_5, twosigma_low_6])
twosigma_up = np.hstack([twosigma_up_1, twosigma_up_2, twosigma_up_3, twosigma_up_4, twosigma_up_5, twosigma_up_6])

### Example
#test_ind = np.random.choice(len(parfiles), 1)

### read the true nebular spectra and decomposite it into PCAs
#true_spectra, pca, spectra_in_pca_basis = PCABasis_H.validate_pca_basis(spectrum_filename = linefiles[test_ind])

### generate the fit spectra from the parameters
#parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16]
#theta = np.loadtxt(parfiles[test_ind])[:,parameter_index]
#fit_spectra = PCABasis_H.PCA.inverse_transform(speculator_H.log_spectrum_(theta)) * \
#speculator_H.log_spectrum_scale_ +\ speculator_H.log_spectrum_shift_
