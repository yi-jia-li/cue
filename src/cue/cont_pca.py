#import __main__
import numpy as np
import dill as pickle #import pickle

#__name__ == "__main__"
### continuum PCA that has a cutoff at 912A
from sklearn.decomposition import IncrementalPCA
class SpectrumPCA():
    """
    SPECULATOR PCA compression class
    """
    
    def __init__(self, n_wavelengths, n_pcas, n_batches, spectrum_filenames, 
                 parameter_filenames, parameter_index=[2,3,4,5,6,7,8,11,12,14,15,16], logCO=True, parameter_selection = None):
        """
        Constructor.
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param n_pcas: number of PCA components
        :param spectrum_filenames: list of .npy filenames for log spectra (each one an [n_samples, n_wavelengths] array)
        :param parameter_filenames: list of .npy filenames for parameters (each one an [n_samples, n_parameters] array)
        """
        
        # input parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.spectrum_filenames = spectrum_filenames
        self.n_parameters = len(parameter_index)
        self.parameter_index = parameter_index
        self.parameter_filenames = parameter_filenames
        self.n_batches = n_batches

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)
        
        # parameter selection (implementing any cuts on strange parts of parameter space)
        self.parameter_selection = parameter_selection
        self.logCONO = logCO
        
        # number of files in a batch
        self.n_files = int(len(self.parameter_filenames)/self.n_batches)
                
    # train PCA incrementally
    def train_pca(self):

        # shift and scale
        self.log_spectrum_shift = np.zeros(self.n_wavelengths)
        self.log_spectrum_scale = np.zeros(self.n_wavelengths)
        self.parameter_shift = np.zeros(self.n_parameters)
        self.parameter_scale = np.zeros(self.n_parameters)
      
        # loop over training data files, increment PCA
        for i in range(self.n_batches):
            
            if self.parameter_selection is None:
                log_spectra = np.vstack([np.load(self.spectrum_filenames[j]) for j in np.arange(self.n_files)+self.n_files*i])
                log_spectra[log_spectra == 0] = 1e-37
                log_spectra = np.log10(log_spectra[:,122:])
                self.log_spectrum_shift += np.mean(log_spectra, axis=0)/self.n_batches
                self.log_spectrum_scale += np.std(log_spectra, axis=0)/self.n_batches
                parameters = np.vstack([np.loadtxt(self.parameter_filenames[j])[:,self.parameter_index] for j in np.arange(self.n_files)+self.n_files*i])
                if self.logCONO == True:
                    parameters[:,-2:] = np.log10(parameters[:,-2:])

                self.parameter_shift += np.mean(parameters, axis=0)/self.n_batches
                self.parameter_scale += np.std(parameters, axis=0)/self.n_batches
            
                # load spectra and shift+scale
                normalized_log_spectra = (log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)
                
            else:
                log_spectra = np.vstack([np.load(self.spectrum_filenames[j]) for j in np.arange(self.n_files)+self.n_files*i])
                log_spectra[log_spectra == 0] = 1e-37
                log_spectra = np.log10(log_spectra[:,122:])
                parameters = np.vstack([np.loadtxt(self.parameter_filenames[j])[:,self.parameter_index] for j in np.arange(self.n_files)+self.n_files*i])
                if self.logCONO == True:
                    parameters[:,-2:] = np.log10(parameters[:,-2:])
                selection = self.parameter_selection(parameters)
                
                # update shifts and scales
                self.log_spectrum_shift += np.mean(log_spectra[selection,:], axis=0)/self.n_batches
                self.log_spectrum_scale += np.std(log_spectra[selection,:], axis=0)/self.n_batches
                self.parameter_shift += np.mean(parameters[selection,:], axis=0)/self.n_batches
                self.parameter_scale += np.std(parameters[selection,:], axis=0)/self.n_batches             

                
                # select based on parameters
                selection = self.parameter_selection(np.load(self.parameter_filenames[i]))
                
                # load spectra and shift+scale
                normalized_log_spectra = (np.load(self.spectrum_filenames[i])[selection,122:] - self.log_spectrum_shift)/self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)
            
        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_
        
    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, filename, retain = False):

        # transform the spectra to PCA basis
        training_pca = list()
        training_parameters = list()
        for i in range(self.n_batches):
            log_spectra = np.vstack([np.load(self.spectrum_filenames[j]) for j in np.arange(self.n_files)+self.n_files*i])
            log_spectra[log_spectra == 0] = 1e-37
            log_spectra = np.log10(log_spectra[:,122:])
            training_pca.append(self.PCA.transform((log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale))
            tmp = np.vstack([np.loadtxt(self.parameter_filenames[j])[:,self.parameter_index] for j in np.arange(self.n_files)+self.n_files*i])
            tmp = np.vstack([np.loadtxt(self.parameter_filenames[j])[:,self.parameter_index] for j in np.arange(self.n_files)+self.n_files*i])
            if self.logCONO == True:
                tmp[:,-2:] = np.log10(tmp[:,-2:])
            training_parameters.append(tmp)
        training_pca = np.concatenate(training_pca)
        training_parameters = np.concatenate(training_parameters)
        
        if self.parameter_selection is not None:
            selection = self.parameter_selection(training_parameters)
            training_pca = training_pca[selection,:]
            training_parameters = training_parameters[selection,:]
            
        # shift and scale of PCA basis
        self.pca_shift = np.mean(training_pca, axis=0)
        self.pca_scale = np.std(training_pca, axis=0)
        
        # save stacked transformed training data
        np.save(filename + '_pca.npy', training_pca)
        np.save(filename + '_parameters.npy', training_parameters)
        
        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters
            
    # make a validation plot of the PCA given some validation data
    def validate_pca_basis(self, spectrum_filename, parameter_filename=None):
        
        # load in the data (and select based on parameter selection if neccessary)
        if self.parameter_selection is None:
            
            # load spectra and shift+scale
            log_spectra = np.vstack([np.load(i) for i in spectrum_filename])
            log_spectra[log_spectra == 0] = 1e-37
            log_spectra = np.log10(log_spectra[:,122:])
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale
                
        else:
                
            # select based on parameters
            selection = self.parameter_selection(np.vstack([np.load(i) for i in parameter_filename]))
                
            # load spectra and shift+scale
            log_spectra = np.vstack([np.load(i) for i in spectrum_filename])
            log_spectra[log_spectra == 0] = 1e-37
            log_spectra = np.log10(log_spectra)[selection,122:]
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale
        
        # transform to PCA basis and back
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = self.PCA.inverse_transform(log_spectra_pca) * self.log_spectrum_scale + self.log_spectrum_shift #(log_spectra_pca@self.pca_transform_matrix+self.PCA.mean_)*self.log_spectrum_scale + self.log_spectrum_shift

        # return raw spectra and spectra in basis
        return log_spectra, log_spectra_pca, log_spectra_in_basis
