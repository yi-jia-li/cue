import jax
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import jax.numpy as jnp
from functools import partial
import importlib
import numpy as np
import pickle

from .sources import SpeculatorANN, PCABasis
from .constants import c_cms, c_AAs, c_kms, Lsun_cgs, lyman_limit


class Emulator():
    def __init__(self,num_lines=138,**kwargs):
        
        self._cont_pca_dict = None
        self._cont_ann_dict = None
        self._cont_norm_dict = None
        self.cont_frac_error = None
        self.cont_wavelengths = None
        self._line_pca_dicts = None
        self._line_ann_dicts = None
        self._line_norm_dicts = None
        self.line_frac_error = None
        self.line_wavelengths = None
        self.line_names = None
        
        files = ['fspse_cue_cont_pca_basis.pkl','fspse_cue_cont_ann.pkl','fspse_cue_cont_norm.pkl',
                 'cont_frac_error.dat','cont_wave.dat','fspse_cue_eline_pca_bases.pkl','fspse_cue_eline_anns.pkl',
                 'fspse_cue_eline_norm.pkl','eline_frac_error.dat','eline_names.dat','prospector_line_mask.dat',
                 'eline_wave_cue.dat']
        attributes = ['_cont_pca_dict','_cont_ann_dict','_cont_norm_dict','cont_frac_error','cont_wavelengths',
                      '_line_pca_dicts','_line_ann_dicts','_line_norm_dicts','line_frac_error',
                      'line_names','prospector_line_mask','line_wavelengths_ann']
        
        for attribute_name,file_name in zip(attributes,files):
            with importlib.resources.files('cuejax.data').joinpath(file_name).open('rb') as file:
                if 'pkl' in file_name:
                    data = pickle.load(file)
                elif 'names' in file_name:
                    data = np.loadtxt(file,dtype=str)
                elif 'prospector' in file_name:
                    data = jnp.array(np.genfromtxt(file),dtype=jnp.int32) # indices must be integers
                else:
                    data = jnp.array(np.genfromtxt(file),dtype=jnp.float32)
            setattr(self, attribute_name, data)

        self._cont_pca = PCABasis(self._cont_pca_dict)
        self._cont_ann = SpeculatorANN(self._cont_ann_dict)

        self._cont_norm_mus = self._cont_norm_dict['mu']
        self._cont_norm_stds = self._cont_norm_dict['std']

        self._line_group_names = np.array(['H1', 'He1', 'He2', 'C1', 'C2C3', 'C4', 'N', 'O1', 'O2', 'O3',
                                           'ionE_1', 'ionE_2', 'S4', 'Ar4', 'Ne3', 'Ne4'])

        if num_lines != 128:
            self.prospector_line_mask = np.arange(len(self.line_wavelengths_ann)) # make the prospector line mask do nothing!

        self.sort_by_wavelength = jnp.argsort(self.line_wavelengths_ann)
        self.line_wavelengths = self.line_wavelengths_ann[self.sort_by_wavelength][self.prospector_line_mask]
        self.line_frac_error = self.line_frac_error[self.prospector_line_mask] # already sorted by wavelength!
        self.line_names = self.line_names[self.sort_by_wavelength][self.prospector_line_mask]
        self.line_names = [' '.join(row) for row in self.line_names]
        self.num_lines_tot = self.line_wavelengths_ann.shape[0] # 138 lines outputted from cue
        self.num_lines = self.line_wavelengths.shape[0] # 128 lines that are in Byler grids (if num_lines = 128)

        self.line_frac_error = self.line_frac_error[self.sort_by_wavelength][self.prospector_line_mask]

        self._line_pca = [PCABasis(self._line_pca_dicts[line]) for line in self._line_group_names]
        self._line_ann = [SpeculatorANN(self._line_ann_dicts[line]) for line in self._line_group_names]

        self._line_norm_mus = [self._line_norm_dicts[line]['mu'] for line in self._line_group_names]
        self._line_norm_stds = [self._line_norm_dicts[line]['std'] for line in self._line_group_names]

        self.free_parameter_order = [ # these are the parameters proposed by the sampler that we are trying to fit for! 12 in tota;
                                    'ionspec_index1', 'ionspec_index2', 'ionspec_index3', 'ionspec_index4',
                                    'ionspec_logLratio1', 'ionspec_logLratio2', 'ionspec_logLratio3',
                                    'gas_logu', 'gas_logn', 'gas_logz', 'gas_logno', 'gas_logco']

        self.ann_parameter_order = [ # these are the parameters that are inputs into the neural networks
                                    'ionspec_index1', 'ionspec_index2', 'ionspec_index3', 'ionspec_index4',
                                    'ionspec_logLratio1', 'ionspec_logLratio2', 'ionspec_logLratio3',
                                    'gas_logq', 'gas_nH', 'gas_logz', 'gas_logno', 'gas_logco']
        
        self.params = {
            'ionspec_index1': 19.7, 
            'ionspec_index2': 5.3, 
            'ionspec_index3': 1.6, 
            'ionspec_index4': 0.6,
            'ionspec_logLratio1': 3.9, 
            'ionspec_logLratio2': 0.01, 
            'ionspec_logLratio3': 0.2,
            'gas_logu': -2.5, 
            'gas_logq':49.1,
            'gas_logn': 2,
            'gas_logz': 0., 
            'gas_logno': 0., 
            'gas_logco': 0.,
            'gas_logqion': 49.1, # gas_logq and gas_logqion are NOT the same... inferred from ionizing source. 
        }      

        theta = [jnp.atleast_1d(self.params[param]) for param in self.free_parameter_order]
        self.theta = jnp.stack(theta, axis=1).astype(jnp.float32)

        for arg, value in self.params.items():
            setattr(self, arg, jnp.atleast_1d(value))
            
        self.update_params(**kwargs)

        self.num_line_groups = len(self._line_group_names)
        self.line_group_sizes = [self._line_pca[i].means.shape[0] for i in range(self.num_line_groups)]
        self.max_line_group_size = int(jnp.max(jnp.array(self.line_group_sizes)))

    def update_params(self,theta=None,**kwargs):
        for arg, value in kwargs.items():
            if arg in self.__dict__ and value is not None:
                setattr(self, arg, jnp.atleast_1d(value).astype(jnp.float32))

        if theta is None:
            # place attributes into theta
            param_values = [jnp.atleast_1d(getattr(self, name)) for name in self.free_parameter_order]
            self.theta = jnp.atleast_2d(jnp.column_stack(param_values))
        else:
            # place theta into attributes
            theta = jnp.atleast_2d(theta)
            for i, name in enumerate(self.free_parameter_order):
                setattr(self, name, theta[:, i])
            self.theta = theta

        self.gas_logq = self.logQ()
        self.gas_nH = 10**self.gas_logn

        theta_values = [jnp.atleast_1d(getattr(self, name)) for name in self.ann_parameter_order]
        self.theta_ann = jnp.atleast_2d(jnp.column_stack(theta_values))
        self.num_concurrent = self.theta_ann.shape[0]


    
    def logQ(self, R=1e19):
        """
        Calculates the ionization parameter (log Q) based on the model parameters `gas_logu` and `gas_logn`. Uses the `cue` defaults
        if not specified.

        Args:
            R (float, optional): Ionization radius. Defaults to 1e19 cm.

        Returns:
            jnp.ndarray: Log10 of the ionization parameter Q.
        """
        return self.gas_logu + jnp.log10(4 * jnp.pi) + 2 * jnp.log10(R) + self.gas_logn + jnp.log10(c_cms)
        
    def predict_lines(self,**kwargs):
        """
        Predicts the non-attenuated line luminosities in Lsun for 138 emission lines.

        Returns:
            jnp.ndarray: Predicted emission line luminosities with shape (N, 138).
        """
        self.update_params(**kwargs)
        
        elines = jnp.zeros((self.num_concurrent, self.num_lines_tot), dtype=jnp.float32)

        j = 0
        for i in range(len(self._line_group_names)): # jax can't handle vectorizing jagged arrays; we instead iterate through the line groups.
            pca_coeffs = self._line_ann[i].predict(self.theta_ann) # passing though ANNs
            elines_norm = self._line_pca[i].inverse_transform(pca_coeffs) # inverse transform through PCA basis
            elines_log = self._denormalize(elines_norm, self._line_norm_mus[i], self._line_norm_stds[i]) # denormalizing into log luminoisity in log erg s^-1
            num_lines_from_ann = elines_log.shape[1]
            elines = elines.at[:, j:j+num_lines_from_ann].set(elines_log)
            j += num_lines_from_ann

        elines = elines[:,self.sort_by_wavelength][:,self.prospector_line_mask] # sorting by wavelength and excluding lines that prospector does not have

        self.line_luminosities = 10**(elines - jnp.log10(Lsun_cgs) - self.gas_logq[:, None] + self.gas_logqion[:, None])  # Lsun
        return self.line_luminosities

    def predict_cont(self,wave,unit='erg/s/Hz',**kwargs):
        """
        Predicts the nebular continuum based on the input parameters (theta) interpolated onto the specified stellar library wavelength grid. 

        Args:
        - wave (jnp.ndarray): Desired output restframe wavelength grid. Input shape of (M,).

        Returns:
        - jnp.ndarray: Predicted nebular continuum, putput shape of (N,M).
        """
        self.update_params(**kwargs)
        
        pca_coeffs = self._cont_ann.predict(self.theta_ann) # passing though ANN
        cont_norm = self._cont_pca.inverse_transform(pca_coeffs) # inverse transform through PCA basis
        cont_log_cgs = self._denormalize(cont_norm, self._cont_norm_mus, self._cont_norm_stds) # denormalizing into log luminosity in log erg s^-1 hz^-1 
        self._cont_lnu_cgs = 10**(cont_log_cgs - self.gas_logq[:, None] + self.gas_logqion[:, None]) # erg s^-1 hz^-1 
        self._cont_lnu_lsun = 10**(jnp.log10(self._cont_lnu_cgs) - jnp.log10(Lsun_cgs)) # Lsun hz^-1
        if unit == 'erg/s/Hz':
            self.cont = self._cont_lnu_cgs
        else:
            self.cont = self._cont_lnu_lsun
        self.spec = self.interpolate_continuum(self.cont,wave)
        return self.spec

    def predict_line_err(self):
        """
        Returns the 1sigma emulation error of the most previously predicted line luminosities. Shape (N,138).
        """
        return self.line_luminosities * self.line_frac_error[None,:]

    def predict_cont_err(self,wave):
        """
        Returns the 1sigma emulation error of the most previously predicted nebular conitnuum interpolated 
        onto the specified restframe wavelength grid, input shape (N,M).
        """
        return  self.interpolate_continuum(self.cont * self.cont_frac_error[None,:],wave)


    def _normalize(self, x, mu, std):
        """
        Normalizes the input array using the provided mean and standard deviation. This is meant for normalizing to be used within the PCA bases.

        Args:
            x (jnp.ndarray): Input array to be normalized.
            mu (jnp.ndarray): Mean values.
            std (jnp.ndarray): Standard deviation values.

        Returns:
            jnp.ndarray: Normalized array.
        """
        return (x - mu) / std

    def _denormalize(self, x, mu, std):
        """
        Denormalizes the input array using the provided mean and standard deviation. This is meant for denormalizing out of the PCA bases.

        Args:
            x (jnp.ndarray): Input array to be denormalized.
            mu (jnp.ndarray): Mean values.
            std (jnp.ndarray): Standard deviation values.

        Returns:
            jnp.ndarray: Denormalized array.
        """
        return (x * std) + mu

    def interpolate_continuum(self, continua, wave_out):
        """
        JAX-vectorized linear interpolation of nebular continua.
    
        Args:
            continua (ndarray): (N, K) continua values.
            wave_out (jnp.array): (M,) output wavelengths.
        
        Returns:
            cont_new (array): (N, M) interpolated continuum.
        """
    
        def interp_fn(cont):
            spec = jnp.interp(wave_out, self.cont_wavelengths, cont)
            return jnp.where(wave_out >= lyman_limit, spec, 0.0)
    
        cont_new = jax.vmap(interp_fn)(continua)
        return cont_new


    # def interpolate_continuum(self,continua,wave_out,k=3):
    #     """
    #     Interpolate multiple nebular continuum vectors from a common wavelength grid onto the stellar library wavelength grid.
    #     Note that this method is not ~actually~ vectorized, but still takes arguments as if it were.
        
    #     Args:
    #     - continua (ndarray): Outputted continua values from emulator. Input shape of (N,K).
    #     - wave_out (jnp.array): The restframe output wavelength grid in AA. Input shape of (M,).
    #     - k (int): degree of interpolation (defualt is cubic, k=3).
        
    #     Returns:
    #         cont_new (array): Interpolated continuum values at wave_out. Output shape of (N,M)
    #     """
    #     cont_new = jnp.zeros((self.num_concurrent,wave_out.shape[0]),dtype=jnp.float32)
    #     spec_temp = jnp.zeros(wave_out.shape[0],dtype=jnp.float32)
    #     for i,cont in enumerate(continua): # for loop since interpolation cannot be vectorized
    #         spline = InterpolatedUnivariateSpline(self.cont_wavelengths, cont, k=k)
    #         spec = jnp.where(wave_out >= lyman_limit,spline(wave_out),spec_temp)
    #         cont_new = cont_new.at[i,:].set(spec) 
    #     return cont_new

@partial(jax.jit, static_argnums=(1))
def fast_line_prediction(x,emul):
    """
    A JIT-compiled line prediction function. 

    Note that you CANNOT pass in the model parameters as kwargs here; JIT will force the function to 
    recompile every time you call it, nullifying in advantage to using JIT. Therefore, only use
    this function if you NEED speed (e.g., during model fitting/sampling) and not flexibility. 
    Additionally note that you won't be able to access all emulator class attributes (e.g., emul.line_wavelengths)
    due to JIT concretiziation.

    Args:
    - x (jnp.ndarray): Input shape of (N,12). The order of parameters that is assumed is accessed via emul.free_parameter_order. 
    - emul (cuejax.Emulator): Instantiated emulator with all optional parameters specified (especially gas_logqion).

    Returns:
    - jnp.ndarray: line luminosities in Lsun, shape of (N,138).
    - jnp.ndarray: line luminosity 1sigma emulation error, shape of (N,138).
    """
    lines = emul.predict_lines(theta=x)
    err = emul.predict_line_err()
    return lines, err

@partial(jax.jit, static_argnums=(2,3))
def fast_cont_prediction(x,wave,emul,unit='erg/s/Hz'):
    """
    A JIT-compiled continuum prediction function. 

    Note that you CANNOT pass in the model parameters as kwargs here; JIT will force the function to 
    recompile every time you call it, nullifying in advantage to using JIT. Therefore, only use
    this function if you NEED speed (e.g., during model fitting/sampling) and not flexibility. 
    Additionally note that you won't be able to access all emulator class attributes (e.g., emul.cont_wavelengths)
    due to JIT concretiziation.

    Args:
    - x (jnp.ndarray): Input shape of (N,12). The order of parameters that is assumed is accessed via emul.free_parameter_order. 
    - emul (cuejax.Emulator): Instantiated emulator with all optional parameters specified (especially gas_logqion).

    Returns:
    - jnp.ndarray: nebular continuum in the unit specifed, shape of (N,M).
    - jnp.ndarray: nebular continuum 1sigma emulation error in the unit specified, shape of (N,M).
    """
    cont = emul.predict_cont(wave,theta=x,unit=unit)
    err = emul.predict_cont_err(wave)
    return cont, err