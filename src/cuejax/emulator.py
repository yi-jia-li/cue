from efsps.sources.constants import lyman_limit
from efsps.sources.transforms import maggies_to_cgs
from efsps.models import priors, NebModel

import jax.numpy as jnp
import numpy as np

def build_cue_model(ionspec_index1=19.7, ionspec_index2=5.3, ionspec_index3=1.6, ionspec_index4=0.6,
                 ionspec_logLratio1=3.9, ionspec_logLratio2=0.01, ionspec_logLratio3=0.2,
                 gas_logu=-2.5, gas_logn=2, gas_logz=0., gas_logno=0., gas_logco=0.,
                 gas_logqion=49.1,**extras):

    params = {}

    params['ionspec_index1'] = {"N": 1, 'isfree': True,
            'init': ionspec_index1, 
            'prior': priors.UniformPrior(min=1, max=42)
                         }
    params['ionspec_index2'] = {"N": 1, 'isfree': True,
            'init': ionspec_index2, 
            'prior': priors.UniformPrior(min=-0.3, max=30)
                         }
    params['ionspec_index3'] = {"N": 1, 'isfree': True,
            'init': ionspec_index3, 
            'prior': priors.UniformPrior(min=-1, max=14)
                         }
    params['ionspec_index4'] = {"N": 1, 'isfree': True,
            'init': ionspec_index4, 
            'prior': priors.UniformPrior(min=-1.7, max=8)
                         }
    params['ionspec_logLratio1'] = {"N": 1, 'isfree': True,
            'init': ionspec_logLratio1, 
            'prior': priors.UniformPrior(min=-1, max=10.1)
                         }
    params['ionspec_logLratio2'] = {"N": 1, 'isfree': True,
            'init':ionspec_logLratio2, 
            'prior': priors.UniformPrior(min=-0.5, max=1.9)
                         }
    params['ionspec_logLratio3'] = {"N": 1, 'isfree': True,
            'init': ionspec_logLratio3, 
            'prior': priors.UniformPrior(min=-0.4, max=2.2)
                         }

    params['gas_logu'] = {"N": 1, 'isfree': True,
            'init': gas_logu, 
            'prior': priors.UniformPrior(min=-4.0, max=-1)
                         }

     # note that this is NOT in log, as this is a direct input into cue in linear units
    params['gas_logn'] = {'N': 1, 'isfree': True, 'init': gas_logn,'prior': priors.UniformPrior(min=1, max=4)}

    params['gas_logz'] = {'N': 1, 'isfree': True,
            'init': gas_logz, 'units': r'log Z/Z_\odot',
            'prior': priors.UniformPrior(min=-2.0, max=0.5)
                         }

    params['gas_logno'] = {"N": 1, 'isfree': True,
            'init': gas_logno, 
            'prior': priors.UniformPrior(min=-1, max=np.log10(5.4))
                         }

    params['gas_logco'] = {"N": 1, 'isfree': True,
            'init': gas_logco, 
            'prior': priors.UniformPrior(min=-1, max=np.log10(5.4))
                         }

    params['gas_logqion'] = {'N': 1, 'isfree': True,
            'init': gas_logqion, 
            'prior': priors.UniformPrior(min=30, max=50)
                         }


    # --- fixed parameters 
    params['add_stars'] = {"N": 1, "isfree": False,"init": False} # don't load in the stellar emulators
    params['use_stellar_ionizing'] = {"N": 1, "isfree": False, "init": False}
    params["add_duste"] = {"N": 1, "isfree": False, "init": False}
    params['add_neb_emission'] = {'N': 1, 'isfree': False, 'init':True}
    params['nebemlineinspec'] = {'N': 1, 'isfree': False, 'init': False}
    params['add_igm_absorption'] = {'N': 1, 'isfree': False, 'init': False}
    params['igm_damping'] = {'N': 1, 'isfree': False, 'init': False}
    params["add_dust_emission"] = {"N": 1, "isfree": False,"init": False}
    params['add_neb_lines'] = {'N': 1, 'isfree': False, 'init':True} # load in line emulators
    params['add_neb_continuum'] = {'N': 1, 'isfree': False, 'init': True} # load in the nebular continuum emulator

    return NebModel(params) # if not predicting stellar continuum, you don't need to set a stellar emulator model path



class Emulator():
    """
    theta must be inputted as a shape (N,12) in order of:
        - ionspec_index1
        - ionspec_index2
        - ionspec_index3
        - ionspec_logLratio1
        - ionspec_logLratio2
        - ionspec_logLratio3
        - gas_logu --> used to calculate gas_logq
        - gas_logn
        - gas_logz
        - gas_logno
        - gas_logco

        - gas_logqion is fixed unless specified by a kwarg
    """
    def __init__(self,**kwargs):
        self.model = build_cue_model(**kwargs)
        self.theta = kwargs.get('theta', self.model.theta)
        self.cont_lam = self.model.spec_wavelengths
        self.line_wavelength = self.model.line_wavelengths

        params = ['ionspec_index1', 'ionspec_index2', 'ionspec_index3', 'ionspec_index4',
                'ionspec_logLratio1', 'ionspec_logLratio2', 'ionspec_logLratio3',
                'gas_logn', 'gas_logz', 'gas_logno', 'gas_logco','gas_logqion']

        for param in params:
            setattr(self, param, self.model.params[param])

    def update(self,theta=None,**kwargs):
        if theta is None:
            theta = self.model.theta.copy()
            for param, value in kwargs.items():
                if value is not None:
                    setattr(self, param, value)
                    idx = jnp.where(self.model.free_parameter_order==param)
                    theta = theta.at[:,idx].set(value) 
        if theta.shape[1] == 11:
            self.theta = jnp.hstack([theta,self.gas_loqion]) # theta vector does not contain logqion 
        if theta.shape[2] == 12:
            self.theta = theta
                

    def predict_lines(self,**kwargs):
        """hard coded to return the 128 FSPS cloudy grid lines. Lines in units of Lsun"""
        self.update(**kwargs)
        return self.model.predict_lines(self.theta) # jit-compiled
        
    def predict_cont(self,wave,**kwargs):
        """continuum in erg/s/cm^2/Hz"""
        self.update(**kwargs)
        cont_maggies = self.model.predict_cont(self.theta,wave) # jit-compiled, interpolates
        cont = maggies_to_cgs(cont_maggies)
        return cont