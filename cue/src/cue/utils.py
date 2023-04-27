### functions for generating ionizing spectrum
import numpy as np
from .constants import Lsun,c,h,HeI_edge,HeII_edge,OII_edge
try:
    from pkg_resources import resource_filename, resource_listdir
except(ImportError):
    pass


cont_lam = np.genfromtxt(resource_filename("cue", "data/FSPSlam.dat"))
cont_nu = c/cont_lam

def get_linewavelength(lines):
    wavelength = np.array([i[5:12] for i in lines], dtype=float)
    unit = np.array([i[12:] for i in lines], dtype=str)
    factor = np.ones(np.shape(unit))
    factor[unit=='m'] = 1e4
    return wavelength*factor
unsorted_line_name = np.genfromtxt(resource_filename("cue", "data/lineList.dat"), delimiter='\n', dtype="S20")
unsorted_line_name = np.array([i.decode() for i in unsorted_line_name])
unsorted_line_lam = get_linewavelength(unsorted_line_name) 
line_name = unsorted_line_name[np.argsort(unsorted_line_lam)]
line_lam = np.sort(unsorted_line_lam)


def linear(logλ, α, logA):
    return logA + α*logλ

def L2norm(wav, index, logL):
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in [HeII_edge, OII_edge, HeI_edge, 912]])+1 #np.array([np.argmin(np.abs(ssp_wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    log_norm = np.zeros(np.shape(index))
    for i in range(len(ind_bin)-1):
        if index[:,i] == 1:
            log_norm[:,i] = logL[:,i]-np.log10(c*Lsun)-np.log10(np.abs(np.log(wav[ind_bin[i+1]-1])-np.log(wav[ind_bin[i]])))
        else:
            log_norm[:,i] = logL[:,i]-np.log10(c*Lsun)-np.log10(np.abs((wav[ind_bin[i+1]-1]**(index[:,i]-1)-wav[ind_bin[i]]**(index[:,i]-1))/(index[:,i]-2)))
    return log_norm

def Ltotal(param=np.zeros((4,2)), wav=None, spec=None):
    """wav in Angstrom; spec in Lnu"""
    Lsun = 3.828e+33 #erg/s
    c = 2.9979e18 #angstrom/s
    edges = np.array([0, HeII_edge, OII_edge, HeI_edge, 912])
    #wav = np.load(resource_filename("cue", "data/Powerlaw_wavelength.npy"))
    #ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in [HeII_edge, OII_edge, HeI_edge, 912]])+1 #np.array([np.argmin(np.abs(ssp_wavelength-λ)) for λ in λ_bin])+1
    #ind_bin = np.insert(ind_bin, 0, 0)
    log_Ltotal = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        #log_Ltotal[i] = param[i,1]+np.log10(np.abs((wav[ind_bin[i+1]-1]**(param[i,0]-1)
        #                                                -wav[ind_bin[i]]**(param[i,0]-1))/(param[i,0]-2)))
        log_Ltotal[i] = param[i,1]+np.log10(np.abs((edges[i+1]**(param[i,0]-1)
                                                    -edges[i]**(param[i,0]-1))/(param[i,0]-2)))
    return log_Ltotal

def get_4loglinear_spectra(wav, param):
    """Return ionizing spectrum given the parameters of the log linear fits.
    """
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in [HeII_edge, OII_edge, HeI_edge, np.max(wav)]]) + 1 #np.array([np.argmin(np.abs(ssp_wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    spec = np.zeros(np.shape(wav))
    for i in range(len(ind_bin)-1):
        spec[ind_bin[i]:ind_bin[i+1]] = 10**linear(np.log10(wav[ind_bin[i]:ind_bin[i+1]]),
                                                   param[i,0], param[i,1])
    return spec

def logQ(logU, R=1e19, lognH=2):
    c = 2.9979e10 #cm/s
    return logU + np.log10(4*np.pi) + 2*np.log10(R) + lognH + np.log10(c)

def spec_normalized(wav, spec):
    """wav in Angstrom; spec in Lnu; return nuLnu
    """
    wav_ind, = np.where(wav<912)
    norm = np.abs(np.trapz(spec[wav_ind]*Lsun, x=c/wav[wav_ind])) #np.abs(np.trapz(spec[:,wav_ind]*Lsun, x=c/wav[wav_ind], axis=1))#*wav/c
    return spec*Lsun*c/wav/norm #spec*Lsun*c/wav/norm.reshape((len(spec),1))


### fit functions
def customize_loss_funtion_loglinear(y_pred, y_true, λ=None, sample_weights=None):
    """Loss function for fitting the powerlaws. 
    loss = 0.5 \sum (y_pred-y_true)^2 + 0.5 N (\log10 Q_true - \log10 Q_pred)^2
    N is the number of data points, Q is the ionizing photon rates of this segment from integrating spectrum/hν
    """
    c = 2.9979e18 #ang/s
    h = 6.626e-27 #erg/s
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(λ) == len(y_true)
    #Q_true = cloudyfsps.generalTools.calcQ(λ, 10**y_true, f_nu=True) #*c.L_sun.cgs.value
    #Q_pred = cloudyfsps.generalTools.calcQ(λ, 10**y_pred, f_nu=True) #*c.L_sun.cgs.value
    Q_true = np.abs(np.trapz(10**y_true*λ/(h*c), x=c/λ))
    Q_pred = np.abs(np.trapz(10**y_pred*λ/(h*c), x=c/λ))
    return 0.5 * np.sum((y_true - y_pred)**2) + \
           0.5 * ((np.log10(Q_true) - np.log10(Q_pred))**2) * len(λ)

from scipy.optimize import minimize
def objective_func_loglinear(params, X, Y):
    return customize_loss_funtion_loglinear(linear(np.log10(X), *params), np.log10(Y), X)

def fit_4loglinear(wav, spec, λ_bin=[HeII_edge, OII_edge, HeI_edge, 912]):
    """Fit 4 powerlaws to the given spectrum.
    :param wav:
        (N,) wavelengths, AA
    :param spec:
        (N,) fluxes, Lsun/Hz
    :param λ_bin:
        edges of the 4 powerlaws (default: ionization edges of HeII, OII, HeI, and HII), AA
    :returns coeff:
        (4,2) index and log of the normalization of the powerlaws
    """
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in λ_bin]) + 1 #np.array([np.argmin(np.abs(wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    coeff = np.zeros((len(ind_bin)-1, 2))
    for i in range(len(ind_bin)-1):
#        if np.min(spec[ind_bin[i]:ind_bin[i+1]])>0:
        pos_ind, = np.where((np.squeeze(spec)[ind_bin[i]:ind_bin[i+1]])>0)
        if np.size(pos_ind)==0:
            coeff[i] = [0, -np.inf]
        else:
            norm = 1e-18/np.median(spec[ind_bin[-1]]) ### normalize the input spec, so that the minimize function can find the right solution from the given initial parameters
            res = minimize(objective_func_loglinear, [10, -30], 
                           args=(wav[ind_bin[i]:ind_bin[i+1]], 
                                 np.clip(np.squeeze(spec*norm), 1e-70, np.inf)[ind_bin[i]:ind_bin[i+1]])
                          )
            coeff[i] = res.x #popt
            coeff[i,1] = coeff[i,1]-np.log10(norm)
    return coeff
