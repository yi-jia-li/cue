### functions for generating ionizing spectrum
import numpy as np
from .constants import Lsun,c,h,HeI_edge,HeII_edge,OII_edge
try:
    from pkg_resources import resource_filename, resource_listdir
except(ImportError):
    pass

def Linear(logλ, α, logA):
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

def get_4logLinear_spectra(wav, param):
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in [HeII_edge, OII_edge, HeI_edge, np.max(wav)]]) + 1 #np.array([np.argmin(np.abs(ssp_wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    spec = np.zeros(len(wav))
    for i in range(len(ind_bin)-1):
        spec[ind_bin[i]:ind_bin[i+1]] = 10**Linear(np.log10(wav[ind_bin[i]:ind_bin[i+1]]),
                                                   param[i,0], param[i,1])
    return spec


def logQ(logU, R=1e19, nH=100):
    return logU + np.log10(4*np.pi) + 2*np.log10(R) + np.log10(nH*c/1e8)


def spec_normalized(wav, spec):
    """wav in Angstrom; spec in Lnu; return nuLnu
    """
    wav_ind, = np.where(wav<912)
    norm = np.abs(np.trapz(spec[wav_ind]*Lsun, x=c/wav[wav_ind])) #np.abs(np.trapz(spec[:,wav_ind]*Lsun, x=c/wav[wav_ind], axis=1))#*wav/c
    return spec*Lsun*c/wav/norm #spec*Lsun*c/wav/norm.reshape((len(spec),1))


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
