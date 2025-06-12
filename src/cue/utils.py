### functions for generating ionizing spectrum
import numpy as np
import dill as pickle
from .constants import Lsun,c,h,HeI_edge,HeII_edge,OII_edge
try:
    from pkg_resources import resource_filename, resource_listdir
except(ImportError):
    pass

from scipy.optimize import minimize

cont_lam = np.genfromtxt(resource_filename("cue", "data/FSPSlam.dat"))
cont_nu = c/cont_lam

#def get_linewavelength(lines):
#    wavelength = np.array([i[5:12] for i in lines], dtype=float)
#    unit = np.array([i[12:] for i in lines], dtype=str)
#    factor = np.ones(np.shape(unit))
#    factor[unit=='m'] = 1e4
#    return wavelength*factor
#unsorted_line_name = np.genfromtxt(resource_filename("cue", "data/lineList128lines.dat"), delimiter='\n', dtype="S20")
#unsorted_line_name = np.array([i.decode() for i in unsorted_line_name])
#unsorted_line_lam = get_linewavelength(unsorted_line_name)
#line_name = unsorted_line_name[np.argsort(unsorted_line_lam)]
#line_lam = np.sort(unsorted_line_lam)

new_unsorted_line_name = np.load(resource_filename("cue", "data/lineList_replaceblnd_name.npy"))
new_unsorted_line_lam = np.load(resource_filename("cue", "data/lineList_wav.npy"))
new_sorted_line_name = new_unsorted_line_name[np.argsort(new_unsorted_line_lam)]
new_sorted_line_lam = np.sort(new_unsorted_line_lam)
new_ele_arr = np.array([i[:4].rstrip() for i in new_sorted_line_name])
line_new_added = np.where((new_sorted_line_lam == 4685.68) | (new_sorted_line_lam == 1550.77) |
                          (new_sorted_line_lam == 1548.19) | (new_sorted_line_lam == 1750.00) |
                          (new_sorted_line_lam == 2424.28) | (new_sorted_line_lam == 1882.71) |
                          (new_sorted_line_lam == 1892.03) | (new_sorted_line_lam == 1406.02) |
                          (new_sorted_line_lam == 4711.26) | (new_sorted_line_lam == 4740.12))[0]
line_old = np.arange(138)[~np.isin(np.arange(138), line_new_added)]
nn_name = np.array(['H1', 'He1', 'He2', 'C1', 'C2C3', 'C4', 'N', 'O1', 'O2', 'O3',
                    'ionE_1', 'ionE_2', 'S4', 'Ar4', 'Ne3', 'Ne4'])
nn_ion = np.array([['H  1'], ['He 1'], ['He 2'], ['C  1'], ['C  2', 'C  3'], ['C  4'],
                               ['N  1', 'N  2', 'N  3'], ['O  1'], ['O  2'], ['O  3'],
                               ['Mg 2', 'Fe 2', 'Si 2', 'Al 2', 'P  2', 'S  2', 'Cl 2', 'Ar 2'],
                               ['Al 3', 'Si 3', 'S  3', 'Cl 3', 'Ar 3', 'Ne 2'],
                               ['S  4'], ['Ar 4'], ['Ne 3'], ['Ne 4']], dtype=object)
nn_wav_selection = list()
for this_line_ion in nn_ion:
    if np.size(this_line_ion) == 1:
        #wav_ind = np.array(new_ele_arr == this_line_ion)
        wav_selection, = np.where(new_ele_arr == this_line_ion)
    else:
        wav_selection = list()
        for i in this_line_ion:
            wav_selection.append(np.where(new_ele_arr == i)[0])
        wav_selection = np.sort(np.concatenate(wav_selection))
    nn_wav_selection.append(wav_selection)
nn_wav_selection = np.array(nn_wav_selection, dtype=object)
nn_wavelength = new_sorted_line_lam[np.concatenate(nn_wav_selection)]

line_name = new_sorted_line_name[line_old]
line_lam = new_sorted_line_lam[line_old]

nn_stats = pickle.load(open(resource_filename("cue", "data/nn_stats_v0.pkl"), "rb"))
sigma_line_for_fsps = 1./nn_stats['SN_quantile'][1][nn_stats['fsps_ind']][np.argsort(nn_stats['wav'][nn_stats['fsps_ind']])]


### power law functions
def linear(logλ, α, logA):
    return logA + α*logλ

def L2norm(index, logL, edges=[HeII_edge, OII_edge, HeI_edge, 911.6], wav=None):
    """
    Transform the log of the integrated fluxes of each segment into the log of the normalizations.
    log Α = log L - log (c*L_sun) - log ( (λ_max^(α-1) - λ_min^(α-1)) / (α-1) )
    :par index:
        (N, M) power law indexes, each spectrum is consist of M power laws
    :par logL:
        (N, M) log of the integrated flux in M bins
    :par edges:
        (M,) ionization edges of the power laws, i.e., upper limit of each bin, default [HeII_edge, OII_edge, HeI_edge, 911.6]
    :returns log of the normalization of each power law (N, M)
    """
    log_norm = np.zeros_like(index)
    if np.any(wav == None):
        edges = np.hstack([1, edges])
        for i in range(len(edges)-1):
            log_norm[:,i] = logL[:,i]-np.log10(c*Lsun)-np.log10(np.abs((edges[i+1]**(index[:,i]-1)-edges[i]**(index[:,i]-1))/(index[:,i]-1)))
            if np.any(index[:,i] == 1):
                one_ind = np.where(index[:,i] == 1)[0]
                log_norm[one_ind,i] = logL[one_ind,i]-np.log10(c*Lsun)-np.log10(np.abs(np.log10(edges[i+1])-np.log10(edges[i])))
    else:
        ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in edges])+1 #np.array([np.argmin(np.abs(ssp_wavelength-λ)) for λ in λ_bin])+1
        ind_bin = np.insert(ind_bin, 0, 0)
        for i in range(len(ind_bin)-1):
            log_norm[:,i] = logL[:,i]-np.log10(c*Lsun)-np.log10(np.abs((wav[ind_bin[i+1]-1]**(index[:,i]-1)-wav[ind_bin[i]]**(index[:,i]-1))/(index[:,i]-1)))
            if np.any(index[:,i] == 1):
                one_ind = np.where(index[:,i] == 1)[0]
                log_norm[one_ind,i] = logL[one_ind,i]-np.log10(c*Lsun)-np.log10(np.abs(np.log10(wav[ind_bin[i+1]-1])-np.log10(wav[ind_bin[i]])))
    return log_norm

def ionparam2norm(ionparam):
    """
    Transform the 7 cue ionizing spectrum parameters into the power law parameters.
    log Α = log L - log (c*L_sun) - log ( (λ_max^(α+1) - λ_min^(α-1)) / (α-1) )
    :par ionparam:
        (7,) power law parameters, 4 power law indexes and 3 flux ratios of the two nearby segments
    :returns {(4,2), (4,2)} of the power law parameters and fluxes of each bin. Note that the spectrum is assumed to start at 1 Angstrom.
     The scale of the returned normalizations and integrated fluxes is arbitrary.
    """
    edges = [1, HeII_edge, OII_edge, HeI_edge, 911.6]
    logL1 = 0 #-30*np.ones(num_runs)
    logL2 = logL1+ionparam[4]
    logL3 = logL2+ionparam[5]
    logL4 = logL3+ionparam[6]
    logL = np.array([logL1, logL2, logL3, logL4])
    index = ionparam[:4]
    log_norm = np.zeros(4)
    for i in range(4):
        if index[i] == 1:
            log_norm[i] = logL[i]-np.log10(c*Lsun)-np.log10(np.abs(np.log10(edges[i+1])-np.log10(edges[i])))
        else:
            log_norm[i] = logL[i]-np.log10(c*Lsun)-np.log10(np.abs((edges[i+1]**(index[i]-1)-edges[i]**(index[i]-1))/(index[i]-1)))
    return np.vstack([ionparam[:4], log_norm]).T, np.vstack([ionparam[:4], logL]).T

def Ltotal(param=np.zeros((1,4,2)), wav=None, spec=None, edges=[HeII_edge, OII_edge, HeI_edge, 911.6]):
    """
    Calculate logL at each bin given the power law parameters.
    log L = log A + log (c*L_sun) + log ( (λ_max^(α-1) - λ_min^(α-1)) / (α-1) )
    :par param:
        (N, M, 2) power law parameters, param[:,:,0] are the indexes α, param[:,:,1] are the log normalizations log A
    :par edges:
        (M,) ionization edges of the power laws, i.e., upper limit of each bin, default [HeII_edge, OII_edge, HeI_edge, 911.6]
    :returns log of the integrating Q in each bin (N, M), the unit is arbitrary unless the power laws are in Fnu/Lsun. Note that the spectrum is assumed to start at 1 Angstrom.
    """
    log_Ltotal = np.zeros((len(param), len(edges)))
    if np.any(wav == None):
        edges = np.hstack([1, edges])
        for i in range(len(edges)-1):
            log_Ltotal[:,i] = param[:,i,1]+ np.log10(c*Lsun) + np.log10(np.abs((edges[i+1]**(param[:,i,0]-1)
                                                                            -edges[i]**(param[:,i,0]-1))/(param[:,i,0]-1)))
    else:
        ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in [HeII_edge, OII_edge, HeI_edge, 911.6]])+1 #np.array([np.argmin(np.abs(ssp_wavelength-λ)) for λ in λ_bin])+1
        ind_bin = np.insert(ind_bin, 0, 0)
        for i in range(len(ind_bin)-1):
            log_Ltotal[:,i] = param[:,i,1]+np.log10(c*Lsun)+\
            np.log10(np.abs((wav[ind_bin[i+1]-1]**(param[:,i,0]-1)-wav[ind_bin[i]]**(param[:,i,0]-1))/(param[:,i,0]-1)))
    return log_Ltotal

def Qtotal(param, edges=[1, HeII_edge, OII_edge, HeI_edge, 911.6]):
    """
    Calculate logQ at each bin given the power law parameters.
    log Q = log A + log L_sun - log h + log ( (λ_max^α - λ_min^α) / α )
    :par param:
        (M, 2) power law parameters, param[:,0] are the indexes α, param[:,1] are the log normalizations log A
    :par edges:
        (M+1,) edges of the power law segments, i.e., lower and upper limit of each bin, default [HeII_edge, OII_edge, HeI_edge, 911.6]
    :returns log of the integrating Q in each bin (M,), the unit is arbitrary unless the power laws are in Fnu/Lsun. Note that the spectrum is assumed to start at 1 Angstrom.
    """
    log_Qtotal = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        log_Qtotal[i] = param[i,1] + np.log10(Lsun) - np.log10(h) + np.log10((edges[i+1]**param[i,0]
                                                                   -edges[i]**param[i,0])/param[i,0])
    return log_Qtotal

def calcQ(lamin0, specin0, mstar=1.0, helium=False, f_nu=True):
    '''
    Calculate the number of lyman ionizing photons for given spectrum
    Input spectrum must be in ergs/s/A if f_nu=False
    Q = int(Lnu/hnu dnu, nu_0, inf)
    '''
    try:
        from scipy.integrate import simps
    except:
        from scipy.integrate import simpson as simps    
    lamin = np.asarray(lamin0)
    specin = np.asarray(specin0)
    if helium:
        lam_0 = HeI_edge
    else:
        lam_0 = 911.6
    if f_nu:
        nu_0 = c/lam_0
        inds, = np.where(c/lamin >= nu_0)
        hlam, hflu = c/lamin[inds], specin[inds]
        nu = hlam[::-1]
        f_nu = hflu[::-1]
        integrand = f_nu/(h*nu)
        Q = simps(integrand, x=nu)
    else:
        inds, = np.nonzero(lamin <= lam_0)
        lam = lamin[inds]
        spec = specin[inds]
        integrand = lam*spec/(h*c)
        Q = simps(integrand, x=lam)*mstar
    return Q

def calcQs(lamin0, specin0, edges=[1, HeII_edge, OII_edge, HeI_edge, 911.6]):
    """
    Calculate the number of ionizing photons for the given spectrum at different segments.
    The default edge of each bin is [1, HeII_edge, OII_edge, HeI_edge, 911.6].
    Input spectrum must be in ergs/s/Hz!!
    Q = int(Lnu/hnu dnu, nu_min, nu_max)
    """
    try:
        from scipy.integrate import simps
    except:
        from scipy.integrate import simpson as simps    
    lamin = np.asarray(lamin0)
    specin = np.asarray(specin0)
    c = 2.9979e18 #ang/s
    h = 6.626e-27 #erg/s
    Qs = list()
    for lam_ind in range(len(edges)-1):
        nu_min = c/edges[lam_ind+1]
        nu_max = c/edges[lam_ind]
        inds, = np.where((c/lamin >= nu_min) & (c/lamin <= nu_max))
        hlam, hflu = c/lamin[inds], specin[inds]
        nu = hlam[::-1]
        f_nu = hflu[::-1]
        integrand = f_nu/(h*nu)
        #cs = scipy.interpolate.CubicSpline(nu, integrand, extrapolate=True)
        #Qs.append(scipy.integrate.quad(lambda x: cs(x), nu_min, nu_max)[0])
        Qs.append(simps(integrand, x=nu))
    return np.array(Qs)

def get_loglinear_spectra(wav, param, ion_edges=[HeII_edge, OII_edge, HeI_edge]):
    """
    Calculate ionizing spectrum given the wavelength and power law parameters.
    The power laws are calculated at the segments [wav[0], ion_edges, wav[-1]].
    :par wav:
        (N,) wavelengths, AA
    :par param:
        (M, 2) power law parameters, param[:,0] are the indexes, param[:,1] are the log normalizations
    :par ion_edges:
        (M-1,) edges of each part of the power law, default [HeII_edge, OII_edge, HeI_edge]
    :returns Fnu at the input wavelength (N,)
    """
    edges = np.hstack([ion_edges, np.max(wav)])
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in edges]) + 1 #np.array([np.argmin(np.abs(ssp_wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    spec = np.zeros_like(wav)
    for i in range(len(ind_bin)-1):
        spec[ind_bin[i]:ind_bin[i+1]] = 10**linear(np.log10(wav[ind_bin[i]:ind_bin[i+1]]),
                                                   param[i,0], param[i,1])
    return spec

def logQ(logU, R=1e19, lognH=2):
    c = 2.9979e10 #cm/s
    return logU + np.log10(4*np.pi) + 2*np.log10(R) + lognH + np.log10(c)

def logU(logQ, R=1e19, lognH=2):
    c = 2.9979e10 #cm/s
    return logQ - np.log10(4*np.pi) - 2*np.log10(R) - lognH - np.log10(c)

def spec_normalized(wav, spec):
    """wav in Angstrom; spec in Lnu; return nuLnu
    """
    wav_ind, = np.where(wav<=911.6)
    if np.array(spec).ndim==1:
        norm = np.abs(np.trapz(spec[wav_ind]*Lsun, x=c/wav[wav_ind]))
        return spec*Lsun*c/wav/norm
    elif np.array(spec).ndim==2:
        norm = np.abs(np.trapz(spec[:,wav_ind]*Lsun, x=c/wav[wav_ind], axis=1))
        return spec*Lsun*c/wav/norm.reshape((len(spec),1))

### fit functions
logh = np.log10(h)
ln10 = np.log(10)

def customize_loss_funtion_loglinear_analytical(params, λmin, λmax, y_pred, y_true, log_Q_true, Ndata, sample_weights=None):
    """Loss function for fitting the powerlaws.
    loss = 0.5 \sum (y_pred-y_true)^2 + 0.5 N (\log10 Q_true - \log10 Q_pred)^2
    N is the number of data points, Q is the ionizing photon rates of this segment from integrating spectrum/hν
    """
    #y_true = np.array(y_true)
    #y_pred = np.array(y_pred)
    #assert len(λ) == len(y_true)
    log_Q_pred = params[1] - logh + np.log10( (λmax**params[0] - λmin**params[0])/params[0] )
    #np.abs(np.trapz(10**y_pred*λ/(h*c), x=c/λ))
    #y_pred = linear(np.log10(λ), *params)
    term1 = (y_true - y_pred)
    term2 = (log_Q_true - log_Q_pred)
    return 0.5 * np.sum(term1*term1) + \
           0.5 * (term2*term2) * Ndata

def objective_func_loglinear_analytical(params, Xmin, Xmax, lnXmin, lnXmax, logX, logY, logQ, Ndata):
    return customize_loss_funtion_loglinear_analytical(params, Xmin, Xmax, linear(logX, *params), logY,
                                                       logQ, Ndata)

def gradient_func_loglinear_analytical(params, Xmin, Xmax, lnXmin, lnXmax, logX, logY, logQ, Ndata):
    #Ndata = len(logX)
    power_xmin = Xmin**params[0]
    power_xmax = Xmax**params[0]
    term_Q = ( params[1] + np.log10( (power_xmax - power_xmin)/params[0] ) - logQ - logh)
    term_sum = (params[1] + params[0] * logX - logY)
    grad_slope = np.sum( term_sum * logX ) + \
    Ndata / ln10 * term_Q *\
    ( (power_xmax * lnXmax - power_xmin * lnXmin) / (power_xmax - power_xmin) - 1./params[0] )
    grad_norm = np.sum( term_sum ) + \
    Ndata * term_Q
    return grad_slope, grad_norm

def fit_4loglinear_ionparam(wav, spec, λ_bin=[HeII_edge, OII_edge, HeI_edge, 911.6]):
    """Fit 4 powerlaws to the given spectrum.
    :param wav:
        (N,) wavelengths, AA
    :param spec:
        (N,) fluxes, Lsun/Hz
    :param λ_bin:
        edges of the 4 powerlaws (default: ionization edges of HeII, OII, HeI, and HII), AA
    :returns ionizing parameters
    """
    ind_bin = np.array([max(np.where(wav<=λ)[0]) for λ in λ_bin]) + 1 #np.array([np.argmin(np.abs(wav-λ)) for λ in λ_bin])+1
    ind_bin = np.insert(ind_bin, 0, 0)
    coeff = np.zeros((len(ind_bin)-1, 2))
    norm = 1e-18/np.median(spec[ind_bin[-1]]) ### normalize the input spec, so that the minimize function can find the right solution from the given initial parameters
    normalized_spec = np.clip(np.squeeze(np.array(spec)*norm), 1e-70*norm, np.inf)
    for i in range(len(ind_bin)-1):
#        if np.min(spec[ind_bin[i]:ind_bin[i+1]])>0:
        pos_ind, = np.where((np.squeeze(spec)[ind_bin[i]:ind_bin[i+1]])>0)
        if np.size(pos_ind)==0:
            coeff[i] = [0, -np.inf]
        else:
            this_x = np.array(wav[ind_bin[i]:ind_bin[i+1]])
            this_xmin = this_x[0]
            this_xmax = this_x[-1]
            this_spec = normalized_spec[ind_bin[i]:ind_bin[i+1]]
            init_slope = np.log10(this_spec[-1]/this_spec[0]) / \
                         np.log10(this_xmax/this_xmin)
            init_norm = np.log10(this_spec[-1]) - init_slope * np.log10(this_xmax)
            Q_true = np.abs(np.trapz(this_spec*this_x/(h*c), x=c/this_x))
            assert len(this_x) == len(this_spec)
            res = minimize(objective_func_loglinear_analytical, [init_slope, init_norm],
                           jac=gradient_func_loglinear_analytical,
                           args=(this_xmin, this_xmax, np.log(this_xmin), np.log(this_xmax),
                                 np.log10(this_x), np.log10(this_spec), np.log10(Q_true), len(this_x)),
                           bounds=[(-40,100), (-200, 100)],
                           method= "SLSQP", #"BFGS", #"L-BFGS-B",
                          )
            coeff[i] = res.x
            coeff[i,1] = coeff[i,1]-np.log10(norm)

    logLratios = np.diff(np.squeeze(Ltotal(param=coeff.reshape(1,4,2))))
    # total QH (log_qion) will be used in normalizing cue outputs and to convert Ls to powerlaw parameters
    logQ = np.log10(calcQ(wav, spec*3.839E33)) #np.log10(np.sum(10**Qtotal(param=coeff)))
    return {'ionspec_index1': np.clip(coeff[0,0], 1, 42), 'ionspec_index2': np.clip(coeff[1,0], -0.3, 30), 'ionspec_index3': np.clip(coeff[2,0], -1, 14), 'ionspec_index4': np.clip(coeff[3,0], -1.7, 8),
            'ionspec_logLratio1': np.clip(logLratios[0], -1, 10.1), 'ionspec_logLratio2': np.clip(logLratios[1], -0.5, 1.9), 'ionspec_logLratio3': np.clip(logLratios[2], -0.4, 2.2),
            "gas_logqion": logQ, 'powerlaw_params': coeff}
