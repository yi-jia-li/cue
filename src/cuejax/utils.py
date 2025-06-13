import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver
from jax import lax
from functools import partial
jax.config.update("jax_debug_nans", False)
import jaxopt

from scipy.optimize import minimize

c = 2.9979e18 #ang/s
Lsun = 3.828e+33 #erg/s
h = 6.626e-27 #erg/s
    
HeI_edge = 1e8/198310.66637
HeII_edge = 1e8/438908.8789
OII_edge = 1e8/283270.9

logh = np.log10(h)
ln10 = np.log(10)

def calculate_Q(wave, spec):
    '''
    spec in Lsun/Hz
    '''
    lam_0 = 911.6
    nu_0 = c/lam_0

    nu = c/wave

    spec_pad = jnp.where(nu>=nu_0,spec,1e-10)    
    ln_spec_cgs = jnp.log(spec_pad) - jnp.log(3.839e33)
    ln_integrand = ln_spec_cgs - jnp.log(h*nu)
    integrand = jnp.exp(ln_integrand)
    integrand = jnp.where(nu>=nu_0,integrand,0)
    Q = jnp.trapezoid(integrand[::-1], x=nu[::-1])
    return Q

def total_log_luminosity(param=jnp.zeros((1,4,2)), wave=None, spec=None, ion_edges=jnp.array([HeII_edge, OII_edge, HeI_edge, 911.6])):

    num_edges = len(ion_edges)
    log_Ltotal = jnp.zeros((len(param),num_edges))
    edges = jnp.hstack([1, ion_edges])
    log_Ltotal = jnp.zeros((len(param), num_edges))

    for i in range(num_edges-1):
        a = ion_edges[i+1]**(param[:, i, 0] - 1)
        b = ion_edges[i]**(param[:, i, 0] - 1)
        d = param[:, i, 0] - 1
        x = jnp.abs((a-b) / d)
        val = param[:, i, 1] + jnp.log10(c) + jnp.log10(Lsun) + jnp.log10(x)
        log_Ltotal = log_Ltotal.at[:, i].set(val)

    return log_Ltotal

@jax.jit
def loss_function(params, Xmin, Xmax, logX, logY, logQ_true, Ndata):
    """
    Combined objective and loss function for fitting power laws analytically.

    Computes:
        loss = 0.5 * Σ (logY - linear(logX))² + 0.5 * N * (logQ_true - logQ_pred)²

    Parameters:
    - params: shape (2,), [slope, intercept] in log-log space
    - Xmin, Xmax: wavelength integration bounds
    - logX: log10(wavelength) values
    - logY: log10(flux) values (true)
    - logQ_true: observed ionizing photon rate (log10)
    - Ndata: number of data points

    Returns:
    - Scalar loss value
    """
    slope, intercept = params
    logY_pred = slope * logX + intercept

    # log10 of Q_pred via analytic integral
    logQ_pred = intercept - logh + jnp.log10((Xmax**slope - Xmin**slope) / slope)

    y_diff = (logY - logY_pred) 
    q_diff = logQ_true - logQ_pred

    return 0.5 * jnp.sum((y_diff) ** 2) + 0.5 * (q_diff ** 2) * Ndata


# optax calculates the gradient under-the-hood
quick_solver = OptaxSolver(
    opt=optax.adamw(learning_rate=1), # crazy high learning rate for fast convergence, but not necessarily more accurate
    fun=loss_function,
    maxiter=150,
    jit=True,
)

# slow_solver = OptaxSolver(
#     opt=optax.adamw(learning_rate=1e-1), 
#     fun=loss_function,
#     maxiter=100,
#     jit=True,
# )

@jax.jit
def do_optimization(this_wave,this_spec):
    this_xmin = this_wave[0]
    this_xmax = this_wave[-1]
    init_slope = jnp.log10(this_spec[-1] / this_spec[0]) / jnp.log10(this_xmax / this_xmin)
    init_norm = jnp.log10(this_spec[-1]) - init_slope * jnp.log10(this_xmax)


    Q_true = jnp.abs(jnp.trapezoid(this_spec * this_wave/ (h * c), x=c / this_wave))
    # Q_true = calculate_Q(this_wave,this_spec)

    args = (
        this_xmin,
        this_xmax,
        jnp.log10(this_wave),
        jnp.log10(this_spec),
        jnp.log10(Q_true),
        len(this_wave)
    )

    init_params = jnp.array([init_slope, init_norm])

    better_solution = quick_solver.run(init_params,*args)
    # final_solution = slow_solver.run(better_solution.params,*args)
    
    return better_solution.params


@partial(jax.jit,static_argnums=(2)) # jit-compiling
def fit_ionizing_continuum(wave, spec, ion_edge_indices):

    print("Tracing + Compiling ...")

    norm = 1/ jnp.median(spec)
    normalized_spec = spec * norm

    num_edges = len(ion_edge_indices) - 1
    coeff = jnp.zeros((num_edges, 2))

    for i in range(num_edges):
        ion_edge_index = ion_edge_indices[i]
        next_ion_edge_index = ion_edge_indices[i + 1]
        length = next_ion_edge_index - ion_edge_index
        this_wave = jax.lax.slice_in_dim(wave,ion_edge_index,next_ion_edge_index)
        this_spec = jax.lax.slice_in_dim(normalized_spec,ion_edge_index,next_ion_edge_index)

        solution = do_optimization(this_wave,this_spec)
        coeff = coeff.at[i,:].set(solution)
        coeff = coeff.at[i,1].subtract(jnp.log10(norm))

    logLratios = jnp.diff(jnp.squeeze(total_log_luminosity(param=coeff.reshape(1, 4, 2))))
    
    logQ = jnp.log10(calculate_Q(wave, spec))

    return coeff, logLratios, logQ

def cuejax_fit(wave,spec):

    ion_edges = np.array([HeII_edge, OII_edge, HeI_edge, 911.6])
    ion_edge_indices = tuple(int(x) for x in np.insert(np.searchsorted(wave, ion_edges), 0, 0))

    coeff, logLratios, logQion =  fit_ionizing_continuum(wave,spec,ion_edge_indices)

    results = {
        'ionspec_index1': jnp.clip(coeff[0, 0], 1, 42),
        'ionspec_index2': jnp.clip(coeff[1, 0], -0.3, 30),
        'ionspec_index3': jnp.clip(coeff[2, 0], -1, 14),
        'ionspec_index4': jnp.clip(coeff[3, 0], -1.7, 8),
        'ionspec_logLratio1': jnp.clip(logLratios[0], -1, 10.1),
        'ionspec_logLratio2': jnp.clip(logLratios[1], -0.5, 1.9),
        'ionspec_logLratio3': jnp.clip(logLratios[2], -0.4, 2.2),
        'log_qion': logQion,
        'powerlaw_params': coeff
    }

    return results

### scipy fit functions in cue, we use it for now as it's accurate and remain less than twice slower
def linear(logλ, α, logA):
    return logA + α*logλ

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

def calcQ(lamin0, specin0, mstar=1.0, helium=False, f_nu=True):
    '''
    Calculate the number of lyman ionizing photons for given spectrum
    Input spectrum must be in ergs/s/A if f_nu=False
    Q = int(Lnu/hnu dnu, nu_0, inf)
    '''
    #from scipy.integrate import simpson as simps    
    lamin = np.asarray(lamin0)
    specin = np.asarray(specin0)
    if helium:
        lam_0 = 304.0
    else:
        lam_0 = 911.6
    if f_nu:
        nu_0 = c/lam_0
        inds, = np.where(c/lamin >= nu_0)
        hlam, hflu = c/lamin[inds], specin[inds]
        nu = hlam[::-1]
        f_nu = hflu[::-1]
        integrand = f_nu/(h*nu)
        #Q = simps(integrand, x=nu)
        Q = np.trapz(integrand, x=nu)
    else:
        inds, = np.nonzero(lamin <= lam_0)
        lam = lamin[inds]
        spec = specin[inds]
        integrand = lam*spec/(h*c)
        #Q = simps(integrand, x=lam)*mstar
        Q = np.trapz(integrand, x=nu)*mstar
    return Q

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