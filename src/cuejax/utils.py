import numpy as np
import jax
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver
from jax import lax
from functools import partial

from cue.constants import Lsun,c,h,HeI_edge,HeII_edge,OII_edge

import jax
jax.config.update("jax_debug_nans", False)
import jax.numpy as jnp
import jaxopt
from jax import lax

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
