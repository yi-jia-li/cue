import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize
from .constants import Lsun, c, h, HeI_edge, HeII_edge, OII_edge

# Load constant data (using JAX-compatible structures)
cont_lam = jnp.array(jnp.loadtxt(resource_filename("cue", "data/FSPSlam.dat")))
cont_nu = c / cont_lam

new_unsorted_line_name = jnp.array(jnp.load(resource_filename("cue", "data/lineList_replaceblnd_name.npy")))
new_unsorted_line_lam = jnp.array(jnp.load(resource_filename("cue", "data/lineList_wav.npy")))
new_sorted_line_name = new_unsorted_line_name[jnp.argsort(new_unsorted_line_lam)]
new_sorted_line_lam = jnp.sort(new_unsorted_line_lam)

new_ele_arr = jnp.array([name[:4].strip() for name in new_sorted_line_name])
line_new_added = jnp.where(
    (new_sorted_line_lam == 4685.68) | (new_sorted_line_lam == 1550.77) |
    (new_sorted_line_lam == 1548.19) | (new_sorted_line_lam == 1750.00) |
    (new_sorted_line_lam == 2424.28) | (new_sorted_line_lam == 1882.71) |
    (new_sorted_line_lam == 1892.03) | (new_sorted_line_lam == 1406.02) |
    (new_sorted_line_lam == 4711.26) | (new_sorted_line_lam == 4740.12)
)[0]

line_old = jnp.array([i for i in range(138) if i not in line_new_added])
nn_name = jnp.array(['H1', 'He1', 'He2', 'C1', 'C2C3', 'C4', 'N', 'O1', 'O2', 'O3',
                     'ionE_1', 'ionE_2', 'S4', 'Ar4', 'Ne3', 'Ne4'])

# Helper function to calculate wavelength ranges for different ions
def calculate_nn_wavelengths(nn_ion, ele_arr, sorted_line_lam):
    selections = []
    for ion in nn_ion:
        indices = []
        if isinstance(ion, str):
            indices = jnp.where(ele_arr == ion)[0]
        else:
            for sub_ion in ion:
                indices.extend(jnp.where(ele_arr == sub_ion)[0])
        selections.append(jnp.sort(jnp.array(indices)))
    return jnp.array(selections, dtype=object), sorted_line_lam[jnp.concatenate(selections)]

nn_ion = [
    ['H  1'], ['He 1'], ['He 2'], ['C  1'], ['C  2', 'C  3'], ['C  4'],
    ['N  1', 'N  2', 'N  3'], ['O  1'], ['O  2'], ['O  3'],
    ['Mg 2', 'Fe 2', 'Si 2', 'Al 2', 'P  2', 'S  2', 'Cl 2', 'Ar 2'],
    ['Al 3', 'Si 3', 'S  3', 'Cl 3', 'Ar 3', 'Ne 2'],
    ['S  4'], ['Ar 4'], ['Ne 3'], ['Ne 4']
]
nn_wav_selection, nn_wavelength = calculate_nn_wavelengths(nn_ion, new_ele_arr, new_sorted_line_lam)

line_name = new_sorted_line_name[line_old]
line_lam = new_sorted_line_lam[line_old]

# Power law functions
@jit
def linear(logλ, α, logA):
    return logA + α * logλ

@jit
def calc_luminosity_norm(index, logL, edges, wav=None):
    edges = jnp.hstack([1, edges])
    log_norm = jnp.zeros_like(index)
    if wav is None:
        for i in range(len(edges) - 1):
            term = (edges[i + 1] ** (index[:, i] - 1) - edges[i] ** (index[:, i] - 1)) / (index[:, i] - 1)
            log_norm = log_norm.at[:, i].set(logL[:, i] - jnp.log10(c * Lsun) - jnp.log10(jnp.abs(term)))
    else:
        ind_bin = jnp.array([jnp.max(jnp.where(wav <= λ)[0]) for λ in edges]) + 1
        ind_bin = jnp.insert(ind_bin, 0, 0)
        for i in range(len(ind_bin) - 1):
            term = (wav[ind_bin[i + 1] - 1] ** (index[:, i] - 1) - wav[ind_bin[i]] ** (index[:, i] - 1)) / (index[:, i] - 1)
            log_norm = log_norm.at[:, i].set(logL[:, i] - jnp.log10(c * Lsun) - jnp.log10(jnp.abs(term)))
    return log_norm

@jit
def Qtotal(param, edges):
    log_Qtotal = jnp.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        term = (edges[i + 1] ** param[i, 0] - edges[i] ** param[i, 0]) / param[i, 0]
        log_Qtotal = log_Qtotal.at[i].set(param[i, 1] + jnp.log10(Lsun) - jnp.log10(h) + jnp.log10(term))
    return log_Qtotal

# Compute ionizing photons (optimized with JAX)
@jit
def calcQ(lamin, spec, mstar=1.0, helium=False, f_nu=True):
    lam_min = 304.0 if helium else 911.6
    nu_min = c / lam_min if f_nu else lam_min
    inds = jnp.where((c / lamin) >= nu_min) if f_nu else jnp.where(lamin <= lam_min)
    x_vals = c / lamin[inds] if f_nu else lamin[inds]
    integrand = spec[inds] / (h * x_vals) if f_nu else lamin[inds] * spec[inds] / (h * c)
    return jnp.trapz(integrand, x=x_vals) * (1.0 if f_nu else mstar)

# Optimizer-based power-law fitting (JAX-compatible)
@jit
def fit_power_laws(wav, spec, bins=[HeII_edge, OII_edge, HeI_edge, 911.6]):
    bins = jnp.insert(jnp.array(bins), 0, 1)
    indices = jnp.array([jnp.max(jnp.where(wav <= edge)[0]) for edge in bins]) + 1
    coefficients = []
    for i in range(len(indices) - 1):
        wave_seg = wav[indices[i]:indices[i + 1]]
        spec_seg = spec[indices[i]:indices[i + 1]]
        init_guess = [0.5, -1.0]
        result = minimize(
            lambda params: jnp.sum((linear(jnp.log10(wave_seg), *params) - jnp.log10(spec_seg)) ** 2),
            init_guess,
        )
        coefficients.append(result.x)
    return jnp.array(coefficients)

