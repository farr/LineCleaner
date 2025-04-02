import numpy as np

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import scipy.signal as sig

def line_model_freq(freqs, d_re, d_im, f_line_guess, f_prior_width, gamma_min, gamma_max, estimate_line=False):
    f0 = numpyro.sample('f0', dist.TruncatedNormal(f_line_guess, f_prior_width, low=f_line_guess - 5*f_prior_width, high=f_line_guess + 5*f_prior_width))
    gamma = numpyro.sample('gamma', dist.Uniform(gamma_min, gamma_max))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    jitter = numpyro.sample('jitter', dist.Exponential(1))

    tau = numpyro.deterministic('tau', 1/gamma)

    A2 = 2.0*jnp.pi*jnp.square(f0)*gamma*jnp.square(sigma)

    w = 2*jnp.pi*freqs
    w0 = 2*jnp.pi*f0
    lor_var = numpyro.deterministic('lor_var',  A2 / (jnp.square(jnp.square(w0) - jnp.square(w)) + 4*jnp.square(w0)*jnp.square(gamma)))

    jvar = jnp.square(jitter)

    data_var = lor_var + jvar
    data_sd = jnp.sqrt(data_var)

    numpyro.sample('obs_re', dist.Normal(0, data_sd), obs=d_re)
    numpyro.sample('obs_im', dist.Normal(0, data_sd), obs=d_im)

    if estimate_line:
        mean_wt = 1 / (1 + jvar/lor_var)
        sd = jnp.sqrt(1 / (1/jvar + 1/lor_var))
        numpyro.sample('line_re', dist.Normal(d_re*mean_wt, sd))
        numpyro.sample('line_im', dist.Normal(d_im*mean_wt, sd))

def clean_strain(times, data, srate, f0s, bandwidths, Twindow, mcmc_seed=None, resample_seed=None, return_mcmcs=False):
    """
        clean_strain(times, data, srate, f0s, bandwidths, Twindow,
        mcmc_seed=None, resample_seed=None, return_mcmcs=False)

    Clean a strain time series by removing Lorentzian lines centered at the
    frequencies `f0s` with (narrow) bandwidths `bandwidths`; the central
    frequency and bandwidth should be chosen to encompass the line to be removed
    as well as a small amount of "continum" data around it, but not too broad.
    
    The strain will be windowed with a Tukey window with a time `Twindow` inside
    the windowed region.  

    `mcmc_seed` and `resample_seed` are for reproducabilty: the former controls
    the MCMC sampling while the latter is used to make the draw from the MCMC
    samples for subtraction to produce the line-cleaned residuals.  

    If `return_mcmcs` is True, the function will return `(times_residual,
    data_residual, mcmcs, pred_samples)`, where `times_residual` are times in
    the original data sequence where the window function is 1, `data_residual`
    are the line-subtracted residuals at those times, `mcmcs` are a list of MCMC
    samples for the lines that have been removed (one for each `f0s` and
    `bandwidths`), and `pred_samples` are the the samples from the predicted
    Lorentzians that have been subtracted from the data.

    If `return_mcmcs` is False, then only `(times_residual, data_residual)` are
    returned; if you do not want to perform further analysis on the Lorentzian
    lines, this is probably the option you want.
    """
    if mcmc_seed is None:
        mcmc_seed = np.random.randint(1<<32)
    if resample_seed is None:
        resample_seed = np.random.randint(1<<32)

    mcmc_rng_key = random.PRNGKey(mcmc_seed)
    resample_rng_key = random.PRNGKey(resample_seed)

    window = sig.windows.tukey(len(data), alpha=Twindow*srate/len(data))
    
    data_freq = np.fft.rfft(data*window)/srate
    data_freq_residual = np.copy(data_freq)
    data_freq_re = np.real(data_freq)
    data_freq_im = np.imag(data_freq)
    freqs = np.fft.rfftfreq(len(data), 1/srate)

    if return_mcmcs:
        mcmcs = []
        pred_sampless = []

    for (f0, bandwidth) in zip(f0s, bandwidths):
        sel = np.abs(freqs - f0) < bandwidth/2
        scale_factor = np.sqrt(np.trapz(np.square(np.abs(data_freq[sel])), freqs[sel]))

        nuts_kernel = NUTS(line_model_freq, dense_mass=True)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=4
        )

        mcmc_rng_key, mk = random.split(mcmc_rng_key)
        mcmc.run(
            mk, 
            freqs[sel], 
            data_freq_re[sel]/scale_factor, 
            data_freq_im[sel]/scale_factor, 
            f0,
            bandwidth/10,
            1/(times[-1]-times[0]), 
            f0/2
        )

        resample_rng_key, rk = random.split(resample_rng_key)
        pred = Predictive(line_model_freq, posterior_samples=mcmc.get_samples())
        pred_samples = pred(rk, freqs[sel], data_freq_re[sel]/scale_factor, data_freq_im[sel]/scale_factor, f0, bandwidth/10, 1/(times[-1]-times[0]), f0/2, estimate_line=True)

        resample_rng_key, rk = random.split(resample_rng_key)
        ind = random.randint(rk, (1,), 0, pred_samples['line_re'].shape[0])[0]

        line_model_re = scale_factor*pred_samples['line_re'][ind,:] 
        line_model_im = scale_factor*pred_samples['line_im'][ind,:]

        data_freq_residual[sel] = data_freq_re[sel] - line_model_re + 1j*(data_freq_im[sel] - line_model_im)

        if return_mcmcs:
            mcmcs.append(mcmc)
            pred_sampless.append(pred_samples)

    data_residual = np.fft.irfft(data_freq_residual)/(times[-1]-times[0])*len(data)
    data_residual = data_residual[window==1]
    times_residual = times[window==1]

    if return_mcmcs:
        return (times_residual, data_residual, mcmcs, pred_sampless, line_model_im, line_model_re, sel, data_freq_re)
    else:
        return (times_residual, data_residual)
