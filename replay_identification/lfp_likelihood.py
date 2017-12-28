import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from spectral_connectivity import Connectivity, Multitaper


def lfp_likelihood_ratio(lfps, is_replay, sampling_frequency):
    '''The likelihood of being in a replay state over time given the
    spectral power of the local field potentials (LFPs).

    Parameters
    ----------
    lfps : ndarray, shape (n_time, n_signals)
    is_replay : bool ndarray, shape (n_time,)
    sampling_frequency : float

    Returns
    -------
    likelihood_ratio : ndarray (n_time,)

    '''
    ripple_band_power = np.log(estimate_ripple_band_power(
        lfps, sampling_frequency))

    out_replay_kde = estimate_kernel_density(
        ripple_band_power[~is_replay])
    in_replay_kde = estimate_kernel_density(
        ripple_band_power[is_replay])

    out_replay_likelihood = bias_zero(out_replay_kde.pdf(ripple_band_power))
    in_replay_likelihood = bias_zero(in_replay_kde.pdf(ripple_band_power))

    return in_replay_likelihood / out_replay_likelihood


def bias_zero(x):
    x[np.isclose(x, 0.0)] = np.spacing(1)
    return x


def estimate_kernel_density(ripple_band_power):
    '''Evaluate a multivariate gaussian kernel for each time point

    Parameters
    ----------
    ripple_band_power : ndarray, shape (n_time, n_signals)

    Returns
    -------
    kernel_density_estimate : function

    '''
    n_time, n_signals = ripple_band_power.shape
    is_not_nan = np.any(~np.isnan(ripple_band_power), axis=1)
    ripple_band_power = ripple_band_power[is_not_nan]
    # replace with np.var?
    power_variances = (np.std(ripple_band_power, axis=0) *
                       (4 / (n_signals + 2) / n_time) **
                       (1 / (n_signals + 4))) ** 2
    return KDEMultivariate(ripple_band_power, bw=power_variances,
                           var_type='c' * n_signals)


def estimate_ripple_band_power(lfps, sampling_frequency):
    '''Estimates the 200 Hz power of each LFP

    Parameters
    ----------
    lfps : ndarray, shape (n_time, n_signals)
    sampling_frequency : float

    Returns
    -------
    ripple_band_power : ndarray (n_time, n_signals)

    '''
    n_time = lfps.shape[0]
    m = Multitaper(lfps, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=1 / sampling_frequency)
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze()
    n_power_time = power.shape[0]
    unobserved = np.full((n_time - n_power_time, *power.shape[1:]), np.nan)
    return np.concatenate((power, unobserved))
