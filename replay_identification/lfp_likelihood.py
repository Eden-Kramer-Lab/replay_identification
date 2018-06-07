import numpy as np
from functools import partial
from sklearn.mixture import GaussianMixture

from spectral_connectivity import Connectivity, Multitaper


def lfp_likelihood_ratio(ripple_band_power, replay_model, no_replay_model):
    """Estimates the likelihood of being in a replay state over time given the
     spectral power of the local field potentials (LFPs).

    Parameters
    ----------
    ripple_band_power : ndarray, shape (n_time, n_signals)
    out_replay_kde : statsmodels.nonparametric.kernel_density.KDEMultivariate
    in_replay_kde : statsmodels.nonparametric.kernel_density.KDEMultivariate

    Returns
    -------
    lfp_likelihood_ratio : ndarray, shape (n_time, 1)

    """
    not_nan = np.all(~np.isnan(ripple_band_power), axis=1)
    n_time = ripple_band_power.shape[0]
    likelihood_ratio = np.ones((n_time, 1))
    no_replay_log_likelihood = no_replay_model.score_samples(
        np.log(ripple_band_power[not_nan]))
    replay_log_likelihood = replay_model.score_samples(
        np.log(ripple_band_power[not_nan]))

    likelihood_ratio[not_nan, 0] = np.exp(
        replay_log_likelihood - no_replay_log_likelihood)
    return likelihood_ratio


def fit_lfp_likelihood_ratio(ripple_band_power, is_replay,
                             model=GaussianMixture,
                             model_kwargs=dict(n_components=3)):
    """Fits the likelihood of being in a replay state over time given the
     spectral power of the local field potentials (LFPs).

    Parameters
    ----------
    ripple_band_power : ndarray, shape (n_time, n_signals)
    is_replay : bool ndarray, shape (n_time,)
    sampling_frequency : float

    Returns
    -------
    likelihood_ratio : function

    """

    not_nan = np.all(~np.isnan(ripple_band_power), axis=1)
    replay_model = model(**model_kwargs).fit(
        np.log(ripple_band_power[is_replay & not_nan]))
    no_replay_model = model(**model_kwargs).fit(
        np.log(ripple_band_power[~is_replay & not_nan]))

    return partial(lfp_likelihood_ratio, replay_model=replay_model,
                   no_replay_model=no_replay_model)


def estimate_ripple_band_power(lfps, sampling_frequency):
    """Estimates the 200 Hz power of each LFP.

    Parameters
    ----------
    lfps : ndarray, shape (n_time, n_signals)
    sampling_frequency : float

    Returns
    -------
    ripple_band_power : ndarray (n_time, n_signals)

    """
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
