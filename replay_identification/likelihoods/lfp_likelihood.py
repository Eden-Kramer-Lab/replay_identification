from functools import partial

import numpy as np
from replay_identification.core import scale_likelihood
from sklearn.mixture import GaussianMixture
from spectral_connectivity import Connectivity, Multitaper


def lfp_likelihood(ripple_band_power, non_local_model, local_model):
    """Estimates the likelihood of being in a non_local state over time given the
     spectral power of the local field potentials (LFPs).

    Parameters
    ----------
    ripple_band_power : ndarray, shape (n_time, n_signals)
    out_non_local_kde : statsmodels.nonparametric.kernel_density.KDEMultivariate
    in_non_local_kde : statsmodels.nonparametric.kernel_density.KDEMultivariate

    Returns
    -------
    lfp_likelihood : ndarray, shape (n_time, 2, 1)

    """
    not_nan = np.all(~np.isnan(ripple_band_power), axis=1)
    n_time = ripple_band_power.shape[0]
    lfp_likelihood = np.ones((n_time, 2))
    lfp_likelihood[not_nan, 0] = local_model.score_samples(
        np.log(ripple_band_power[not_nan])
    )
    lfp_likelihood[not_nan, 1] = non_local_model.score_samples(
        np.log(ripple_band_power[not_nan])
    )

    return scale_likelihood(lfp_likelihood[..., np.newaxis])


def fit_lfp_likelihood(
    ripple_band_power,
    is_non_local,
    model=GaussianMixture,
    model_kwargs=dict(n_components=3),
):
    """Fits the likelihood of being in a non_local state over time given the
     spectral power of the local field potentials (LFPs).

    Parameters
    ----------
    ripple_band_power : ndarray, shape (n_time, n_signals)
    is_non_local : bool ndarray, shape (n_time,)
    sampling_frequency : float

    Returns
    -------
    likelihood_ratio : function

    """
    is_non_local = np.asarray(is_non_local).astype(bool)
    not_nan = np.all(~np.isnan(ripple_band_power), axis=1)
    non_local_model = model(**model_kwargs).fit(
        np.log(ripple_band_power[is_non_local & not_nan] + np.spacing(1))
    )
    local_model = model(**model_kwargs).fit(
        np.log(ripple_band_power[~is_non_local & not_nan] + np.spacing(1))
    )

    return partial(
        lfp_likelihood, non_local_model=non_local_model, local_model=local_model
    )


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
    m = Multitaper(
        lfps,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.020,
        time_window_step=1 / sampling_frequency,
    )
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze()
    n_power_time = power.shape[0]
    unobserved = np.full((n_time - n_power_time, *power.shape[1:]), np.nan)
    return np.concatenate((power, unobserved))
