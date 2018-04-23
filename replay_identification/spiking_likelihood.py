"""Fitting and predicting the likelihood of replay events based on place field
spiking patterns.
"""

from functools import partial
from logging import getLogger

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from statsmodels.api import families

import dask.array as da
from regularized_glm import penalized_IRLS

from .core import combined_likelihood

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


logger = getLogger(__name__)


def fit_glm_model(spikes, design_matrix, penalty=1E1):
    """Fits the Poisson model to the spikes from a neuron.

    Parameters
    ----------
    spikes : array_like
    design_matrix : array_like or pandas DataFrame
    ind : int
    penalty : float, optional

    Returns
    -------
    fitted_model : statsmodel results

    """
    penalty = np.ones((design_matrix.shape[1],)) * penalty
    penalty[0] = 0.0
    results = penalized_IRLS(
        np.array(design_matrix), np.array(spikes),
        family=families.Poisson(), penalty=penalty)
    return np.squeeze(results.coefficients)


def create_predict_design_matrix(position, design_matrix):
    is_nan = np.isnan(position)
    position[is_nan] = 0
    predictors = {'position': position}
    design_matrix = build_design_matrices(
        [design_matrix.design_info], predictors)[0]
    design_matrix[is_nan] = np.nan
    return design_matrix


def get_conditional_intensity(coefficients, design_matrix):
    """Predict the model's response given a design matrix and the model
    parameters.

    Parameters
    ----------
    coefficients : ndarray, shape (n_coefficients, n_neurons)
    design_matrix : ndarray, shape (n_obs, n_coefficients)

    Returns
    -------
    conditional_intensity : ndarray, shape (n_coefficients, n_neurons)

    """
    return np.exp(design_matrix @ coefficients)


def atleast_kd(array, k):
    """
    https://stackoverflow.com/questions/42516569/numpy-add-variable-number-of-dimensions-to-an-array
    """
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)


@combined_likelihood
def poisson_log_likelihood(is_spike, conditional_intensity=None,
                           time_bin_size=1):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    is_spike : array_like with values in {0, 1}, shape (n_signals,)
        Indicator of spike or no spike at current time.
    conditional_intensity : array_like, shape (n_signals, n_states,
                                               n_place_bins)
        Instantaneous probability of observing a spike
    time_bin_size : float, optional

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_signals, n_states,
                                                n_place_bins)

    """
    probability_no_spike = -conditional_intensity * time_bin_size
    is_spike = atleast_kd(is_spike, conditional_intensity.ndim)
    return (np.log(conditional_intensity) * is_spike +
            probability_no_spike) + np.spacing(1)


def spiking_likelihood_ratio(
        is_spike, position, design_matrix, place_field_coefficients,
        place_conditional_intensity, time_bin_size=1, chunks=1E3):
    """Computes the likelihood ratio between replay and not replay events.

    Parameters
    ----------
    is_spike : ndarray, shape (n_time, n_neurons)
    position : ndarray, shape (n_time,)
    design_matrix : ndarray, shape (n_time, n_coefficients)
    place_field_coefficients : ndarray, shape (n_coefficients, n_neurons)
    place_conditional_intensity : ndarray, shape (1, n_place_bins, n_neurons)
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood_ratio : ndarray, shape (n_place_bins,)

    """
    no_replay_design_matrix = create_predict_design_matrix(
        position, design_matrix)
    no_replay_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, no_replay_design_matrix)
    no_replay_log_likelihood = da.from_array(poisson_log_likelihood(
        is_spike, no_replay_conditional_intensity,
        time_bin_size)[:, np.newaxis], chunks)
    is_spike = da.from_array(is_spike[:, np.newaxis, :], chunks)
    place_conditional_intensity = da.from_array(
        place_conditional_intensity[np.newaxis, ...], chunks)
    replay_log_likelihood = poisson_log_likelihood(
        is_spike, place_conditional_intensity, time_bin_size)
    return np.exp(replay_log_likelihood - no_replay_log_likelihood).compute()


def fit_spiking_likelihood_ratio(position, spikes, is_replay,
                                 place_bin_centers, penalty=1E1,
                                 time_bin_size=1, knot_spacing=30):
    """Estimate the place field model.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    spikes : ndarray, shape (n_time, n_neurons)
    place_bin_centers : ndarray, shape (n_place_bins,)
    penalty : float, optional
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood_ratio : function

    """
    min_position, max_position = np.nanmin(position), np.nanmax(position)
    n_steps = (max_position - min_position) // knot_spacing
    position_knots = min_position + np.arange(1, n_steps) * knot_spacing
    FORMULA = ('1 + cr(position, knots=position_knots, constraints="center")')
    training_data = pd.DataFrame(
        dict(position=position[~is_replay])).dropna()
    design_matrix = dmatrix(
        FORMULA, training_data, return_type='dataframe')
    place_field_coefficients = np.stack(
        [fit_glm_model(
            pd.DataFrame(s).loc[design_matrix.index],
            design_matrix, penalty=penalty)
         for s in tqdm(spikes[~is_replay].T, desc='neurons')], axis=1)
    place_design_matrix = create_predict_design_matrix(
        place_bin_centers, design_matrix)
    place_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, place_design_matrix)
    return partial(
        spiking_likelihood_ratio,
        design_matrix=design_matrix,
        place_field_coefficients=place_field_coefficients,
        place_conditional_intensity=place_conditional_intensity,
        time_bin_size=time_bin_size)
