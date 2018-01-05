"""Fitting and predicting the likelihood of replay events based on place field
spiking patterns.
"""

from functools import partial
from logging import getLogger

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from statsmodels.api import GLM, families

from .core import combined_likelihood

logger = getLogger(__name__)


def fit_glm_model(spikes, design_matrix, penalty=1E-5):
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
    model = GLM(spikes, design_matrix, family=families.Poisson(),
                drop='missing')
    if penalty is None:
        return model.fit()
    else:
        regularization_weights = np.ones((design_matrix.shape[1],)) * penalty
        regularization_weights[0] = 0.0
        return model.fit_regularized(alpha=regularization_weights, L1_wt=0)


def create_predict_design_matrix(position, design_matrix):
    predictors = {'position': position}
    return build_design_matrices(
        [design_matrix.design_info], predictors)[0]


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
    return np.exp(np.dot(design_matrix, coefficients))


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
    conditional_intensity[
        np.isclose(conditional_intensity, 0.0)] = np.spacing(1)
    return (np.log(conditional_intensity) * is_spike +
            probability_no_spike)


def spiking_likelihood_ratio(
        is_spike, position, design_matrix, place_field_coefficients,
        place_conditional_intensity, time_bin_size=1):
    """Computes the likelihood ratio between replay and not replay events.

    Parameters
    ----------
    is_spike : ndarray, shape (n_neurons,)
    position : float
    design_matrix : ndarray, shape (n_time, n_coefficients)
    place_field_coefficients : ndarray, shape (n_coefficients, n_neurons)
    place_conditional_intensity : ndarray, shape (n_neurons, n_place_bins)
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood_ratio : ndarray, shape (n_place_bins,)

    """
    no_replay_design_matrix = create_predict_design_matrix(
        position, design_matrix)
    no_replay_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, no_replay_design_matrix).T
    no_replay_log_likelihood = poisson_log_likelihood(
        is_spike, no_replay_conditional_intensity, time_bin_size)
    replay_log_likelihood = poisson_log_likelihood(
        is_spike, place_conditional_intensity, time_bin_size)
    return np.exp(replay_log_likelihood - no_replay_log_likelihood)


def fit_spiking_likelihood_ratio(training_position, training_spikes,
                                 place_bin_centers, penalty=1E-5,
                                 knot_spacing=30, time_bin_size=1):
    """Estimate the place field model.

    Parameters
    ----------
    training_position : ndarray, shape (n_time,)
    training_spikes : ndarray, shape (n_neurons, n_time)
    place_bin_centers : ndarray, shape (n_place_bins,)
    penalty : float, optional
    knot_spacing : float, optional
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood_ratio : function

    """
    knots = place_bin_centers.copy()
    knots = knots[(knots >= np.min(training_position)) &
                  (knots <= np.max(training_position))]
    formula = ('1 + cr(position, knots=knots, constraints="center")')

    training_data = pd.DataFrame(dict(position=training_position))
    design_matrix = dmatrix(
        formula, training_data, return_type='dataframe')
    place_field_coefficients = np.stack(
        [fit_glm_model(
            pd.DataFrame(spikes).loc[design_matrix.index], design_matrix,
            penalty=penalty).params
         for spikes in training_spikes], axis=1)
    place_design_matrix = create_predict_design_matrix(
        place_bin_centers, design_matrix)
    place_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, place_design_matrix).T
    return partial(
        spiking_likelihood_ratio,
        design_matrix=design_matrix,
        place_field_coefficients=place_field_coefficients,
        place_conditional_intensity=place_conditional_intensity,
        time_bin_size=time_bin_size)