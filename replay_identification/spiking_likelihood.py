"""Fitting and predicting the likelihood of non_local events based on place field
spiking patterns.
"""

from functools import partial
from logging import getLogger

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from patsy import build_design_matrices, dmatrix
from regularized_glm import penalized_IRLS
from statsmodels.api import families

from .core import atleast_2d, scale_likelihood

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
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
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
    intensity = np.exp(design_matrix @ coefficients)
    intensity[np.isnan(intensity)] = np.spacing(1)
    return intensity


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
    return (np.log(conditional_intensity + np.spacing(1)) * is_spike
            - conditional_intensity * time_bin_size)


def spiking_likelihood(
        is_spike, position, design_matrix, place_field_coefficients,
        place_conditional_intensity, is_track_interior, time_bin_size=1):
    """Computes the likelihood of non-local and local events.

    Parameters
    ----------
    is_spike : ndarray, shape (n_time, n_neurons)
    position : ndarray, shape (n_time,)
    design_matrix : ndarray, shape (n_time, n_coefficients)
    place_field_coefficients : ndarray, shape (n_coefficients, n_neurons)
    place_conditional_intensity : ndarray, shape (n_place_bins, n_neurons)
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood : ndarray, shape (n_time, n_place_bins,)

    """
    local_design_matrix = create_predict_design_matrix(
        position, design_matrix)
    local_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, local_design_matrix)
    n_time = is_spike.shape[0]
    n_place_bins = place_conditional_intensity.shape[0]
    spiking_likelihood = np.zeros((n_time, 2, n_place_bins))
    spiking_likelihood[:, 1, :] = (combined_likelihood(
        is_spike.T[..., np.newaxis],
        place_conditional_intensity.T[:, np.newaxis, :], time_bin_size))
    spiking_likelihood[:, 0, :] = (combined_likelihood(
        is_spike.T, local_conditional_intensity.T, time_bin_size)
    )
    spiking_likelihood[:, :, ~is_track_interior] = np.nan

    return scale_likelihood(spiking_likelihood)


def combined_likelihood(spikes, conditional_intensity, time_bin_size=1):
    n_time = spikes.shape[1]
    n_bins = (conditional_intensity.shape[-1]
              if conditional_intensity.ndim > 2 else 1)
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(tqdm(spikes, desc='neurons'),
                            conditional_intensity):
        log_likelihood += atleast_2d(
            poisson_log_likelihood(is_spike, ci, time_bin_size))

    return log_likelihood


def fit_spiking_likelihood(position, spikes, is_training,
                           place_bin_centers, is_track_interior, penalty=1E1,
                           knot_spacing=30):
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
    spiking_likelihood : function

    """
    min_position, max_position = np.nanmin(position), np.nanmax(position)
    n_steps = (max_position - min_position) // knot_spacing
    position_knots = min_position + np.arange(1, n_steps) * knot_spacing  # noqa: F841, E501
    FORMULA = ('1 + cr(position, knots=position_knots, constraints="center")')
    training_data = pd.DataFrame(
        dict(position=position[is_training].squeeze())).dropna()
    design_matrix = dmatrix(
        FORMULA, training_data, return_type='dataframe')
    place_field_coefficients = np.stack(
        [fit_glm_model(
            pd.DataFrame(s).loc[design_matrix.index],
            design_matrix, penalty=penalty)
         for s in tqdm(spikes[is_training].T, desc='neurons')], axis=1)
    place_design_matrix = create_predict_design_matrix(
        place_bin_centers, design_matrix)
    place_conditional_intensity = get_conditional_intensity(
        place_field_coefficients, place_design_matrix)
    return partial(
        spiking_likelihood,
        design_matrix=design_matrix,
        place_field_coefficients=place_field_coefficients,
        place_conditional_intensity=place_conditional_intensity,
        is_track_interior=is_track_interior)
