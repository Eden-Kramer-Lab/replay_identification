import numpy as np
from patsy import dmatrices
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter
from statsmodels.api import GLM, families
from statsmodels.tsa.tsatools import lagmat


def estimate_movement_variance(position, speed, speed_threshold=4.0):
    is_above_threshold = speed > speed_threshold

    lagged_position = lagmat(position, maxlag=1)

    data = {
        'position': position[is_above_threshold],
        'lagged_position': lagged_position[is_above_threshold]
    }

    MODEL_FORMULA = 'position ~ lagged_position - 1'
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    fit = GLM(response, design_matrix, family=families.Gaussian()).fit()

    return np.sqrt(np.sum(fit.resid_response ** 2) / fit.df_resid)


def estimate_position_state_transition(place_bins, position, variance):
    """Zero mean random walk with covariance based on movement.

    p(x_{k} | x_{k-1}, I_{k}, I_{k-1})
    """
    state_transition_matrix = norm.pdf(
        place_bins[:, np.newaxis], loc=place_bins[np.newaxis, :],
        scale=variance)
    return _normalize_row_probability(state_transition_matrix)


def fit_position_state_transition(position, speed, place_bins,
                                  speed_threshold=4.0, speed_up_factor=20):
    movement_variance = estimate_movement_variance(
        position, speed, speed_threshold)
    return np.linalg.matrix_power(estimate_position_state_transition(
        place_bins, position, movement_variance), speed_up_factor)


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    return x / x.sum(axis=1, keepdims=True)


def empirical_movement_transition_matrix(place, place_bin_edges, speed,
                                         sequence_compression_factor=16,
                                         speed_threshold=4.0):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `sequence_compression_factor`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    place : array_like, shape (n_time,)
        Linearized position of the animal over time
    place_bin_edges : array_like, shape (n_bins,)
    sequence_compression_factor : int, optional
        How much the movement is sped-up during a replay event
    is_movement : array_like, shape (n_time,)
        Boolean indicator for an experimental condition.
    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    movement_variance = estimate_movement_variance(
        place, speed, speed_threshold)

    is_movement = speed > speed_threshold

    place = np.stack((place[1:], place[:-1]))
    place = place[:, is_movement[1:]]

    movement_bins, _, _ = np.histogram2d(place[0], place[1],
                                         bins=(place_bin_edges,
                                               place_bin_edges),
                                         normed=False)

    smoothed_movement_bins_probability = gaussian_filter(
        _normalize_row_probability(
            _fix_zero_bins(movement_bins)), sigma=movement_variance)
    return np.linalg.matrix_power(
        smoothed_movement_bins_probability,
        sequence_compression_factor)


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins
