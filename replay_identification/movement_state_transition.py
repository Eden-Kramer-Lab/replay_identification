import numpy as np
from patsy import dmatrices
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from statsmodels.api import GLM, families
from statsmodels.tsa.tsatools import lagmat


def estimate_movement_variance(position, lagged_position):

    data = {
        'position': position,
        'lagged_position': lagged_position
    }

    MODEL_FORMULA = 'position ~ lagged_position - 1'
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    fit = GLM(response, design_matrix, family=families.Gaussian()).fit()

    return np.sqrt(fit.scale)


def estimate_movement_state_transition(place_bins, position, variance):
    """Zero mean random walk with covariance based on movement.

    p(x_{k} | x_{k-1}, I_{k}, I_{k-1})
    """
    state_transition_matrix = norm.pdf(
        place_bins[:, np.newaxis], loc=place_bins[np.newaxis, :],
        scale=variance)
    return _normalize_row_probability(state_transition_matrix)


def fit_movement_state_transition(position, speed, place_bins,
                                  movement_threshold=4.0, speed_up_factor=20):
    movement_variance = estimate_movement_variance(
        position, speed, movement_threshold)
    return np.linalg.matrix_power(estimate_movement_state_transition(
        place_bins, position, movement_variance), speed_up_factor)


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    return x / x.sum(axis=1, keepdims=True)


def empirical_movement_transition_matrix(position, place_bin_edges, speed,
                                         replay_speedup=16,
                                         movement_threshold=4.0):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `sequence_compression_factor`

    position cell firing during a hippocampal replay event is a "sped-up"
    version of position cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which position
    cell is likely to fire next.

    Parameters
    ----------
    position : array_like, shape (n_time,)
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

    is_movement = speed > movement_threshold
    lagged_position = lagmat(position, 1)[is_movement].squeeze()
    position = position[is_movement]

    movement_bins, _, _ = np.histogram2d(lagged_position, position,
                                         bins=(place_bin_edges,
                                               place_bin_edges),
                                         normed=False)
    smoothed_movement_bins_probability = gaussian_filter(
        _normalize_row_probability(
            _fix_zero_bins(movement_bins)), sigma=0.5)
    return np.linalg.matrix_power(
        smoothed_movement_bins_probability, replay_speedup)


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins
