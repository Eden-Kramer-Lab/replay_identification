import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from statsmodels.api import GLM, families

from .core import atleast_2d


def estimate_movement_std(position):
    '''Estimates the movement standard deviation based on position.

    WARNING: Need to use on original position, not interpolated position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dim)

    Returns
    -------
    movement_std : ndarray, shape (n_position_dim,)

    '''
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_std = []
    for p in position.T:
        fit = GLM(p[:-1], p[1:], family=families.Gaussian()).fit()
        movement_std.append(np.sqrt(fit.scale))
    return np.array(movement_std)


def estimate_movement_state_transition(place_bins, position, variance):
    """Zero mean random walk with covariance based on movement.

    p(x_{k} | x_{k-1}, I_{k}, I_{k-1})
    """
    state_transition_matrix = norm.pdf(
        place_bins[:, np.newaxis], loc=place_bins[np.newaxis, :],
        scale=variance)
    return _normalize_row_probability(state_transition_matrix)


def fit_movement_state_transition(position, place_bin_centers, speed,
                                  replay_speedup=16, movement_threshold=4.0,
                                  movement_std=0.5):
    is_movement = speed > movement_threshold
    position = position[is_movement]

    return np.linalg.matrix_power(estimate_movement_state_transition(
        place_bin_centers, position, movement_std), replay_speedup)


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    return x / x.sum(axis=1, keepdims=True)


def empirical_movement_transition_matrix(position, place_bin_edges, speed,
                                         replay_speed=20,
                                         movement_threshold=4.0,
                                         movement_std=0.5):
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
    movement_std : None or float, optional
        If `movement_std` is None, then will estimate the movement standard
        deviation from the the data.

    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    is_movement = speed > movement_threshold
    position_info = pd.DataFrame({'position': position,
                                  'is_movement': is_movement})
    position_info['lagged_position'] = position_info.position.shift(1)
    position_info = position_info.loc[position_info.is_movement].dropna()

    movement_bins, _, _ = np.histogram2d(
        position_info.position, position_info.lagged_position,
        bins=(place_bin_edges, place_bin_edges))

    movement_bins = _fix_zero_bins(movement_bins)
    movement_bins = _normalize_row_probability(movement_bins)
    movement_bins = gaussian_filter(movement_bins, sigma=movement_std)
    return np.linalg.matrix_power(movement_bins, replay_speed)


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins
