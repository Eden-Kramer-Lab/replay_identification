import numpy as np
from scipy.stats import multivariate_normal
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


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    n_bins = movement_bins.shape[0]
    movement_bins[movement_bins.sum(axis=1) == 0] = 1 / n_bins
    return movement_bins


def empirical_movement_transition_matrix(position, edges, is_training,
                                         replay_speed=20,
                                         combine_with_uniform=False):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `replay_speed`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    edges : sequence
        A sequence of arrays describing the bin edges along each dimension.
    is_training  : ndarray, shape (n_time,)
    replay_speed : int, optional
        How much the movement is sped-up during a replay event
    combine_with_uniform : bool

    Returns
    -------
    transition_matrix : ndarray, shape (n_position_bins, n_position_bins)

    '''
    position = atleast_2d(position)[is_training]
    movement_bins, _ = np.histogramdd(
        np.concatenate((position[1:], position[:-1]), axis=1),
        bins=edges * 2)
    original_shape = movement_bins.shape
    n_position_dims = position.shape[1]
    shape_2d = np.product(original_shape[:n_position_dims])
    movement_bins = _normalize_row_probability(
        movement_bins.reshape((shape_2d, shape_2d), order='F'))
    movement_bins = np.linalg.matrix_power(movement_bins, replay_speed)
    if combine_with_uniform:
        movement_bins = _fix_zero_bins(movement_bins)

    return movement_bins


def random_walk_state_transition(place_bin_centers, covariance,
                                 replay_speed=20):
    '''Zero mean random walk with covariance.
    '''
    transition_matrix = np.stack(
        [multivariate_normal(mean=bin, cov=covariance).pdf(place_bin_centers)
         for bin in place_bin_centers], axis=1)
    transition_matrix = _normalize_row_probability(transition_matrix)
    return np.linalg.matrix_power(transition_matrix, replay_speed)
