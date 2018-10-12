"""Calculates the evidence of being in a replay state based on the
current speed and the speed in the previous time step.

"""
from functools import partial

import numpy as np
from patsy import dmatrices
from statsmodels.api import GLM, families
from statsmodels.tsa.tsatools import lagmat

FAMILY = families.Gaussian(link=families.links.log)
FORMULA = 'speed ~ lagged_speed - 1'


def speed_likelihood(speed, lagged_speed, replay_coefficients,
                     replay_scale, no_replay_coefficients,
                     no_replay_scale, speed_threshold=4.0):
    """Calculates the evidence of being in a replay state based on the
    current speed and the speed in the previous time step.

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    lagged_speed : ndarray, shape (n_time,)
    replay_speed_std : float
    no_replay_speed_std : float
    speed_threshold : float, optional

    Returns
    -------
    speed_likelihood : ndarray, shape (n_time, 2, 1)

    """
    no_replay_prediction = _predict(no_replay_coefficients, lagged_speed)
    replay_prediction = _predict(replay_coefficients, lagged_speed)
    n_time = speed.shape[0]
    speed_likelihood = np.zeros((n_time, 2, 1))
    speed_likelihood[:, 0, :] = np.exp(FAMILY.loglike_obs(
        speed, no_replay_prediction, scale=no_replay_scale))[:, np.newaxis]
    speed_likelihood[:, 1, :] = np.exp(FAMILY.loglike_obs(
        speed, replay_prediction, scale=replay_scale))[:, np.newaxis]

    return speed_likelihood


def fit_speed_likelihood(speed, is_replay, speed_threshold=4.0):
    """Fits the standard deviation of the change in speed for the replay and
    non-replay state.

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : ndarray, shape (n_time,)
    speed_threshold : float, optional

    Returns
    -------
    speed_likelihood : function

    """
    lagged_speed = lagmat(speed, 1)
    replay_coefficients, replay_scale = fit_speed_model(
        speed[is_replay], lagged_speed[is_replay])
    no_replay_coefficients, no_replay_scale = fit_speed_model(
        speed[~is_replay], lagged_speed[~is_replay])
    return partial(speed_likelihood,
                   replay_coefficients=replay_coefficients,
                   replay_scale=replay_scale,
                   no_replay_coefficients=no_replay_coefficients,
                   no_replay_scale=no_replay_scale,
                   speed_threshold=speed_threshold)


def fit_speed_model(speed, lagged_speed):
    response, design_matrix = dmatrices(
        FORMULA, dict(speed=speed, lagged_speed=lagged_speed))
    results = GLM(response, design_matrix, family=FAMILY).fit()
    return results.params, results.scale


def _predict(coefficients, lagged_speed):
    return FAMILY.link.inverse(lagged_speed * coefficients)
