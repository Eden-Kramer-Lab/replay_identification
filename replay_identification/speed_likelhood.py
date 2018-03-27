"""Calculates the evidence of being in a replay state based on the
current speed and the speed in the previous time step.

"""
from functools import partial

import numpy as np
from statsmodels.api import GLM, families
from statsmodels.tsa.tsatools import lagmat
from patsy import dmatrices


def speed_log_likelihood(response, predicted_response, weights=1., scale=1.):
    '''Gaussian log likelihood by observation

    Parameters
    ----------
    response : ndarray, shape (n_observations,)
    predicted_response : ndarray, shape (n_observations,)
    weights : ndarray or float, shape (n_observations,)
    scale : float

    Returns
    -------
    log_likelihood : ndarray, shape (n_observations,)

    '''
    ll_obs = -weights * (response - predicted_response) ** 2 / scale
    ll_obs += -np.log(scale / weights) - np.log(2 * np.pi)
    ll_obs /= 2
    return ll_obs


def speed_likelihood_ratio(speed, lagged_speed, replay_fit,
                           no_replay_fit, speed_threshold=4.0):
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
    speed_likelihood_ratio : ndarray, shape (n_time, 1)

    """
    no_replay_prediction = no_replay_fit.predict(lagged_speed)
    replay_prediction = replay_fit.predict(lagged_speed)

    replay_log_likelihood = speed_log_likelihood(
        speed, replay_prediction, scale=replay_fit.scale)
    no_replay_log_likelihood = speed_log_likelihood(
        speed, no_replay_prediction, scale=no_replay_fit.scale)
    log_likelihood_ratio = replay_log_likelihood - no_replay_log_likelihood

    likelihood_ratio = np.exp(log_likelihood_ratio)
    likelihood_ratio[np.isposinf(likelihood_ratio)] = 1

    return likelihood_ratio[:, np.newaxis]


def fit_speed_likelihood_ratio(speed, is_replay, speed_threshold=4.0):
    """Fits the standard deviation of the change in speed for the replay and
    non-replay state.

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : ndarray, shape (n_time,)
    speed_threshold : float, optional

    Returns
    -------
    speed_likelihood_ratio : function

    """
    lagged_speed = lagmat(speed, 1)
    replay_fit = fit_speed_model(speed[is_replay], lagged_speed[is_replay])
    no_replay_fit = fit_speed_model(
        speed[~is_replay], lagged_speed[~is_replay])
    return partial(speed_likelihood_ratio,
                   replay_fit=replay_fit,
                   no_replay_fit=no_replay_fit,
                   speed_threshold=speed_threshold)


def fit_speed_model(speed, lagged_speed):
    FORMULA = 'speed ~ lagged_speed - 1'
    response, design_matrix = dmatrices(
        FORMULA, dict(speed=speed, lagged_speed=lagged_speed))
    family = families.Gaussian(link=families.links.log)
    return GLM(response, design_matrix, family=family).fit()
