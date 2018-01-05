"""Calculates the evidence of being in a replay state based on the
current speed and the speed in the previous time step.

"""
import numpy as np
from functools import partial


def speed_log_likelihood(speed_change, speed_std):
    """The likelihood based on a Gaussian random walk.

    Parameters
    ----------
    speed_change : ndarray
    speed_std : float

    Returns
    -------
    log_likelihood : ndarray

    """
    return -np.log(speed_std) - 0.5 * speed_change ** 2 / speed_std ** 2


def speed_likelihood_ratio(speed, lagged_speed, replay_speed_std,
                           no_replay_speed_std, speed_threshold=4.0):
    """Calculates the evidence of being in a replay state based on the
    current speed and the speed in the previous time step.

    Parameters
    ----------
    speed : ndarray
    lagged_speed : ndarray
    replay_speed_std : float
    no_replay_speed_std : float
    speed_threshold : float, optional

    Returns
    -------
    speed_likelihood_ratio : ndarray

    """
    speed_change = np.squeeze(speed) - np.squeeze(lagged_speed)
    replay_log_likelihood = speed_log_likelihood(
        speed_change, replay_speed_std)
    no_replay_log_likelihood = speed_log_likelihood(
        speed_change, no_replay_speed_std)
    log_likelihood_ratio = replay_log_likelihood - no_replay_log_likelihood
    # Ask long tao about this line
    log_likelihood_ratio[speed > speed_threshold] = -speed[
        speed > speed_threshold]
    likelihood_ratio = np.exp(log_likelihood_ratio)
    likelihood_ratio[np.isposinf(likelihood_ratio)] = 1
    return likelihood_ratio


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
    speed_change = np.insert(np.diff(speed), 0, np.nan)
    replay_speed_std = np.nanstd(speed_change[is_replay])
    no_replay_speed_std = np.nanstd(speed_change[~is_replay])
    return partial(speed_likelihood_ratio,
                   replay_speed_std=replay_speed_std,
                   no_replay_speed_std=no_replay_speed_std,
                   speed_threshold=speed_threshold)
