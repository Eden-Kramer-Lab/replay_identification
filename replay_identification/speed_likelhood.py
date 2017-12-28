import numpy as np


def log_likelihood(speed, is_state):
    '''Gaussian random walk with state dependent variance.

    Note: 2pi?

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_state, ndarray, bool, shape (n_time,)

    Returns
    -------
    log_likelihood : ndarray, shape (n_time)

    '''
    speed_change = np.diff(speed)
    speed_change = np.insert(speed_change, 0, np.nan)
    speed_std = np.nanstd(speed_change[is_state])
    return -np.log(speed_std) - 0.5 * speed_change ** 2 / speed_std ** 2


def estimate_speed_likelihood_ratio(speed, is_replay, speed_threshold=4):
    '''p(v_t|v_{t-1}, I_t)

    l_vel in Long Tao's code

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : boolean ndarray, shape (n_time,)
    speed_threshold : float, optional

    Returns
    -------
    likelihood_ratio : ndarray (n_time,)

    '''
    log_likelihood_ratio = (
        log_likelihood(speed, is_replay) -
        log_likelihood(speed, ~is_replay & (speed <= speed_threshold)))
    log_likelihood_ratio[speed > speed_threshold] = -speed[
        speed > speed_threshold]
    likelihood_ratio = np.exp(log_likelihood_ratio)
    likelihood_ratio[np.isposinf(likelihood_ratio)] = 1
    return likelihood_ratio
