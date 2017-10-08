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
    likelihood_ratio = (log_likelihood(speed[is_candidate_replay]) -
                        log_likelihood(speed[~is_candidate_replay]))
    likelihood_ratio[speed > speed_threshold] = -speed[
        speed > speed_threshold]
    return likelihood_ratio

    speed_change = np.diff(speed)
    speed_change = np.insert(speed_change, 0, np.nan)
    speed_std = np.nanstd(speed_change[is_state])
    return (-np.log(speed_std) - 0.5 * speed_change ** 2) / speed_std ** 2

def estimate_indicator_probability(speed, is_candidate_replay):
    '''Estimate the predicted probablity of replay given speed and whether
    it was a replay in the previous time step.

    p(I_t | I_t-1, v_t-1)

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
