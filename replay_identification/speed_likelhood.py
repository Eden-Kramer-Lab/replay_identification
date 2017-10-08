import numpy as np
from statsmodels.api import GLM, families
from patsy import build_design_matrices, dmatrix


def log_likelihood(speed):
    speed_std = np.std(speed[1:] - speed[:-1])
    return (-np.log(speed_std) - 0.5 * (speed[1:] -
            speed[:-1]) ** 2) / speed_std ** 2


    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_state, ndarray, bool, shape (n_time,)

    Returns
    -------
    likelihood_ratio : ndarray (n_time,)

    '''
    likelihood_ratio = (log_likelihood(speed[is_candidate_replay]) -
                        log_likelihood(speed[~is_candidate_replay]))
    likelihood_ratio[speed > speed_threshold] = -speed[
        speed > speed_threshold]
    return likelihood_ratio


def estimate_indicator_probability(speed, is_candidate_replay):
    '''Estimate the predicted probablity of replay given speed and whether
    it was a replay in the previous time step.

    p(I_t | I_t-1, v_t-1)

    p_I_0, p_I_1 in Long Tao's code

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : boolean ndarray, shape (n_time,)
    speed_threshold : float, optional

    Returns
    -------
    probability_replay : ndarray, shape (2, n_time)

    '''

    design_matrix = dmatrix('is_replay + bs(speed, knots=[1, 2, 3, 20])',
                            dict(speed=speed[1:],
                                 is_replay=is_candidate_replay))
    fit = GLM(is_candidate_replay[1:], design_matrix,
              family=families.Binomial()).fit()

    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info],
        dict(is_replay=np.unique(is_candidate_replay)))[0]
    return (np.exp(np.dot(predict_design_matrix, fit.params)) /
            (1 + np.exp(np.dot(predict_design_matrix, fit.params))))
