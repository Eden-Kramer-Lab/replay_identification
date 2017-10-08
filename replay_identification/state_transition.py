import numpy as np
from statsmodels.api import GLM, families
from patsy import build_design_matrices, dmatrices


def estimate_indicator_probability(speed, is_replay):
    '''Estimate the predicted probablity of replay given speed and whether
    it was a replay in the previous time step.

    p(I_t | I_t-1, v_t-1)

    p_I_0, p_I_1 in Long Tao's code

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : boolean ndarray, shape (n_time,)

    Returns
    -------
    probability_replay : ndarray, shape (n_time, 2)

    '''
    data = {
        'is_replay': is_replay[:-1],
        'lagged_is_replay': is_replay[1:],
        'lagged_speed': speed[1:]
    }
    MODEL_FORMULA = ('is_replay ~ lagged_is_replay + '
                     'bs(lagged_speed, knots=[1, 2, 3, 20])')
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    family = families.Binomial()
    fit = GLM(response, design_matrix, family=family).fit()

    probability_replay_given_no_previous_replay = predict_probability(
        0, design_matrix, fit, speed, family)
    probability_replay_given_previous_replay = predict_probability(
        1, design_matrix, fit, speed, family)

    return np.stack((probability_replay_given_no_previous_replay,
                     probability_replay_given_previous_replay), axis=1)


def predict_probability(lagged_is_replay, design_matrix, fit, speed,
                        family):
    '''Predict probability from model.

    Parameters
    ----------
    lagged_is_replay : 0 | 1
    design_matrix : patsy design matrix
    fit : statsmodels fitted model
    speed : ndarray, shape (n_time,)
    family : statsmodels family

    Returns
    -------
    predicted_probabilities : ndarray, shape (n_time,)

    '''
    predict_data = {
        'lagged_is_replay': lagged_is_replay * np.ones_like(speed[1:]),
        'lagged_speed': speed[1:]
    }
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]
    linear_predictor = np.dot(predict_design_matrix, fit.params)
    probability = family.link.inverse(linear_predictor)
    return np.insert(np.nan, 1, probability)
