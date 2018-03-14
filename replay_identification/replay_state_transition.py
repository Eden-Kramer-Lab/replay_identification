from functools import partial

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices
from statsmodels.api import families
from statsmodels.tsa.tsatools import lagmat
from regularized_glm import penalized_IRLS


def fit_replay_state_transition(speed, is_replay, penalty=1E-1):
    """Estimate the predicted probablity of replay given speed and whether
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

    """
    data = pd.DataFrame({
        'is_replay': is_replay.astype(np.float64),
        'lagged_is_replay': lagmat(
            is_replay, maxlag=1).astype(np.float64).squeeze(),
        'lagged_speed': lagmat(speed, maxlag=1).squeeze()
    }).dropna()
    MODEL_FORMULA = (
        'is_replay ~ 1 + lagged_is_replay + '
        'cr(lagged_speed, knots=[1, 2, 3, 20], constraints="center")')
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    family = families.Binomial()
    penalty = np.ones((design_matrix.shape[1],)) * penalty
    penalty[0] = 0.0
    fit = penalized_IRLS(design_matrix, response, family=family,
                         penalty=penalty)
    return partial(predict_probability, design_matrix=design_matrix, fit=fit)


def make_design_matrix(lagged_is_replay, lagged_speed, design_matrix):
    no_previous_replay_predict_data = {
        'lagged_is_replay': lagged_is_replay * np.ones_like(lagged_speed),
        'lagged_speed': lagged_speed
    }
    return build_design_matrices(
        [design_matrix.design_info], no_previous_replay_predict_data)[0]


def predict_probability(lagged_speed, design_matrix, fit):
    """Predict probability of replay state given speed and whether it was a
    replay in the previous time step.

    Parameters
    ----------
    design_matrix : patsy design matrix
    fit : statsmodels fitted model
    speed : ndarray, shape (n_time,)

    Returns
    -------
    replay_probability : ndarray, shape (n_time,)

    """
    no_previous_replay_design_matrix = make_design_matrix(
        0, lagged_speed, design_matrix)

    previous_replay_design_matrix = make_design_matrix(
        1, lagged_speed, design_matrix)
    coefficients = np.squeeze(fit.coefficients)

    return np.stack((predict(no_previous_replay_design_matrix, coefficients),
                     predict(previous_replay_design_matrix, coefficients)),
                    axis=1)


def predict(design_matrix, coefficients):
    family = families.Binomial()
    return family.link.inverse(design_matrix @ np.squeeze(coefficients))
