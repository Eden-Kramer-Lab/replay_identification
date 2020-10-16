from functools import partial
from logging import getLogger

import numpy as np
import pandas as pd
from patsy import NAAction, build_design_matrices, dmatrices
from regularized_glm import penalized_IRLS
from statsmodels.api import families
from statsmodels.tsa.tsatools import lagmat

FAMILY = families.Binomial()

logger = getLogger(__name__)


def fit_replay_state_transition(speed, is_replay, penalty=1E-5,
                                speed_knots=None, diagonal=None):
    """Estimate the predicted probablity of replay given speed and whether
    it was a replay in the previous time step.

    p(I_t | I_t-1, v_t-1)

    p_I_0, p_I_1 in Long Tao's code

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : boolean ndarray, shape (n_time,)
    speed_knots : ndarray, shape (n_knots,)

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

    if speed_knots is None:
        speed_mid_point = np.nanmedian(speed[speed > 10])
        speed_knots = [1., 2., 3., speed_mid_point]

    MODEL_FORMULA = (
        'is_replay ~ 1 + lagged_is_replay + '
        'cr(lagged_speed, knots=speed_knots, constraints="center")')
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    penalty = np.ones((design_matrix.shape[1],)) * penalty
    penalty[0] = 0.0
    fit = penalized_IRLS(design_matrix, response, family=FAMILY,
                         penalty=penalty)
    if np.isnan(fit.AIC):
        logger.error("Discrete state transition failed to fit properly. "
                     "Try specifying `speed_knots`")
    return partial(predict_probability, design_matrix=design_matrix,
                   coefficients=fit.coefficients)


def fit_replay_state_transition_no_speed(
        speed, is_replay, penalty=1E-5, speed_knots=None, diagonal=None):
    """Estimate the predicted probablity of replay and whether
    it was a replay in the previous time step.

    p(I_t | I_t-1, v_t-1)

    p_I_0, p_I_1 in Long Tao's code

    Parameters
    ----------
    speed : ndarray, shape (n_time,)
    is_replay : boolean ndarray, shape (n_time,)
    speed_knots : ndarray, shape (n_knots,)

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

    MODEL_FORMULA = 'is_replay ~ 1 + lagged_is_replay'
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    penalty = np.ones((design_matrix.shape[1],)) * penalty
    penalty[0] = 0.0
    fit = penalized_IRLS(design_matrix, response, family=FAMILY,
                         penalty=penalty)
    return partial(predict_probability, design_matrix=design_matrix,
                   coefficients=fit.coefficients)


def constant_transition(
        speed, is_replay, penalty=1E-5, speed_knots=None, diagonal=None):
    if diagonal is None:
        diagonal = np.array([0.00003, 0.98])
    return partial(_constant_probability, diagonal=diagonal)


def make_design_matrix(lagged_is_replay, lagged_speed, design_matrix):
    predict_data = {
        'lagged_is_replay': lagged_is_replay * np.ones_like(lagged_speed),
        'lagged_speed': lagged_speed
    }
    return build_design_matrices(
        [design_matrix.design_info], predict_data,
        NA_action=NAAction(NA_types=[]))[0]


def make_design_matrix_no_speed(lagged_is_replay, lagged_speed, design_matrix):
    predict_data = {
        'lagged_is_replay': lagged_is_replay * np.ones_like(lagged_speed),
    }
    return build_design_matrices(
        [design_matrix.design_info], predict_data,
        NA_action=NAAction(NA_types=[]))[0]


def predict_probability(lagged_speed, design_matrix, coefficients):
    """Predict probability of replay state given speed and whether it was a
    replay in the previous time step.

    Parameters
    ----------
    design_matrix : patsy design matrix
    fit : statsmodels fitted model
    speed : ndarray, shape (n_time,)

    Returns
    -------
    replay_probability : ndarray, shape (n_time, 2)

    """
    no_previous_replay_design_matrix = make_design_matrix(
        0, lagged_speed, design_matrix)

    previous_replay_design_matrix = make_design_matrix(
        1, lagged_speed, design_matrix)

    replay_probability = np.stack(
        (predict(no_previous_replay_design_matrix, coefficients),
         predict(previous_replay_design_matrix, coefficients)),
        axis=1)

    replay_probability[np.isnan(replay_probability)] = 0.0

    return replay_probability


def predict_probability_no_speed(lagged_speed, design_matrix, coefficients):
    """Predict probability of replay state given speed and whether it was a
    replay in the previous time step.

    Parameters
    ----------
    design_matrix : patsy design matrix
    fit : statsmodels fitted model
    speed : ndarray, shape (n_time,)

    Returns
    -------
    replay_probability : ndarray, shape (n_time, 2)

    """
    no_previous_replay_design_matrix = make_design_matrix_no_speed(
        0, lagged_speed, design_matrix)

    previous_replay_design_matrix = make_design_matrix_no_speed(
        1, lagged_speed, design_matrix)

    replay_probability = np.stack(
        (predict(no_previous_replay_design_matrix, coefficients),
         predict(previous_replay_design_matrix, coefficients)),
        axis=1)

    replay_probability[np.isnan(replay_probability)] = 0.0

    return replay_probability


def _constant_probability(lagged_speed, diagonal=None):
    if diagonal is None:
        diagonal = np.array([0.00003, 0.98])
    n_time = lagged_speed.shape[0]
    replay_probability = np.zeros((n_time, 2))
    replay_probability[:, 0] = diagonal[0]
    replay_probability[:, 1] = diagonal[1]

    return replay_probability


def predict(design_matrix, coefficients):
    return FAMILY.link.inverse(design_matrix @ np.squeeze(coefficients))


_DISCRETE_STATE_TRANSITIONS = {
    'ripples_with_speed_threshold': fit_replay_state_transition,
    'ripples_no_speed_threshold': fit_replay_state_transition_no_speed,
    'constant': constant_transition,
}
