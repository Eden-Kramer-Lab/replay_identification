import numpy as np
from statsmodels.api import GLM, families
from patsy import build_design_matrices, dmatrices
from statsmodels.tsa.tsatools import lagmat
from functools import partial


def fit_speed_state_transition(speed, is_replay, penalty=1E-5):
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
    data = {
        'is_replay': is_replay.astype(np.float64),
        'lagged_is_replay': lagmat(is_replay, maxlag=1).astype(np.float64),
        'lagged_speed': lagmat(speed, maxlag=1)
    }
    MODEL_FORMULA = (
        'is_replay ~ 1 + lagged_is_replay + '
        'cr(lagged_speed, knots=[1, 2, 3, 20], constraints="center")')
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    family = families.Binomial()
    regularization_weights = np.ones((design_matrix.shape[1],)) * penalty
    regularization_weights[0] = 0.0
    model = GLM(response, design_matrix, family=family)
    fit = model.fit_regularized(alpha=regularization_weights, L1_wt=0)
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

    return np.stack((fit.predict(no_previous_replay_design_matrix),
                     fit.predict(previous_replay_design_matrix)), axis=1)
