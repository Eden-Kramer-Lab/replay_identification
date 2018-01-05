import numpy as np
from statsmodels.api import GLM, families
from patsy import dmatrices
from scipy.stats import norm


def estimate_movement_variance(position, speed, speed_threshold=4):
    is_above_threshold = speed[1:] > speed_threshold

    data = {
        'position': position[1:][is_above_threshold],
        'lagged_position': position[:-1][is_above_threshold]
    }

    MODEL_FORMULA = 'position ~ lagged_position - 1'
    response, design_matrix = dmatrices(MODEL_FORMULA, data)
    fit = GLM(response, design_matrix, family=families.Gaussian()).fit()

    sigma = np.sqrt(np.sum(fit.resid_response ** 2) / fit.df_resid)
    return fit.params[0], sigma


def estimate_position_state_transition(place_bins, position, variance):
    """Zero mean random walk with covariance based on movement

    p(x_{k} | x_{k-1}, I_{k}, I_{k-1})
    """
    position_bin_size = np.diff(place_bins)[0]
    state_transition_matrix = norm.pdf(
        place_bins[:, np.newaxis], loc=place_bins[np.newaxis, :],
        scale=variance)
    state_transition_matrix /= np.sum(
        state_transition_matrix, axis=0, keepdims=True) / position_bin_size

    return state_transition_matrix
