import numpy as np
from functools import wraps


def scaled_likelihood(log_likelihood_func):
    '''Converts a log likelihood to a scaled likelihood with its max value at
    1.

    Used primarily to keep the likelihood numerically stable because more
    observations at a time point will lead to a smaller overall likelihood
    and this can exceed the floating point accuarcy of a machine.

    Parameters
    ----------
    log_likelihood_func : function

    Returns
    -------
    scaled_likelihood : function

    '''
    @wraps(log_likelihood_func)
    def decorated_function(*args, **kwargs):
        log_likelihood = log_likelihood_func(*args, **kwargs)
        axis = tuple(range(log_likelihood.ndim))[1:]
        return np.exp(log_likelihood - np.max(log_likelihood, axis=axis))

    return decorated_function


def combined_likelihood(log_likelihood_function):
    @wraps(log_likelihood_function)
    def decorated_function(*args, **kwargs):
        try:
            return np.sum(log_likelihood_function(*args, **kwargs),
                          axis=-1)
        except ValueError:
            return log_likelihood_function(*args, **kwargs).squeeze()
    return decorated_function


def get_place_bins(position, place_bin_size):
    not_nan_position = position[~np.isnan(position)]
    n_bins = (np.floor(np.ptp(not_nan_position) / place_bin_size) + 2
              ).astype(np.int)
    return np.linspace(
        np.min(not_nan_position), np.max(not_nan_position), n_bins)


def get_place_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2
