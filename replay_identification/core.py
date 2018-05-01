import numpy as np
from functools import wraps


def combined_likelihood(log_likelihood_function):
    @wraps(log_likelihood_function)
    def decorated_function(*args, **kwargs):
        try:
            return np.nansum(log_likelihood_function(*args, **kwargs), axis=-1)
        except ValueError:
            return log_likelihood_function(*args, **kwargs).squeeze()
    return decorated_function


def get_place_bins(position, place_bin_size):
    not_nan_position = position[~np.isnan(position)]
    n_bins = (np.floor(np.ptp(not_nan_position) / place_bin_size) + 2
              ).astype(np.int)
    return np.linspace(
        np.min(not_nan_position), np.max(not_nan_position) + 1E-3, n_bins)


def get_place_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2
