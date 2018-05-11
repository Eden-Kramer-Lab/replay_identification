import numpy as np


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


def atleast_2d(x):
    """Adds a dimension to the last axis if the array is 1D."""
    return np.atleast_2d(x).T if x.ndim < 2 else x
