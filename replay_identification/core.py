import numpy as np


def get_bin_edges(x, n_bins=None, bin_size=None):
    not_nan_x = x[~np.isnan(x)]
    if bin_size is not None:
        n_bins = (
            np.round(np.ceil(np.ptp(not_nan_x) / bin_size))).astype(np.int)
    return np.linspace(np.min(not_nan_x), np.max(not_nan_x), n_bins + 1,
                       endpoint=True)


def get_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def atleast_2d(x):
    """Adds a dimension to the last axis if the array is 1D."""
    return np.atleast_2d(x).T if x.ndim < 2 else x
