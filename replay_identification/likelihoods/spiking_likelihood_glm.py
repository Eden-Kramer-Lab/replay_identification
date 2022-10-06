"""Fitting and predicting the likelihood of non_local events based on place
field spiking patterns.
"""

import logging
from functools import partial
from logging import getLogger

import dask
import numpy as np
import scipy.stats
import statsmodels.api as sm
from dask.distributed import Client, get_client
from patsy import build_design_matrices, dmatrix
from replay_identification.bins import atleast_2d, get_n_bins
from replay_identification.core import scale_likelihood
from statsmodels.api import families
from tqdm.autonotebook import tqdm

logger = getLogger(__name__)


@dask.delayed
def fit_glm(response, design_matrix, is_training=None, penalty=None, tolerance=1e-5):
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(np.float).eps

    glm = sm.GLM(
        response.squeeze(),
        design_matrix,
        family=families.Poisson(),
        var_weights=is_training.squeeze(),
    )

    return glm.fit_regularized(alpha=penalty, L1_wt=0, cnvrg_tol=tolerance)


def poisson_log_likelihood(spikes, conditional_intensity):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : ndarray, shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    return scipy.stats.poisson.logpmf(spikes, conditional_intensity + np.spacing(1))


def spiking_likelihood(
    spikes,
    position,
    design_matrix,
    place_field_coefficients,
    place_conditional_intensity,
    is_track_interior,
    set_no_spike_to_equally_likely=False,
):
    """Computes the likelihood of non-local and local events.

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_neurons)
    position : ndarray, shape (n_time,)
    design_matrix : ndarray, shape (n_time, n_coefficients)
    place_field_coefficients : ndarray, shape (n_coefficients, n_neurons)
    place_conditional_intensity : ndarray, shape (n_place_bins, n_neurons)
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood : ndarray, shape (n_time, n_place_bins,)

    """
    local_design_matrix = make_spline_predict_matrix(
        design_matrix.design_info, position
    )
    local_conditional_intensity = get_firing_rate(
        local_design_matrix, place_field_coefficients, sampling_frequency=1
    )
    n_time = spikes.shape[0]
    n_place_bins = place_conditional_intensity.shape[0]
    spiking_log_likelihood = np.zeros((n_time, 2, n_place_bins))

    # Non-Local
    spiking_log_likelihood[:, 1, :] = combined_likelihood(
        spikes.T[..., np.newaxis], place_conditional_intensity.T[:, np.newaxis, :]
    )

    # Local
    spiking_log_likelihood[:, 0, :] = combined_likelihood(
        spikes.T, local_conditional_intensity.T
    )

    if set_no_spike_to_equally_likely:
        no_spike = np.isclose(spikes.sum(axis=1), 0.0)
        spiking_log_likelihood[no_spike] = 0.0

    is_track_interior = is_track_interior.ravel(order="F")
    spiking_log_likelihood[:, :, ~is_track_interior] = np.nan

    return scale_likelihood(spiking_log_likelihood)


def combined_likelihood(spikes, conditional_intensity):
    n_time = spikes.shape[1]
    n_bins = conditional_intensity.shape[-1] if conditional_intensity.ndim > 2 else 1
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(tqdm(spikes, desc="neurons"), conditional_intensity):
        log_likelihood += atleast_2d(poisson_log_likelihood(is_spike, ci))

    return log_likelihood


def make_spline_design_matrix(position, place_bin_edges, knot_spacing=10):
    inner_knots = []
    for pos, edges in zip(position.T, place_bin_edges.T):
        n_points = get_n_bins(edges, bin_size=knot_spacing)
        knots = np.linspace(edges.min(), edges.max(), n_points)[1:-1]
        knots = knots[(knots > pos.min()) & (knots < pos.max())]
        inner_knots.append(knots)

    inner_knots = np.meshgrid(*inner_knots)

    data = {}
    formula = "1 + te("
    for ind in range(position.shape[1]):
        formula += f"cr(x{ind}, knots=inner_knots[{ind}])"
        formula += ", "
        data[f"x{ind}"] = position[:, ind]

    formula += 'constraints="center")'

    return dmatrix(formula, data)


def make_spline_predict_matrix(design_info, position):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position[is_nan] = 0.0

    predict_data = {}
    for ind in range(position.shape[1]):
        predict_data[f"x{ind}"] = position[:, ind]

    design_matrix = build_design_matrices([design_info], predict_data)[0]
    design_matrix[is_nan] = np.nan

    return design_matrix


def get_firing_rate(design_matrix, coefficients, sampling_frequency=1):
    rate = np.exp(design_matrix @ coefficients) * sampling_frequency
    rate[np.isnan(rate)] = np.spacing(1)
    return rate


def fit_spiking_likelihood_glm(
    position,
    spikes,
    is_training,
    place_bin_centers,
    place_bin_edges,
    is_track_interior,
    spike_model_penalty=1e-6,
    spike_model_knot_spacing=30,
):
    """Estimate the place field model.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    spikes : ndarray, shape (n_time, n_neurons)
    place_bin_centers : ndarray, shape (n_place_bins,)
    penalty : float, optional
    time_bin_size : float, optional

    Returns
    -------
    spiking_likelihood : function

    """
    if np.any(np.ptp(place_bin_edges, axis=0) <= spike_model_knot_spacing):
        logging.warning("Range of position is smaller than knot spacing.")

    is_training = np.asarray(is_training).astype(float)
    include = ~np.isclose(is_training, 0.0) & ~np.any(np.isnan(position), axis=1)
    is_training = is_training[include]
    position = position[include]
    spikes = spikes[include]

    design_matrix = make_spline_design_matrix(
        position, place_bin_edges, spike_model_knot_spacing
    )
    try:
        client = get_client()
    except ValueError:
        client = Client()
    dm = client.scatter(np.asarray(design_matrix), broadcast=True)

    place_field_coefficients = [
        fit_glm(is_spike, dm, is_training, spike_model_penalty).params
        for is_spike in spikes.T
    ]
    place_field_coefficients = np.stack(dask.compute(*place_field_coefficients), axis=1)

    predict_matrix = make_spline_predict_matrix(
        design_matrix.design_info, place_bin_centers
    )
    place_conditional_intensity = get_firing_rate(
        predict_matrix, place_field_coefficients, sampling_frequency=1
    )

    return partial(
        spiking_likelihood,
        design_matrix=design_matrix,
        place_field_coefficients=place_field_coefficients,
        place_conditional_intensity=place_conditional_intensity,
        is_track_interior=is_track_interior,
    )
