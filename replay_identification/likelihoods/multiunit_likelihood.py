"""
"""

from functools import partial

import numba
import numpy as np
from replay_identification.bins import atleast_2d
from replay_identification.core import scale_likelihood
from sklearn.base import BaseEstimator, DensityMixin
from tqdm.auto import tqdm

SQRT_2PI = np.float64(np.sqrt(2.0 * np.pi))


def multiunit_likelihood(
    multiunits,
    position,
    place_bin_centers,
    occupancy_model,
    joint_models,
    marginal_models,
    mean_rates,
    is_track_interior,
    time_bin_size=1,
    set_no_spike_to_equally_likely=False,
):
    """The likelihood of being in a non-local state vs. a local state based
    on whether the multiunits correspond to the current position of the animal.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_marks, n_electrodes)
    position : ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    occupancy_model : list of fitted density models, len (n_electrodes)
    joint_models : list of fitted density models, len (n_electrodes)
    marginal_models : list of fitted density models, len (n_electrodes)
    mean_rates : list of floats, len (n_electrodes)
    is_track_interior : ndarray, shape (n_bins, n_position_dim)
    time_bin_size : float, optional

    Returns
    -------
    multiunit_likelihood : ndarray, shape (n_time, 2, n_place_bins)

    """
    n_time = multiunits.shape[0]
    n_place_bins = place_bin_centers.shape[0]
    multiunit_likelihood = np.zeros((n_time, 2, n_place_bins))
    multiunit_likelihood[:, 1, :] = estimate_non_local_log_likelihood(
        np.moveaxis(multiunits, -1, 0),
        place_bin_centers,
        occupancy_model,
        joint_models,
        marginal_models,
        mean_rates,
        is_track_interior,
        time_bin_size,
    )
    multiunit_likelihood[:, 0, :] = estimate_local_log_likelihood(
        np.moveaxis(multiunits, -1, 0),
        position,
        occupancy_model,
        joint_models,
        marginal_models,
        mean_rates,
        time_bin_size,
    )

    if set_no_spike_to_equally_likely:
        no_spike = np.all(np.isnan(multiunits), axis=(1, 2))
        multiunit_likelihood[no_spike] = 0.0
    multiunit_likelihood[:, :, ~is_track_interior] = np.nan

    return scale_likelihood(multiunit_likelihood)


def estimate_non_local_log_likelihood(
    multiunits,
    place_bin_centers,
    occupancy_model,
    joint_models,
    marginal_models,
    mean_rates,
    is_track_interior,
    time_bin_size,
):
    """Estimate the log likelihood of being at any position.

    Parameters
    ----------
    multiunits : ndarray, shape (n_electrodes, n_time, n_features)
    place_bin_centers : ndarray, shape (n_place_bins,)
    occupancy_model : fitted density model
    marginal_models : list of fitted density models, len (n_electrodes,)
    mean_rates : list of floats, shape (n_electrodes,)
    is_track_interior : ndarray, shape (n_bins, n_position_dim)
    time_bin_size : float

    Returns
    -------
    non_local_log_likelihood : ndarray, shape (n_time, n_place_bins)

    """

    n_bin = place_bin_centers.shape[0]
    n_time = multiunits.shape[1]
    log_likelihood = np.zeros((n_time, n_bin))

    occupancy = estimate_occupancy(place_bin_centers, occupancy_model)

    for multiunit, joint_model, marginal_model, mean_rate in zip(
        tqdm(multiunits, desc="electrodes"), joint_models, marginal_models, mean_rates
    ):
        ground_process_intensity = np.atleast_2d(
            estimate_ground_process_intensity(
                place_bin_centers, occupancy, marginal_model, mean_rate
            )
        )
        log_joint_mark_intensity = np.ones((n_time, n_bin))
        log_joint_mark_intensity[:, is_track_interior] = np.stack(
            [
                estimate_log_joint_mark_intensity(
                    multiunit,
                    place_bin * np.ones((n_time, 1)),
                    joint_model,
                    mean_rate,
                    occ * np.ones((n_time,)),
                )
                for occ, place_bin in zip(
                    occupancy[is_track_interior], place_bin_centers[is_track_interior]
                )
            ],
            axis=1,
        )
        log_likelihood += poisson_mark_log_likelihood(
            log_joint_mark_intensity, ground_process_intensity, time_bin_size
        )

    return log_likelihood


def estimate_local_log_likelihood(
    multiunits,
    position,
    occupancy_model,
    joint_models,
    marginal_models,
    mean_rates,
    time_bin_size,
):
    """Estimate the log likelihood of being at the current position.

    Parameters
    ----------
    multiunits : ndarray, shape (n_electrodes, n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    occupancy_model : fitted density model
    joint_models : list of fitted density models, len (n_electrodes,)
    marginal_models : list of fitted density models, len (n_electrodes,)
    mean_rates : list of floats, len (n_electrodes,)
    time_bin_size : float

    Returns
    -------
    local_log_likelihood : ndarray, shape (n_time,)

    """
    n_time = multiunits.shape[1]
    log_likelihood = np.zeros((n_time, 1))

    occupancy = estimate_occupancy(position, occupancy_model)

    for multiunit, joint_model, marginal_model, mean_rate in zip(
        tqdm(multiunits, desc="electrodes"), joint_models, marginal_models, mean_rates
    ):
        ground_process_intensity = estimate_ground_process_intensity(
            position, occupancy, marginal_model, mean_rate
        )[:, np.newaxis]
        log_joint_mark_intensity = estimate_log_joint_mark_intensity(
            multiunit, position, joint_model, mean_rate, occupancy
        )[:, np.newaxis]
        log_likelihood += poisson_mark_log_likelihood(
            log_joint_mark_intensity, ground_process_intensity, time_bin_size
        )

    return log_likelihood


def poisson_mark_log_likelihood(
    log_joint_mark_intensity, ground_process_intensity, time_bin_size=1
):
    """Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    log_joint_mark_intensity : ndarray, shape (n_time, n_position)
    ground_process_intensity : ndarray, shape (n_time, n_position)
        Probability of observing a spike regardless of multiunit.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_log_likelihood : ndarray, shape (n_time, n_position)

    """
    return (
        np.log(time_bin_size)
        + np.nan_to_num(log_joint_mark_intensity)
        - (ground_process_intensity * time_bin_size)
    )


def estimate_occupancy(position, occupancy_model):
    """Computes the spatial occupancy.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    occupancy_model : fitted density model

    Returns
    -------
    occupancy : ndarray, shape (n_time,)

    """
    position = atleast_2d(position)
    not_nan_position = np.all(~np.isnan(position), axis=1)
    occupancy = np.full((position.shape[0],), np.nan)
    occupancy[not_nan_position] = np.exp(
        occupancy_model.score_samples(position[not_nan_position])
    )
    return occupancy


def estimate_ground_process_intensity(position, occupancy, marginal_model, mean_rate):
    """Computes the rate function of position marginalized over mark.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    occupancy : ndarray, shape (n_position_dims,)
    marginal_model : fitted density model
    mean_rate : float

    Returns
    -------
    ground_process_intensity : ndarray, shape (n_time,)

    """
    place_field = marginal_model.score_samples(atleast_2d(position))
    return np.exp(np.log(mean_rate) + place_field - np.log(occupancy))


def estimate_log_joint_mark_intensity(
    multiunit, position, joint_model, mean_rate, occupancy
):
    """Computes the rate function of position and mark.

    Parameters
    ----------
    multiunit : ndarray, shape  (n_time, n_features)
    position : ndarray, shape  (n_time, n_position_dims)
    joint_model : fitted density model
    mean_rate : float
    occupancy : ndarray, shape (n_time,)

    Returns
    -------
    log_joint_mark_intensity : ndarray, shape (n_time,)

    """
    multiunit, position = atleast_2d(multiunit), atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1) & np.all(
        ~np.isnan(position), axis=1
    )
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)

    log_joint_mark_intensity = np.zeros((position.shape[0],))
    log_joint_mark_intensity[is_spike] = (
        np.log(mean_rate)
        + joint_model.score_samples(
            np.concatenate(
                (multiunit[is_spike][:, not_nan_marks], position[is_spike]), axis=1
            )
        )
        - np.log(occupancy[is_spike])
    )
    return log_joint_mark_intensity


def train_marginal_model(multiunit, position, density_model, model_kwargs):
    """

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_marginal_model : density_model class instance

    """
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan_position = np.all(~np.isnan(atleast_2d(position)), axis=1)
    return density_model(**model_kwargs).fit(
        atleast_2d(position)[is_spike & not_nan_position]
    )


def train_occupancy_model(position, density_model, model_kwargs):
    """Fits a density model for computing the spatial occupancy.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_occupancy_model : density_model class instance

    """
    position = atleast_2d(position)
    not_nan_position = np.all(~np.isnan(atleast_2d(position)), axis=1)
    return density_model(**model_kwargs).fit(position[not_nan_position])


def train_joint_model(multiunit, position, density_model, model_kwargs):
    """Fits a density model to the joint pdf of position and mark.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_joint_model : density_model class instance

    """
    multiunit, position = atleast_2d(multiunit), atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1) & np.all(
        ~np.isnan(position), axis=1
    )
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)

    return density_model(**model_kwargs).fit(
        np.concatenate(
            (multiunit[is_spike][:, not_nan_marks], position[is_spike]), axis=1
        )
    )


def estimate_mean_rate(multiunit, position):
    """Mean rate of multiunit.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)

    Returns
    -------
    mean_rate : float

    """
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan = np.all(~np.isnan(atleast_2d(position)), axis=1)
    return np.mean(is_spike[not_nan])


def fit_multiunit_likelihood(
    position,
    multiunits,
    is_training,
    place_bin_centers,
    density_model,
    model_kwargs,
    occupancy_marginal_model,
    occupancy_kwargs,
    is_track_interior=None,
):
    """Precompute quantities to fit the multiunit likelihood to new data.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    multiunits : ndarray, shape (n_time, n_features, n_electrodes)
    is_training : bool ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict
    is_track_interior : ndarray, shape (n_bins, n_position_dim)

    Returns
    -------
    multiunit_likelihood : function

    """
    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=np.bool)
    joint_models = []
    marginal_models = []
    mean_rates = []
    occupancy_model = train_occupancy_model(
        position[is_training], occupancy_marginal_model, occupancy_kwargs
    )

    for multiunit in tqdm(
        np.moveaxis(multiunits[is_training], -1, 0), desc="electrodes"
    ):
        mean_rates.append(estimate_mean_rate(multiunit, position[is_training]))
        joint_models.append(
            train_joint_model(
                multiunit, position[is_training], density_model, model_kwargs
            )
        )
        marginal_models.append(
            train_marginal_model(
                multiunit,
                position[is_training],
                occupancy_marginal_model,
                occupancy_kwargs,
            )
        )

    return partial(
        multiunit_likelihood,
        place_bin_centers=place_bin_centers,
        occupancy_model=occupancy_model,
        joint_models=joint_models,
        marginal_models=marginal_models,
        mean_rates=mean_rates,
        is_track_interior=is_track_interior,
    )


@numba.njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def numba_kde(eval_points, samples, bandwidths):
    n_eval_points, n_bandwidths = eval_points.shape
    result = np.zeros((n_eval_points,))
    n_samples = len(samples)

    for i in numba.prange(n_eval_points):
        for j in range(n_samples):
            product_kernel = 1.0
            for k in range(n_bandwidths):
                bandwidth = bandwidths[k]
                eval_point = eval_points[i, k]
                sample = samples[j, k]
                product_kernel *= (
                    np.exp(-0.5 * ((eval_point - sample) / bandwidth) ** 2)
                    / (bandwidth * SQRT_2PI)
                ) / bandwidth
            result[i] += product_kernel
        result[i] /= n_samples

    return result


class NumbaKDE(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y=None, sample_weight=None):
        self.training_data = X
        return self

    def score_samples(self, X):
        return np.log(numba_kde(X, self.training_data, self.bandwidth[-X.shape[1] :]))
