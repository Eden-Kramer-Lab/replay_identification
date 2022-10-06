"""Estimates a Poisson likelihood using place fields estimated with a KDE"""

from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from replay_identification.bins import atleast_2d
from replay_identification.core import scale_likelihood
from scipy.interpolate import griddata
from tqdm.auto import tqdm


def _interpolate_value(place_bin_centers, likelihood, pos):
    value = griddata(place_bin_centers, likelihood, pos, method="linear")
    if np.isnan(value):
        value = griddata(place_bin_centers, likelihood, pos, method="nearest")
    return value


def interpolate_local_likelihood(place_bin_centers, non_local_likelihood, position):

    return np.asarray(
        [
            _interpolate_value(place_bin_centers, likelihood, pos)
            for likelihood, pos in zip(non_local_likelihood, tqdm(position))
        ]
    )


def gaussian_pdf(x, mean, sigma):
    """Compute the value of a Gaussian probability density function at x with
    given mean and sigma."""
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def estimate_position_distance(
    place_bin_centers: np.ndarray,
    positions: np.ndarray,
    position_std: np.ndarray,
) -> np.ndarray:
    """Estimates the Euclidean distance between positions and position bins.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
    positions : np.ndarray, shape (n_time, n_position_dims)
    position_std : array_like, shape (n_position_dims,)

    Returns
    -------
    position_distance : np.ndarray, shape (n_time, n_position_bins)

    """
    n_time, n_position_dims = positions.shape
    n_position_bins = place_bin_centers.shape[0]

    position_distance = np.ones((n_time, n_position_bins), dtype=np.float32)

    if isinstance(position_std, (int, float)):
        position_std = [position_std] * n_position_dims

    for position_ind, std in enumerate(position_std):
        position_distance *= gaussian_pdf(
            np.expand_dims(place_bin_centers[:, position_ind], axis=0),
            np.expand_dims(positions[:, position_ind], axis=1),
            std,
        )

    return position_distance


def estimate_position_density(
    place_bin_centers: np.ndarray,
    positions: np.ndarray,
    position_std: np.ndarray,
    block_size: int = 100,
    sample_weights: np.ndarray = None,
) -> np.ndarray:
    """Estimates a kernel density estimate over position bins using
    Euclidean distances.

    Parameters
    ----------
    place_bin_centers : np.ndarray, shape (n_position_bins, n_position_dims)
    positions : np.ndarray, shape (n_time, n_position_dims)
    position_std : float or array_like, shape (n_position_dims,)
    sample_weights : None or np.ndarray, shape (n_time,)

    Returns
    -------
    position_density : np.ndarray, shape (n_position_bins,)

    """
    n_position_bins = place_bin_centers.shape[0]

    if block_size is None:
        block_size = n_position_bins

    position_density = np.empty((n_position_bins,))
    for start_ind in range(0, n_position_bins, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        position_density[block_inds] = np.average(
            estimate_position_distance(
                place_bin_centers[block_inds], positions, position_std
            ),
            axis=0,
            weights=sample_weights,
        )
    return position_density


def get_firing_rate(
    is_spike: np.ndarray,
    position: np.ndarray,
    place_bin_centers: np.ndarray,
    is_track_interior: np.ndarray,
    not_nan_position: np.ndarray,
    occupancy: np.ndarray,
    position_std: np.ndarray,
    block_size: int = None,
    weights: np.ndarray = None,
) -> np.ndarray:
    if is_spike.sum() > 0:
        mean_rate = is_spike.mean()
        marginal_density = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)

        marginal_density[is_track_interior] = estimate_position_density(
            place_bin_centers[is_track_interior],
            np.asarray(position[is_spike & not_nan_position], dtype=np.float32),
            position_std,
            block_size=block_size,
            sample_weights=np.asarray(
                weights[is_spike & not_nan_position], dtype=np.float32
            ),
        )
        return np.exp(np.log(mean_rate) + np.log(marginal_density) - np.log(occupancy))
    else:
        return np.zeros_like(occupancy)


def estimate_place_conditional_intensity_kde(
    position: np.ndarray,
    spikes: np.ndarray,
    place_bin_centers: np.ndarray,
    position_std: np.ndarray,
    is_track_interior: np.ndarray = None,
    block_size: int = None,
    is_training: np.ndarray = None,
) -> np.ndarray:
    """Gives the conditional intensity of the neurons' spiking with respect to
    position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dims)
    spikes : np.ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    place_bin_centers : np.ndarray, shape (n_bins, n_position_dims)
    position_std : float or array_like, shape (n_position_dims,)
        Amount of smoothing for position.  Standard deviation of kernel.
    is_track_interior : None or np.ndarray, shape (n_bins,)
    block_size : int
        Size of data to process in chunks

    Returns
    -------
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """

    position = atleast_2d(position).astype(np.float32)
    place_bin_centers = atleast_2d(place_bin_centers).astype(np.float32)
    not_nan_position = np.all(~np.isnan(position), axis=1)

    occupancy = np.zeros((place_bin_centers.shape[0],), dtype=np.float32)
    occupancy[is_track_interior.ravel(order="F")] = estimate_position_density(
        place_bin_centers[is_track_interior.ravel(order="F")],
        position[not_nan_position],
        position_std,
        block_size=block_size,
        sample_weights=np.asarray(is_training),
    )
    place_conditional_intensity = np.stack(
        [
            get_firing_rate(
                is_spike=is_spike,
                position=position,
                place_bin_centers=place_bin_centers,
                is_track_interior=is_track_interior.ravel(order="F"),
                not_nan_position=not_nan_position,
                occupancy=occupancy,
                position_std=position_std,
                weights=np.asarray(is_training),
            )
            for is_spike in np.asarray(spikes, dtype=bool).T
        ],
        axis=1,
    )

    DIMS = ["position", "neuron"]
    if position.shape[1] == 1:
        names = ["position"]
        coords = {"position": place_bin_centers.squeeze()}
    elif position.shape[1] == 2:
        names = ["x_position", "y_position"]
        coords = {
            "position": pd.MultiIndex.from_arrays(
                place_bin_centers.T.tolist(), names=names
            )
        }

    return xr.DataArray(data=place_conditional_intensity, coords=coords, dims=DIMS)


def poisson_log_likelihood(
    spikes: np.ndarray, conditional_intensity: np.ndarray
) -> np.ndarray:
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : np.ndarray, shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    return (
        np.log(conditional_intensity[np.newaxis, :] + np.spacing(1))
        * spikes[:, np.newaxis]
        - conditional_intensity[np.newaxis, :]
        + np.spacing(1)
    )


def combined_likelihood(
    spikes: np.ndarray, conditional_intensity: np.ndarray
) -> np.ndarray:
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    for is_spike, ci in zip(tqdm(spikes.T), conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood


def estimate_non_local_spiking_likelihood(
    spikes: np.ndarray,
    conditional_intensity: np.ndarray,
    is_track_interior: np.ndarray = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)
    is_track_interior : None or np.ndarray, optional, shape (n_x_position_bins,
                                                             n_y_position_bins)
    Returns
    -------
    likelihood : np.ndarray, shape (n_time, n_bins)
    """
    spikes = np.asarray(spikes, dtype=np.float32)

    if is_track_interior is not None:
        is_track_interior = is_track_interior.ravel(order="F")
    else:
        n_bins = conditional_intensity.shape[0]
        is_track_interior = np.ones((n_bins,), dtype=np.bool)

    log_likelihood = combined_likelihood(spikes, conditional_intensity)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return log_likelihood * mask


def estimate_local_spiking_likelihood(
    spikes,
    position,
    encoding_spikes,
    encoding_position,
    position_std,
    is_training,
    block_size=1,
    disable_progress_bar=False,
):
    position = atleast_2d(position)
    encoding_position = atleast_2d(encoding_position)

    occupancy = estimate_position_density(
        np.asarray(position, dtype=np.float32),
        np.asarray(encoding_position, dtype=np.float32),
        position_std,
        block_size=block_size,
        sample_weights=np.asarray(is_training, dtype=np.float32),
    )
    log_likelihood = np.zeros_like(occupancy)

    for neuron_ind, is_spike in tqdm(enumerate(spikes.T), disable=disable_progress_bar):
        is_enc_spike = encoding_spikes[:, neuron_ind].astype(bool)
        mean_rate = is_enc_spike.mean()

        if (is_spike.sum() > 0) & (is_enc_spike.sum() > 0):
            enc_pos_at_spike = encoding_position[is_enc_spike]
            sample_weights = is_training[is_enc_spike]
            marginal_density = estimate_position_density(
                np.asarray(position, dtype=np.float32),
                np.asarray(enc_pos_at_spike, dtype=np.float32),
                position_std,
                block_size=block_size,
                sample_weights=sample_weights,
            )
        else:
            marginal_density = np.zeros_like(occupancy)
        intensity = np.spacing(1) + (mean_rate * marginal_density / occupancy)
        log_likelihood += np.log(intensity) * is_spike - intensity

    return log_likelihood


def spiking_likelihood(
    spikes,
    position,
    place_bin_centers,
    place_conditional_intensity,
    position_std,
    encoding_spikes,
    encoding_position,
    is_track_interior,
    is_training,
    set_no_spike_to_equally_likely=False,
    block_size=100,
    disable_progress_bar=False,
    interpolate_local_likelihood=False,
):
    """The likelihood of being in a replay state vs. not a replay state based
    on whether the multiunits correspond to the current position of the animal.

    Parameters
    ----------
    multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
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
    spiking_likelihood : ndarray, shape (n_time, 2, n_place_bins)

    """
    n_time = spikes.shape[0]
    n_place_bins = place_bin_centers.shape[0]
    spiking_likelihood = np.zeros((n_time, 2, n_place_bins), dtype=np.float32)
    place_conditional_intensity = np.asarray(place_conditional_intensity)
    spiking_likelihood[:, 1, :] = estimate_non_local_spiking_likelihood(
        spikes,
        place_conditional_intensity,
        is_track_interior,
    )
    if interpolate_local_likelihood:
        spiking_likelihood[:, 0, :] = interpolate_local_likelihood(
            place_bin_centers, spiking_likelihood[:, 1, :], position
        )
    else:
        spiking_likelihood[:, 0, :] = estimate_local_spiking_likelihood(
            spikes,
            position,
            encoding_spikes,
            encoding_position,
            position_std,
            is_training,
            block_size=1,
            disable_progress_bar=disable_progress_bar,
        )[:, np.newaxis]

    if set_no_spike_to_equally_likely:
        no_spike = np.all(np.isnan(spikes), axis=1)
        spiking_likelihood[no_spike] = 0.0
    spiking_likelihood[:, :, ~is_track_interior] = np.nan

    return scale_likelihood(spiking_likelihood)


def fit_spiking_likelihood_kde(
    position: np.ndarray,
    spikes: np.ndarray,
    is_training: np.ndarray,
    place_bin_centers: np.ndarray,
    is_track_interior: np.ndarray = None,
    position_std: np.ndarray = 6.0,
    block_size: int = None,
    **kwargs
) -> np.ndarray:

    is_training = np.asarray(is_training).astype(float)
    include = ~np.isclose(is_training, 0.0) & ~np.any(np.isnan(position), axis=1)

    is_training = is_training[include]
    position = position[include]
    spikes = spikes[include]

    place_conditional_intensity = estimate_place_conditional_intensity_kde(
        position,
        spikes,
        place_bin_centers,
        position_std,
        is_track_interior,
        block_size,
        is_training,
    )

    return partial(
        spiking_likelihood,
        place_bin_centers=place_bin_centers,
        place_conditional_intensity=place_conditional_intensity,
        position_std=position_std,
        encoding_spikes=spikes,
        encoding_position=position,
        is_track_interior=is_track_interior,
        is_training=is_training,
        **kwargs
    )
