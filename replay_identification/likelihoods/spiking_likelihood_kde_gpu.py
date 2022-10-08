"""Estimates a Poisson likelihood using place fields estimated with a KDE
using GPUs"""
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from replay_identification.bins import atleast_2d
from replay_identification.core import scale_likelihood
from scipy.interpolate import griddata
from tqdm.auto import tqdm

try:
    import cupy as cp

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

    @cp.fuse
    def gaussian_pdf(x, mean, sigma):
        """Compute the value of a Gaussian probability density function at x with
        given mean and sigma."""
        return cp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * cp.sqrt(2.0 * cp.pi))

    def estimate_position_distance(
        place_bin_centers: cp.ndarray,
        positions: cp.ndarray,
        position_std: cp.ndarray,
    ) -> cp.ndarray:
        """Estimates the Euclidean distance between positions and position bins.

        Parameters
        ----------
        place_bin_centers : cp.ndarray, shape (n_position_bins, n_position_dims)
        positions : cp.ndarray, shape (n_time, n_position_dims)
        position_std : array_like, shape (n_position_dims,)

        Returns
        -------
        position_distance : cp.ndarray, shape (n_time, n_position_bins)

        """
        n_time, n_position_dims = positions.shape
        n_position_bins = place_bin_centers.shape[0]

        position_distance = cp.ones((n_time, n_position_bins), dtype=cp.float32)

        if isinstance(position_std, (int, float)):
            position_std = [position_std] * n_position_dims

        for position_ind, std in enumerate(position_std):
            position_distance *= gaussian_pdf(
                cp.expand_dims(place_bin_centers[:, position_ind], axis=0),
                cp.expand_dims(positions[:, position_ind], axis=1),
                std,
            )

        return position_distance

    def estimate_position_density(
        place_bin_centers: cp.ndarray,
        positions: cp.ndarray,
        position_std: cp.ndarray,
        block_size: int = 100,
        sample_weights: cp.ndarray = None,
    ) -> cp.ndarray:
        """Estimates a kernel density estimate over position bins using
        Euclidean distances.

        Parameters
        ----------
        place_bin_centers : cp.ndarray, shape (n_position_bins, n_position_dims)
        positions : cp.ndarray, shape (n_time, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
        sample_weights : None or cp.ndarray, shape (n_time,)

        Returns
        -------
        position_density : cp.ndarray, shape (n_position_bins,)

        """
        n_position_bins = place_bin_centers.shape[0]

        if block_size is None:
            block_size = n_position_bins

        position_density = cp.empty((n_position_bins,))
        for start_ind in range(0, n_position_bins, block_size):
            block_inds = slice(start_ind, start_ind + block_size)
            position_density[block_inds] = cp.average(
                estimate_position_distance(
                    place_bin_centers[block_inds], positions, position_std
                ),
                axis=0,
                weights=sample_weights,
            )
        return position_density

    def get_firing_rate(
        is_spike: cp.ndarray,
        position: cp.ndarray,
        place_bin_centers: cp.ndarray,
        is_track_interior: cp.ndarray,
        not_nan_position: cp.ndarray,
        occupancy: cp.ndarray,
        position_std: cp.ndarray,
        block_size: int = None,
        weights: cp.ndarray = None,
    ) -> cp.ndarray:
        if is_spike.sum() > 0:
            mean_rate = is_spike.mean()
            marginal_density = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)

            marginal_density[is_track_interior] = estimate_position_density(
                place_bin_centers[is_track_interior],
                cp.asarray(position[is_spike & not_nan_position], dtype=cp.float32),
                position_std,
                block_size=block_size,
                sample_weights=cp.asarray(
                    weights[is_spike & not_nan_position], dtype=cp.float32
                ),
            )
            return cp.exp(
                cp.log(mean_rate) + cp.log(marginal_density) - cp.log(occupancy)
            )
        else:
            return cp.zeros_like(occupancy)

    def estimate_place_conditional_intensity_kde(
        position: cp.ndarray,
        spikes: cp.ndarray,
        place_bin_centers: cp.ndarray,
        position_std: cp.ndarray,
        is_track_interior: cp.ndarray = None,
        block_size: int = None,
        is_training: cp.ndarray = None,
    ) -> xr.DataArray:
        """Gives the conditional intensity of the neurons' spiking with respect to
        position.

        Parameters
        ----------
        position : cp.ndarray, shape (n_time, n_position_dims)
        spikes : cp.ndarray, shape (n_time,)
            Indicator of spike or no spike at current time.
        place_bin_centers : cp.ndarray, shape (n_bins, n_position_dims)
        position_std : float or array_like, shape (n_position_dims,)
            Amount of smoothing for position.  Standard deviation of kernel.
        is_track_interior : None or cp.ndarray, shape (n_bins,)
        block_size : int
            Size of data to process in chunks

        Returns
        -------
        conditional_intensity : cp.ndarray, shape (n_bins, n_neurons)

        """

        position = atleast_2d(position).astype(cp.float32)
        place_bin_centers = atleast_2d(place_bin_centers).astype(cp.float32)
        not_nan_position = cp.all(~cp.isnan(position), axis=1)

        occupancy = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)
        occupancy[is_track_interior.ravel(order="F")] = estimate_position_density(
            place_bin_centers[is_track_interior.ravel(order="F")],
            position[not_nan_position],
            position_std,
            block_size=block_size,
            sample_weights=cp.asarray(is_training, dtype=cp.float32),
        )
        place_conditional_intensity = cp.stack(
            [
                get_firing_rate(
                    is_spike=is_spike,
                    position=position,
                    place_bin_centers=place_bin_centers,
                    is_track_interior=is_track_interior.ravel(order="F"),
                    not_nan_position=not_nan_position,
                    occupancy=occupancy,
                    position_std=position_std,
                    weights=cp.asarray(is_training, dtyp=cp.float32),
                )
                for is_spike in cp.asarray(spikes, dtype=bool).T
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

        return xr.DataArray(
            data=cp.asnumpy(place_conditional_intensity), coords=coords, dims=DIMS
        )

    def poisson_log_likelihood(
        spikes: cp.ndarray, conditional_intensity: cp.ndarray
    ) -> cp.ndarray:
        """Probability of parameters given spiking at a particular time.

        Parameters
        ----------
        spikes : cp.ndarray, shape (n_time,)
            Indicator of spike or no spike at current time.
        conditional_intensity : cp.ndarray, shape (n_place_bins,)
            Instantaneous probability of observing a spike

        Returns
        -------
        poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

        """
        return (
            cp.log(conditional_intensity[cp.newaxis, :] + np.spacing(1))
            * spikes[:, cp.newaxis]
            - conditional_intensity[cp.newaxis, :]
            + np.spacing(1)
        )

    def combined_likelihood(
        spikes: cp.ndarray, conditional_intensity: cp.ndarray
    ) -> cp.ndarray:
        """

        Parameters
        ----------
        spikes : cp.ndarray, shape (n_time, n_neurons)
        conditional_intensity : cp.ndarray, shape (n_bins, n_neurons)

        """
        n_time = spikes.shape[0]
        n_bins = conditional_intensity.shape[0]
        log_likelihood = cp.zeros((n_time, n_bins))

        mempool = cp.get_default_memory_pool()

        for is_spike, ci in zip(
            tqdm(cp.asarray(spikes).T), cp.asarray(conditional_intensity).T
        ):
            log_likelihood += poisson_log_likelihood(is_spike, ci)
            mempool.free_all_blocks()

        return cp.asnumpy(log_likelihood)

    def estimate_non_local_spiking_likelihood(
        spikes: cp.ndarray,
        conditional_intensity: cp.ndarray,
        is_track_interior: cp.ndarray = None,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        spikes : cp.ndarray, shape (n_time, n_neurons)
        conditional_intensity : cp.ndarray, shape (n_bins, n_neurons)
        is_track_interior : None or cp.ndarray, optional, shape (n_x_position_bins,
                                                                n_y_position_bins)
        Returns
        -------
        likelihood : cp.ndarray, shape (n_time, n_bins)
        """
        spikes = np.asarray(spikes, dtype=cp.float32)

        if is_track_interior is not None:
            is_track_interior = is_track_interior.ravel(order="F")
        else:
            n_bins = conditional_intensity.shape[0]
            is_track_interior = np.ones((n_bins,), dtype=cp.bool)

        log_likelihood = combined_likelihood(spikes, conditional_intensity)

        mask = np.ones_like(is_track_interior, dtype=cp.float)
        mask[~is_track_interior] = cp.nan

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
            cp.asarray(position, dtype=cp.float32),
            cp.asarray(encoding_position, dtype=cp.float32),
            position_std,
            block_size=block_size,
            sample_weights=cp.asarray(is_training, dtype=cp.float32),
        )
        log_likelihood = cp.zeros_like(occupancy)

        for neuron_ind, is_spike in tqdm(
            enumerate(spikes.T), disable=disable_progress_bar
        ):
            is_enc_spike = encoding_spikes[:, neuron_ind].astype(bool)
            mean_rate = is_enc_spike.mean()

            if (is_spike.sum() > 0) & (is_enc_spike.sum() > 0):
                marginal_density = estimate_position_density(
                    cp.asarray(position, dtype=cp.float32),
                    cp.asarray(encoding_position[is_enc_spike], dtype=cp.float32),
                    position_std,
                    block_size=block_size,
                    sample_weights=is_training[is_enc_spike],
                )
            else:
                marginal_density = cp.zeros_like(occupancy)
            intensity = np.spacing(1) + (mean_rate * marginal_density / occupancy)
            log_likelihood += cp.log(intensity) * is_spike - intensity

        return cp.asnumpy(log_likelihood)

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
        place_conditional_intensity = cp.asarray(place_conditional_intensity)
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

    def fit_spiking_likelihood_kde_gpu(
        position: np.ndarray,
        spikes: np.ndarray,
        is_training: np.ndarray,
        place_bin_centers: np.ndarray,
        is_track_interior: np.ndarray = None,
        position_std: np.ndarray = 6.0,
        block_size: int = None,
        **kwargs
    ):

        is_training = np.asarray(is_training).astype(np.float32)
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

except ImportError:

    def fit_spiking_likelihood_kde_gpu(*args, **kwargs):
        pass
