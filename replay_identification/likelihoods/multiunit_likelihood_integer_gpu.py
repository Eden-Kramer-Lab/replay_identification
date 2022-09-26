from functools import partial

import numpy as np
from replay_identification.bins import atleast_2d
from replay_identification.core import scale_likelihood
from scipy.interpolate import griddata
from tqdm.autonotebook import tqdm

try:
    import cupy as cp

    @cp.fuse
    def gaussian_pdf(x, mean, sigma):
        """Compute the value of a Gaussian probability density function at x with
        given mean and sigma."""
        return cp.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * cp.sqrt(2.0 * cp.pi))

    def estimate_position_distance(place_bin_centers, positions, position_std):
        """

        Parameters
        ----------
        place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
        positions : ndarray, shape (n_time, n_position_dims)
        position_std : float

        Returns
        -------
        position_distance : ndarray, shape (n_time, n_position_bins)

        """
        n_time, n_position_dims = positions.shape
        n_position_bins = place_bin_centers.shape[0]

        if isinstance(position_std, (int, float)):
            position_std = [position_std] * n_position_dims

        position_distance = cp.ones((n_time, n_position_bins), dtype=cp.float32)

        for position_ind, std in enumerate(position_std):
            position_distance *= gaussian_pdf(
                cp.expand_dims(place_bin_centers[:, position_ind], axis=0),
                cp.expand_dims(positions[:, position_ind], axis=1),
                std,
            )

        return position_distance

    def estimate_position_density(
        place_bin_centers, positions, position_std, block_size=100, sample_weights=None
    ):
        """

        Parameters
        ----------
        place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
        positions : ndarray, shape (n_time, n_position_dims)
        position_std : float

        Returns
        -------
        position_density : ndarray, shape (n_position_bins,)

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

    def estimate_log_intensity(density, occupancy, mean_rate):
        return cp.log(mean_rate) + cp.log(density) - cp.log(occupancy)

    def estimate_intensity(density, occupancy, mean_rate):
        """

        Parameters
        ----------
        density : ndarray, shape (n_bins,)
        occupancy : ndarray, shape (n_bins,)
        mean_rate : float

        Returns
        -------
        intensity : ndarray, shape (n_bins,)

        """
        return cp.exp(estimate_log_intensity(density, occupancy, mean_rate))

    def normal_pdf_integer_lookup(x, mean, std=20, max_value=6000):
        """Fast density evaluation for integers by precomputing a hash table of
        values.

        Parameters
        ----------
        x : int
        mean : int
        std : float
        max_value : int

        Returns
        -------
        probability_density : int

        """
        normal_density = gaussian_pdf(cp.arange(-max_value, max_value), 0, std).astype(
            cp.float32
        )

        return normal_density[(x - mean) + max_value]

    def estimate_log_joint_mark_intensity(
        decoding_marks,
        encoding_marks,
        mark_std,
        occupancy,
        mean_rate,
        place_bin_centers=None,
        encoding_positions=None,
        position_std=None,
        max_mark_value=6000,
        set_diag_zero=False,
        position_distance=None,
        sample_weights=None,
    ):
        """

        Parameters
        ----------
        decoding_marks : ndarray, shape (n_decoding_spikes, n_features)
        encoding_marks : ndarray, shape (n_encoding_spikes, n_features)
        mark_std : float or ndarray, shape (n_features,)
        occupancy : ndarray, shape (n_position_bins,)
        mean_rate : float
        place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
        encoding_positions : ndarray, shape (n_decoding_spikes, n_position_dims)
        position_std : float
        is_track_interior : None or ndarray, shape (n_position_bins,)
        max_mark_value : int
        set_diag_zero : bool

        Returns
        -------
        log_joint_mark_intensity : ndarray, shape (n_decoding_spikes, n_position_bins)

        """

        n_encoding_spikes, n_marks = encoding_marks.shape
        n_decoding_spikes = decoding_marks.shape[0]

        if sample_weights is None:
            sample_weights = cp.ones((1, n_decoding_spikes), dtype=cp.float32)
            denominator = n_encoding_spikes
        else:
            sample_weights = cp.atleast_2d(sample_weights)
            denominator = cp.sum(sample_weights)

        mark_distance = (
            cp.ones((n_decoding_spikes, n_encoding_spikes), dtype=cp.float32)
            * sample_weights
        )

        for mark_ind in range(n_marks):
            mark_distance *= normal_pdf_integer_lookup(
                cp.expand_dims(decoding_marks[:, mark_ind], axis=1),
                cp.expand_dims(encoding_marks[:, mark_ind], axis=0),
                std=mark_std,
                max_value=max_mark_value,
            )

        if set_diag_zero:
            diag_ind = (cp.arange(n_decoding_spikes), cp.arange(n_decoding_spikes))
            mark_distance[diag_ind] = 0.0

        if position_distance is None:
            position_distance = estimate_position_distance(
                place_bin_centers, encoding_positions, position_std
            ).astype(cp.float32)

        return cp.asnumpy(
            estimate_log_intensity(
                mark_distance @ position_distance / denominator, occupancy, mean_rate
            )
        )

    def fit_multiunit_likelihood_gpu(
        position,
        multiunits,
        is_training,
        place_bin_centers,
        mark_std,
        position_std,
        is_track_interior=None,
        **kwargs
    ):
        """

        Parameters
        ----------
        position : ndarray, shape (n_time, n_position_dims)
        multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
        place_bin_centers : ndarray, shape ( n_bins, n_position_dims)
        model : sklearn model
        model_kwargs : dict
        occupancy_model : sklearn model
        occupancy_kwargs : dict
        is_track_interior : None or ndarray, shape (n_bins,)

        Returns
        -------
        joint_pdf_models : list of sklearn models, shape (n_electrodes,)
        ground_process_intensities : list of ndarray, shape (n_electrodes,)
        occupancy : ndarray, (n_bins, n_position_dims)
        mean_rates : ndarray, (n_electrodes,)

        """
        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=bool)

        is_zero = np.isclose(is_training.astype(float), 0.0)

        # Exclude from dataset if is training is exactly zero
        is_training = is_training[~is_zero]
        position = atleast_2d(position)[~is_zero]
        multiunits = multiunits[~is_zero]

        place_bin_centers = atleast_2d(place_bin_centers)
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior.ravel(order="F")], dtype=cp.float32
        )
        gpu_is_track_interior = cp.asarray(is_track_interior.ravel(order="F"))

        not_nan_position = np.all(~np.isnan(position), axis=1)

        occupancy = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)
        occupancy[gpu_is_track_interior] = estimate_position_density(
            interior_place_bin_centers,
            cp.asarray(position[not_nan_position], dtype=cp.float32),
            position_std,
            sample_weights=cp.asarray(is_training, dtype=cp.float32),
        )

        mean_rates = []
        summed_ground_process_intensity = cp.zeros(
            (place_bin_centers.shape[0],), dtype=cp.float32
        )
        encoding_marks = []
        encoding_positions = []
        encoding_weights = []

        for multiunit in np.moveaxis(multiunits, -1, 0):

            # ground process intensity
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            mean_rates.append(np.average(is_spike, weights=is_training))
            marginal_density = cp.zeros((place_bin_centers.shape[0],), dtype=cp.float32)

            if is_spike.sum() > 0:
                marginal_density[gpu_is_track_interior] = estimate_position_density(
                    interior_place_bin_centers,
                    cp.asarray(position[is_spike & not_nan_position], dtype=cp.float32),
                    position_std,
                    sample_weights=cp.asarray(
                        is_training[is_spike & not_nan_position], dtype=cp.float32
                    ),
                )

            summed_ground_process_intensity += estimate_intensity(
                marginal_density, occupancy, mean_rates[-1]
            )

            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            encoding_marks.append(
                cp.asarray(
                    multiunit[np.ix_(is_spike & not_nan_position, is_mark_features)],
                    dtype=cp.int16,
                )
            )
            encoding_positions.append(
                cp.asarray(position[is_spike & not_nan_position], dtype=cp.float32)
            )
            encoding_weights.append(
                cp.asarray(is_training[is_spike & not_nan_position], dtype=cp.float32)
            )

        summed_ground_process_intensity = cp.asnumpy(
            summed_ground_process_intensity
        ) + np.spacing(1)

        return partial(
            multiunit_likelihood,
            place_bin_centers=place_bin_centers,
            encoding_marks=encoding_marks,
            encoding_marks_position=encoding_positions,
            encoding_weights=encoding_weights,
            encoding_position=position,
            summed_ground_process_intensity=summed_ground_process_intensity,
            occupancy=occupancy,
            mean_rates=mean_rates,
            mark_std=mark_std,
            position_std=position_std,
            is_track_interior=is_track_interior,
            is_training=cp.asarray(is_training, dtype=cp.float32),
            **kwargs,
        )

    def estimate_non_local_multiunit_likelihood(
        multiunits,
        encoding_marks,
        encoding_weights,
        mark_std,
        place_bin_centers,
        encoding_positions,
        position_std,
        occupancy,
        mean_rates,
        summed_ground_process_intensity,
        max_mark_value=6000,
        set_diag_zero=False,
        is_track_interior=None,
        time_bin_size=1,
        block_size=100,
        disable_progress_bar=False,
    ):
        """

        Parameters
        ----------
        multiunits : ndarray, shape (n_time, n_marks, n_electrodes)
        place_bin_centers : ndarray, (n_bins, n_position_dims)
        joint_pdf_models : list of sklearn models, shape (n_electrodes,)
        ground_process_intensities : list of ndarray, shape (n_electrodes,)
        occupancy : ndarray, (n_bins, n_position_dims)
        mean_rates : ndarray, (n_electrodes,)

        Returns
        -------
        log_likelihood : (n_time, n_bins)

        """

        if is_track_interior is None:
            is_track_interior = np.ones((place_bin_centers.shape[0],), dtype=np.bool)
        else:
            is_track_interior = is_track_interior.ravel(order="F")

        n_time = multiunits.shape[0]
        log_likelihood = (
            -time_bin_size
            * summed_ground_process_intensity
            * np.ones((n_time, 1), dtype=np.float32)
        )

        multiunits = np.moveaxis(multiunits, -1, 0)
        n_position_bins = is_track_interior.sum()
        interior_place_bin_centers = cp.asarray(
            place_bin_centers[is_track_interior], dtype=cp.float32
        )
        gpu_is_track_interior = cp.asarray(is_track_interior)
        interior_occupancy = occupancy[gpu_is_track_interior]

        for multiunit, enc_marks, enc_pos, enc_weights, mean_rate in zip(
            tqdm(multiunits, desc="n_electrodes", disable=disable_progress_bar),
            encoding_marks,
            encoding_positions,
            encoding_weights,
            mean_rates,
        ):
            is_spike = np.any(~np.isnan(multiunit), axis=1)
            is_mark_features = np.any(~np.isnan(multiunit), axis=0)
            decoding_marks = cp.asarray(
                multiunit[np.ix_(is_spike, is_mark_features)], dtype=cp.int16
            )
            n_decoding_marks = decoding_marks.shape[0]
            log_joint_mark_intensity = np.zeros(
                (n_decoding_marks, n_position_bins), dtype=np.float32
            )

            if block_size is None:
                block_size = n_decoding_marks

            position_distance = estimate_position_distance(
                interior_place_bin_centers, enc_pos, position_std
            ).astype(cp.float32)

            for start_ind in range(0, n_decoding_marks, block_size):
                block_inds = slice(start_ind, start_ind + block_size)
                log_joint_mark_intensity[
                    block_inds
                ] = estimate_log_joint_mark_intensity(
                    decoding_marks[block_inds],
                    enc_marks,
                    mark_std,
                    interior_occupancy,
                    mean_rate,
                    max_mark_value=max_mark_value,
                    set_diag_zero=set_diag_zero,
                    position_distance=position_distance,
                    sample_weights=enc_weights,
                )
            log_likelihood[np.ix_(is_spike, is_track_interior)] += np.nan_to_num(
                log_joint_mark_intensity
            )

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        log_likelihood[:, ~is_track_interior] = np.nan

        return log_likelihood

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

    def estimate_local_occupancy(
        train_position,
        test_position,
        position_std,
        sample_weights=None,
        block_size=None,
    ):
        return estimate_position_density(
            cp.asarray(test_position, dtype=cp.float32),
            cp.asarray(train_position, dtype=cp.float32),
            position_std,
            block_size=block_size,
            sample_weights=sample_weights,
        )

    def estimate_local_gpi(
        test_position,
        enc_pos,
        occupancy,
        mean_rate,
        position_std,
        block_size=None,
        sample_weights=None,
    ):
        marginal_density = estimate_position_density(
            cp.asarray(test_position, dtype=cp.float32),
            cp.asarray(enc_pos, dtype=cp.float32),
            position_std,
            block_size=block_size,
            sample_weights=sample_weights,
        )
        return estimate_intensity(
            marginal_density, cp.asarray(occupancy, dtype=cp.float32), mean_rate
        )

    def estimate_local_log_joint_mark_intensity(
        decoding_marks,
        encoding_marks,
        mark_std,
        occupancy,
        mean_rate,
        decoding_position=None,
        encoding_position=None,
        position_std=None,
        max_mark_value=6000,
        set_diag_zero=False,
        position_distance=None,
        sample_weights=None,
    ):
        """
        Parameters
        ----------
        decoding_marks : ndarray, shape (n_decoding_spikes, n_features)
        encoding_marks : ndarray, shape (n_encoding_spikes, n_features)
        mark_std : float or ndarray, shape (n_features,)
        occupancy : ndarray, shape (n_position_bins,)
        mean_rate : float
        place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
        encoding_positions : ndarray, shape (n_decoding_spikes, n_position_dims)
        position_std : float
        is_track_interior : None or ndarray, shape (n_position_bins,)
        max_mark_value : int
        set_diag_zero : bool
        Returns
        -------
        log_joint_mark_intensity : ndarray, shape (n_decoding_spikes, n_position_bins)
        """
        n_encoding_spikes, n_marks = encoding_marks.shape
        n_decoding_spikes = decoding_marks.shape[0]

        if sample_weights is None:
            sample_weights = cp.ones((1, n_decoding_spikes), dtype=cp.float32)
            denominator = n_encoding_spikes
        else:
            sample_weights = cp.atleast_2d(sample_weights)
            denominator = cp.sum(sample_weights)

        mark_distance = (
            cp.ones((n_decoding_spikes, n_encoding_spikes), dtype=cp.float32)
            * sample_weights
        )

        for mark_ind in range(n_marks):
            mark_distance *= normal_pdf_integer_lookup(
                cp.expand_dims(decoding_marks[:, mark_ind], axis=1),
                cp.expand_dims(encoding_marks[:, mark_ind], axis=0),
                std=mark_std,
                max_value=max_mark_value,
            )

        if set_diag_zero:
            diag_ind = cp.diag_indices_from(mark_distance)
            mark_distance[diag_ind] = 0.0

        if position_distance is None:
            position_distance = estimate_position_distance(
                decoding_position, encoding_position, position_std
            ).astype(cp.float32)

        return cp.asnumpy(
            estimate_log_intensity(
                cp.sum(mark_distance * position_distance.T, axis=1) / denominator,
                occupancy,
                mean_rate,
            )
        )

    def estimate_local_multiunit_likelihood(
        decoding_multiunit: cp.ndarray,
        decoding_position: cp.ndarray,
        encoding_marks: cp.ndarray,
        encoding_weights: cp.ndarray,
        mark_std: cp.ndarray,
        encoding_position: cp.ndarray,
        encoding_marks_position: cp.ndarray,
        position_std: cp.ndarray,
        mean_rates: cp.ndarray,
        max_mark_value=6000,
        set_diag_zero=False,
        time_bin_size=1,
        block_size=100,
        is_training=None,
        disable_progress_bar=False,
    ) -> cp.ndarray:
        """_summary_

        Parameters
        ----------
        decoding_multiunit : cp.ndarray
            _description_
        decoding_position : cp.ndarray
            _description_
        encoding_marks : cp.ndarray
            _description_
        encoding_weights : cp.ndarray
            _description_
        mark_std : cp.ndarray
            _description_
        encoding_position : cp.ndarray
            _description_
        encoding_marks_position : cp.ndarray
            _description_
        position_std : cp.ndarray
            _description_
        mean_rates : cp.ndarray
            _description_
        max_mark_value : int, optional
            _description_, by default 6000
        set_diag_zero : bool, optional
            _description_, by default False
        time_bin_size : int, optional
            _description_, by default 1
        block_size : int, optional
            _description_, by default 100
        is_training : _type_, optional
            _description_, by default None
        disable_progress_bar : bool, optional
            _description_, by default False

        Returns
        -------
        cp.ndarray
            _description_
        """

        n_time = decoding_multiunit.shape[0]
        log_likelihood = np.zeros((n_time,), dtype=np.float32)

        decoding_multiunits = np.moveaxis(decoding_multiunit, -1, 0)
        decoding_position_gpu = cp.asarray(decoding_position, dtype=cp.float32)

        local_occupancy = estimate_local_occupancy(
            train_position=cp.asarray(encoding_position, dtype=cp.float32),
            test_position=decoding_position_gpu,
            position_std=position_std,
            block_size=block_size,
            sample_weights=cp.asarray(is_training, dtype=cp.float32),
        )

        for decoding_multiunit, enc_marks, enc_pos, enc_weights, mean_rate in zip(
            tqdm(
                decoding_multiunits, desc="n_electrodes", disable=disable_progress_bar
            ),
            encoding_marks,
            encoding_marks_position,
            encoding_weights,
            mean_rates,
        ):

            is_decoding_spike = np.any(~np.isnan(decoding_multiunit), axis=1)
            is_decoding_spike_gpu = cp.asarray(is_decoding_spike)
            decoding_marks = cp.asarray(
                decoding_multiunit[is_decoding_spike], dtype=cp.int16
            )
            n_decoding_marks = decoding_marks.shape[0]
            enc_pos_gpu = cp.asarray(enc_pos, dtype=cp.float32)

            if block_size is None:
                block_size = n_decoding_marks

            log_likelihood -= time_bin_size * cp.asnumpy(
                estimate_local_gpi(
                    test_position=decoding_position_gpu,
                    enc_pos=enc_pos_gpu,
                    occupancy=local_occupancy,
                    mean_rate=mean_rate,
                    position_std=position_std,
                    block_size=block_size,
                    sample_weights=enc_weights,
                )
            )

            log_joint_mark_intensity = np.zeros((n_decoding_marks,), dtype=np.float32)

            for start_ind in range(0, n_decoding_marks, block_size):
                block_inds = slice(start_ind, start_ind + block_size)
                position_distance = estimate_position_distance(
                    decoding_position_gpu[is_decoding_spike_gpu][block_inds],
                    enc_pos_gpu,
                    position_std,
                ).astype(
                    cp.float32
                )  # n_encoding_spikes, n_decoding_spikes
                log_joint_mark_intensity[
                    block_inds
                ] = estimate_local_log_joint_mark_intensity(
                    decoding_marks[block_inds],
                    cp.asarray(enc_marks, cp.int16),
                    mark_std,
                    local_occupancy[is_decoding_spike_gpu][block_inds],
                    mean_rate,
                    max_mark_value=max_mark_value,
                    set_diag_zero=set_diag_zero,
                    position_distance=position_distance,
                    sample_weights=enc_weights,
                )
            log_likelihood[is_decoding_spike] += np.nan_to_num(log_joint_mark_intensity)

        return log_likelihood

    def multiunit_likelihood(
        multiunits,
        position,
        place_bin_centers,
        encoding_marks,
        mark_std,
        encoding_marks_position,
        encoding_weights,
        position_std,
        occupancy,
        summed_ground_process_intensity,
        encoding_position,
        mean_rates,
        is_track_interior,
        set_no_spike_to_equally_likely=False,
        is_training=None,
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

        Returns
        -------
        multiunit_likelihood : ndarray, shape (n_time, 2, n_place_bins)

        """
        n_time = multiunits.shape[0]
        n_place_bins = place_bin_centers.shape[0]
        multiunit_likelihood = np.zeros((n_time, 2, n_place_bins), dtype=np.float32)
        multiunit_likelihood[:, 1, :] = estimate_non_local_multiunit_likelihood(
            multiunits,
            encoding_marks,
            encoding_weights,
            mark_std,
            place_bin_centers,
            encoding_marks_position,
            position_std,
            occupancy,
            mean_rates,
            summed_ground_process_intensity,
            block_size=block_size,
            disable_progress_bar=disable_progress_bar,
        )
        if interpolate_local_likelihood:
            multiunit_likelihood[:, 0, :] = interpolate_local_likelihood(
                place_bin_centers, multiunit_likelihood[:, 1, :], position
            )
        else:
            multiunit_likelihood[:, 0, :] = estimate_local_multiunit_likelihood(
                multiunits,
                position,
                encoding_marks,
                encoding_weights,
                mark_std,
                encoding_position,
                encoding_marks_position,
                position_std,
                mean_rates,
                block_size=block_size,
                is_training=is_training,
                disable_progress_bar=disable_progress_bar,
            )[:, np.newaxis]

        if set_no_spike_to_equally_likely:
            no_spike = np.all(np.isnan(multiunits), axis=(1, 2))
            multiunit_likelihood[no_spike] = 0.0
        multiunit_likelihood[:, :, ~is_track_interior] = np.nan

        return scale_likelihood(multiunit_likelihood)

except ImportError:

    def multiunit_likelihood(*args, **kwargs):
        pass

    def fit_multiunit_likelihood_gpu(*args, **kwargs):
        pass
