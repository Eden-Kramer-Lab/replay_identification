from functools import partial

import numpy as np
from replay_identification.bins import atleast_2d
from scipy.interpolate import griddata
from tqdm.autonotebook import tqdm

from .core import scale_likelihood

SQRT_2PI = np.sqrt(2.0 * np.pi)


def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.'''
    return np.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)


def estimate_position_distance(place_bin_centers, positions, position_std):
    '''

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
    positions : ndarray, shape (n_time, n_position_dims)
    position_std : float

    Returns
    -------
    position_distance : ndarray, shape (n_time, n_position_bins)

    '''
    n_time, n_position_dims = positions.shape
    n_position_bins = place_bin_centers.shape[0]

    position_distance = np.ones((n_time, n_position_bins))

    for position_ind in range(n_position_dims):
        position_distance *= gaussian_pdf(
            np.expand_dims(place_bin_centers[:, position_ind], axis=0),
            np.expand_dims(positions[:, position_ind], axis=1),
            position_std)

    return position_distance


def estimate_position_density(place_bin_centers, positions, position_std,
                              block_size=None):
    '''

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_position_bins, n_position_dims)
    positions : ndarray, shape (n_time, n_position_dims)
    position_std : float

    Returns
    -------
    position_density : ndarray, shape (n_position_bins,)

    '''
    n_time = positions.shape[0]
    n_position_bins = place_bin_centers.shape[0]

    if block_size is None:
        block_size = n_time

    position_density = np.empty((n_position_bins,))
    for start_ind in range(0, n_position_bins, block_size):
        block_inds = slice(start_ind, start_ind + block_size)
        position_density[block_inds] = np.mean(estimate_position_distance(
            place_bin_centers[block_inds], positions, position_std), axis=0)
    return position_density


def estimate_log_intensity(density, occupancy, mean_rate):
    return np.log(mean_rate) + np.log(density) - np.log(occupancy)


def estimate_intensity(density, occupancy, mean_rate):
    '''

    Parameters
    ----------
    density : ndarray, shape (n_bins,)
    occupancy : ndarray, shape (n_bins,)
    mean_rate : float

    Returns
    -------
    intensity : ndarray, shape (n_bins,)

    '''
    return np.exp(estimate_log_intensity(density, occupancy, mean_rate))


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
    normal_density = gaussian_pdf(
        np.arange(-max_value, max_value), 0, std).astype(np.float32)

    return normal_density[(x - mean) + max_value]


def estimate_log_joint_mark_intensity(decoding_marks,
                                      encoding_marks,
                                      mark_std,
                                      occupancy,
                                      mean_rate,
                                      place_bin_centers=None,
                                      encoding_positions=None,
                                      position_std=None,
                                      max_mark_value=6000,
                                      set_diag_zero=False,
                                      position_distance=None):
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
    mark_distance = np.ones(
        (n_decoding_spikes, n_encoding_spikes), dtype=np.float32)

    for mark_ind in range(n_marks):
        mark_distance *= normal_pdf_integer_lookup(
            np.expand_dims(decoding_marks[:, mark_ind], axis=1),
            np.expand_dims(encoding_marks[:, mark_ind], axis=0),
            std=mark_std,
            max_value=max_mark_value
        )

    if set_diag_zero:
        diag_ind = np.diag_indices_from(mark_distance)
        mark_distance[diag_ind] = 0.0

    if position_distance is None:
        position_distance = estimate_position_distance(
            place_bin_centers, encoding_positions, position_std
        ).astype(np.float32)

    return estimate_log_intensity(
        mark_distance @ position_distance / n_encoding_spikes,
        occupancy,
        mean_rate)


def fit_multiunit_likelihood_integer(position,
                                     multiunits,
                                     is_training,
                                     place_bin_centers,
                                     mark_std,
                                     position_std,
                                     is_track_interior=None,
                                     **kwargs):
    '''

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

    '''
    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)
    position = atleast_2d(position)[is_training]
    multiunits = multiunits[is_training]
    place_bin_centers = atleast_2d(place_bin_centers)
    interior_place_bin_centers = np.asarray(
        place_bin_centers[is_track_interior], dtype=np.float32)

    not_nan_position = np.all(~np.isnan(position), axis=1)

    occupancy = np.zeros((place_bin_centers.shape[0],))
    occupancy[is_track_interior] = estimate_position_density(
        interior_place_bin_centers,
        np.asarray(position[not_nan_position], dtype=np.float32),
        position_std)

    mean_rates = []
    ground_process_intensities = []
    encoding_marks = []
    encoding_positions = []

    for multiunit in np.moveaxis(multiunits, -1, 0):

        # ground process intensity
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        mean_rates.append(is_spike.mean())
        marginal_density = np.zeros((place_bin_centers.shape[0],))

        if is_spike.sum() > 0:
            marginal_density[is_track_interior] = estimate_position_density(
                interior_place_bin_centers,
                np.asarray(
                    position[is_spike & not_nan_position], dtype=np.float32),
                position_std)

        ground_process_intensities.append(
            estimate_intensity(marginal_density, np.asarray(
                occupancy, dtype=np.float32), mean_rates[-1]))

        encoding_marks.append(
            np.asarray(multiunit[is_spike & not_nan_position], dtype=np.int16))
        encoding_positions.append(np.asarray(
            position[is_spike & not_nan_position], dtype=np.float32))

    summed_ground_process_intensity = np.sum(
        np.stack(ground_process_intensities, axis=0), axis=0, keepdims=True)
    summed_ground_process_intensity += np.spacing(1)

    return partial(
        multiunit_likelihood,
        place_bin_centers=place_bin_centers,
        encoding_marks=encoding_marks,
        encoding_marks_position=encoding_positions,
        encoding_position=position,
        summed_ground_process_intensity=summed_ground_process_intensity,
        occupancy=occupancy,
        mean_rates=mean_rates,
        mark_std=mark_std,
        position_std=position_std,
        is_track_interior=is_track_interior,
        **kwargs,
    )


def estimate_non_local_multiunit_likelihood(
        multiunits,
        encoding_marks,
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
        block_size=None):
    '''

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

    '''

    if is_track_interior is None:
        is_track_interior = np.ones((place_bin_centers.shape[0],),
                                    dtype=np.bool)

    n_time = multiunits.shape[0]
    log_likelihood = (-time_bin_size * summed_ground_process_intensity *
                      np.ones((n_time, 1), dtype=np.float32))

    multiunits = np.moveaxis(multiunits, -1, 0)
    n_position_bins = is_track_interior.sum()
    interior_place_bin_centers = np.asarray(
        place_bin_centers[is_track_interior], dtype=np.float32)
    interior_occupancy = occupancy[is_track_interior]

    for multiunit, enc_marks, enc_pos, mean_rate in zip(
            tqdm(multiunits, desc='n_electrodes'),
            encoding_marks, encoding_positions, mean_rates):
        is_spike = np.any(~np.isnan(multiunit), axis=1)
        decoding_marks = np.asarray(multiunit[is_spike], dtype=np.int16)
        n_decoding_marks = decoding_marks.shape[0]
        log_joint_mark_intensity = np.zeros(
            (n_decoding_marks, n_position_bins), dtype=np.float32)

        if block_size is None:
            block_size = n_decoding_marks

        position_distance = estimate_position_distance(
            interior_place_bin_centers,
            enc_pos,
            position_std
        ).astype(np.float32)

        for start_ind in range(0, n_decoding_marks, block_size):
            block_inds = slice(start_ind, start_ind + block_size)
            log_joint_mark_intensity[block_inds] = estimate_log_joint_mark_intensity(
                decoding_marks[block_inds],
                enc_marks,
                mark_std,
                interior_occupancy,
                mean_rate,
                max_mark_value=max_mark_value,
                set_diag_zero=set_diag_zero,
                position_distance=position_distance,
            )
        log_likelihood[np.ix_(is_spike, is_track_interior)] += (
            log_joint_mark_intensity + np.spacing(1))

    log_likelihood[:, ~is_track_interior] = np.nan

    return log_likelihood


def estimate_local_log_joint_mark_intensity(decoding_marks,
                                            encoding_marks,
                                            mark_std,
                                            occupancy,
                                            mean_rate,
                                            decoding_position=None,
                                            encoding_position=None,
                                            position_std=None,
                                            max_mark_value=6000,
                                            set_diag_zero=False,
                                            position_distance=None):
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
    mark_distance = np.ones(
        (n_decoding_spikes, n_encoding_spikes), dtype=np.float32)

    for mark_ind in range(n_marks):
        mark_distance *= normal_pdf_integer_lookup(
            np.expand_dims(decoding_marks[:, mark_ind], axis=1),
            np.expand_dims(encoding_marks[:, mark_ind], axis=0),
            std=mark_std,
            max_value=max_mark_value
        )

    if set_diag_zero:
        diag_ind = np.diag_indices_from(mark_distance)
        mark_distance[diag_ind] = 0.0

    if position_distance is None:
        position_distance = estimate_position_distance(
            decoding_position, encoding_position, position_std
        ).astype(np.float32)

    return estimate_log_intensity(
        np.sum(mark_distance * position_distance.T,
               axis=1) / n_encoding_spikes,
        occupancy,
        mean_rate)


def estimate_local_multiunit_likelihood(
        place_bin_centers, non_local_likelihood, position):

    return np.asarray(
        [griddata(place_bin_centers, likelihood, pos)
         for likelihood, pos in zip(non_local_likelihood, tqdm(position))])


def multiunit_likelihood(multiunit, position, place_bin_centers, encoding_marks,
                         mark_std,
                         encoding_marks_position, position_std, occupancy,
                         summed_ground_process_intensity, encoding_position,
                         mean_rates, is_track_interior, time_bin_size=1,
                         set_no_spike_to_equally_likely=True,
                         block_size=None):
    '''The likelihood of being in a replay state vs. not a replay state based
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

    '''
    n_time = multiunit.shape[0]
    n_place_bins = place_bin_centers.size
    multiunit_likelihood = np.zeros((n_time, 2, n_place_bins))
    multiunit_likelihood[:, 1, :] = estimate_non_local_multiunit_likelihood(
        multiunit,
        encoding_marks,
        mark_std,
        place_bin_centers,
        encoding_marks_position,
        position_std,
        occupancy,
        mean_rates,
        summed_ground_process_intensity,
        block_size=block_size)
    multiunit_likelihood[:, 0, :] = estimate_local_multiunit_likelihood(
        place_bin_centers,
        multiunit_likelihood[:, 1, :],
        position)[:, np.newaxis]

    if set_no_spike_to_equally_likely:
        no_spike = np.all(np.isnan(multiunit), axis=(1, 2))
        multiunit_likelihood[no_spike] = 0.0
    multiunit_likelihood[:, :, ~is_track_interior] = np.nan

    return scale_likelihood(multiunit_likelihood)
