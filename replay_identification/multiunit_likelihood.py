"""
"""

from functools import partial

import numpy as np

from .core import atleast_2d

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def multiunit_likelihood_ratio(multiunit, position, place_bin_centers,
                               place_bins, joint_mark_intensity_functions,
                               ground_process_intensity, time_bin_size=1):
    """The ratio of being in a replay state vs. not a replay state based on
    whether the multiunits correspond to the current position of the animal.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_marks, n_signals)
    position : ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    joint_mark_intensity_functions : list of functions
    ground_process_intensity : ndarray, shape (n_place_bins, n_signals)
    time_bin_size : float, optional

    Returns
    -------
    multiunit_likelihood_ratio : ndarray, shape (n_time, n_place_bins)

    """
    position = position
    n_time = position.shape[0]

    replay_log_likelihood = combined_likelihood(
        multiunit, place_bin_centers,
        joint_mark_intensity_functions, ground_process_intensity,
        time_bin_size)
    bin_ind = np.digitize(position, place_bins)
    is_nan = np.isnan(np.squeeze(position))
    bin_ind[is_nan] = 0
    no_replay_log_likelihood = replay_log_likelihood[
        (np.arange(n_time), bin_ind - 1)][:, np.newaxis]
    no_replay_log_likelihood[is_nan] = np.nan

    return np.exp(replay_log_likelihood - no_replay_log_likelihood)


def combined_likelihood(multiunit, position, joint_mark_intensity_functions,
                        ground_process_intensity, time_bin_size):
    """Combined poisson mark likelihood over all signals.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_marks, n_signals)
    position : ndarray, shape (n_bins, ...)
    joint_mark_intensity_functions : list of functions
    ground_process_intensity : ndarray, shape (n_place_bins, n_signals)
    time_bin_size : float

    Returns
    -------
    combined_likelihood : ndarray, shape (n_time, n_place_bins)

    """
    n_bin = ground_process_intensity.shape[1]
    n_time = multiunit.shape[0]
    log_likelihood = np.zeros((n_time, n_bin))
    multiunit = np.moveaxis(multiunit, -1, 0)

    for signal_marks, jmi, gpi in zip(
            tqdm(multiunit), joint_mark_intensity_functions,
            ground_process_intensity):
        log_likelihood += poisson_mark_log_likelihood(
            jmi(signal_marks, position), gpi,
            time_bin_size)

    return log_likelihood


def poisson_mark_log_likelihood(joint_mark_intensity,
                                ground_process_intensity,
                                time_bin_size=1):
    """Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    multiunit : array, shape (n_signals, n_marks)
    joint_mark_intensity : function
        Instantaneous probability of observing a spike given mark vector
        from data. The parameters for this function should already be set,
        before it is passed to `poisson_mark_log_likelihood`.
    ground_process_intensity : array, shape (n_signals, n_states,
                                             n_place_bins)
        Probability of observing a spike regardless of multiunit.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_log_likelihood : array_like, shape (n_signals, n_states,
                                                     n_time, n_place_bins)

    """
    return np.log(joint_mark_intensity) - (
        ground_process_intensity * time_bin_size)


def joint_mark_intensity(
        multiunit, place_bin_centers, place_occupancy, fitted_model,
        mean_rate):
    """Evaluate the multivariate density function of the marks and place.

    Parameters
    ----------
    multiunit : array, shape (n_time, n_marks)
    place_bin_centers : ndarray, shape (n_place_bins,)
    place_occupancy : ndarray, shape (n_place_bins,)
    fitted_model : a fitted instance of a Class with fit, score_samples, and
                   sample methods.
    mean_rate : float

    Returns
    -------
    joint_mark_intensity : ndarray, shape (n_time, n_place_bins)

    """
    multiunit = np.atleast_2d(multiunit)
    n_place_bins = place_bin_centers.shape[0]
    n_time = multiunit.shape[0]
    is_spike = np.all(~np.isnan(multiunit), axis=1)
    n_spikes = np.sum(is_spike)
    joint_mark_intensity = np.ones((n_time, n_place_bins))

    for ind, position in enumerate(place_bin_centers):
        predict_data = np.concatenate(
            (multiunit[is_spike], position * np.ones((n_spikes, 1))), axis=1)
        joint_mark_intensity[is_spike, ind] = mean_rate * np.exp(
            fitted_model.score_samples(predict_data)) / place_occupancy[ind]

    return joint_mark_intensity


def estimate_place_occupancy(position, place_bin_centers, model, model_kwargs):
    """Probability of being at a position.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict

    Returns
    -------
    place_occupancy : ndarray, shape (n_place_bins,)

    """
    return np.exp(model(**model_kwargs)
                  .fit(atleast_2d(position[~np.isnan(position)]))
                  .score_samples(place_bin_centers[:, np.newaxis]))


def estimate_ground_process_intensity(
        position, multiunit, place_bin_centers, model, model_kwargs,
        place_occupancy=None):
    """The marginal rate function.

    The probability of observing a spike regardless of mark.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_marks)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict

    Returns
    -------
    ground_process_intensity : shape (1, n_place_bins)

    """
    if place_occupancy is None:
        place_occupancy = estimate_place_occupancy(
            position, place_bin_centers, model, model_kwargs)

    is_spike = np.all(~np.isnan(multiunit), axis=1)
    not_nan = ~np.isnan(position)
    place_field = np.exp(model(**model_kwargs)
                         .fit(atleast_2d(position)[is_spike & not_nan])
                         .score_samples(place_bin_centers[:, np.newaxis]))
    mean_rate = np.mean(is_spike)
    return np.atleast_2d(mean_rate * place_field / place_occupancy)


def build_joint_mark_intensity(
        position, multiunit, place_bin_centers, model, model_kwargs,
        place_occupancy=None):
    """Make a joint mark intensity function with precalculated quauntities
    set.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_marks)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict

    Returns
    -------
    joint_mark_intensity : function

    """
    if place_occupancy is None:
        place_occupancy = estimate_place_occupancy(
            position, place_bin_centers, model, model_kwargs)

    multiunit = atleast_2d(multiunit)
    position = atleast_2d(position)

    is_spike = np.all(~np.isnan(multiunit), axis=1)
    not_nan = ~np.isnan(np.squeeze(position))
    mean_rate = np.mean(is_spike, dtype=np.float)

    training_data = np.concatenate(
        (multiunit[is_spike & not_nan], position[is_spike & not_nan]), axis=1)
    fitted_model = model(**model_kwargs).fit(training_data)

    return partial(joint_mark_intensity,
                   place_occupancy=place_occupancy,
                   fitted_model=fitted_model,
                   mean_rate=mean_rate)


def fit_multiunit_likelihood_ratio(position, multiunit, is_replay,
                                   place_bin_centers, place_bins,
                                   model, model_kwargs):
    """Precompute quantities to fit the multiunit likelihood ratio to new data.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_marks, n_signals)
    is_replay : bool ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict

    Returns
    -------
    multiunit_likelihood_ratio : function

    """
    joint_mark_intensity_functions = []
    ground_process_intensity = []
    place_occupancy = estimate_place_occupancy(
        position[~is_replay], place_bin_centers, model, model_kwargs)

    for m in tqdm(np.moveaxis(multiunit[~is_replay], -1, 0),
                  desc='electrodes'):
        joint_mark_intensity_functions.append(
            build_joint_mark_intensity(
                position[~is_replay], m, place_bin_centers, model,
                model_kwargs, place_occupancy=place_occupancy))
        ground_process_intensity.append(
            estimate_ground_process_intensity(
                position[~is_replay], m, place_bin_centers, model,
                model_kwargs, place_occupancy=place_occupancy))

    ground_process_intensity = np.concatenate(
        ground_process_intensity, axis=0)

    return partial(
        multiunit_likelihood_ratio,
        place_bin_centers=place_bin_centers,
        place_bins=place_bins,
        joint_mark_intensity_functions=joint_mark_intensity_functions,
        ground_process_intensity=ground_process_intensity,
    )
