"""
"""

from functools import partial

import numpy as np

from .core import combined_likelihood


def multiunit_likelihood_ratio(marks, position, place_bin_centers,
                               joint_mark_intensity_functions=None,
                               ground_process_intensity=None, time_bin_size=1):

    position = np.atleast_2d(np.squeeze(position))

    no_replay_log_likelihood = poisson_mark_log_likelihood(
        marks, position, joint_mark_intensity_functions,
        ground_process_intensity, time_bin_size)
    replay_log_likelihood = poisson_mark_log_likelihood(
        marks, place_bin_centers,
        joint_mark_intensity_functions, ground_process_intensity,
        time_bin_size)
    return np.exp(replay_log_likelihood - no_replay_log_likelihood)


@combined_likelihood
def poisson_mark_log_likelihood(marks, position,
                                joint_mark_intensity_functions=None,
                                ground_process_intensity=None,
                                time_bin_size=1):
    '''Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    marks : array, shape (n_signals, n_marks)
    joint_mark_intensity : function
        Instantaneous probability of observing a spike given mark vector
        from data. The parameters for this function should already be set,
        before it is passed to `poisson_mark_log_likelihood`.
    ground_process_intensity : array, shape (n_signals, n_states,
                                             n_place_bins)
        Probability of observing a spike regardless of marks.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_log_likelihood : array_like, shape (n_signals, n_states,
                                                     n_time, n_place_bins)

    '''
    marks = np.moveaxis(marks, -1, 0)
    probability_no_spike = -ground_process_intensity * time_bin_size
    joint_mark_intensity = np.stack(
        [jmi(signal_marks, position) for signal_marks, jmi
         in zip(marks, joint_mark_intensity_functions)], axis=-1)
    return np.log(joint_mark_intensity) + probability_no_spike


def joint_mark_intensity(
        marks, place_bin_centers, place_occupancy, fitted_model, mean_rate):
    marks = np.atleast_2d(marks)
    n_place_bins = place_bin_centers.shape[0]
    n_time = marks.shape[0]
    is_nan = np.any(np.isnan(marks), axis=1)
    n_spikes = np.sum(~is_nan)
    density = np.zeros((n_time, n_place_bins))

    if n_spikes > 0:
        for bin_ind, bin in enumerate(place_bin_centers):
            bin = atleast_2d(bin * np.ones((n_time,)))
            not_nan = ~is_nan & ~np.isnan(bin.squeeze())
            predict_data = np.concatenate(
                (marks[not_nan], bin[not_nan]), axis=1)
            density[not_nan, bin_ind] = np.exp(
                fitted_model.score_samples(predict_data))

    joint_mark_intensity = mean_rate * density / place_occupancy
    joint_mark_intensity[is_nan] = 1.0
    return joint_mark_intensity


def estimate_place_occupancy(position, place_bin_centers, model, model_kwargs):
    return np.exp(model(**model_kwargs).fit(position)
                  .score_samples(place_bin_centers[:, np.newaxis]))


def estimate_ground_process_intensity(
        position, marks, place_bin_centers, model, model_kwargs):
    is_spike = np.any(~np.isnan(marks), axis=1)
    not_nan = ~np.isnan(position)
    position = atleast_2d(position)
    place_field = np.exp(model(**model_kwargs)
                         .fit(position[is_spike & not_nan])
                         .score_samples(place_bin_centers[:, np.newaxis]))
    place_occupancy = estimate_place_occupancy(
        position[not_nan], place_bin_centers, model, model_kwargs)
    mean_rate = np.mean(is_spike)
    return np.atleast_2d(mean_rate * place_field / place_occupancy)


def build_joint_mark_intensity(
        position, training_marks, place_bin_centers, model, model_kwargs):
    training_marks = atleast_2d(training_marks)[~np.isnan(position)]
    position = atleast_2d(position)[~np.isnan(position)]

    is_spike = np.any(~np.isnan(training_marks), axis=1)
    mean_rate = np.mean(is_spike, dtype=np.float)

    training_data = np.concatenate(
        (training_marks[is_spike], position[is_spike]), axis=1)
    fitted_model = model(**model_kwargs).fit(training_data)
    place_occupancy = estimate_place_occupancy(
        position, place_bin_centers, model, model_kwargs)

    return partial(joint_mark_intensity,
                   place_occupancy=place_occupancy,
                   fitted_model=fitted_model,
                   mean_rate=mean_rate)


def fit_multiunit_likelihood_ratio(position, spike_marks, is_replay,
                                   place_bin_centers, model, model_kwargs):
    joint_mark_intensity_functions = []
    ground_process_intensity = []

    position = position[~is_replay]

    for marks in np.moveaxis(spike_marks[~is_replay], -1, 0):
        joint_mark_intensity_functions.append(
            build_joint_mark_intensity(
                position, marks, place_bin_centers, model, model_kwargs))
        ground_process_intensity.append(
            estimate_ground_process_intensity(
                position, marks, place_bin_centers, model, model_kwargs))

    ground_process_intensity = np.concatenate(
        ground_process_intensity, axis=0).T

    return partial(
        multiunit_likelihood_ratio,
        place_bin_centers=place_bin_centers,
        joint_mark_intensity_functions=joint_mark_intensity_functions,
        ground_process_intensity=ground_process_intensity,
        )


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x
