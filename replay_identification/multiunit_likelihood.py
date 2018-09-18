"""
"""

from functools import partial

import numpy as np

from .core import atleast_2d

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def multiunit_likelihood_ratio(multiunit, position, place_bin_centers,
                               occupancy_model, joint_models, marginal_models,
                               mean_rates, time_bin_size=1):
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
    replay_log_likelihood = estimate_replay_log_likelihood(
        np.moveaxis(multiunit, -1, 0), place_bin_centers,
        occupancy_model, joint_models, marginal_models, mean_rates,
        time_bin_size)
    no_replay_log_likelihood = estimate_no_replay_log_likelihood(
        np.moveaxis(multiunit, -1, 0), position, occupancy_model,
        joint_models, marginal_models, mean_rates, time_bin_size)

    return np.exp(replay_log_likelihood - no_replay_log_likelihood)


def estimate_replay_log_likelihood(
        multiunit, place_bin_centers, occupancy_model,
        joint_models, marginal_models, mean_rates, time_bin_size):

    n_bin = place_bin_centers.size
    n_time = multiunit.shape[1]
    log_likelihood = np.zeros((n_time, n_bin))

    occupancy = estimate_occupancy(place_bin_centers, occupancy_model)

    for m, joint_model, marginal_model, mean_rate in zip(
            tqdm(multiunit), joint_models, marginal_models, mean_rates):
        ground_process_intensity = np.atleast_2d(
            estimate_ground_process_intensity(
                place_bin_centers, occupancy, marginal_model, mean_rate))
        joint_mark_intensity = []
        for occ, place_bin in zip(occupancy, place_bin_centers):
            position = place_bin * np.ones((n_time, 1))
            joint_mark_intensity.append(estimate_joint_mark_intensity(
                m, position, joint_model, mean_rate, occ))
        joint_mark_intensity = np.stack(joint_mark_intensity, axis=1)
        log_likelihood += poisson_mark_log_likelihood(
            joint_mark_intensity, ground_process_intensity,
            time_bin_size)

    return log_likelihood


def estimate_no_replay_log_likelihood(
        multiunit, position, occupancy_model,
        joint_models, marginal_models, mean_rates, time_bin_size):
    n_time = multiunit.shape[1]
    log_likelihood = np.zeros((n_time, 1))

    occupancy = estimate_occupancy(position, occupancy_model)

    for m, joint_model, marginal_model, mean_rate in zip(
            tqdm(multiunit), joint_models, marginal_models, mean_rates):
        ground_process_intensity = estimate_ground_process_intensity(
            position, occupancy, marginal_model, mean_rate)[:, np.newaxis]
        joint_mark_intensity = estimate_joint_mark_intensity(
            m, position, joint_model, mean_rate, occupancy)[:, np.newaxis]
        log_likelihood += poisson_mark_log_likelihood(
            joint_mark_intensity, ground_process_intensity,
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
    joint_mark_intensity += np.spacing(1)
    ground_process_intensity += np.spacing(1)
    return np.log(joint_mark_intensity) - (
        ground_process_intensity * time_bin_size)


def estimate_occupancy(position, occupancy_model):
    position = atleast_2d(position)
    not_nan_position = ~np.isnan(np.squeeze(position))
    occupancy_probability = np.full((position.shape[0],), np.nan)
    occupancy_probability[not_nan_position] = np.exp(
        occupancy_model.score_samples(position[not_nan_position]))
    return occupancy_probability


def estimate_ground_process_intensity(position, place_occupancy,
                                      place_field_model, mean_rate):
    place_field = np.exp(place_field_model
                         .score_samples(atleast_2d(position)))
    return mean_rate * place_field / place_occupancy


def estimate_joint_mark_intensity(multiunit, position, joint_model, mean_rate,
                                  place_occupancy):
    multiunit = atleast_2d(multiunit)
    position = atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)
    not_nan_position = ~np.isnan(np.squeeze(position))
    n_time = position.shape[0]
    joint_mark_intensity = np.ones((n_time,))

    joint_data = np.concatenate(
        (multiunit[is_spike & not_nan_position][:, not_nan_marks],
         position[is_spike & not_nan_position]), axis=1)
    joint_mark_intensity[is_spike & not_nan_position] = np.exp(
        joint_model.score_samples(joint_data))
    joint_mark_intensity /= place_occupancy
    joint_mark_intensity *= mean_rate
    joint_mark_intensity[~is_spike | ~not_nan_position] = 1.0
    return joint_mark_intensity


def train_marginal_model(multiunit, position, density_model, model_kwargs):
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan_position = ~np.isnan(position)
    return (density_model(**model_kwargs)
            .fit(atleast_2d(position)[is_spike & not_nan_position]))


def train_occupancy_model(position, density_model, model_kwargs):
    position = atleast_2d(position)
    not_nan_position = ~np.isnan(np.squeeze(position))
    return density_model(**model_kwargs).fit(position[not_nan_position])


def train_joint_model(multiunit, position, density_model, model_kwargs):
    multiunit = atleast_2d(multiunit)
    position = atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)
    not_nan_position = ~np.isnan(np.squeeze(position))

    joint_data = np.concatenate(
        (multiunit[is_spike & not_nan_position][:, not_nan_marks],
         position[is_spike & not_nan_position]), axis=1)
    return density_model(**model_kwargs).fit(joint_data)


def estimate_mean_rate(multiunit, position):
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan = ~np.isnan(position)
    return np.mean(is_spike[not_nan])


def fit_multiunit_likelihood_ratio(position, multiunit, is_replay,
                                   place_bin_centers,
                                   density_model, model_kwargs):
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
    joint_models = []
    marginal_models = []
    mean_rates = []
    occupancy_model = train_occupancy_model(
        position[~is_replay], density_model, model_kwargs)

    for m in tqdm(np.moveaxis(multiunit[~is_replay], -1, 0),
                  desc='electrodes'):
        mean_rates.append(estimate_mean_rate(m, position[~is_replay]))
        joint_models.append(
            train_joint_model(m, position[~is_replay], density_model,
                              model_kwargs))
        marginal_models.append(
            train_marginal_model(m, position[~is_replay], density_model,
                                 model_kwargs))

    return partial(
        multiunit_likelihood_ratio,
        place_bin_centers=place_bin_centers,
        occupancy_model=occupancy_model,
        joint_models=joint_models,
        marginal_models=marginal_models,
        mean_rates=mean_rates
    )
