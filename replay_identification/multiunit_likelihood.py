'''
'''

from functools import partial

import numpy as np
from tqdm.auto import tqdm

from .core import atleast_2d


def multiunit_likelihood(multiunit, position, place_bin_centers,
                         occupancy_model, joint_models, marginal_models,
                         mean_rates, time_bin_size=1):
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
    time_bin_size : float, optional

    Returns
    -------
    multiunit_likelihood : ndarray, shape (n_time, 2, n_place_bins)

    '''
    n_time = multiunit.shape[0]
    n_place_bins = place_bin_centers.size
    multiunit_likelihood = np.zeros((n_time, 2, n_place_bins))
    multiunit_likelihood[:, 1, :] = np.exp(estimate_replay_log_likelihood(
        np.moveaxis(multiunit, -1, 0), place_bin_centers,
        occupancy_model, joint_models, marginal_models, mean_rates,
        time_bin_size))
    multiunit_likelihood[:, 0, :] = np.exp(estimate_no_replay_log_likelihood(
        np.moveaxis(multiunit, -1, 0), position, occupancy_model,
        joint_models, marginal_models, mean_rates, time_bin_size))

    return multiunit_likelihood


def estimate_replay_log_likelihood(
        multiunit, place_bin_centers, occupancy_model,
        joint_models, marginal_models, mean_rates, time_bin_size):
    '''Estimate the log likelihood of being at any position.

    Parameters
    ----------
    multiunit : ndarray, shape (n_electrodes, n_time, n_features)
    place_bin_centers : ndarray, shape (n_place_bins,)
    occupancy_model : fitted density model
    marginal_models : list of fitted density models, len (n_electrodes,)
    mean_rates : list of floats, shape (n_electrodes,)
    time_bin_size : float

    Returns
    -------
    replay_log_likelihood : ndarray, shape (n_time, n_place_bins)

    '''

    n_bin = place_bin_centers.size
    n_time = multiunit.shape[1]
    log_likelihood = np.zeros((n_time, n_bin))

    occupancy = estimate_occupancy(place_bin_centers, occupancy_model)

    for m, joint_model, marginal_model, mean_rate in zip(
            tqdm(multiunit, desc='electrodes'), joint_models, marginal_models,
            mean_rates):
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
    '''Estimate the log likelihood of being at the current position.

    Parameters
    ----------
    multiunit : ndarray, shape (n_electrodes, n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    occupancy_model : fitted density model
    joint_models : list of fitted density models, len (n_electrodes,)
    marginal_models : list of fitted density models, len (n_electrodes,)
    mean_rates : list of floats, len (n_electrodes,)
    time_bin_size : float

    Returns
    -------
    no_replay_log_likelihood : ndarray, shape (n_time,)

    '''
    n_time = multiunit.shape[1]
    log_likelihood = np.zeros((n_time, 1))

    occupancy = estimate_occupancy(position, occupancy_model)

    for m, joint_model, marginal_model, mean_rate in zip(
            tqdm(multiunit, desc='electrodes'), joint_models, marginal_models,
            mean_rates):
        ground_process_intensity = estimate_ground_process_intensity(
            position, occupancy, marginal_model, mean_rate)[:, np.newaxis]
        joint_mark_intensity = estimate_joint_mark_intensity(
            m, position, joint_model, mean_rate, occupancy)[:, np.newaxis]
        log_likelihood += poisson_mark_log_likelihood(
            joint_mark_intensity, ground_process_intensity,
            time_bin_size)

    return log_likelihood


def poisson_mark_log_likelihood(log_joint_mark_intensity,
                                ground_process_intensity, time_bin_size=1):
    '''Probability of parameters given spiking indicator at a particular
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
    joint_mark_intensity += np.spacing(1)
    ground_process_intensity += np.spacing(1)
    return np.log(joint_mark_intensity) - (
    '''
        ground_process_intensity * time_bin_size)


def estimate_occupancy(position, occupancy_model):
    '''Computes the spatial occupancy.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    occupancy_model : fitted density model

    Returns
    -------
    occupancy : ndarray, shape (n_time,)

    '''
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
                                      marginal_model, mean_rate):
    '''Computes the rate function of position marginalized over mark.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    occupancy : ndarray, shape (n_position_dims,)
    marginal_model : fitted density model
    mean_rate : float

def estimate_joint_mark_intensity(multiunit, position, joint_model, mean_rate,
                                  place_occupancy):
    multiunit = atleast_2d(multiunit)
    position = atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    Returns
    -------
    ground_process_intensity : ndarray, shape (n_time,)

    '''
    '''Computes the rate function of position and mark.

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

    '''
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
    '''

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_marginal_model : density_model class instance

    '''
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan_position = ~np.isnan(position)
    return (density_model(**model_kwargs)
            .fit(atleast_2d(position)[is_spike & not_nan_position]))


def train_occupancy_model(position, density_model, model_kwargs):
    '''Fits a density model for computing the spatial occupancy.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_occupancy_model : density_model class instance

    '''
    position = atleast_2d(position)
    not_nan_position = ~np.isnan(np.squeeze(position))
    return density_model(**model_kwargs).fit(position[not_nan_position])


def train_joint_model(multiunit, position, density_model, model_kwargs):
    multiunit = atleast_2d(multiunit)
    position = atleast_2d(position)
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    '''Fits a density model to the joint pdf of position and mark.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)
    density_model : class
    model_kwargs : dict

    Returns
    -------
    fitted_joint_model : density_model class instance

    '''
    not_nan_marks = np.any(~np.isnan(multiunit), axis=0)
    not_nan_position = ~np.isnan(np.squeeze(position))

    joint_data = np.concatenate(
        (multiunit[is_spike & not_nan_position][:, not_nan_marks],
         position[is_spike & not_nan_position]), axis=1)
    return density_model(**model_kwargs).fit(joint_data)


def estimate_mean_rate(multiunit, position):
    '''Mean rate of multiunit.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_features)
    position : ndarray, shape (n_time, n_position_dims)

    Returns
    -------
    mean_rate : float

    '''
    is_spike = np.any(~np.isnan(multiunit), axis=1)
    not_nan = ~np.isnan(position)
    return np.mean(is_spike[not_nan])


def fit_multiunit_likelihood(position, multiunit, is_replay,
                             place_bin_centers,
                             density_model, model_kwargs):
    '''Precompute quantities to fit the multiunit likelihood to new data.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    multiunit : ndarray, shape (n_time, n_features, n_electrodes)
    is_replay : bool ndarray, shape (n_time,)
    place_bin_centers : ndarray, shape (n_place_bins,)
    model : Class
    model_kwargs : dict

    Returns
    -------
    multiunit_likelihood : function

    '''
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
        multiunit_likelihood,
        place_bin_centers=place_bin_centers,
        occupancy_model=occupancy_model,
        joint_models=joint_models,
        marginal_models=marginal_models,
        mean_rates=mean_rates
    )
