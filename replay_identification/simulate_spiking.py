"""Tools for simulating place field spiking"""
import numpy as np
from replay_identification.bins import atleast_2d
from scipy.stats import multivariate_normal, norm


def simulate_time(n_samples, sampling_frequency):
    return np.arange(n_samples) / sampling_frequency


def simulate_linear_distance(time, track_height, running_speed=10):
    half_height = track_height / 2
    return (
        half_height * np.sin(2 * np.pi * time / running_speed - np.pi / 2) + half_height
    )


def simulate_linear_distance_with_pauses(
    time, track_height, running_speed=10, pause=0.5, sampling_frequency=1
):
    linear_distance = simulate_linear_distance(time, track_height, running_speed)
    peaks = np.nonzero(linear_distance == track_height)[0]
    n_pause_samples = int(pause * sampling_frequency)
    pause_linear_distance = np.zeros((time.size + n_pause_samples * peaks.size,))
    pause_ind = peaks[:, np.newaxis] + np.arange(n_pause_samples)
    pause_ind += np.arange(peaks.size)[:, np.newaxis] * n_pause_samples

    pause_linear_distance[pause_ind.ravel()] = track_height
    pause_linear_distance[pause_linear_distance == 0] = linear_distance
    return pause_linear_distance[: time.size]


def get_trajectory_direction(linear_distance):
    is_inbound = np.insert(np.diff(linear_distance) < 0, 0, False)
    return np.where(is_inbound, "Inbound", "Outbound"), is_inbound


def simulate_poisson_spikes(rate, sampling_frequency):
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)


def create_place_field(
    place_field_mean,
    linear_distance,
    sampling_frequency,
    is_condition=None,
    place_field_std_deviation=12.5,
    max_firing_rate=20,
    baseline_firing_rate=0.001,
):
    if is_condition is None:
        is_condition = np.ones_like(linear_distance, dtype=bool)
    field_firing_rate = norm(place_field_mean, place_field_std_deviation).pdf(
        linear_distance
    )
    field_firing_rate /= np.nanmax(field_firing_rate)
    field_firing_rate[~is_condition] = 0
    return baseline_firing_rate + max_firing_rate * field_firing_rate


def simulate_place_field_firing_rate(
    means, position, max_rate=15, variance=10, is_condition=None
):
    """Simulates the firing rate of a neuron with a place field at `means`.

    Parameters
    ----------
    means : ndarray, shape (n_position_dims,)
    position : ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional
    is_condition : None or ndarray, (n_time,)

    Returns
    -------
    firing_rate : ndarray, shape (n_time,)

    """
    if is_condition is None:
        is_condition = np.ones(position.shape[0], dtype=bool)
    position = atleast_2d(position)
    firing_rate = multivariate_normal(means, variance).pdf(position)
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    firing_rate[~is_condition] = 0.0

    return firing_rate


def simulate_neuron_with_place_field(
    means, position, max_rate=15, variance=36, sampling_frequency=500, is_condition=None
):
    """Simulates the spiking of a neuron with a place field at `means`.

    Parameters
    ----------
    means : ndarray, shape (n_position_dims,)
    position : ndarray, shape (n_time, n_position_dims)
    max_rate : float, optional
    variance : float, optional
    sampling_frequency : float, optional
    is_condition : None or ndarray, (n_time,)

    Returns
    -------
    spikes : ndarray, shape (n_time,)

    """
    firing_rate = simulate_place_field_firing_rate(
        means, position, max_rate, variance, is_condition
    )
    return simulate_poisson_spikes(firing_rate, sampling_frequency)
