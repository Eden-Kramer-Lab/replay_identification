'''Tools for simulating place field spiking'''
import numpy as np
from scipy.stats import norm, multivariate_normal


def simulate_time(n_samples, sampling_frequency):
    return np.arange(n_samples) / sampling_frequency


def simulate_linear_distance(time, track_height, running_speed=10):
    half_height = (track_height / 2)
    return (half_height * np.sin(2 * np.pi * time / running_speed - np.pi / 2)
            + half_height)


def simulate_linear_distance_with_pauses(time, track_height, running_speed=10,
                                         pause=0.5, sampling_frequency=1):
    linear_distance = simulate_linear_distance(
        time, track_height, running_speed)
    peaks = np.nonzero(linear_distance == track_height)[0]
    n_pause_samples = int(pause * sampling_frequency)
    pause_linear_distance = np.zeros(
        (time.size + n_pause_samples * peaks.size,))
    pause_ind = (peaks[:, np.newaxis] + np.arange(n_pause_samples))
    pause_ind += np.arange(peaks.size)[:, np.newaxis] * n_pause_samples

    pause_linear_distance[pause_ind.ravel()] = track_height
    pause_linear_distance[pause_linear_distance == 0] = linear_distance
    return pause_linear_distance[:time.size]


def get_trajectory_direction(linear_distance):
    is_inbound = np.insert(np.diff(linear_distance) < 0, 0, False)
    return np.where(is_inbound, 'Inbound', 'Outbound'), is_inbound


def simulate_poisson_spikes(rate, sampling_frequency):
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)


def create_place_field(
    place_field_mean, linear_distance, sampling_frequency, is_condition=None,
        place_field_std_deviation=12.5, max_firing_rate=20,
        baseline_firing_rate=0.5):
    if is_condition is None:
        is_condition = np.ones_like(linear_distance, dtype=bool)
    field_firing_rate = norm(
        place_field_mean, place_field_std_deviation).pdf(linear_distance)
    field_firing_rate /= field_firing_rate.max()
    field_firing_rate[~is_condition] = 0
    return baseline_firing_rate + max_firing_rate * field_firing_rate


def generate_marks(spikes, mark_mean, mark_std_deviation, n_marks=4):
    '''Generate a place field with an associated mark'''
    spikes[spikes == 0] = np.nan
    marks = multivariate_normal(
        mean=[mark_mean] * n_marks,
        cov=[mark_std_deviation] * n_marks).rvs(size=(spikes.size,))
    return marks * spikes[:, np.newaxis]


def simulate_multiunit(
    place_field_means, mark_means, linear_distance, sampling_frequency,
        mark_std_deviation=20, n_marks=4, **kwargs):
    '''Simulate a single tetrode assuming each tetrode picks up several
    neurons with different place fields with distinguishing marks.'''
    unit = []
    for place_field_mean, mark_mean in zip(place_field_means, mark_means):
        rate = create_place_field(
            place_field_mean, linear_distance, sampling_frequency, **kwargs)
        spikes = simulate_poisson_spikes(rate, sampling_frequency)
        marks = generate_marks(
            spikes, mark_mean, mark_std_deviation, n_marks=n_marks)
        unit.append(marks)

    return np.nanmean(np.stack(unit, axis=0), axis=0)
