import numpy as np
from scipy.stats import multivariate_normal, norm


def simulate_poisson_spikes(rate, sampling_frequency):
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)


def create_place_field(
    place_field_mean, linear_distance, sampling_frequency, is_condition=None,
        place_field_std_deviation=12.5, max_firing_rate=20,
        baseline_firing_rate=0.0):
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
