import numpy as np

from spectral_connectivity import Connectivity, Multitaper

from .simulate_LFP import simulate_LFP
from .simulate_multiunit import simulate_multiunit
from .simulate_spiking import (create_place_field,
                               simulate_linear_distance_with_pauses,
                               simulate_poisson_spikes, simulate_time)


def estimate_ripple_band_power(lfps, sampling_frequency):
    """Estimates the 200 Hz power of each LFP.

    Parameters
    ----------
    lfps : ndarray, shape (n_time, n_signals)
    sampling_frequency : float

    Returns
    -------
    ripple_band_power : ndarray (n_time, n_signals)

    """
    n_time = lfps.shape[0]
    m = Multitaper(lfps, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=1 / sampling_frequency)
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze()
    n_power_time = power.shape[0]
    unobserved = np.full((n_time - n_power_time, *power.shape[1:]), np.nan)
    return np.concatenate((power, unobserved))


def simulate_dataset(track_height=170, sampling_frequency=1000,
                     time_duration=65, pause_duration=3.0,
                     ripple_duration=0.100,
                     place_field_means=np.arange(0, 200, 25)):
    time = simulate_time(
        sampling_frequency * time_duration, sampling_frequency)
    linear_distance = simulate_linear_distance_with_pauses(
        time, track_height, sampling_frequency=sampling_frequency,
        pause=pause_duration)
    linear_distance += np.random.randn(*time.shape) * 1E-4

    speed = np.abs(np.diff(linear_distance) / np.diff(time))
    speed = np.insert(speed, 0, 0.0)

    pause_ind = np.nonzero(
        np.diff(np.isclose(linear_distance, track_height, atol=1E-5)))[0] + 1

    pause_times = time[np.reshape(pause_ind, (-1, 2))]
    pause_width = np.diff(pause_times)[0]
    lfps = np.stack((
        simulate_LFP(
            time, pause_times[:3, 0] + pause_width / 2, noise_amplitude=1.2,
            ripple_amplitude=1, ripple_width=ripple_duration),
        simulate_LFP(
            time, pause_times[:2, 0] + pause_width / 2, noise_amplitude=1.2,
            ripple_amplitude=1.1, ripple_width=ripple_duration)), axis=1
    )
    power = estimate_ripple_band_power(lfps, sampling_frequency)
    mid_ripple_time = pause_times[:3, 0] + pause_width / 2
    ripple_times = (mid_ripple_time +
                    np.array([-0.5, 0.5])[:, np.newaxis] * ripple_duration).T

    place_fields = np.stack([create_place_field(
        place_field_mean, linear_distance, sampling_frequency)
        for place_field_mean in place_field_means])

    spikes = simulate_poisson_spikes(place_fields, sampling_frequency).T

    # Add replay
    is_toward = np.array([0, 1], dtype=bool)
    n_neurons = spikes.shape[1]
    n_samples_between_spikes = 20

    for (start_time, end_time), flip in zip(ripple_times[[0, -1]], is_toward):
        is_ripple_time = (time >= start_time) & (time <= end_time)
        ripple_ind = np.nonzero(is_ripple_time)[0]
        spikes[is_ripple_time] = 0
        neuron_order = np.flip(np.arange(n_neurons),
                               axis=0) if flip else np.arange(n_neurons)
        replay_ind = (
            ripple_ind[0] +
            np.arange(0, n_neurons * n_samples_between_spikes,
                      n_samples_between_spikes),
            neuron_order)
        spikes[replay_ind] = 1

    # Add no spikes condition
    no_spike_time = pause_times[3, 0] + pause_duration / 2
    start_time, end_time = (no_spike_time + np.array([-0.5, 0.5]))
    is_no_spike_time = (time >= start_time) & (time <= end_time)
    spikes[is_no_spike_time, :] = 0.0

    mark_means = np.array([200, 125, 325, 275])
    place_field_means = np.stack((np.arange(0, 200, 50),
                                  np.arange(25, 200, 50)))

    multiunit = np.stack(
        [simulate_multiunit(place_field_means[0], mark_means, linear_distance,
                            sampling_frequency),
         simulate_multiunit(
             place_field_means[1], mark_means, linear_distance,
            sampling_frequency),
         ], axis=-1)

    # Add replay
    n_units = multiunit.shape[1]

    n_samples_between_spikes = 40
    is_toward = np.array([0, 1], dtype=bool)

    tetrode_ind = np.meshgrid(np.arange(multiunit.shape[1]), np.arange(
        multiunit.shape[2]))[1].flatten(order='F')
    mean_order = np.stack((mark_means, mark_means)).flatten(order='F')

    for start_time, end_time in ripple_times[[1]]:
        is_ripple_time = (time >= start_time) & (time <= end_time)
        clear_ind = (time >= start_time - 0.200) & (time <= end_time + 0.200)
        ripple_ind = np.nonzero(is_ripple_time)[0]
        multiunit[clear_ind] = np.nan
        time_ind = (ripple_ind[0] +
                    np.arange(0, 2 * n_units * n_samples_between_spikes,
                              n_samples_between_spikes))
        multiunit[time_ind, :, tetrode_ind] = mean_order[:, np.newaxis]

    is_training = np.any(power > 0.0004, axis=1)

    return {
        'time': time,
        'linear_distance': linear_distance,
        'speed': speed,
        'lfps': lfps,
        'ripple_power': power,
        'ripple_times': ripple_times,
        'spikes': spikes,
        'multiunit': multiunit,
        'is_training': is_training,
        'sampling_frequency': sampling_frequency,
    }
