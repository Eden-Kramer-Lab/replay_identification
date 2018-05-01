import numpy as np
from scipy.signal import convolve, gaussian
from scipy.stats import norm


def simulate_speed(time, sampling_frequency, ripple_times, sigma=0.010,
                   ripple_width=0.050):
    ripple_start_end = (ripple_times[:, np.newaxis]
                        + ripple_width * np.array([-1, 1]))

    is_replay = np.zeros_like(time, dtype=bool)

    for start, end in ripple_start_end:
        is_replay[(time >= start) & (time <= end)] = True
    speed = (is_replay.copy() * -19.0) + 20
    bandwidth = sigma * sampling_frequency
    n_time_window_samples = int(bandwidth * 8)
    kernel = gaussian(n_time_window_samples, bandwidth)
    speed = convolve(speed, kernel, mode='same') / kernel.sum()
    return norm.rvs(loc=speed)
