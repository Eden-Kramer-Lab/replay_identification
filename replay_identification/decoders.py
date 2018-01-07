from functools import partial

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .core import get_place_bin_centers, get_place_bins
from .lfp_likelihood import fit_lfp_likelihood_ratio
from .position_state_transition import fit_position_state_transition
from .speed_likelhood import fit_speed_likelihood_ratio
from .speed_state_transition import fit_speed_state_transition
from .spiking_likelihood import fit_spiking_likelihood_ratio

_DEFAULT_LIKELIHOODS = ['spikes', 'lfp_power', 'speed']


class ReplayDetector(object):
    def __init__(self, speed_threshold=4.0, spike_model_penalty=1E-5,
                 time_bin_size=1, speed_state_transition_penalty=1E-5,
                 place_bin_size=30):
        self.speed_threshold = speed_threshold
        self.spike_model_penalty = spike_model_penalty
        self.time_bin_size = time_bin_size
        self.speed_state_transition_penalty = speed_state_transition_penalty
        self.place_bin_size = place_bin_size

    def fit(self, is_replay, speed, lfp_power, position,
            spikes=None, multiunit=None):
        """Train the model on replay and non-replay periods.

        Parameters
        ----------
        is_replay : ndarray, shape (n_time,)
        speed : ndarray, shape (n_time,)
        lfp_power : ndarray, shape (n_time, n_signals)
        position : ndarray, shape (n_time,)
        spikes : ndarray or None, shape (n_time, n_neurons)
        multiunit : ndarray or None, shape (n_time,)

        """

        self.place_bins = get_place_bins(position, self.place_bin_size)
        self.place_bin_centers = get_place_bin_centers(self.place_bins)

        self._speed_likelihood_ratio = fit_speed_likelihood_ratio(
            speed, is_replay, self.speed_threshold)
        self._lfp_likelihood_ratio = fit_lfp_likelihood_ratio(
            lfp_power, is_replay)
        if spikes is not None:
            self._spiking_likelihood_ratio = fit_spiking_likelihood_ratio(
                position, spikes, self.place_bin_centers,
                self.spike_model_penalty, self.time_bin_size)
        else:
            self._spiking_likelihood_ratio = return_None

        self._position_state_transition = fit_position_state_transition(
            position, speed, spikes, self.place_bins, self.speed_threshold)
        self._speed_state_transition = fit_speed_state_transition(
            speed, is_replay, self.speed_state_transition_penalty)

    def predict(self, speed, lfp_power, position, spikes=None, multiunit=None,
                use_likelihoods=_DEFAULT_LIKELIHOODS, sampling_frequency=1):
        """Predict the probability of replay and replay position/position.

        Parameters
        ----------
        speed : ndarray, shape (n_time,)
        lfp_power : ndarray, shape (n_time, n_signals)
        position : ndarray, shape (n_time,)
        spikes : ndarray or None, shape (n_time,)
        multiunit : ndarray or None, shape (n_time,)
        use_likelihoods : speed | lfp_power | spikes, optional
        sampling_frequency : float, optional

        Returns
        -------
        decoding_results

        """
        n_time = speed.shape[0]
        time = np.arange(n_time) / sampling_frequency
        lagged_speed = lagmat(speed, maxlag=1).squeeze()

        n_place_bins = self.place_bin_centers.size
        place_bins = self.place_bin_centers
        place_bin_size = np.diff(place_bins)[0]

        posterior_density = np.zeros((n_time,))
        replay_posterior = np.zeros((n_time, n_place_bins))
        uniform = np.ones((n_place_bins,)) / n_place_bins
        likelihood = np.ones((n_time, 1))

        likelihoods = {
            'speed': partial(self._speed_likelihood_ratio, speed=speed,
                             lagged_speed=lagged_speed),
            'lfp_power': partial(self._lfp_likelihood_ratio,
                                 lfp_power=lfp_power),
            'spikes': partial(self.spikes_likelihood_ratio,
                              spikes=spikes, position=position),
        }

        for name, likelihood_func in likelihoods.items():
            if name.lower() in use_likelihoods:
                likelihood = likelihood * likelihood_func()
        probability_replay = self._speed_state_transition(lagged_speed)

        for time_ind in np.arange(1, n_time):
            replay_prior = (
                probability_replay[time_ind, 1] *
                (self._position_state_transition @
                 replay_posterior[time_ind - 1]) +
                probability_replay[time_ind, 0] *
                (1 - posterior_density[time_ind - 1]) * uniform)
            updated_posterior = likelihood[time_ind] * replay_prior
            non_replay_posterior = ((1 - probability_replay[time_ind - 1, 0]) *
                                    (1 - posterior_density[time_ind - 1]) +
                                    (1 - probability_replay[time_ind - 1, 1]) *
                                    posterior_density[time_ind - 1])
            s = np.sum(updated_posterior * place_bin_size / n_place_bins)
            norm = s + non_replay_posterior
            posterior_density[time_ind] = s / norm
            replay_posterior[time_ind] = updated_posterior / norm

        return time, replay_posterior, posterior_density


class DecodingResults():
    def __init__():
        pass

    def plot_replay_probability():
        pass

    def plot_replay_position():
        pass


def get_n_time(*args):
    for arg in args:
        try:
            return np.shape(arg)[0]
        except IndexError:
            continue
    else:
        raise AttributeError('All of the data is None')


def replace_None_with_NaN(n_time, *args):
    new_args = []
    for arg in args:
        if arg is None:
            new_args.append(np.full((n_time,), np.nan))
        else:
            new_args.append(arg.copy())
    return new_args


def return_None():
    pass
