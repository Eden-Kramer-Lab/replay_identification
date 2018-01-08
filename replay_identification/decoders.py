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
    """

    Attributes
    ----------
    speed_threshold : float, optional
    spike_model_penalty : float, optional
    time_bin_size : float, optional
    speed_state_transition_penalty : float, optional
    place_bin_size : float, optional

    """

    def __init__(self, speed_threshold=4.0, spike_model_penalty=1E-5,
                 time_bin_size=1, speed_state_transition_penalty=1E-5,
                 place_bin_size=30, replay_speed=20, spike_glm_df=5):
        self.speed_threshold = speed_threshold
        self.spike_model_penalty = spike_model_penalty
        self.time_bin_size = time_bin_size
        self.speed_state_transition_penalty = speed_state_transition_penalty
        self.place_bin_size = place_bin_size
        self.replay_speed = replay_speed
        self.spike_glm_df = spike_glm_df

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

        self.place_bin_centers = get_place_bin_centers(
            get_place_bins(position, self.place_bin_size))

        self._speed_likelihood_ratio = fit_speed_likelihood_ratio(
            speed, is_replay, self.speed_threshold)
        self._lfp_likelihood_ratio = fit_lfp_likelihood_ratio(
            lfp_power, is_replay)
        if spikes is not None:
            self._spiking_likelihood_ratio = fit_spiking_likelihood_ratio(
                position, spikes, is_replay, self.place_bin_centers,
                self.spike_model_penalty, self.time_bin_size,
                self.spike_glm_df)
        else:
            self._spiking_likelihood_ratio = return_None

        self._position_state_transition = fit_position_state_transition(
            position, speed, self.place_bin_centers,
            self.speed_threshold, self.replay_speed)
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
        spikes : ndarray or None, shape (n_time, n_neurons)
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

        replay_probability = np.zeros((n_time,))
        replay_posterior = np.zeros((n_time, n_place_bins))
        uniform = np.ones((n_place_bins,)) / n_place_bins
        likelihood = np.ones((n_time, 1))

        likelihoods = {
            'speed': partial(self._speed_likelihood_ratio, speed=speed,
                             lagged_speed=lagged_speed),
            'lfp_power': partial(self._lfp_likelihood_ratio,
                                 ripple_band_power=lfp_power),
            'spikes': partial(self._spiking_likelihood_ratio,
                              is_spike=spikes, position=position),
        }

        for name, likelihood_func in likelihoods.items():
            if name.lower() in use_likelihoods:
                likelihood = likelihood * replace_NaN(likelihood_func())
        probability_replay = self._speed_state_transition(lagged_speed)

        for time_ind in np.arange(1, n_time):
            replay_prior = (
                probability_replay[time_ind, 1] *
                (self._position_state_transition @
                 replay_posterior[time_ind - 1] * place_bin_size) +
                probability_replay[time_ind, 0] *
                uniform * (1 - replay_probability[time_ind - 1]))
            updated_posterior = likelihood[time_ind] * replay_prior
            non_replay_posterior = ((1 - probability_replay[time_ind - 1, 0]) *
                                    (1 - replay_probability[time_ind - 1]) +
                                    (1 - probability_replay[time_ind - 1, 1]) *
                                    replay_probability[time_ind - 1])
            s = np.sum(updated_posterior * place_bin_size / n_place_bins)
            norm = s + non_replay_posterior
            replay_probability[time_ind] = s / norm
            replay_posterior[time_ind] = updated_posterior / norm

        return time, replay_posterior, replay_probability, likelihood


class DecodingResults():
    def __init__():
        pass

    def plot_replay_probability():
        pass

    def plot_replay_position():
        pass


def return_None():
    pass


def replace_NaN(x):
    x[np.isnan(x)] = 1
    return x
