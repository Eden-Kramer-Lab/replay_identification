from functools import partial
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numba import jit
from sklearn.mixture import BayesianGaussianMixture
from statsmodels.tsa.tsatools import lagmat

from .core import get_place_bin_centers, get_place_bins
from .lfp_likelihood import fit_lfp_likelihood_ratio
from .movement_state_transition import empirical_movement_transition_matrix
from .multiunit_likelihood import fit_multiunit_likelihood_ratio
from .replay_state_transition import fit_replay_state_transition
from .speed_likelhood import fit_speed_likelihood_ratio
from .spiking_likelihood import fit_spiking_likelihood_ratio

logger = getLogger(__name__)

_DEFAULT_LIKELIHOODS = ['spikes', 'lfp_power', 'speed']


class ReplayDetector(object):
    """Find replay events using information from spikes, lfp ripple band power,
    speed, and/or multiunit.

    Attributes
    ----------
    speed_threshold : float, optional
        Speed cutoff that denotes when the animal is moving vs. not moving.
    spike_model_penalty : float, optional
    time_bin_size : float, optional
    replay_state_transition_penalty : float, optional
    place_bin_size : float, optional
    replay_speed : int, optional
        The amount of speedup expected from the replay events vs.
        normal movement.
    spike_model_knot_spacing : float, optional
        Determines how far apart to place to the spline knots over position.
    multiunit_density_model : Class, optional
        Fits the mark space vs. position density. Can be any class with a fit,
        score_samples, and a sample method. For example, density estimators
        from scikit-learn such as sklearn.neighbors.KernelDensity,
        sklearn.mixture.GaussianMixture, and
        sklearn.mixture.BayesianGaussianMixture.
    multiunit_model_kwargs : dict, optional
        Arguments for the `multiunit_density_model`

    Methods
    -------
    fit
        Fits the model to the training data.
    predict
        Predicts the replay probability and posterior density to new data.
    plot_fitted_place_fields
        Plot the place fields from the fitted spiking data.
    plot_fitted_multiunit_model
        Plot position by mark from the fitted multiunit data.
    plot_replay_state_transition
        Plot the replay state transition model over speed lags.
    plot_movement_state_transition
        Plot the semi-latent state movement transition model.

    """

    def __init__(self, speed_threshold=4.0, spike_model_penalty=1E1,
                 time_bin_size=1, replay_state_transition_penalty=1E-5,
                 place_bin_size=1, replay_speed=20,
                 spike_model_knot_spacing=30,
                 multiunit_density_model=BayesianGaussianMixture,
                 multiunit_model_kwargs=dict(n_components=10)):
        self.speed_threshold = speed_threshold
        self.spike_model_penalty = spike_model_penalty
        self.time_bin_size = time_bin_size
        self.replay_state_transition_penalty = replay_state_transition_penalty
        self.place_bin_size = place_bin_size
        self.replay_speed = replay_speed
        self.spike_model_knot_spacing = spike_model_knot_spacing
        self.multiunit_density_model = multiunit_density_model
        self.multiunit_model_kwargs = multiunit_model_kwargs

    def __dir__(self):
        return self.keys()

    def fit(self, is_replay, speed, lfp_power, position,
            spikes=None, multiunit=None):
        """Train the model on replay and non-replay periods.

        Parameters
        ----------
        is_replay : bool ndarray, shape (n_time,)
        speed : ndarray, shape (n_time,)
        lfp_power : ndarray, shape (n_time, n_signals)
        position : ndarray, shape (n_time,)
        spikes : ndarray or None, shape (n_time, n_neurons)
        multiunit : ndarray or None, shape (n_time, n_marks, n_signals)
            np.nan represents times with no multiunit activity.

        """
        self.place_bin_edges = get_place_bins(position, self.place_bin_size)
        self.place_bin_centers = get_place_bin_centers(self.place_bin_edges)

        logger.info('Fitting speed model...')
        self._speed_likelihood_ratio = fit_speed_likelihood_ratio(
            speed, is_replay, self.speed_threshold)
        logger.info('Fitting LFP power model...')
        self._lfp_likelihood_ratio = fit_lfp_likelihood_ratio(
            lfp_power, is_replay)
        if spikes is not None:
            logger.info('Fitting spiking model...')
            self._spiking_likelihood_ratio = fit_spiking_likelihood_ratio(
                position, spikes, is_replay, self.place_bin_centers,
                self.spike_model_penalty, self.time_bin_size,
                self.spike_model_knot_spacing)
        else:
            self._spiking_likelihood_ratio = return_None

        if multiunit is not None:
            logger.info('Fitting multiunit model...')
            self._multiunit_likelihood_ratio = fit_multiunit_likelihood_ratio(
                position, multiunit, is_replay, self.place_bin_centers,
                self.multiunit_density_model, self.multiunit_model_kwargs)
        else:
            self._multiunit_likelihood_ratio = return_None

        logger.info('Fitting movement state transition...')
        self._movement_state_transition = empirical_movement_transition_matrix(
            position, self.place_bin_edges, speed, self.replay_speed)
        logger.info('Fitting replay state transition...')
        self._replay_state_transition = fit_replay_state_transition(
            speed, is_replay, self.replay_state_transition_penalty)

    def predict(self, speed, lfp_power, position, spikes=None, multiunit=None,
                use_likelihoods=_DEFAULT_LIKELIHOODS, time=None):
        """Predict the probability of replay and replay position/position.

        Parameters
        ----------
        speed : ndarray, shape (n_time,)
        lfp_power : ndarray, shape (n_time, n_signals)
        position : ndarray, shape (n_time,)
        spikes : ndarray or None, shape (n_time, n_neurons), optional
        multiunit : ndarray or None, shape (n_time, n_marks, n_signals),
                    optional
        use_likelihoods : list of str, optional
            Valid strings in the list are:
             (speed | lfp_power | spikes | multiunit)
        time : ndarray or None, shape (n_time,), optional
            Experiment time will be included in the results if specified.


        Returns
        -------
        decoding_results : xarray.Dataset
            Includes replay probability and posterior density.

        """
        n_time = speed.shape[0]
        if time is None:
            time = np.arange(n_time)
        lagged_speed = lagmat(speed, maxlag=1).squeeze()

        n_place_bins = self.place_bin_centers.size
        place_bins = self.place_bin_centers
        place_bin_size = np.diff(place_bins)[0]

        likelihood = np.ones((n_time, 1))

        likelihoods = {
            'speed': partial(self._speed_likelihood_ratio, speed=speed,
                             lagged_speed=lagged_speed),
            'lfp_power': partial(self._lfp_likelihood_ratio,
                                 ripple_band_power=lfp_power),
            'spikes': partial(self._spiking_likelihood_ratio,
                              is_spike=spikes, position=position),
            'multiunit': partial(self._multiunit_likelihood_ratio,
                                 multiunit=multiunit, position=position)
        }

        for name, likelihood_func in likelihoods.items():
            if name.lower() in use_likelihoods:
                logger.info('Predicting {0} likelihood...'.format(name))
                likelihood = likelihood * replace_NaN(likelihood_func())

        replay_state_transition = self._replay_state_transition(lagged_speed)

        logger.info('Predicting replay probability and density...')
        replay_probability, replay_posterior = _predict(
            likelihood, self._movement_state_transition,
            replay_state_transition, n_time, n_place_bins, place_bin_size)

        likelihood_dims = (['time', 'position'] if likelihood.shape[1] > 1
                           else ['time'])

        return xr.Dataset(
            {'replay_probability': (['time'], replay_probability),
             'replay_posterior': (['time', 'position'], replay_posterior),
             'likelihood': (likelihood_dims, likelihood.squeeze())},
            coords={'time': time, 'position': place_bins})

    def plot_fitted_place_fields(self, ax=None, sampling_frequency=1):
        """Plot the place fields from the fitted spiking data.

        Parameters
        ----------
        ax : matplotlib axes or None, optional
        sampling_frequency : float, optional

        """
        if ax is None:
            ax = plt.gca()

        place_conditional_intensity = (
            self._spiking_likelihood_ratio
            .keywords['place_conditional_intensity']).squeeze()
        ax.plot(self.place_bin_centers,
                place_conditional_intensity * sampling_frequency)
        ax.set_title('Estimated Place Fields')
        ax.set_ylabel('Spikes / s')
        ax.set_xlabel('Position')

    def plot_fitted_multiunit_model(self, sampling_frequency=1,
                                    n_samples=1E4,
                                    mark_edges=np.linspace(0, 400, 100),
                                    is_histogram=True):
        """Plot position by mark from the fitted multiunit data.

        Parameters
        ----------
        sampling_frequency : float, optional
            If 'is_histogram' is True, then used for computing the intensity.
        n_samples : int, optional
            Number of samples to generate from the fitted model.
        mark_edges : ndarray, shape (n_edges,)
            If `is_histogram` is True, then the edges that define the mark bins
        is_histogram : bool, optional
            If True, plots the joint mark intensity of the samples. Otherwise,
            a scatter plot of the samples is returned.

        Returns
        -------
        axes : matplotlib.pyplot axes

        """
        joint_mark_intensity_functions = (
            self._multiunit_likelihood_ratio.keywords[
                'joint_mark_intensity_functions'])
        n_signals = len(joint_mark_intensity_functions)
        n_marks = (joint_mark_intensity_functions[0]
                   .keywords['fitted_model'].means_.shape[1] - 1)
        bins = (self.place_bin_edges, mark_edges)

        fig, axes = plt.subplots(n_signals, n_marks,
                                 figsize=(n_marks * 3, n_signals * 3),
                                 sharex=True, sharey=True)
        for jmi, row_axes in zip(joint_mark_intensity_functions, axes):
            samples = jmi.keywords['fitted_model'].sample(n_samples)[0]
            place_occupancy = jmi.keywords['place_occupancy']
            mean_rate = jmi.keywords['mean_rate']

            for mark_ind, ax in enumerate(row_axes):
                if is_histogram:
                    H = np.histogram2d(samples[:, -1], samples[:, mark_ind],
                                       bins=bins, normed=True)[0]
                    H = sampling_frequency * mean_rate * H.T / place_occupancy
                    X, Y = np.meshgrid(*bins)
                    ax.pcolormesh(X, Y, H, vmin=0, vmax=1)
                else:
                    ax.scatter(samples[:, -1], samples[:, mark_ind], alpha=0.1)

        plt.xlim((bins[0].min(), bins[0].max()))
        plt.ylim((bins[1].min(), bins[1].max()))
        plt.tight_layout()

        return axes

    def plot_replay_state_transition(self):
        """Plot the replay state transition model over speed lags."""
        lagged_speeds = np.arange(0, 30, .1)
        probablity_replay = self._replay_state_transition(lagged_speeds)

        fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        axes[0].plot(lagged_speeds, probablity_replay[:, 1])
        axes[0].set_ylabel('Probability Replay')
        axes[0].set_title('Previous time step is replay')

        axes[1].plot(lagged_speeds, probablity_replay[:, 0])
        axes[1].set_xlabel('Speed t - 1')
        axes[1].set_ylabel('Probability Replay')
        axes[1].set_title('Previous time step is not replay')

        plt.tight_layout()

    def plot_movement_state_transition(self, ax=None):
        """Plot the sped up empirical movement state transition.

        Parameters
        ----------
        ax : matplotlib axis or None, optional

        """
        if ax is None:
            ax = plt.gca()
        place_t, place_t_1 = np.meshgrid(self.place_bin_edges,
                                         self.place_bin_edges)
        vmax = np.percentile(self._movement_state_transition, 97.5)
        cax = ax.pcolormesh(place_t, place_t_1,
                            self._movement_state_transition, vmin=0, vmax=vmax)
        ax.set_xlabel('position t')
        ax.set_ylabel('position t - 1')
        ax.set_title('Movement State Transition')
        plt.colorbar(cax, label='probability')


@jit(nopython=True)
def _predict(likelihood, movement_state_transition, replay_state_transition,
             n_time, n_place_bins, place_bin_size):
    replay_probability = np.zeros((n_time,))
    replay_posterior = np.zeros((n_time, n_place_bins))
    uniform = np.ones((n_place_bins,)) / n_place_bins

    for time_ind in np.arange(1, n_time):
        replay_prior = (
            replay_state_transition[time_ind, 1]
            * np.dot(movement_state_transition, replay_posterior[time_ind - 1])
            * place_bin_size
            + replay_state_transition[time_ind, 0]
            * uniform * (1 - replay_probability[time_ind - 1]))
        updated_posterior = likelihood[time_ind] * replay_prior
        non_replay_posterior = (
            (1 - replay_state_transition[time_ind, 0]) *
            (1 - replay_probability[time_ind - 1]) +
            (1 - replay_state_transition[time_ind, 1]) *
            replay_probability[time_ind - 1])
        integrated_posterior = np.sum(updated_posterior) * place_bin_size
        norm = integrated_posterior + non_replay_posterior
        replay_probability[time_ind] = integrated_posterior / norm
        replay_posterior[time_ind] = updated_posterior / norm

    return replay_probability, replay_posterior


def replace_NaN(x):
    x[np.isnan(x)] = 1
    return x


def return_None(*args, **kwargs):
    return None
