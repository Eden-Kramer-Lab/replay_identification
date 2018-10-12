from functools import partial
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.externals import joblib
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.tsatools import lagmat

from .core import (_filter, _smoother, get_bin_centers, get_bin_edges,
                   get_observed_position_bin, replace_NaN, return_None)
from .lfp_likelihood import fit_lfp_likelihood
from .movement_state_transition import empirical_movement_transition_matrix
from .multiunit_likelihood import fit_multiunit_likelihood
from .replay_state_transition import fit_replay_state_transition
from .speed_likelhood import fit_speed_likelihood
from .spiking_likelihood import fit_spiking_likelihood

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
    speed_knots : ndarray, shape (n_knots,), optional
        Spline knots for lagged speed in replay state transition.
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

    def __init__(self, speed_threshold=4.0, spike_model_penalty=1E-1,
                 time_bin_size=1, replay_state_transition_penalty=1E-5,
                 place_bin_size=1, n_place_bins=None, replay_speed=20,
                 spike_model_knot_spacing=15, speed_knots=None,
                 multiunit_density_model=KernelDensity,
                 multiunit_model_kwargs=dict(bandwidth=10, leaf_size=1000,
                                             rtol=1E-3)):
        self.speed_threshold = speed_threshold
        self.spike_model_penalty = spike_model_penalty
        self.time_bin_size = time_bin_size
        self.replay_state_transition_penalty = replay_state_transition_penalty
        self.place_bin_size = place_bin_size
        self.n_place_bins = n_place_bins
        self.replay_speed = replay_speed
        self.spike_model_knot_spacing = spike_model_knot_spacing
        self.multiunit_density_model = multiunit_density_model
        self.multiunit_model_kwargs = multiunit_model_kwargs
        self.speed_knots = speed_knots

    def __dir__(self):
        return self.keys()

    def fit(self, is_replay, speed, position, lfp_power=None,
            spikes=None, multiunit=None, place_bin_edges=None):
        """Train the model on replay and non-replay periods.

        Parameters
        ----------
        is_replay : bool ndarray, shape (n_time,)
        speed : ndarray, shape (n_time,)
        position : ndarray, shape (n_time,)
        lfp_power : ndarray or None, shape (n_time, n_signals), optional
        spikes : ndarray or None, shape (n_time, n_neurons), optional
        multiunit : ndarray or None, shape (n_time, n_marks, n_signals), optional
            np.nan represents times with no multiunit activity.

        """
        if place_bin_edges is not None:
            self.place_bin_edges = place_bin_edges
        else:
            self.place_bin_edges = get_bin_edges(
                position, self.n_place_bins, self.place_bin_size)
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

        logger.info('Fitting speed model...')
        self._speed_likelihood = fit_speed_likelihood(
            speed, is_replay, self.speed_threshold)
        if lfp_power is not None:
            logger.info('Fitting LFP power model...')
            self._lfp_likelihood = fit_lfp_likelihood(
                lfp_power, is_replay)
        else:
            self._lfp_likelihood = return_None

        if spikes is not None:
            logger.info('Fitting spiking model...')
            self._spiking_likelihood = fit_spiking_likelihood(
                position, spikes, is_replay, self.place_bin_centers,
                self.spike_model_penalty, self.time_bin_size,
                self.spike_model_knot_spacing)
        else:
            self._spiking_likelihood = return_None

        if multiunit is not None:
            logger.info('Fitting multiunit model...')
            self._multiunit_likelihood = fit_multiunit_likelihood(
                position, multiunit, is_replay, self.place_bin_centers,
                self.multiunit_density_model, self.multiunit_model_kwargs)
        else:
            self._multiunit_likelihood = return_None

        logger.info('Fitting movement state transition...')
        self._movement_state_transition = empirical_movement_transition_matrix(
            position, self.place_bin_edges, speed, self.replay_speed)
        logger.info('Fitting replay state transition...')
        self._replay_state_transition = fit_replay_state_transition(
            speed, is_replay, self.replay_state_transition_penalty,
            self.speed_knots)

        return self

    def predict(self, speed, position, lfp_power=None, spikes=None,
                multiunit=None, use_likelihoods=_DEFAULT_LIKELIHOODS,
                time=None, use_smoother=True):
        """Predict the probability of replay and replay position/position.

        Parameters
        ----------
        speed : ndarray, shape (n_time,)
        position : ndarray, shape (n_time,)
        lfp_power : ndarray, shape (n_time, n_signals)
        spikes : ndarray or None, shape (n_time, n_neurons), optional
        multiunit : ndarray or None, shape (n_time, n_marks, n_signals),
                    optional
        use_likelihoods : list of str, optional
            Valid strings in the list are:
             (speed | lfp_power | spikes | multiunit)
        time : ndarray or None, shape (n_time,), optional
            Experiment time will be included in the results if specified.
        use_smoother : bool, True

        Returns
        -------
        decoding_results : xarray.Dataset
            Includes replay probability and posterior density.

        """
        n_time = speed.shape[0]
        if time is None:
            time = np.arange(n_time)
        lagged_speed = lagmat(speed, maxlag=1).squeeze()

        place_bins = self.place_bin_centers
        place_bin_size = np.diff(place_bins)[0]

        likelihood = np.ones((n_time, 2, 1))

        likelihoods = {
            'speed': partial(self._speed_likelihood, speed=speed,
                             lagged_speed=lagged_speed),
            'lfp_power': partial(self._lfp_likelihood,
                                 ripple_band_power=lfp_power),
            'spikes': partial(self._spiking_likelihood,
                              is_spike=spikes, position=position),
            'multiunit': partial(self._multiunit_likelihood,
                                 multiunit=multiunit, position=position)
        }

        for name, likelihood_func in likelihoods.items():
            if name.lower() in use_likelihoods:
                logger.info('Predicting {0} likelihood...'.format(name))
                likelihood = likelihood * replace_NaN(likelihood_func())

        replay_state_transition = self._replay_state_transition(lagged_speed)
        observed_position_bin = get_observed_position_bin(
            position, self.place_bin_edges)

        logger.info('Predicting replay probability and density...')
        posterior, state_probability, _ = _filter(
            likelihood, self._movement_state_transition,
            replay_state_transition, observed_position_bin, place_bin_size)
        if use_smoother:
            logger.info('Smoothing...')
            posterior, state_probability, _, _ = _smoother(
                posterior, self._movement_state_transition,
                replay_state_transition, observed_position_bin,
                place_bin_size)
        if likelihood.shape[-1] > 1:
            likelihood_dims = ['time', 'state', 'position']
        else:
            likelihood_dims = ['time', 'state']
        coords = {'time': time,
                  'position': place_bins,
                  'state': ['No Replay', 'Replay']}

        return xr.Dataset(
            {'replay_probability': (['time'], state_probability[:, 1]),
             'posterior': (['time', 'state', 'position'], posterior),
             'likelihood': (likelihood_dims, likelihood.squeeze())},
            coords=coords)

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
            self._spiking_likelihood
            .keywords['place_conditional_intensity']).squeeze()
        ax.plot(self.place_bin_centers,
                place_conditional_intensity * sampling_frequency)
        ax.set_title('Estimated Place Fields')
        ax.set_ylabel('Spikes / s')
        ax.set_xlabel('Position')

    def plot_fitted_multiunit_model(self, sampling_frequency=1,
                                    n_samples=10000,
                                    mark_edges=np.linspace(0, 400, 100),
                                    is_histogram=False):
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
        joint_models = (self._multiunit_likelihood
                        .keywords['joint_models'])
        mean_rates = self._multiunit_likelihood.keywords['mean_rates']
        bins = (self.place_bin_edges, mark_edges)
        if is_histogram:
            place_occupancy = np.exp(
                self._multiunit_likelihood
                .keywords['occupancy_model']
                .score_samples(self.place_bin_centers[:, np.newaxis]))
        n_signals = len(joint_models)
        try:
            n_marks = joint_models[0].sample().shape[1] - 1
        except AttributeError:
            n_marks = joint_models[0].sample()[0].shape[1] - 1

        fig, axes = plt.subplots(n_signals, n_marks,
                                 figsize=(n_marks * 3, n_signals * 3),
                                 sharex=True, sharey=True)
        for model, mean_rate, row_axes in zip(joint_models, mean_rates, axes):
            try:
                samples, _ = model.sample(n_samples)
            except ValueError:
                samples = model.sample(n_samples)

            for mark_ind, ax in enumerate(row_axes):
                if is_histogram:
                    H = np.histogram2d(samples[:, -1], samples[:, mark_ind],
                                       bins=bins, normed=True)[0]
                    H = sampling_frequency * mean_rate * H.T / place_occupancy
                    X, Y = np.meshgrid(*bins)
                    ax.pcolormesh(X, Y, H, vmin=0)
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

    @staticmethod
    def plot_multiunit(multiunit, linear_distance, axes=None):
        '''Plot the multiunit training data for comparison with the
        fitted model.
        '''
        if axes is None:
            _, n_marks, n_signals = multiunit.shape
            _, axes = plt.subplots(n_signals, n_marks,
                                   figsize=(n_marks * 3, n_signals * 3),
                                   sharex=True, sharey=True)

        for row_axes, m in zip(axes, np.moveaxis(multiunit, 2, 0)):
            not_nan = np.any(~np.isnan(m), axis=-1)
            for mark_ind, ax in enumerate(row_axes):
                ax.scatter(linear_distance[not_nan],
                           m[not_nan, mark_ind], alpha=0.1, zorder=-1)

        plt.ylim((0, 400))
        plt.xlim((np.nanmin(linear_distance), np.nanmax(linear_distance)))

    def save_model(self, filename='model.pkl'):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename='model.pkl'):
        return joblib.load(filename)
