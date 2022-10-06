from functools import partial
from itertools import combinations_with_replacement
from logging import getLogger

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.mixture import BayesianGaussianMixture
from statsmodels.tsa.tsatools import lagmat

from replay_identification.bins import (
    atleast_2d,
    get_centers,
    get_grid,
    get_observed_position_bin,
    get_track_grid,
    get_track_interior,
)
from replay_identification.core import (
    _acausal_classifier,
    _acausal_classifier_gpu,
    _causal_classifier,
    _causal_classifier_gpu,
    check_converged,
    replace_NaN,
    return_None,
)
from replay_identification.discrete_state_transition import (
    _DISCRETE_STATE_TRANSITIONS,
    _constant_probability,
    estimate_discrete_state_transition,
)
from replay_identification.likelihoods import (
    NumbaKDE,
    fit_lfp_likelihood,
    fit_multiunit_likelihood,
    fit_multiunit_likelihood_gpu,
    fit_multiunit_likelihood_integer,
    fit_speed_likelihood,
    fit_spiking_likelihood_glm,
)
from replay_identification.movement_state_transition import (
    empirical_movement,
    random_walk,
    random_walk_on_track_graph,
)

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)

_DEFAULT_LIKELIHOODS = ["multiunit"]
_DEFAULT_MULTIUNIT_KWARGS = dict(bandwidth=np.array([24.0, 24.0, 24.0, 24.0, 6.0, 6.0]))
_DEFAULT_LFP_KWARGS = dict(n_components=1, max_iter=200, tol=1e-6)
_DEFAULT_OCCUPANCY_KWARGS = dict(bandwidth=np.array([6.0, 6.0]))


class ReplayDetector(BaseEstimator):
    """Find replay events using information from spikes, lfp ripple band power,
    speed, and/or multiunit.

    Attributes
    ----------
    speed_threshold : float, optional
        Speed cutoff that denotes when the animal is moving vs. not moving.
    spike_model_penalty : float, optional
    discrete_state_transition_penalty : float, optional
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
        Predicts the non-local probability and posterior density to new data.
    plot_fitted_place_fields
        Plot the place fields from the fitted spiking data.
    plot_fitted_multiunit_model
        Plot position by mark from the fitted multiunit data.
    plot_discrete_state_transition
        Plot the replay state transition model over speed lags.
    plot_movement_state_transition
        Plot the semi-latent state movement transition model.

    """

    def __init__(
        self,
        speed_threshold=4.0,
        spike_model_penalty=1e-5,
        discrete_state_transition_penalty=1e-5,
        place_bin_size=2.0,
        position_range=None,
        is_track_interior=None,
        infer_track_interior=True,
        replay_speed=1,
        movement_var=4.0,
        spike_model_knot_spacing=12.5,
        speed_knots=None,
        multiunit_density_model=NumbaKDE,
        multiunit_model_kwargs=_DEFAULT_MULTIUNIT_KWARGS,
        multiunit_occupancy_model=NumbaKDE,
        multiunit_occupancy_kwargs=_DEFAULT_OCCUPANCY_KWARGS,
        lfp_model=BayesianGaussianMixture,
        lfp_model_kwargs=_DEFAULT_LFP_KWARGS,
        movement_state_transition_type="empirical",
        discrete_state_transition_type="ripples_with_speed_threshold",
        discrete_diagonal=None,
    ):
        self.speed_threshold = speed_threshold
        self.spike_model_penalty = spike_model_penalty
        self.discrete_state_transition_penalty = discrete_state_transition_penalty
        self.place_bin_size = place_bin_size
        self.position_range = position_range
        self.is_track_interior = is_track_interior
        self.infer_track_interior = infer_track_interior
        self.replay_speed = replay_speed
        self.movement_var = movement_var
        self.spike_model_knot_spacing = spike_model_knot_spacing
        self.speed_knots = speed_knots
        self.multiunit_density_model = multiunit_density_model
        self.multiunit_model_kwargs = multiunit_model_kwargs
        self.multiunit_occupancy_model = multiunit_occupancy_model
        self.multiunit_occupancy_kwargs = multiunit_occupancy_kwargs
        self.lfp_model = lfp_model
        self.lfp_model_kwargs = lfp_model_kwargs
        self.movement_state_transition_type = movement_state_transition_type
        self.discrete_state_transition_type = discrete_state_transition_type
        self.discrete_diagonal = discrete_diagonal

    def fit_place_grid(
        self,
        position=None,
        track_graph=None,
        edge_order=None,
        edge_spacing=None,
        infer_track_interior=True,
    ):
        self.track_graph = track_graph
        if self.track_graph is None:
            (
                self.edges_,
                self.place_bin_edges_,
                self.place_bin_centers_,
                self.centers_shape_,
            ) = get_grid(
                position,
                self.place_bin_size,
                self.position_range,
                self.infer_track_interior,
            )

            self.infer_track_interior = infer_track_interior

            if self.is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(position, self.edges_)
            elif self.is_track_interior is None and not self.infer_track_interior:
                self.is_track_interior_ = np.ones(self.centers_shape_, dtype=np.bool)
        else:
            (
                self.place_bin_centers_,
                self.place_bin_edges_,
                self.is_track_interior_,
                self.distance_between_nodes_,
                self.centers_shape_,
                self.edges_,
                self.track_graph_with_bin_centers_edges_,
                self.original_nodes_df_,
                self.place_bin_edges_nodes_df_,
                self.place_bin_centers_nodes_df_,
                self.nodes_df_,
            ) = get_track_grid(
                self.track_graph, edge_order, edge_spacing, self.place_bin_size
            )

        return self

    def estimate_parameters(
        self,
        fit_args,
        predict_args,
        tolerance=1e-4,
        max_iter=10,
        estimate_state_transition=True,
        estimate_likelihood=True,
    ):

        self.fit(**fit_args)
        results = self.predict(**predict_args)

        data_log_likelihoods = [results.data_log_likelihood]
        log_likelihood_change = np.inf
        converged = False
        increasing = True
        n_iter = 0

        logger.info(f"iteration {n_iter}, likelihood: {data_log_likelihoods[-1]}")

        while not converged and (n_iter < max_iter):
            if estimate_state_transition:
                discrete_state_transition = estimate_discrete_state_transition(
                    self, results
                )
                self.discrete_state_transition_ = partial(
                    _constant_probability, diagonal=discrete_state_transition[:, 1]
                )
            if estimate_likelihood:
                fit_args["is_training"] = 1 - results.non_local_probability
                fit_args["refit"] = True
                self.fit(**fit_args)
            results = self.predict(**predict_args)
            data_log_likelihoods.append(results.data_log_likelihood)
            log_likelihood_change = data_log_likelihoods[-1] - data_log_likelihoods[-2]
            n_iter += 1

            converged, increasing = check_converged(
                data_log_likelihoods[-1], data_log_likelihoods[-2], tolerance
            )

            logger.info(
                f"iteration {n_iter}, "
                f"likelihood: {data_log_likelihoods[-1]}, "
                f"change: {log_likelihood_change}"
            )

        return results, data_log_likelihoods

    def fit(
        self,
        is_ripple,
        speed,
        position,
        lfp_power=None,
        spikes=None,
        multiunits=None,
        is_track_interior=None,
        track_graph=None,
        edge_order=None,
        edge_spacing=None,
        is_training=None,
        refit=False,
        use_gpu=False,
        integer_marks=False,
    ):
        """Train the model on replay and non-replay periods.

        Parameters
        ----------
        is_ripple : bool ndarray, shape (n_time,)
        speed : ndarray, shape (n_time,)
        position : ndarray, shape (n_time,)
        lfp_power : ndarray or None, shape (n_time, n_signals), optional
        spikes : ndarray or None, shape (n_time, n_neurons), optional
        multiunit : ndarray or None, shape (n_time, n_marks, n_signals), optional
            np.nan represents times with no multiunit activity.
        is_track_interior : ndarray, shape (n_place_bins, n_position_dims)
        track_graph : networkx.Graph
        center_well_id : object
        edge_order : array_like
        edge_spacing : None, float or array_like

        """
        speed = np.asarray(speed).squeeze()
        position = atleast_2d(np.asarray(position))
        is_ripple = np.asarray(is_ripple).squeeze()
        if is_training is None:
            is_training = speed > self.speed_threshold

        self.fit_place_grid(position, track_graph, edge_order, edge_spacing)
        is_track_interior = self.is_track_interior_.ravel(order="F")

        try:
            logger.info("Fitting speed model...")
            self._speed_likelihood = fit_speed_likelihood(
                speed, is_ripple, self.speed_threshold
            )
        except ValueError:
            self._speed_likelihood = return_None

        if lfp_power is not None:
            logger.info("Fitting LFP power model...")
            lfp_power = np.asarray(lfp_power)
            self._lfp_likelihood = fit_lfp_likelihood(
                lfp_power, is_ripple, self.lfp_model, self.lfp_model_kwargs
            )
        else:
            self._lfp_likelihood = return_None

        if spikes is not None:
            logger.info("Fitting spiking model...")
            spikes = np.asarray(spikes)
            self._spiking_likelihood = fit_spiking_likelihood_glm(
                position,
                spikes,
                is_training,
                self.place_bin_centers_,
                self.place_bin_edges_,
                is_track_interior,
                self.spike_model_penalty,
                self.spike_model_knot_spacing,
            )
        else:
            self._spiking_likelihood = return_None

        if multiunits is not None:
            logger.info("Fitting multiunit model...")
            multiunits = np.asarray(multiunits)
            if not use_gpu:
                if not integer_marks:
                    self._multiunit_likelihood = fit_multiunit_likelihood(
                        position,
                        multiunits,
                        is_training,
                        self.place_bin_centers_,
                        self.multiunit_density_model,
                        self.multiunit_model_kwargs,
                        self.multiunit_occupancy_model,
                        self.multiunit_occupancy_kwargs,
                        is_track_interior,
                    )
                else:
                    self._multiunit_likelihood = fit_multiunit_likelihood_integer(
                        position,
                        multiunits,
                        is_training,
                        self.place_bin_centers_,
                        is_track_interior=is_track_interior,
                        **self.multiunit_model_kwargs,
                    )
            else:
                self._multiunit_likelihood = fit_multiunit_likelihood_gpu(
                    position,
                    multiunits,
                    is_training,
                    self.place_bin_centers_,
                    is_track_interior=is_track_interior,
                    **self.multiunit_model_kwargs,
                )
        else:
            self._multiunit_likelihood = return_None

        logger.info("Fitting continuous state transition...")
        if self.movement_state_transition_type == "empirical":
            self.movement_state_transition_ = empirical_movement(
                position, self.edges_, is_training, self.replay_speed
            )
        elif (self.movement_state_transition_type == "random_walk") & (
            track_graph is not None
        ):
            place_bin_center_ind_to_node = np.asarray(
                self.place_bin_centers_nodes_df_.node_id
            )
            self.movement_state_transition_ = random_walk_on_track_graph(
                self.place_bin_centers_,
                0.0,
                self.movement_var,
                place_bin_center_ind_to_node,
                self.distance_between_nodes_,
            )
        elif self.movement_state_transition_type == "random_walk":
            self.movement_state_transition_ = random_walk(
                self.place_bin_centers_,
                self.movement_var,
                is_track_interior,
                self.replay_speed,
            )

        if not refit:
            logger.info("Fitting discrete state transition...")
            self.discrete_state_transition_ = _DISCRETE_STATE_TRANSITIONS[
                self.discrete_state_transition_type
            ](
                speed,
                is_ripple,
                self.discrete_state_transition_penalty,
                self.speed_knots,
                self.discrete_diagonal,
            )

        return self

    def predict(
        self,
        speed,
        position,
        lfp_power=None,
        spikes=None,
        multiunits=None,
        use_likelihoods=_DEFAULT_LIKELIHOODS,
        time=None,
        use_acausal=True,
        set_no_spike_to_equally_likely=True,
        use_gpu=False,
    ):
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
        use_acausal : bool, True
        set_no_spike_to_equally_likely : bool, True
            If there are no spikes in a time bin, likelihood is 1 for all
            positions.

        Returns
        -------
        decoding_results : xarray.Dataset
            Includes non-local probability and posterior density.

        """
        n_time = speed.shape[0]
        speed = np.asarray(speed).squeeze()
        position = atleast_2d(np.asarray(position))
        if lfp_power is not None:
            lfp_power = np.asarray(lfp_power)
        if spikes is not None:
            spikes = np.asarray(spikes)
        if multiunits is not None:
            multiunits = np.asarray(multiunits)

        if time is None:
            time = np.arange(n_time)
        lagged_speed = lagmat(speed, maxlag=1).squeeze()
        lagged_speed[0] = speed[0]

        place_bins = self.place_bin_centers_
        is_track_interior = self.is_track_interior_.ravel(order="F")

        likelihood = np.ones((n_time, 2, 1))

        likelihoods = {
            "speed": partial(
                self._speed_likelihood, speed=speed, lagged_speed=lagged_speed
            ),
            "lfp_power": partial(self._lfp_likelihood, ripple_band_power=lfp_power),
            "spikes": partial(
                self._spiking_likelihood,
                spikes=spikes,
                position=position,
                set_no_spike_to_equally_likely=set_no_spike_to_equally_likely,
            ),
            "multiunit": partial(
                self._multiunit_likelihood,
                multiunits=multiunits,
                position=position,
                set_no_spike_to_equally_likely=set_no_spike_to_equally_likely,
            ),
        }

        for name, likelihood_func in likelihoods.items():
            if name.lower() in use_likelihoods:
                logger.info("Predicting {0} likelihood...".format(name))
                likelihood = likelihood * replace_NaN(likelihood_func())
                if (name == "spikes") or (name == "multiunit"):
                    likelihood[:, :, ~is_track_interior] = 0.0
        discrete_state_transition = self.discrete_state_transition_(lagged_speed)
        observed_position_bin = get_observed_position_bin(
            position, self.edges_, place_bins, is_track_interior
        )

        uniform = np.ones((place_bins.shape[0],))
        uniform[~is_track_interior] = 0.0
        uniform /= uniform.sum()

        logger.info("Finding causal non-local probability and position...")
        if not use_gpu:
            (
                causal_posterior,
                state_probability,
                data_log_likelihood,
            ) = _causal_classifier(
                likelihood,
                self.movement_state_transition_,
                discrete_state_transition,
                observed_position_bin,
                uniform,
            )
        else:
            (
                causal_posterior,
                state_probability,
                data_log_likelihood,
            ) = _causal_classifier_gpu(
                likelihood,
                self.movement_state_transition_,
                discrete_state_transition,
                observed_position_bin,
                uniform,
            )

        n_position_dims = place_bins.shape[1]

        if (likelihood.shape[-1] > 1) & (n_position_dims > 1):
            likelihood_dims = ["time", "state", "x_position", "y_position"]
            posterior_dims = likelihood_dims
            coords = dict(
                time=time,
                x_position=get_centers(self.edges_[0]),
                y_position=get_centers(self.edges_[1]),
                state=["Local", "Non-Local"],
            )
            likelihood_shape = (n_time, 2, *self.centers_shape_)
            posterior_shape = likelihood_shape
        elif (likelihood.shape[-1] > 1) & (n_position_dims == 1):
            likelihood_dims = ["time", "state", "position"]
            posterior_dims = likelihood_dims
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=["Local", "Non-Local"],
            )
            likelihood_shape = (n_time, 2, *self.centers_shape_)
            posterior_shape = likelihood_shape
        else:
            likelihood_dims = ["time", "state"]
            posterior_dims = ["time", "state", "position"]
            likelihood_shape = (n_time, 2)
            posterior_shape = (n_time, 2, *self.centers_shape_)
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=["Local", "Non-Local"],
            )

        try:
            results = xr.Dataset(
                {
                    "causal_posterior": (
                        posterior_dims,
                        causal_posterior.reshape(posterior_shape).swapaxes(3, 2),
                    ),
                    "likelihood": (
                        likelihood_dims,
                        likelihood.reshape(likelihood_shape).swapaxes(3, 2),
                    ),
                },
                coords=coords,
                attrs=dict(data_log_likelihood=data_log_likelihood),
            )
        except np.AxisError:
            results = xr.Dataset(
                {
                    "causal_posterior": (posterior_dims, causal_posterior.squeeze()),
                    "likelihood": (likelihood_dims, likelihood.squeeze()),
                },
                coords=coords,
                attrs=dict(data_log_likelihood=data_log_likelihood),
            )
        if use_acausal:
            logger.info("Finding acausal non-local probability and position...")

            if not use_gpu:
                acausal_posterior, state_probability = _acausal_classifier(
                    causal_posterior,
                    self.movement_state_transition_,
                    discrete_state_transition,
                    observed_position_bin,
                    uniform,
                )
            else:
                acausal_posterior, state_probability = _acausal_classifier_gpu(
                    causal_posterior,
                    self.movement_state_transition_,
                    discrete_state_transition,
                    observed_position_bin,
                    uniform,
                )

            try:
                results["acausal_posterior"] = (
                    posterior_dims,
                    acausal_posterior.reshape(posterior_shape).swapaxes(3, 2),
                )
            except np.AxisError:
                results["acausal_posterior"] = (
                    posterior_dims,
                    acausal_posterior.squeeze(),
                )
        results["non_local_probability"] = (["time"], state_probability[:, 1])

        return results

    def plot_fitted_place_fields(self, sampling_frequency=1, col_wrap=5, axes=None):
        """Plot the place fields from the fitted spiking data.

        Parameters
        ----------
        ax : matplotlib axes or None, optional
        sampling_frequency : float, optional

        """
        place_conditional_intensity = (
            self._spiking_likelihood.keywords["place_conditional_intensity"]
        ).squeeze()
        n_neurons = place_conditional_intensity.shape[1]
        n_rows = np.ceil(n_neurons / col_wrap).astype(np.int)

        if axes is None:
            fig, axes = plt.subplots(
                n_rows,
                col_wrap,
                sharex=True,
                figsize=(col_wrap * 2, n_rows * 2),
                constrained_layout=True,
            )

        for ind, ax in enumerate(axes.flat):
            if ind < n_neurons:
                mask = np.ones_like(self.place_bin_centers_.squeeze())
                mask[~self.is_track_interior_.ravel(order="F")] = np.nan
                ax.plot(
                    self.place_bin_centers_,
                    place_conditional_intensity[:, ind] * sampling_frequency * mask,
                    color="black",
                    linewidth=1,
                    label="fitted model",
                )
                ax.set_title(f"Neuron #{ind + 1}")
                ax.set_ylabel("Spikes / s")
                ax.set_xlabel("Position")
            else:
                ax.axis("off")

    @staticmethod
    def plot_spikes(
        spikes, position, is_ripple, sampling_frequency=1, col_wrap=5, bins="auto"
    ):
        is_ripple = np.asarray(is_ripple.copy()).squeeze()
        position = np.asarray(position.copy()).squeeze()[~is_ripple]
        spikes = np.asarray(spikes.copy())[~is_ripple]

        position_occupancy, bin_edges = np.histogram(position, bins=bins)
        bin_size = np.diff(bin_edges)[0]

        time_ind, neuron_ind = np.nonzero(spikes)
        n_neurons = spikes.shape[1]

        n_rows = np.ceil(n_neurons / col_wrap).astype(np.int)

        fig, axes = plt.subplots(
            n_rows, col_wrap, sharex=True, figsize=(col_wrap * 2, n_rows * 2)
        )

        for ind, ax in enumerate(axes.flat):
            if ind < n_neurons:
                hist, _ = np.histogram(
                    position[time_ind[neuron_ind == ind]], bins=bin_edges
                )
                rate = sampling_frequency * hist / position_occupancy
                ax.bar(bin_edges[:-1], rate, width=bin_size)
                ax.set_title(f"Neuron #{ind + 1}")
                ax.set_ylabel("Spikes / s")
                ax.set_xlabel("Position")
            else:
                ax.axis("off")

        plt.tight_layout()

        return axes

    def plot_fitted_multiunit_model(
        self,
        sampling_frequency=1,
        n_samples=10000,
        mark_edges=np.linspace(0, 400, 100),
        is_histogram=False,
    ):
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
        joint_models = self._multiunit_likelihood.keywords["joint_models"]
        mean_rates = self._multiunit_likelihood.keywords["mean_rates"]
        bins = (self.place_bin_edges_.squeeze(), mark_edges)
        if is_histogram:
            place_occupancy = np.exp(
                self._multiunit_likelihood.keywords["occupancy_model"].score_samples(
                    self.place_bin_centers_
                )
            )
        n_signals = len(joint_models)
        try:
            n_marks = joint_models[0].sample().shape[1] - 1
        except AttributeError:
            n_marks = joint_models[0].sample()[0].shape[1] - 1

        fig, axes = plt.subplots(
            n_signals,
            n_marks,
            figsize=(n_marks * 3, n_signals * 3),
            sharex=True,
            sharey=True,
        )
        zipped = zip(joint_models, mean_rates, axes)
        for electrode_ind, (model, mean_rate, row_axes) in enumerate(zipped):
            try:
                samples, _ = model.sample(n_samples)
            except ValueError:
                samples = model.sample(n_samples)

            for mark_ind, ax in enumerate(row_axes):
                if is_histogram:
                    H = np.histogram2d(
                        samples[:, -1], samples[:, mark_ind], bins=bins, normed=True
                    )[0]
                    H = sampling_frequency * mean_rate * H.T / place_occupancy
                    X, Y = np.meshgrid(*bins)
                    ax.pcolormesh(X, Y, H, vmin=0)
                else:
                    ax.scatter(samples[:, -1], samples[:, mark_ind], alpha=0.1)
                ax.set_title(f"Electrode {electrode_ind + 1}, Feature {mark_ind + 1}")

        plt.xlim((bins[0].min(), bins[0].max()))
        plt.ylim((bins[1].min(), bins[1].max()))
        plt.tight_layout()

        return axes

    def plot_discrete_state_transition(self):
        """Plot the replay state transition model over speed lags."""
        lagged_speeds = np.arange(0, 30, 0.1)
        probablity_replay = self.discrete_state_transition_(lagged_speeds)

        fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        axes[0].plot(lagged_speeds, probablity_replay[:, 1])
        axes[0].set_ylabel("Probability Non-Local")
        axes[0].set_title("Previous time step is non-local")

        axes[1].plot(lagged_speeds, probablity_replay[:, 0])
        axes[1].set_xlabel("Speed t - 1")
        axes[1].set_ylabel("Probability Non-Local")
        axes[1].set_title("Previous time step is local")

        plt.tight_layout()

    def plot_movement_state_transition(self, ax=None, vmax_percent=95):
        """Plot the sped up empirical movement state transition.

        Parameters
        ----------
        ax : matplotlib axis or None, optional

        """
        if ax is None:
            ax = plt.gca()
        place_t, place_t_1 = np.meshgrid(self.place_bin_edges_, self.place_bin_edges_)
        vmax = np.percentile(self.movement_state_transition_, vmax_percent)
        cax = ax.pcolormesh(
            place_t, place_t_1, self.movement_state_transition_, vmin=0, vmax=vmax
        )
        ax.set_xlabel("position t")
        ax.set_ylabel("position t - 1")
        ax.set_title("Movement State Transition")
        plt.colorbar(cax, label="probability")

    @staticmethod
    def plot_multiunit(multiunit, position, is_ripple, axes=None):
        """Plot the multiunit training data for comparison with the
        fitted model."""
        multiunit = np.asarray(multiunit.copy())
        position = atleast_2d(np.asarray(position.copy()))
        is_ripple = np.asarray(is_ripple.copy()).squeeze()

        if axes is None:
            _, n_marks, n_signals = multiunit.shape
            _, axes = plt.subplots(
                n_signals,
                n_marks,
                figsize=(n_marks * 3, n_signals * 3),
                sharex=True,
                sharey=True,
            )
        zipped = zip(axes, np.moveaxis(multiunit, 2, 0))
        for electrode_ind, (row_axes, m) in enumerate(zipped):
            not_nan = np.any(~np.isnan(m), axis=-1)
            for mark_ind, ax in enumerate(row_axes):
                ax.scatter(
                    position[not_nan & ~is_ripple],
                    m[not_nan & ~is_ripple, mark_ind],
                    alpha=0.1,
                    zorder=-1,
                )
                ax.set_title(f"Electrode {electrode_ind + 1}, Feature {mark_ind + 1}")

        plt.xlim((np.nanmin(position), np.nanmax(position)))

    @staticmethod
    def plot_lfp_power(lfp_power, is_ripple):
        """Plot the lfp power training data for comparison with the
        fitted model."""
        lfp_power = np.log(np.asarray(lfp_power.copy()))
        is_ripple = np.asarray(is_ripple.copy()).squeeze()
        n_lfps = lfp_power.shape[1]
        lfp_ind = np.arange(n_lfps)

        fig, axes = plt.subplots(
            n_lfps, n_lfps, figsize=(2 * n_lfps, 2 * n_lfps), sharex=True, sharey=True
        )
        combinations_ind = combinations_with_replacement(lfp_ind, 2)
        for (ind1, ind2) in combinations_ind:
            axes[ind1, ind2].scatter(
                lfp_power[~is_ripple, ind1],
                lfp_power[~is_ripple, ind2],
                label="Local",
                alpha=0.5,
            )
            axes[ind1, ind2].scatter(
                lfp_power[is_ripple, ind1],
                lfp_power[is_ripple, ind2],
                label="Replay",
                alpha=0.5,
            )
            axes[ind1, ind2].set_title(f"Electrode {ind1 + 1} vs. {ind2 + 1}")
            if ind1 != ind2:
                axes[ind2, ind1].axis("off")

        axes[0, 0].legend()
        plt.tight_layout()

    def plot_fitted_lfp_power_model(self, n_samples=1000):
        replay_model = self._lfp_likelihood.keywords["replay_model"]
        no_replay_model = self._lfp_likelihood.keywords["no_replay_model"]
        try:
            replay_samples, _ = replay_model.sample(n_samples=n_samples)
            no_replay_samples, _ = no_replay_model.sample(n_samples=n_samples)
            samples = np.concatenate((replay_samples, no_replay_samples), axis=0)
        except ValueError:
            samples = np.concatenate(
                (
                    replay_model.sample(n_samples=n_samples),
                    no_replay_model.sample(n_samples=n_samples),
                ),
                axis=0,
            )

        is_ripple = np.zeros((n_samples * 2,), dtype=np.bool)
        is_ripple[:n_samples] = True

        self.plot_lfp_power(np.exp(samples), is_ripple)

    def save_model(self, filename="model.pkl"):
        raise NotImplementedError
        # Won't work until patsy designInfo becomes pickleable
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename="model.pkl"):
        raise NotImplementedError
        # Won't work until patsy designInfo becomes pickleable
        return joblib.load(filename)
