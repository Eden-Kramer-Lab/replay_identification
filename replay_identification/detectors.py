from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import xarray as xr
from replay_identification.bins import (atleast_2d, get_centers, get_grid,
                                        get_observed_position_bin,
                                        get_track_grid, get_track_interior)
from replay_identification.core import (_acausal_classifier,
                                        _acausal_classifier_gpu,
                                        _causal_classifier,
                                        _causal_classifier_gpu,
                                        check_converged)
from replay_identification.discrete_state_transition import (
    estimate_discrete_state_transition, infer_discrete_state_transition,
    make_discrete_state_transition_from_diagonal)
from replay_identification.movement_state_transition import (
    empirical_movement, random_walk, random_walk_on_track_graph)
from replay_identification.spiking_likelihood import fit_spiking_likelihood
from sklearn.base import BaseEstimator

logger = getLogger(__name__)
sklearn.set_config(print_changed_only=False)

_DISCRETE_DIAGONAL = np.asarray([0.999, 0.98])


class _BaseDetector(BaseEstimator):
    def __init__(
        self,
        place_bin_size=2.0,
        position_range=None,
        track_graph=None,
        edge_order=None,
        edge_spacing=None,
        continuous_state_transition_type='random_walk',
        random_walk_variance=6.0,
        discrete_state_transition_type='infer',
        discrete_transition_diagonal=_DISCRETE_DIAGONAL,
        is_track_interior=None,
        infer_track_interior=True,
    ):
        self.place_bin_size = place_bin_size
        self.position_range = position_range
        self.track_graph = track_graph
        self.edge_order = None
        self.edge_spacing = None
        self.continuous_state_transition_type = continuous_state_transition_type
        self.random_walk_variance = random_walk_variance
        self.discrete_state_transition_type = discrete_state_transition_type
        self.discrete_transition_diagonal = discrete_transition_diagonal
        self.is_track_interior = is_track_interior
        self.infer_track_interior = infer_track_interior

    def fit_place_grid(
            self,
            position=None,
            track_graph=None,
            edge_order=None,
            edge_spacing=None,
            infer_track_interior=True
    ):
        position = atleast_2d(np.asarray(position))
        self.track_graph = track_graph
        if self.track_graph is None:
            (self.edges_,
             self.place_bin_edges_,
             self.place_bin_centers_,
             self.centers_shape_
             ) = get_grid(position, self.place_bin_size, self.position_range,
                          self.infer_track_interior)

            self.infer_track_interior = infer_track_interior

            if self.is_track_interior is None and self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(
                    position, self.edges_)
            elif (self.is_track_interior is None and
                  not self.infer_track_interior):
                self.is_track_interior_ = np.ones(
                    self.centers_shape_, dtype=np.bool)
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
                self.nodes_df_
            ) = get_track_grid(self.track_graph, edge_order,
                               edge_spacing, self.place_bin_size)

        return self

    def fit_discrete_state_transition(
            self,
            discrete_transition_diagonal=None,
            is_training=None):
        logger.info('Fitting discrete state transition...')
        if discrete_transition_diagonal is not None:
            self.discrete_transition_diagonal = discrete_transition_diagonal

        if self.discrete_state_transition_type == 'infer':
            self.discrete_state_transition_ = infer_discrete_state_transition(
                is_training)
        else:
            self.discrete_state_transition_ = make_discrete_state_transition_from_diagonal(
                discrete_transition_diagonal)

    def plot_discrete_state_transition(
            self, state_names=None, cmap='Oranges', ax=None,
            convert_to_seconds=False, sampling_frequency=1):

        if ax is None:
            ax = plt.gca()

        if state_names is None:
            state_names = ['Local', 'Non-Local']

        if convert_to_seconds:
            discrete_state_transition = (
                1 / (1 - self.discrete_state_transition_)) / sampling_frequency
            vmin, vmax, fmt = 0.0, None, '0.03f'
            label = 'Seconds'
        else:
            discrete_state_transition = self.discrete_state_transition_
            vmin, vmax, fmt = 0.0, 1.0, '0.03f'
            label = 'Probability'

        sns.heatmap(data=discrete_state_transition,
                    vmin=vmin, vmax=vmax, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=state_names, yticklabels=state_names, ax=ax,
                    cbar_kws={'label': label})
        ax.set_ylabel('Previous State', fontsize=12)
        ax.set_xlabel('Current State', fontsize=12)
        ax.set_title('Discrete State Transition', fontsize=16)

    def fit_continuous_state_transition(
        self,
        position=None,
        is_training=None,
    ):
        logger.info('Fitting continuous state transition...')
        if self.continuous_state_transition_type == 'empirical':
            if is_training is None:
                n_time = position.shape[0]
                is_training = np.ones((n_time,), dtype=bool)
            if position is not None:
                position = atleast_2d(np.asarray(position))
            self.continuous_state_transition_ = empirical_movement(
                position,
                self.edges_,
                is_training,
                replay_speed=20)
        elif ((self.continuous_state_transition_type == 'random_walk') &
              (self.track_graph is not None)):
            place_bin_center_ind_to_node = np.asarray(
                self.place_bin_centers_nodes_df_.node_id)
            self.continuous_state_transition_ = random_walk_on_track_graph(
                self.place_bin_centers_,
                0.0,
                self.random_walk_variance,
                place_bin_center_ind_to_node,
                self.distance_between_nodes_
            )
        elif self.continuous_state_transition_type == 'random_walk':
            self.continuous_state_transition_ = random_walk(
                self.place_bin_centers_,
                self.random_walk_variance,
                self.is_track_interior_.ravel(order='F'),
                1)

    def plot_continuous_state_transition(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.pcolormesh(self.continuous_state_transition_.T)

    def _get_results(
        self,
        position,
        likelihood,
        time=None,
        is_compute_acausal=True,
        use_gpu=False
    ):

        n_time = likelihood.shape[0]
        if time is None:
            time = np.arange(n_time)

        observed_position_bin = get_observed_position_bin(
            position,
            self.edges_,
            self.place_bin_centers_,
            self.is_track_interior_.ravel(order='F')
        )

        uniform = np.ones((self.place_bin_centers_.shape[0],))
        uniform[~self.is_track_interior_.ravel(order='F')] = 0.0
        uniform /= uniform.sum()

        logger.info('Finding causal non-local probability and position...')
        if not use_gpu:
            (causal_posterior, state_probability,
             data_log_likelihood) = _causal_classifier(
                likelihood,
                self.continuous_state_transition_,
                np.repeat(self.discrete_state_transition_[
                          :, [1]].T, n_time, axis=0),
                observed_position_bin,
                uniform)
        else:
            (causal_posterior, state_probability,
             data_log_likelihood) = _causal_classifier_gpu(
                likelihood,
                self.continuous_state_transition_,
                np.repeat(self.discrete_state_transition_[
                          :, [1]].T, n_time, axis=0),
                observed_position_bin,
                uniform)

        n_position_dims = self.place_bin_centers_.shape[1]

        if (likelihood.shape[-1] > 1) & (n_position_dims > 1):
            likelihood_dims = ['time', 'state', 'x_position', 'y_position']
            posterior_dims = likelihood_dims
            coords = dict(
                time=time,
                x_position=get_centers(self.edges_[0]),
                y_position=get_centers(self.edges_[1]),
                state=['Local', 'Non-Local'],
            )
            likelihood_shape = (n_time, 2, *self.centers_shape_)
            posterior_shape = likelihood_shape
        elif (likelihood.shape[-1] > 1) & (n_position_dims == 1):
            likelihood_dims = ['time', 'state', 'position']
            posterior_dims = likelihood_dims
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=['Local', 'Non-Local'],
            )
            likelihood_shape = (n_time, 2, *self.centers_shape_)
            posterior_shape = likelihood_shape
        else:
            likelihood_dims = ['time', 'state']
            posterior_dims = ['time', 'state', 'position']
            likelihood_shape = (n_time, 2)
            posterior_shape = (n_time, 2, *self.centers_shape_)
            coords = dict(
                time=time,
                position=get_centers(self.edges_[0]),
                state=['Local', 'Non-Local'],
            )

        try:
            results = xr.Dataset(
                {'causal_posterior': (
                    posterior_dims,
                    causal_posterior.reshape(posterior_shape).swapaxes(3, 2)),
                 'likelihood': (
                    likelihood_dims,
                    likelihood.reshape(likelihood_shape).swapaxes(3, 2))},
                coords=coords,
                attrs=dict(data_log_likelihood=data_log_likelihood))
        except np.AxisError:
            results = xr.Dataset(
                {'causal_posterior': (
                    posterior_dims,
                    causal_posterior.squeeze()),
                 'likelihood': (
                    likelihood_dims,
                    likelihood.squeeze())},
                coords=coords,
                attrs=dict(data_log_likelihood=data_log_likelihood))
        if is_compute_acausal:
            logger.info(
                'Finding acausal non-local probability and position...')

            if not use_gpu:
                acausal_posterior, state_probability = _acausal_classifier(
                    causal_posterior,
                    self.continuous_state_transition_,
                    np.repeat(self.discrete_state_transition_[
                              :, [1]].T, n_time, axis=0),
                    observed_position_bin,
                    uniform)
            else:
                acausal_posterior, state_probability = _acausal_classifier_gpu(
                    causal_posterior,
                    self.continuous_state_transition_,
                    np.repeat(self.discrete_state_transition_[
                              :, [1]].T, n_time, axis=0),
                    observed_position_bin,
                    uniform)

            try:
                results['acausal_posterior'] = (
                    posterior_dims,
                    acausal_posterior.reshape(posterior_shape).swapaxes(3, 2))
            except np.AxisError:
                results['acausal_posterior'] = (
                    posterior_dims,
                    acausal_posterior.squeeze())
        results['non_local_probability'] = (['time'], state_probability[:, 1])

        return results

    def estimate_parameters(self, fit_args, predict_args, tolerance=1E-4,
                            max_iter=10, estimate_state_transition=True,
                            estimate_likelihood=True):

        self.fit(**fit_args)
        results = self.predict(**predict_args)

        data_log_likelihoods = [results.data_log_likelihood]
        log_likelihood_change = np.inf
        converged = False
        increasing = True
        n_iter = 0

        logger.info(
            f'iteration {n_iter}, likelihood: {data_log_likelihoods[-1]}')

        while not converged and (n_iter < max_iter):
            if estimate_state_transition:
                self.discrete_state_transition_ = estimate_discrete_state_transition(
                    self, results)
            if estimate_likelihood:
                fit_args['is_training'] = 1 - results.non_local_probability
                fit_args['refit'] = True
                self.fit(**fit_args)

            results = self.predict(**predict_args)
            data_log_likelihoods.append(results.data_log_likelihood)
            log_likelihood_change = (
                data_log_likelihoods[-1] - data_log_likelihoods[-2])
            n_iter += 1

            converged, increasing = check_converged(
                data_log_likelihoods[-1], data_log_likelihoods[-2], tolerance)

            logger.info(
                f'iteration {n_iter}, '
                f'likelihood: {data_log_likelihoods[-1]}, '
                f'change: {log_likelihood_change}'
            )

        return results, data_log_likelihoods


class SortedSpikesDetector(_BaseDetector):
    def __init__(
        self,
        place_bin_size=2.0,
        position_range=None,
        track_graph=None,
        edge_order=None,
        edge_spacing=None,
        continuous_state_transition_type='random_walk',
        random_walk_variance=6.0,
        discrete_state_transition_type='infer',
        discrete_transition_diagonal=_DISCRETE_DIAGONAL,
        is_track_interior=None,
        infer_track_interior=True,
        spike_model_penalty=1E-5,
        spike_model_knot_spacing=12.5,
    ):
        super().__init__(
            place_bin_size,
            position_range,
            track_graph,
            edge_order,
            edge_spacing,
            continuous_state_transition_type,
            random_walk_variance,
            discrete_state_transition_type,
            discrete_transition_diagonal,
            is_track_interior,
            infer_track_interior,
        )
        self.spike_model_knot_spacing = spike_model_knot_spacing
        self.spike_model_penalty = spike_model_penalty

    def fit_place_fields(self, position, spikes, is_training=None):
        logger.info('Fitting place fields...')
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)
        if is_training is None:
            is_training = np.ones((spikes.shape[0],), dtype=bool)

        self.encoding_model_ = fit_spiking_likelihood(
            position,
            spikes,
            is_training,
            self.place_bin_centers_,
            self.place_bin_edges_,
            self.is_track_interior_.ravel(order='F'),
            self.spike_model_penalty,
            self.spike_model_knot_spacing)

    def plot_place_fields(
        self,
        sampling_frequency=1,
        col_wrap=5,
        axes=None
    ):
        """Plot the place fields from the fitted spiking data.

        Parameters
        ----------
        ax : matplotlib axes or None, optional
        sampling_frequency : float, optional

        """
        place_conditional_intensity = (
            self.encoding_model_
            .keywords['place_conditional_intensity']).squeeze()
        n_neurons = place_conditional_intensity.shape[1]
        n_rows = np.ceil(n_neurons / col_wrap).astype(np.int)

        if axes is None:
            fig, axes = plt.subplots(n_rows, col_wrap, sharex=True,
                                     figsize=(col_wrap * 2, n_rows * 2),
                                     constrained_layout=True)

        for ind, ax in enumerate(axes.flat):
            if ind < n_neurons:
                mask = np.ones_like(self.place_bin_centers_.squeeze())
                mask[~self.is_track_interior_.ravel(order='F')] = np.nan
                ax.plot(self.place_bin_centers_,
                        place_conditional_intensity[:, ind] *
                        sampling_frequency * mask, color='black', linewidth=1,
                        label='fitted model')
                ax.set_title(f'Neuron #{ind + 1}')
                ax.set_ylabel('Spikes / s')
                ax.set_xlabel('Position')
            else:
                ax.axis('off')

    @staticmethod
    def plot_spikes(position, spikes, is_training=None, sampling_frequency=1,
                    col_wrap=5, bins='auto'):
        if is_training is None:
            is_training = np.ones((spikes.shape[0], ), dtype=bool)
        else:
            is_training = np.asarray(is_training.copy()).squeeze()
        position = np.asarray(position.copy()).squeeze()[is_training]
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes.copy())[is_training]

        position_occupancy, bin_edges = np.histogram(position, bins=bins)
        bin_size = np.diff(bin_edges)[0]

        time_ind, neuron_ind = np.nonzero(spikes)
        n_neurons = spikes.shape[1]

        n_rows = np.ceil(n_neurons / col_wrap).astype(np.int)

        fig, axes = plt.subplots(n_rows, col_wrap, sharex=True,
                                 figsize=(col_wrap * 2, n_rows * 2))

        for ind, ax in enumerate(axes.flat):
            if ind < n_neurons:
                hist, _ = np.histogram(position[time_ind[neuron_ind == ind]],
                                       bins=bin_edges)
                rate = sampling_frequency * hist / position_occupancy
                ax.bar(bin_edges[:-1], rate, width=bin_size)
                ax.set_title(f'Neuron #{ind + 1}')
                ax.set_ylabel('Spikes / s')
                ax.set_xlabel('Position')
            else:
                ax.axis('off')

        plt.tight_layout()

        return axes

    def fit(self, position, spikes, is_training=None, refit=False):
        position = atleast_2d(np.asarray(position))
        if is_training is None:
            is_training = np.ones((position.shape[0],), dtype=bool)

        if not refit:
            self.fit_place_grid(
                position=position,
                track_graph=self.track_graph,
                edge_order=self.edge_order,
                edge_spacing=self.edge_spacing,
                infer_track_interior=self.infer_track_interior,
            )

            self.fit_discrete_state_transition(
                discrete_transition_diagonal=self.discrete_transition_diagonal,
                is_training=is_training
            )
            self.fit_continuous_state_transition(
                position=position,
                is_training=is_training,
            )

        self.fit_place_fields(
            position,
            spikes,
            is_training
        )

        return self

    def predict(
        self,
        position,
        spikes,
        time=None,
        is_compute_acausal=True,
        set_no_spike_to_equally_likely=False,
        use_gpu=False,
        store_likelihood=False
    ):
        position = atleast_2d(np.asarray(position))
        spikes = np.asarray(spikes)

        logger.info('Estimating likelihood...')
        likelihood = self.encoding_model_(
            spikes=spikes,
            position=position,
            set_no_spike_to_equally_likely=set_no_spike_to_equally_likely)

        if store_likelihood:
            self.likelihood_ = likelihood

        return self._get_results(
            position,
            likelihood,
            time=time,
            is_compute_acausal=is_compute_acausal,
            use_gpu=use_gpu,
        )
