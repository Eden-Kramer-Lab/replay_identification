import numpy as np


def get_bin_edges(x, n_bins=None, bin_size=None):
    not_nan_x = x[~np.isnan(x)]
    if bin_size is not None:
        n_bins = (
            np.round(np.ceil(np.ptp(not_nan_x) / bin_size))).astype(np.int)
    return np.linspace(np.min(not_nan_x), np.max(not_nan_x), n_bins + 1,
                       endpoint=True)


def get_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def atleast_2d(x):
    """Adds a dimension to the last axis if the array is 1D."""
    return np.atleast_2d(x).T if x.ndim < 2 else x


def normalize_to_probability(distribution, bin_size):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / (np.nansum(distribution) * bin_size)


def _filter(likelihood, replay_state_transition, movement_state_transition,
            observed_position_bin, position_bin_size):
    n_position_bins = movement_state_transition.shape[0]
    n_time = likelihood.shape[0]
    n_states = 2

    posterior = np.zeros((n_time, n_states, n_position_bins))
    prior = np.zeros_like(posterior)
    uniform = 1 / (n_position_bins * position_bin_size)
    state_probability = np.zeros((n_time, n_states))

    # Initial Conditions
    posterior[0, 0, observed_position_bin[0]] = 1.0 / position_bin_size
    state_probability[0] = np.sum(posterior[0], axis=1) * position_bin_size

    for k in np.arange(1, n_time):
        # I_{k - 1} = 0, I_{k} = 0
        prior[k, 0, observed_position_bin[k]] = (
            (1 - replay_state_transition[k, 0]) * state_probability[k - 1, 0])
        # I_{k - 1} = 1, I_{k} = 0
        prior[k, 0, observed_position_bin[k]] += (
            (1 - replay_state_transition[k, 1]) * state_probability[k - 1, 1])

        # I_{k - 1} = 0, I_{k} = 1
        prior[k, 1] = (
            replay_state_transition[k, 0] * uniform *
            state_probability[k - 1, 0])
        # I_{k - 1} = 1, I_{k} = 1
        prior[k, 1] += (
            replay_state_transition[k, 1] *
            (movement_state_transition @ posterior[k - 1, 1]) *
            position_bin_size)

        posterior[k] = normalize_to_probability(
            prior[k] * likelihood[k], position_bin_size)

        state_probability[k] = np.sum(posterior[k], axis=1) * position_bin_size

    return posterior, state_probability, prior


def _smoother(filter_posterior, movement_state_transition,
              replay_state_transition, position_bin_size,
              observed_position_bin):
    '''
    '''
    filter_probability = np.sum(filter_posterior, axis=2) * position_bin_size

    smoother_posterior = np.zeros_like(filter_posterior)
    smoother_posterior[-1] = filter_posterior[-1].copy()
    smoother_prior = np.zeros_like(filter_posterior)
    weights = np.zeros_like(filter_posterior)
    n_time, _, n_position_bins = filter_posterior.shape
    uniform = 1 / (n_position_bins * position_bin_size)

    for k in np.arange(n_time - 2, -1, -1):
        position_ind = observed_position_bin[k]
        # position_ind = observed_position_bin[k + 1]

        # Predict p(x_{k + 1}, I_{k + 1} \vert H_{1:k})
        # I_{k} = 0, I_{k + 1} = 0
        smoother_prior[k, 0, position_ind] = (
            (1 - replay_state_transition[k, 0]) * filter_probability[k, 0])

        # I_{k} = 1, I_{k + 1} = 0
        smoother_prior[k, 0, position_ind] += (
            (1 - replay_state_transition[k, 1]) * filter_probability[k, 1])

        # I_{k + 1} = 1, I_{k} = 0
        smoother_prior[k, 1] = (
            replay_state_transition[k, 0] * uniform * filter_probability[k, 0])

        # I_{k + 1} = 1, I_{k} = 1
        smoother_prior[k, 1] += (
            replay_state_transition[k, 1] *
            (movement_state_transition @ filter_posterior[k, 1]) *
            position_bin_size)

        smoother_prior[k] += np.spacing(1)

        # Update p(x_{k}, I_{k} \vert H_{1:k})
        # I_{k} = 0, I_{k + 1} = 0
        weights[k, 0, position_ind] = (
            (1 - replay_state_transition[k, 0]) *
            np.sum(smoother_posterior[k + 1, 0] / smoother_prior[k, 0]) *
            position_bin_size)

        # I_{k} = 0, I_{k + 1} = 1
        weights[k, 0] += (
            uniform * replay_state_transition[k, 0] *
            np.sum(smoother_posterior[k + 1, 1] / smoother_prior[k, 1]) *
            position_bin_size)

        # I_{k + 1} = 0, I_{k} = 1
        weights[k, 1, position_ind] = (
            (1 - replay_state_transition[k, 1]) *
            np.sum(smoother_posterior[k + 1, 0] / smoother_prior[k, 0]) *
            position_bin_size)

        # I_{k + 1} = 1, I_{k} = 1
        weights[k, 1] += (
            replay_state_transition[k, 1] *
            np.sum(movement_state_transition * smoother_posterior[k + 1, 1] /
                   smoother_prior[k, 1], axis=1) * position_bin_size)

        smoother_posterior[k] = normalize_to_probability(
            weights[k] * filter_posterior[k], position_bin_size)

    return smoother_posterior, smoother_prior, weights
