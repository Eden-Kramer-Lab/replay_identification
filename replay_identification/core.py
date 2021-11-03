import numpy as np
from numba import njit


def replace_NaN(x):
    x[np.isnan(x)] = 1
    return x


def return_None(*args, **kwargs):
    return None


@njit(cache=True, nogil=True)
def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)


@njit(cache=True, nogil=True, error_model='numpy')
def _causal_classifier(likelihood, movement_state_transition, replay_state_transition,
                       observed_position_bin, uniform):
    '''
    Parameters
    ----------
    likelihood : ndarray, shape (n_time, ...)
    movement_state_transition : ndarray, shape (n_position_bins,
                                                n_position_bins)
    replay_state_transition : ndarray, shape (n_time, 2)
        replay_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
        replay_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
    observed_position_bin : ndarray, shape (n_time,)
        Which position bin is the animal in.
    position_bin_size : float

    Returns
    -------
    posterior : ndarray, shape (n_time, 2, n_position_bins)
    state_probability : ndarray, shape (n_time, 2)
        state_probability[:, 0] = Pr(I_{1:T} = 0)
        state_probability[:, 1] = Pr(I_{1:T} = 1)
    prior : ndarray, shape (n_time, 2, n_position_bins)

    '''
    n_position_bins = movement_state_transition.shape[0]
    n_time = likelihood.shape[0]
    n_states = 2

    posterior = np.zeros((n_time, n_states, n_position_bins))
    prior = np.zeros_like(posterior)
    state_probability = np.zeros((n_time, n_states))

    # Initial Conditions
    posterior[0, 0, observed_position_bin[0]] = 1.0
    state_probability[0] = np.sum(posterior[0], axis=1)

    for k in np.arange(1, n_time):
        position_ind = observed_position_bin[k]
        # I_{k - 1} = 0, I_{k} = 0
        prior[k, 0, position_ind] = (
            (1 - replay_state_transition[k, 0]) * state_probability[k - 1, 0])
        # I_{k - 1} = 1, I_{k} = 0
        prior[k, 0, position_ind] += (
            (1 - replay_state_transition[k, 1]) * state_probability[k - 1, 1])

        # I_{k - 1} = 0, I_{k} = 1
        prior[k, 1] = (replay_state_transition[k, 0] * uniform *
                       state_probability[k - 1, 0])
        # I_{k - 1} = 1, I_{k} = 1
        prior[k, 1] += (
            replay_state_transition[k, 1] *
            (movement_state_transition.T @ posterior[k - 1, 1]))

        posterior[k] = normalize_to_probability(
            prior[k] * likelihood[k])

        state_probability[k] = np.sum(posterior[k], axis=1)

    return posterior, state_probability, prior


@njit(cache=True, nogil=True, error_model='numpy')
def _acausal_classifier(filter_posterior, movement_state_transition,
                        replay_state_transition, observed_position_bin, uniform):
    '''
    Parameters
    ----------
    filter_posterior : ndarray, shape (n_time, 2, n_position_bins)
    movement_state_transition : ndarray, shape (n_position_bins,
                                                n_position_bins)
    replay_state_transition : ndarray, shape (n_time, 2)
        replay_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
        replay_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
    observed_position_bin : ndarray, shape (n_time,)
        Which position bin is the animal in.
    position_bin_size : float

    Returns
    -------
    smoother_posterior : ndarray, shape (n_time, 2, n_position_bins)
        p(x_{k + 1}, I_{k + 1} \vert H_{1:T})
    smoother_probability : ndarray, shape (n_time, 2)
        smoother_probability[:, 0] = Pr(I_{1:T} = 0)
        smoother_probability[:, 1] = Pr(I_{1:T} = 1)
    smoother_prior : ndarray, shape (n_time, 2, n_position_bins)
        p(x_{k + 1}, I_{k + 1} \vert H_{1:k})
    weights : ndarray, shape (n_time, 2, n_position_bins)
        \sum_{I_{k+1}} \int \Big[ \frac{p(x_{k+1} \mid x_{k}, I_{k}, I_{k+1}) *
        Pr(I_{k + 1} \mid I_{k}, v_{k}) * p(x_{k+1}, I_{k+1} \mid H_{1:T})}
        {p(x_{k + 1}, I_{k + 1} \mid H_{1:k})} \Big] dx_{k+1}
    '''  # noqa
    filter_probability = np.sum(filter_posterior, axis=2)

    smoother_posterior = np.zeros_like(filter_posterior)
    smoother_prior = np.zeros_like(filter_posterior)
    weights = np.zeros_like(filter_posterior)
    n_time, _, n_position_bins = filter_posterior.shape

    smoother_posterior[-1] = filter_posterior[-1].copy()

    for k in np.arange(n_time - 2, -1, -1):
        position_ind = observed_position_bin[k + 1]

        # Predict p(x_{k + 1}, I_{k + 1} \vert H_{1:k})
        # I_{k} = 0, I_{k + 1} = 0
        smoother_prior[k, 0, position_ind] = (
            (1 - replay_state_transition[k + 1, 0]) * filter_probability[k, 0])

        # I_{k} = 1, I_{k + 1} = 0
        smoother_prior[k, 0, position_ind] += (
            (1 - replay_state_transition[k + 1, 1]) * filter_probability[k, 1])

        # I_{k} = 0, I_{k + 1} = 1
        smoother_prior[k, 1] = (
            replay_state_transition[k + 1, 0] * uniform *
            filter_probability[k, 0])

        # I_{k} = 1, I_{k + 1} = 1
        smoother_prior[k, 1] += (
            replay_state_transition[k + 1, 1] *
            (movement_state_transition.T @ filter_posterior[k, 1]))

        # Update p(x_{k}, I_{k} \vert H_{1:k})
        ratio = np.exp(
            np.log(smoother_posterior[k + 1] + np.spacing(1)) -
            np.log(smoother_prior[k] + np.spacing(1)))
        integrated_ratio = np.sum(ratio, axis=1)
        # I_{k} = 0, I_{k + 1} = 0
        weights[k, 0] = (
            (1 - replay_state_transition[k + 1, 0]) * ratio[0, position_ind])

        # I_{k} = 0, I_{k + 1} = 1
        weights[k, 0] += (
            uniform * replay_state_transition[k + 1, 0] * integrated_ratio[1])

        # I_{k} = 1, I_{k + 1} = 0
        weights[k, 1] = (
            (1 - replay_state_transition[k + 1, 1]) * ratio[0, position_ind])

        # I_{k} = 1, I_{k + 1} = 1
        weights[k, 1] += (
            replay_state_transition[k + 1, 1] *
            ratio[1] @ movement_state_transition)

        smoother_posterior[k] = normalize_to_probability(
            weights[k] * filter_posterior[k])

    smoother_probability = (
        np.sum(smoother_posterior, axis=2))

    return smoother_posterior, smoother_probability, smoother_prior, weights


def scale_likelihood(log_likelihood):
    '''Scales the likelihood to its max value to prevent overflow and underflow.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_states, n_position_bins)

    Returns
    -------
    scaled_likelihood : ndarray, shape (n_time, n_states, n_position_bins)

    '''
    return np.exp(log_likelihood -
                  np.nanmax(log_likelihood, axis=(1, 2), keepdims=True))
