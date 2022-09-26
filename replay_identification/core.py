import numpy as np
from numba import njit


def replace_NaN(x):
    x[np.isnan(x)] = 1
    return x


def return_None(*args, **kwargs):
    return None


@njit(cache=True, nogil=True)
def normalize_to_probability(distribution):
    """Ensure the distribution integrates to 1 so that it is a probability
    distribution
    """
    return distribution / np.nansum(distribution)


@njit(cache=True, nogil=True, error_model="numpy")
def _causal_classifier(
    likelihood,
    movement_state_transition,
    discrete_state_transition,
    observed_position_bin,
    uniform,
):
    """
    Parameters
    ----------
    likelihood : ndarray, shape (n_time, ...)
    movement_state_transition : ndarray, shape (n_position_bins,
                                                n_position_bins)
    discrete_state_transition : ndarray, shape (n_time, 2)
        discrete_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
        discrete_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
    observed_position_bin : ndarray, shape (n_time,)
        Which position bin is the animal in.
    position_bin_size : float

    Returns
    -------
    posterior : ndarray, shape (n_time, 2, n_position_bins)
    state_probability : ndarray, shape (n_time, 2)
        state_probability[:, 0] = Pr(I_{1:T} = 0), Local
        state_probability[:, 1] = Pr(I_{1:T} = 1), Non-Local
    prior : ndarray, shape (n_time, 2, n_position_bins)

    """
    n_position_bins = movement_state_transition.shape[0]
    n_time = likelihood.shape[0]
    n_states = 2

    posterior = np.zeros((n_time, n_states, n_position_bins))
    state_probability = np.zeros((n_time, n_states))

    # Initial Conditions
    posterior[0, 0, observed_position_bin[0]] = likelihood[
        0, 0, observed_position_bin[0]
    ]
    norm = np.nansum(posterior[0])
    data_log_likelihood = np.log(norm)
    posterior[0] /= norm
    state_probability[0] = np.sum(posterior[0], axis=1)

    for k in np.arange(1, n_time):
        prior = np.zeros((n_states, n_position_bins))
        position_ind = observed_position_bin[k]
        # I_{k - 1} = 0, I_{k} = 0
        prior[0, position_ind] = (
            1 - discrete_state_transition[k, 0]
        ) * state_probability[k - 1, 0]
        # I_{k - 1} = 1, I_{k} = 0
        prior[0, position_ind] += (
            1 - discrete_state_transition[k, 1]
        ) * state_probability[k - 1, 1]

        # I_{k - 1} = 0, I_{k} = 1
        prior[1] = (
            discrete_state_transition[k, 0] * uniform * state_probability[k - 1, 0]
        )
        # I_{k - 1} = 1, I_{k} = 1
        prior[1] += discrete_state_transition[k, 1] * (
            movement_state_transition.T @ posterior[k - 1, 1]
        )

        posterior[k] = prior * likelihood[k]
        norm = np.nansum(posterior[k])
        data_log_likelihood += np.log(norm)
        posterior[k] /= norm

        state_probability[k] = np.sum(posterior[k], axis=1)

    return posterior, state_probability, data_log_likelihood


@njit(cache=True, nogil=True, error_model="numpy")
def _acausal_classifier(
    filter_posterior,
    movement_state_transition,
    discrete_state_transition,
    observed_position_bin,
    uniform,
):
    """
    Parameters
    ----------
    filter_posterior : ndarray, shape (n_time, 2, n_position_bins)
    movement_state_transition : ndarray, shape (n_position_bins,
                                                n_position_bins)
    discrete_state_transition : ndarray, shape (n_time, 2)
        discrete_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
        discrete_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
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
    """  # noqa
    filter_probability = np.sum(filter_posterior, axis=2)

    smoother_posterior = np.zeros_like(filter_posterior)
    n_time, _, n_position_bins = filter_posterior.shape

    smoother_posterior[-1] = filter_posterior[-1]

    for k in np.arange(n_time - 2, -1, -1):
        smoother_prior = np.zeros((2, n_position_bins))
        weights = np.zeros((2, n_position_bins))
        position_ind = observed_position_bin[k + 1]

        # Predict p(x_{k + 1}, I_{k + 1} \vert H_{1:k})
        # I_{k} = 0, I_{k + 1} = 0
        smoother_prior[0, position_ind] = (
            1 - discrete_state_transition[k + 1, 0]
        ) * filter_probability[k, 0]

        # I_{k} = 1, I_{k + 1} = 0
        smoother_prior[0, position_ind] += (
            1 - discrete_state_transition[k + 1, 1]
        ) * filter_probability[k, 1]

        # I_{k} = 0, I_{k + 1} = 1
        smoother_prior[1] = (
            discrete_state_transition[k + 1, 0] * uniform * filter_probability[k, 0]
        )

        # I_{k} = 1, I_{k + 1} = 1
        smoother_prior[1] += discrete_state_transition[k + 1, 1] * (
            movement_state_transition.T @ filter_posterior[k, 1]
        )

        # Update p(x_{k}, I_{k} \vert H_{1:k})
        ratio = np.exp(
            np.log(smoother_posterior[k + 1]) - np.log(smoother_prior + np.spacing(1))
        )
        integrated_ratio = np.sum(ratio, axis=1)
        # I_{k} = 0, I_{k + 1} = 0
        weights[0] = (1 - discrete_state_transition[k + 1, 0]) * ratio[0, position_ind]

        # I_{k} = 0, I_{k + 1} = 1
        weights[0] += (
            uniform * discrete_state_transition[k + 1, 0] * integrated_ratio[1]
        )

        # I_{k} = 1, I_{k + 1} = 0
        weights[1] = (1 - discrete_state_transition[k + 1, 1]) * ratio[0, position_ind]

        # I_{k} = 1, I_{k + 1} = 1
        weights[1] += (
            discrete_state_transition[k + 1, 1] * ratio[1] @ movement_state_transition
        )

        smoother_posterior[k] = normalize_to_probability(weights * filter_posterior[k])

    smoother_probability = np.sum(smoother_posterior, axis=2)

    return smoother_posterior, smoother_probability


def scale_likelihood(log_likelihood, axis=(1, 2)):
    """Scales the likelihood to its max value to prevent overflow and underflow.

    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_states, n_position_bins)

    Returns
    -------
    scaled_likelihood : ndarray, shape (n_time, n_states, n_position_bins)

    """
    max_log_likelihood = np.nanmax(log_likelihood, axis=axis, keepdims=True)
    # If maximum is infinity, set to zero
    if max_log_likelihood.ndim > 0:
        max_log_likelihood[~np.isfinite(max_log_likelihood)] = 0.0
    elif not np.isfinite(max_log_likelihood):
        max_log_likelihood = 0.0

    # Maximum likelihood is always 1
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    # avoid zero likelihood
    likelihood += np.spacing(1, dtype=likelihood.dtype)

    return likelihood


def check_converged(loglik, previous_loglik, tolerance=1e-4):
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.spacing(1)) / 2

    is_increasing = loglik - previous_loglik >= -1e-3
    is_converged = (delta_loglik / avg_loglik) < tolerance

    return is_converged, is_increasing


try:
    import cupy as cp

    def _causal_classifier_gpu(
        likelihood,
        movement_state_transition,
        discrete_state_transition,
        observed_position_bin,
        uniform,
    ):
        """
        Parameters
        ----------
        likelihood : ndarray, shape (n_time, ...)
        movement_state_transition : ndarray, shape (n_position_bins,
                                                    n_position_bins)
        discrete_state_transition : ndarray, shape (n_time, 2)
            discrete_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
            discrete_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
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

        """

        likelihood = cp.asarray(likelihood, dtype=cp.float32)
        movement_state_transition = cp.asarray(
            movement_state_transition, dtype=cp.float32
        )
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32
        )
        observed_position_bin = cp.asarray(observed_position_bin)
        uniform = cp.asarray(uniform, dtype=cp.float32)

        n_position_bins = movement_state_transition.shape[0]
        n_time = likelihood.shape[0]
        n_states = 2

        posterior = cp.zeros((n_time, n_states, n_position_bins), dtype=cp.float32)
        state_probability = cp.zeros((n_time, n_states), dtype=cp.float32)

        # Initial Conditions
        posterior[0, 0, observed_position_bin[0]] = likelihood[
            0, 0, observed_position_bin[0]
        ]
        norm = cp.nansum(posterior[0])
        data_log_likelihood = cp.log(norm)
        posterior[0] /= norm
        state_probability[0] = cp.sum(posterior[0], axis=1)

        for k in np.arange(1, n_time):
            prior = cp.zeros((n_states, n_position_bins), dtype=cp.float32)
            position_ind = observed_position_bin[k]
            # I_{k - 1} = 0, I_{k} = 0
            prior[0, position_ind] = (
                1 - discrete_state_transition[k, 0]
            ) * state_probability[k - 1, 0]
            # I_{k - 1} = 1, I_{k} = 0
            prior[0, position_ind] += (
                1 - discrete_state_transition[k, 1]
            ) * state_probability[k - 1, 1]

            # I_{k - 1} = 0, I_{k} = 1
            prior[1] = (
                discrete_state_transition[k, 0] * uniform * state_probability[k - 1, 0]
            )
            # I_{k - 1} = 1, I_{k} = 1
            prior[1] += discrete_state_transition[k, 1] * (
                movement_state_transition.T @ posterior[k - 1, 1]
            )

            posterior[k] = prior * likelihood[k]
            norm = cp.nansum(posterior[k])
            data_log_likelihood += cp.log(norm)
            posterior[k] /= norm

            state_probability[k] = cp.sum(posterior[k], axis=1)

        return (
            cp.asnumpy(posterior),
            cp.asnumpy(state_probability),
            data_log_likelihood,
        )

    def _acausal_classifier_gpu(
        filter_posterior,
        movement_state_transition,
        discrete_state_transition,
        observed_position_bin,
        uniform,
    ):
        """
        Parameters
        ----------
        filter_posterior : ndarray, shape (n_time, 2, n_position_bins)
        movement_state_transition : ndarray, shape (n_position_bins,
                                                    n_position_bins)
        discrete_state_transition : ndarray, shape (n_time, 2)
            discrete_state_transition[k, 0] = Pr(I_{k} = 1 | I_{k-1} = 0, v_{k})
            discrete_state_transition[k, 1] = Pr(I_{k} = 1 | I_{k-1} = 1, v_{k})
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
        """  # noqa

        filter_posterior = cp.asarray(filter_posterior, dtype=cp.float32)
        movement_state_transition = cp.asarray(
            movement_state_transition, dtype=cp.float32
        )
        discrete_state_transition = cp.asarray(
            discrete_state_transition, dtype=cp.float32
        )
        observed_position_bin = cp.asarray(observed_position_bin)
        uniform = cp.asarray(uniform, dtype=cp.float32)
        EPS = cp.asarray(np.spacing(1), dtype=cp.float32)

        filter_probability = cp.sum(filter_posterior, axis=2)

        smoother_posterior = cp.zeros_like(filter_posterior)
        n_time, _, n_position_bins = filter_posterior.shape

        smoother_posterior[-1] = filter_posterior[-1]

        for k in cp.arange(n_time - 2, -1, -1):
            smoother_prior = cp.zeros((2, n_position_bins), dtype=cp.float32)
            weights = cp.zeros((2, n_position_bins), dtype=cp.float32)

            position_ind = observed_position_bin[k + 1]

            # Predict p(x_{k + 1}, I_{k + 1} \vert H_{1:k})
            # I_{k} = 0, I_{k + 1} = 0
            smoother_prior[0, position_ind] = (
                1 - discrete_state_transition[k + 1, 0]
            ) * filter_probability[k, 0]

            # I_{k} = 1, I_{k + 1} = 0
            smoother_prior[0, position_ind] += (
                1 - discrete_state_transition[k + 1, 1]
            ) * filter_probability[k, 1]

            # I_{k} = 0, I_{k + 1} = 1
            smoother_prior[1] = (
                discrete_state_transition[k + 1, 0] * uniform * filter_probability[k, 0]
            )

            # I_{k} = 1, I_{k + 1} = 1
            smoother_prior[1] += discrete_state_transition[k + 1, 1] * (
                movement_state_transition.T @ filter_posterior[k, 1]
            )

            # Update p(x_{k}, I_{k} \vert H_{1:k})
            ratio = cp.exp(
                cp.log(smoother_posterior[k + 1]) - cp.log(smoother_prior + EPS)
            )
            integrated_ratio = cp.sum(ratio, axis=1)
            # I_{k} = 0, I_{k + 1} = 0
            weights[0] = (1 - discrete_state_transition[k + 1, 0]) * ratio[
                0, position_ind
            ]

            # I_{k} = 0, I_{k + 1} = 1
            weights[0] += (
                uniform * discrete_state_transition[k + 1, 0] * integrated_ratio[1]
            )

            # I_{k} = 1, I_{k + 1} = 0
            weights[1] = (1 - discrete_state_transition[k + 1, 1]) * ratio[
                0, position_ind
            ]

            # I_{k} = 1, I_{k + 1} = 1
            weights[1] += (
                discrete_state_transition[k + 1, 1]
                * ratio[1]
                @ movement_state_transition
            )

            smoother_posterior[k] = weights * filter_posterior[k]
            smoother_posterior[k] /= cp.nansum(smoother_posterior[k])

        smoother_probability = cp.sum(smoother_posterior, axis=2)

        return (cp.asnumpy(smoother_posterior), cp.asnumpy(smoother_probability))

except ImportError:

    def _causal_classifier_gpu(*args, **kwargs):
        pass

    def _acausal_classifier_gpu(*args, **kwargs):
        pass
