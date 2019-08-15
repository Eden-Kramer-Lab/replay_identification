import numpy as np
from numba import jit
from scipy import ndimage


def get_n_bins(position, bin_size=2.5, position_range=None):
    '''Get number of bins need to span a range given a bin size.

    Parameters
    ----------
    position : ndarray, shape (n_time,)
    bin_size : float, optional

    Returns
    -------
    n_bins : int

    '''
    if position_range is not None:
        extent = np.diff(position_range, axis=1).squeeze()
    else:
        extent = np.ptp(position, axis=0)
    return np.ceil(extent / bin_size).astype(np.int)


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x


def get_centers(edge):
    return edge[:-1] + np.diff(edge) / 2


def add_zero_end_bins(hist, edges):
    new_edges = []

    for edge_ind, edge in enumerate(edges):
        bin_size = np.diff(edge)[0]
        try:
            if hist.sum(axis=edge_ind)[0] != 0:
                edge = np.insert(edge, 0, edge[0] - bin_size)
            if hist.sum(axis=edge_ind)[-1] != 0:
                edge = np.append(edge, edge[-1] + bin_size)
        except IndexError:
            if hist[0] != 0:
                edge = np.insert(edge, 0, edge[0] - bin_size)
            if hist[-1] != 0:
                edge = np.append(edge, edge[-1] + bin_size)
        new_edges.append(edge)

    return new_edges


def get_grid(position, bin_size=2.5, position_range=None,
             infer_track_interior=True):
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    n_bins = get_n_bins(position, bin_size, position_range)
    hist, edges = np.histogramdd(position, bins=n_bins, range=position_range)
    if infer_track_interior:
        edges = add_zero_end_bins(hist, edges)
    mesh_edges = np.meshgrid(*edges)
    place_bin_edges = np.stack([edge.ravel() for edge in mesh_edges], axis=1)

    mesh_centers = np.meshgrid(
        *[get_centers(edge) for edge in edges])
    place_bin_centers = np.stack(
        [center.ravel() for center in mesh_centers], axis=1)
    centers_shape = mesh_centers[0].shape

    return edges, place_bin_edges, place_bin_centers, centers_shape


def replace_NaN(x):
    x[np.isnan(x)] = 1
    return x


def return_None(*args, **kwargs):
    return None


@jit(nopython=True, cache=True, nogil=True)
def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)


def get_observed_position_bin(position, bin_edges):
    position = position.squeeze()
    is_too_big = position >= bin_edges[-1]
    bin_size = np.diff(bin_edges, axis=0)[0][0]
    position[is_too_big] = position[is_too_big] - (bin_size / 2)
    return np.digitize(position.squeeze(), bin_edges.squeeze()) - 1


@jit(nopython=True, cache=True, nogil=True)
def _filter(likelihood, movement_state_transition, replay_state_transition,
            observed_position_bin):
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
    posterior : ndarray, shape (n_time, 3, n_position_bins)
    state_probability : ndarray, shape (n_time, 2)
        state_probability[:, 0] = Pr(I_{1:T} = 0)
        state_probability[:, 1] = Pr(I_{1:T} = 1)
    prior : ndarray, shape (n_time, 2, n_position_bins)

    '''
    n_time, n_states, n_bins = likelihood.shape

    posterior = np.zeros((n_time, n_states, n_bins))
    prior = np.zeros_like(posterior)
    uniform = 1 / n_bins  # exclude places where there is no position
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
            (movement_state_transition @ posterior[k - 1, 1]))

        posterior[k] = normalize_to_probability(
            prior[k] * likelihood[k])

        state_probability[k] = np.sum(posterior[k], axis=1)

    return posterior, state_probability, prior


@jit(nopython=True, cache=True, nogil=True)
def _smoother(filter_posterior, movement_state_transition,
              replay_state_transition, observed_position_bin):
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
    '''
    filter_probability = np.sum(filter_posterior, axis=2)

    smoother_posterior = np.zeros_like(filter_posterior)
    smoother_prior = np.zeros_like(filter_posterior)
    weights = np.zeros_like(filter_posterior)
    n_time, _, n_position_bins = filter_posterior.shape
    uniform = 1 / n_position_bins

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
            (movement_state_transition @ filter_posterior[k, 1]))

        # Update p(x_{k}, I_{k} \vert H_{1:k})
        ratio = np.exp(
            np.log(smoother_posterior[k + 1] + np.spacing(1)) -
            np.log(smoother_prior[k]) + np.spacing(1))
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


@jit(nopython=True, nogil=True)
def _causal_classify(initial_conditions, continuous_state_transition,
                     discrete_state_transition, likelihood,
                     observed_position_bin):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)
    likelihood : ndarray, shape (n_time, n_states, n_bins, 1)

    Returns
    -------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    n_time, n_states, n_bins, _ = likelihood.shape
    posterior = np.zeros_like(likelihood)

    posterior[0] = normalize_to_probability(
        initial_conditions.copy() * likelihood[0])

    for k in np.arange(1, n_time):
        prior = np.zeros((n_states, n_bins, 1))
        position_ind = observed_position_bin[k]
        continuous_state_transition[:, 0] = 0.0
        continuous_state_transition[:, 0, :, position_ind] = 1.0

        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                prior[state_k, :] += (
                    discrete_state_transition[k, state_k_1, state_k] *
                    continuous_state_transition[state_k_1, state_k] @
                    posterior[k - 1, state_k_1])
        posterior[k] = normalize_to_probability(prior * likelihood[k])

    return posterior


@jit(nopython=True, nogil=True)
def _acausal_classify(causal_posterior, continuous_state_transition,
                      discrete_state_transition, observed_position_bin):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_states, n_bins, _ = causal_posterior.shape

    for k in np.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = np.zeros((n_states, n_bins, 1))
        position_ind = observed_position_bin[k]
        continuous_state_transition[0, 0] = 0.0
        continuous_state_transition[1, 0] = 0.0
        continuous_state_transition[0, 0, :, position_ind] = 1.0
        continuous_state_transition[1, 0, :, position_ind] = 1.0
        for state_k_1 in np.arange(n_states):
            for state_k in np.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[k, state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    causal_posterior[k, state_k])

        # Backwards Update
        weights = np.zeros((n_states, n_bins, 1))
        ratio = np.exp(
            np.log(acausal_posterior[k + 1] + np.spacing(1)) -
            np.log(prior + np.spacing(1)))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[k, state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    ratio[state_k_1])

        acausal_posterior[k] = normalize_to_probability(
            weights * causal_posterior[k])

    return acausal_posterior


def get_track_interior(position, bins):
    '''

    position : ndarray, shape (n_time, n_position_dims)
    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    '''
    bin_counts, edges = np.histogramdd(position, bins=bins)
    is_maze = (bin_counts > 0).astype(int)
    n_position_dims = position.shape[1]
    structure = np.ones([1] * n_position_dims)
    is_maze = ndimage.binary_closing(is_maze, structure=structure)
    is_maze = ndimage.binary_fill_holes(is_maze)
    return ndimage.binary_dilation(is_maze, structure=structure)


@jit(nopython=True, nogil=True, parallel=True)
def _causal_classify2(continuous_likelihood, discrete_likelihood,
                      continuous_state_transition,
                      discrete_state_transition):
    '''

    Parameters
    ----------
    continuous_likelihood : ndarray, shape (n_time, n_continuous_states,
                                            n_bins)
    discrete_likelihood : ndarray, shape (n_time, n_discrete_states)
    continuous_state_transition : ndarray, shape (n_continuous_states,
                                                  n_continuous_states, n_bins,
                                                  n_bins)
    discrete_state_transition : ndarray, shape (n_time,
                                                n_discrete_states +
                                                n_continuous_states,
                                                n_discrete_states +
                                                n_continuous_states)

    Returns
    -------
    continuous_posterior : ndarray, shape (n_time, n_continuous_states, n_bins)
    state_probability : ndarray, shape (n_time, n_discrete_states +
                                        n_continuous_states)

    '''
    n_time, n_continuous_states, n_bins = continuous_likelihood.shape
    n_discrete_states = discrete_likelihood.shape[1]

    continuous_posterior = np.zeros((n_time, n_continuous_states, n_bins))
    uniform = 1.0 / n_bins

    state_probability = np.zeros(
        (n_time, n_discrete_states + n_continuous_states))
    # 0 = Local, 1 = No Spike, 2 = Non-Local

    # Initial Conditions
    state_probability[0, 0] = 1.0

    for k in np.arange(1, n_time):
        # discrete -> discrete or continuous -> discrete
        for to_state_ind in range(n_discrete_states):
            for from_state_ind in range(
                    n_discrete_states + n_continuous_states):
                # I_{k - 1} = from_state_ind, I_{k} = to_state_ind
                state_probability[k, to_state_ind] += (
                    discrete_state_transition[k, from_state_ind, to_state_ind]
                    * state_probability[k - 1, from_state_ind])

        # discrete -> continuous
        for to_state_ind in range(n_continuous_states):
            for from_state_ind in range(n_discrete_states):
                continuous_posterior[k, to_state_ind] += (
                    uniform * discrete_state_transition[
                        k, from_state_ind, n_discrete_states + to_state_ind] *
                    state_probability[k - 1, from_state_ind])

        # continuous -> continuous
        for to_state_ind in range(n_continuous_states):
            for from_state_ind in range(n_continuous_states):
                continuous_posterior[k, to_state_ind, :] += (
                    discrete_state_transition[
                        k, n_discrete_states + from_state_ind,
                        n_discrete_states + to_state_ind]
                    * continuous_state_transition[from_state_ind, to_state_ind]
                    @ continuous_posterior[k - 1, from_state_ind])

        continuous_posterior[k] *= continuous_likelihood[k]
        state_probability[k, :n_discrete_states] *= discrete_likelihood[k]

        # Normalize
        denominator = (np.sum(state_probability[k, :n_discrete_states]) +
                       np.sum(continuous_posterior[k]))

        continuous_posterior[k] /= denominator
        state_probability[k, :n_discrete_states] /= denominator
        state_probability[k, n_discrete_states:] = np.sum(
            continuous_posterior[k], axis=1)

    return continuous_posterior, state_probability


def _acausal_classify2(causal_continuous_posterior, causal_state_probability,
                       continuous_state_transition, discrete_state_transition):
    '''

    Parameters
    ----------
    causal_continuous_posterior : ndarray, shape (n_time, n_continuous_states,
                                                  n_bins)
    causal_state_probability : ndarray, shape (n_time, n_discrete_states +
                                               n_continuous_states)
    continuous_state_transition : ndarray, shape (n_continuous_states,
                                                  n_continuous_states, n_bins,
                                                  n_bins)
    discrete_state_transition : ndarray, shape (n_time,
                                                n_discrete_states +
                                                n_continuous_states,
                                                n_discrete_states +
                                                n_continuous_states)

    Returns
    -------
    acausal_posterior : ndarray, shape (n_time, n_continuous_states, n_bins)
    acausal_state_probability : ndarray, shape (n_time, n_discrete_states +
                                                n_continuous_states)

    '''
    acausal_continuous_posterior = np.zeros_like(causal_continuous_posterior)
    acausal_continuous_posterior[-1] = causal_continuous_posterior[-1].copy()
    n_time, n_continuous_states, n_bins = causal_continuous_posterior.shape

    acausal_state_probability = np.zeros_like(causal_state_probability)
    acausal_state_probability[-1] = causal_state_probability[-1].copy()
    n_discrete_states = causal_state_probability.shape[1] - n_continuous_states

    continuous_prior = np.zeros((n_continuous_states, n_bins))
    uniform = 1.0 / n_bins

    for k in np.arange(n_time - 2, -1, -1):
        # Discrete/Continuous -> Discrete (n_discrete_states,)
        state_probability_prior = (
            causal_state_probability[k] @
            discrete_state_transition[k, :, n_discrete_states])
        # Discrete -> Continuous
        continuous_prior = (
            uniform * causal_state_probability[k, :n_discrete_states] @
            discrete_state_transition[k, :n_discrete_states,
                                      n_discrete_states:])
        # Continuous -> Continuous
        for from_state_ind in range(n_continuous_states):
            for to_state_ind in range(n_continuous_states):
                continuous_prior[to_state_ind, :] += (
                    discrete_state_transition[
                        k, n_discrete_states + from_state_ind,
                        n_discrete_states + to_state_ind]
                    * continuous_state_transition[from_state_ind, to_state_ind]
                    @ causal_continuous_posterior[k, from_state_ind])

        # Backwards Update
        # I_{k} = 0, I_{k+1} = 0 (discrete -> discrete)
        acausal_state_probability[k, 0] += (
            discrete_state_transition[k, 0, 0] *
            acausal_state_probability[k + 1, 0] / state_probability_prior[0])
        # I_{k} = 0, I_{k+1} = 1 (discrete -> discrete)
        acausal_state_probability[k, 0] += (
            discrete_state_transition[k, 0, 1] *
            acausal_state_probability[k + 1, 1] / state_probability_prior[1]
        )
        # I_{k} = 0, I_{k+1} = 2 (discrete -> continuous)
        acausal_state_probability[k, 0] += (
            uniform * discrete_state_transition[k, 0, 2] *
            acausal_state_probability[k + 1, 2] / state_probability_prior[2]
        )

        # I_{k} = 1, I_{k+1} = 0
        acausal_state_probability[k, 1] += (
            discrete_state_transition[k, 1, 0] *
            acausal_state_probability[k + 1, 0] / state_probability_prior[0]
        )
        # I_{k} = 1, I_{k+1} = 1
        acausal_state_probability[k, 1] += (
            discrete_state_transition[k, 1, 1] *
            acausal_state_probability[k + 1, 1] / state_probability_prior[1]
        )
        # I_{k} = 1, I_{k+1} = 2
        acausal_state_probability[k, 1] += (
            uniform * discrete_state_transition[k, 1, 2] *
            acausal_state_probability[k + 1, 2] / state_probability_prior[2]
        )

        # I_{k} = 2, I_{k+1} = 0
        acausal_state_probability[k, 2] += (
            discrete_state_transition[k, 2, 0] *
            acausal_state_probability[k + 1, 0] / state_probability_prior[0]
        )

        # I_{k} = 2, I_{k+1} = 1
        acausal_state_probability[k, 2] += (
            discrete_state_transition[k, 2, 1] *
            acausal_state_probability[k + 1, 1] / state_probability_prior[1]
        )

        # I_{k} = 2, I_{k+1} = 2
        acausal_state_probability[k, 2] += (
            discrete_state_transition[k, 2, 1] *
            continuous_state_transition[0, 0] @
            acausal_continuous_posterior[k + 1, 2] / continuous_prior[0]
        )
