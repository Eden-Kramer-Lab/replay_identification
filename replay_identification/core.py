import networkx as nx
import numpy as np
from numba import njit
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


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


def order_border(border):
    '''
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    '''
    n_points = border.shape[0]
    clf = NearestNeighbors(2).fit(border)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    paths = [list(nx.dfs_preorder_nodes(T, i))
             for i in range(n_points)]
    min_idx, min_dist = 0, np.inf

    for idx, path in enumerate(paths):
        ordered = border[path]    # ordered nodes
        cost = np.sum(np.diff(ordered) ** 2)
        if cost < min_dist:
            min_idx, min_dist = idx, cost

    opt_order = paths[min_idx]
    return border[opt_order][:-1]


def get_track_border(is_maze, edges):
    '''

    Parameters
    ----------
    is_maze : ndarray, shape (n_x_bins, n_y_bins)
    edges : list of ndarray

    '''
    structure = ndimage.generate_binary_structure(2, 2)
    border = ndimage.binary_dilation(is_maze, structure=structure) ^ is_maze

    inds = np.nonzero(border)
    centers = [get_centers(x) for x in edges]
    border = np.stack([center[ind] for center, ind in zip(centers, inds)],
                      axis=1)
    return order_border(border)


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


def get_observed_position_bin(position, bin_edges):
    position = position.squeeze()
    is_too_big = position >= bin_edges[-1]
    bin_size = np.diff(bin_edges, axis=0)[0][0]
    position[is_too_big] = position[is_too_big] - (bin_size / 2)
    return np.digitize(position.squeeze(), bin_edges.squeeze()) - 1


@njit(cache=True, nogil=True)
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
    uniform = 1 / n_position_bins
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


@njit(cache=True, nogil=True)
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
