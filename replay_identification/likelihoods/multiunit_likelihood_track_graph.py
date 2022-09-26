import dask
import networkx as nx
import numba
import numpy as np
from scipy.stats import multivariate_normal

SQRT_2PI = np.float64(np.sqrt(2.0 * np.pi))


def _setup_distance(linear_position, nodes_df):
    linear_position = linear_position.squeeze()
    bin_ind = np.searchsorted(nodes_df.linear_position.values, linear_position)
    is_same_edge = (
        nodes_df.iloc[bin_ind - 1].edge_id.values
        == nodes_df.iloc[bin_ind].edge_id.values
    )

    left_node_ind = bin_ind - 1
    right_node_ind = bin_ind

    right_node_ind[~is_same_edge] = left_node_ind[~is_same_edge] = np.argmin(
        np.abs(
            linear_position[~is_same_edge, np.newaxis] - nodes_df.linear_position.values
        ),
        axis=1,
    )

    left_node_id = nodes_df.node_ids.values[left_node_ind]
    right_node_id = nodes_df.node_ids.values[right_node_ind]
    left_dist = np.abs(nodes_df.linear_position.values[left_node_ind] - linear_position)
    right_dist = np.abs(
        nodes_df.linear_position.values[left_node_ind] - linear_position
    )

    return left_node_id, right_node_id, left_dist, right_dist


def get_distance2(
    track_graph,
    left_node_id,
    right_node_id,
    left_dist,
    right_dist,
    place_bin_center_to_node_id,
):
    nx.add_path(track_graph, [left_node_id, "a"], distance=left_dist)
    nx.add_path(track_graph, ["a", right_node_id], distance=right_dist)
    dist = nx.single_source_dijkstra_path_length(
        track_graph, source="a", weight="distance"
    )
    track_graph.remove_node("a")
    r = np.array([dist[id] for id in place_bin_center_to_node_id])

    return r


@dask.delayed
def batch_distance(
    track_graph,
    left_node_id,
    right_node_id,
    left_dist,
    right_dist,
    place_bin_center_to_node_id,
):
    copy_graph = track_graph.copy()

    return np.stack(
        [
            get_distance2(copy_graph, l_id, r_id, l_d, r_d, place_bin_center_to_node_id)
            for l_id, r_id, l_d, r_d in zip(
                left_node_id, right_node_id, left_dist, right_dist
            )
        ],
        axis=0,
    )


def batch(n_samples, batch_size=1):
    for ind in range(0, n_samples, batch_size):
        yield range(ind, min(ind + batch_size, n_samples))


def convert_linear_position_to_track_distances(
    linear_position, track_graph, nodes_df, place_bin_center_to_node_id
):
    """

    Parameters
    ----------

    Returns
    -------
    track_distances : np.ndarray, shape (n_time, n_place_bins)

    """
    (left_node_id, right_node_id, left_dist, right_dist) = _setup_distance(
        linear_position, nodes_df
    )

    n_time = linear_position.shape[0]
    track_distances = []
    for time_ind in batch(n_time, batch_size=10_000):
        track_distances.append(
            batch_distance(
                track_graph,
                left_node_id[time_ind],
                right_node_id[time_ind],
                left_dist[time_ind],
                right_dist[time_ind],
                place_bin_center_to_node_id,
            )
        )

    return np.concatenate(dask.compute(*track_distances, scheduler="processes"), axis=0)


def get_gaussian_track_distances(track_distances, variance=8):
    return (
        multivariate_normal(mean=0, cov=variance)
        .pdf(track_distances.ravel())
        .reshape(track_distances.shape)
    )


@numba.njit(nogil=True, cache=False, parallel=True, error_model="numpy")
def numba_product(eval_point, samples, bandwidths):
    """
    Parameters
    ----------
    eval_point : np.ndarray, shape (n_marks,)
    samples : np.ndarray, shape (n_train, n_marks)
    bandwidths : np.ndarray, shape (n_marks,)
    Returns
    -------
    product_kernel : shape (n_train,)

    """
    n_samples, n_bandwidths = samples.shape
    product_kernel = np.ones((n_samples,))

    for j in numba.prange(n_samples):
        for k in range(n_bandwidths):
            bandwidth = bandwidths[k]
            sample = samples[j, k]
            product_kernel[j] *= (
                np.exp(-0.5 * ((eval_point[k] - sample) / bandwidth) ** 2)
                / (bandwidth * SQRT_2PI)
            ) / bandwidth

    return product_kernel


def get_kde(
    test_multiunit,
    train_multiunit,
    is_track_interior,
    bandwidths,
    gaussian_track_distances,
):
    """

    Parameters
    ----------
    test_multiunit : np.ndarray, shape (n_test, n_marks)
    train_multiunit : np.ndarray, shape (n_train, n_marks)
    is_track_interior : np.ndarray, shape (n_bins, 1)
    bandwidths : np.ndarray, shape (n_marks,)
    gaussian_track_distances : np.ndarray, shape (n_train, n_bins)

    Returns
    -------
    kde : np.ndarray, shape (n_test, n_bins)

    """
    n_test, n_bins = test_multiunit.shape[0], gaussian_track_distances.shape[1]
    kde = np.zeros((n_test, n_bins))
    for ind in range(n_test):
        kde[ind, is_track_interior] = (
            numba_product(test_multiunit[ind], train_multiunit, bandwidths)
            @ gaussian_track_distances
        )

    return kde


@numba.njit(nogil=True, cache=False, parallel=True)
def numba_kde(eval_points, samples, bandwidths, precalculated_kernel):
    n_eval_points, n_bandwidths = eval_points.shape
    n_grid = precalculated_kernel.shape[1]
    result = np.zeros((n_eval_points, n_grid))
    n_samples = len(samples)

    for eval_ind in range(n_eval_points):
        for sample_ind in range(n_samples):
            product_kernel = 1.0
            for bandwidth_ind in range(n_bandwidths):
                bandwidth = bandwidths[bandwidth_ind]
                eval_point = eval_points[eval_ind, bandwidth_ind]
                sample = samples[sample_ind, bandwidth_ind]
                product_kernel *= (
                    np.exp(-0.5 * ((eval_point - sample) / bandwidth) ** 2)
                    / (bandwidth * SQRT_2PI)
                ) / bandwidth
            for grid_ind in range(n_grid):
                result[eval_ind, grid_ind] += (
                    product_kernel
                    * precalculated_kernel[sample_ind, grid_ind]
                    / n_samples
                )

    return result


@numba.njit(nogil=True, cache=False, parallel=True, fastmath=True)
def gaussian_kernel(eval_point, bandwidths, sample, n_bandwidths):
    product_kernel = 1.0
    for k in range(n_bandwidths):
        bandwidth = bandwidths[k]
        product_kernel *= (
            np.exp(-0.5 * ((eval_point[k] - sample[k]) / bandwidth) ** 2)
            / (bandwidth * SQRT_2PI)
        ) / bandwidth
    return product_kernel
