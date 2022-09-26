import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from statsmodels.api import GLM, families

from replay_identification.bins import atleast_2d


def estimate_movement_std(position):
    """Estimates the movement standard deviation based on position.

    WARNING: Need to use on original position, not interpolated position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dim)

    Returns
    -------
    movement_std : ndarray, shape (n_position_dim,)

    """
    position = atleast_2d(position)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_std = []
    for p in position.T:
        fit = GLM(p[:-1], p[1:], family=families.Gaussian()).fit()
        movement_std.append(np.sqrt(fit.scale))
    return np.array(movement_std)


def _normalize_row_probability(x):
    """Ensure the state transition matrix rows sum to 1"""
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x


def _fix_zero_bins(movement_bins):
    """If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    """
    n_bins = movement_bins.shape[0]
    movement_bins[movement_bins.sum(axis=1) == 0] = 1 / n_bins
    return movement_bins


def empirical_movement(
    position, edges, is_training=None, replay_speed=20, position_extent=None
):
    """Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `replay_speed`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dims)
    edges : sequence
        A sequence of arrays describing the bin edges along each dimension.
    is_training : None or bool ndarray, shape (n_time,), optional
    replay_speed : int, optional
        How much the movement is sped-up during a replay event
    position_extent : sequence, optional
        A sequence of `n_position_dims`, each an optional (lower, upper)
        tuple giving the outer bin edges for position.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of
        `n_position_dims` None values.

    Returns
    -------
    transition_matrix : ndarray, shape (n_position_bins, n_position_bins)

    """
    if is_training is None:
        is_training = np.ones((position.shape[0]), dtype=np.bool)
    position = atleast_2d(position)[is_training]
    movement_bins, _ = np.histogramdd(
        np.concatenate((position[1:], position[:-1]), axis=1),
        bins=edges * 2,
        range=position_extent,
    )
    original_shape = movement_bins.shape
    n_position_dims = position.shape[1]
    shape_2d = np.product(original_shape[:n_position_dims])
    movement_bins = _normalize_row_probability(
        movement_bins.reshape((shape_2d, shape_2d), order="F")
    )
    movement_bins = np.linalg.matrix_power(movement_bins, replay_speed)

    return movement_bins


def random_walk(place_bin_centers, movement_var, is_track_interior, replay_speed=20):
    """Zero mean random walk.

    This version makes the sped up gaussian without constraints and then
    imposes the constraints.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins, n_position_dims)
    movement_var : float,
    is_track_interior : bool ndarray, shape (n_x_bins, n_y_bins)
    replay_speed : int

    Returns
    -------
    transition_matrix : ndarray, shape (n_bins, n_bins)

    """
    transition_matrix = np.stack(
        [
            multivariate_normal(mean=bin, cov=movement_var * replay_speed).pdf(
                place_bin_centers
            )
            for bin in place_bin_centers
        ],
        axis=1,
    )
    is_track_interior = is_track_interior.ravel(order="F")
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0

    return _normalize_row_probability(transition_matrix)


def _center_arm(row, bin_labels, before_gaussian, after_gaussian):
    arm_ind = bin_labels.loc[(bin_labels == "Right Arm")].index
    n_samples = min(arm_ind.size, after_gaussian.size)
    row[arm_ind[:n_samples]] = after_gaussian[:n_samples] / 2

    arm_ind = bin_labels.loc[(bin_labels == "Left Arm")].index
    n_samples = min(arm_ind.size, after_gaussian.size)
    row[arm_ind[:n_samples]] = after_gaussian[:n_samples] / 2
    return row


def _left_arm(row, bin_labels, before_gaussian, after_gaussian):
    arm_ind = bin_labels.loc[(bin_labels == "Center Arm")].index
    n_samples = min(arm_ind.size, before_gaussian.size)
    row[arm_ind[-n_samples:]] = before_gaussian[-n_samples:] / 2

    arm_ind = bin_labels.loc[(bin_labels == "Right Arm")].index
    n_samples = min(arm_ind.size, before_gaussian.size)
    row[arm_ind[:n_samples]] = before_gaussian[-n_samples:][::-1] / 2
    return row


def _right_arm(row, bin_labels, before_gaussian, after_gaussian):
    arm_ind = bin_labels.loc[(bin_labels == "Center Arm")].index
    n_samples = min(arm_ind.size, before_gaussian.size)
    row[arm_ind[-n_samples:]] = before_gaussian[-n_samples:] / 2

    arm_ind = bin_labels.loc[(bin_labels == "Left Arm")].index
    n_samples = min(arm_ind.size, before_gaussian.size)
    row[arm_ind[:n_samples]] = before_gaussian[-n_samples:][::-1] / 2
    return row


_ARM_FUNCS = {
    "Left Arm": _left_arm,
    "Right Arm": _right_arm,
    "Center Arm": _center_arm,
}


def w_track_1D_random_walk(
    position,
    place_bin_edges,
    place_bin_centers,
    labels,
    movement_var,
    is_track_interior,
    replay_speed=200,
):
    position = position.squeeze()
    place_bin_edges = place_bin_edges.squeeze()
    place_bin_centers = place_bin_centers.squeeze()
    bin_inds = np.digitize(position, bins=place_bin_edges) - 1
    n_bins = place_bin_edges.size - 1
    bin_inds[bin_inds >= n_bins] = n_bins - 1
    bin_labels = pd.Series(labels).groupby(bin_inds).unique().apply(lambda s: s[0])

    transition_matrix = []

    for bin_ind in np.arange(n_bins):
        bin_center = place_bin_centers[bin_ind]
        row = np.zeros((n_bins,))
        try:
            bin_label = bin_labels.loc[bin_ind]
            is_same_bin = bin_labels == bin_label
            is_same_bin_ind = is_same_bin[is_same_bin].index
            is_same_bin = np.zeros_like(place_bin_centers.squeeze(), dtype=np.bool)
            is_same_bin[is_same_bin_ind] = True

            gaussian = multivariate_normal(
                mean=bin_center, cov=movement_var * replay_speed
            ).pdf(place_bin_centers)
            row[is_same_bin] = gaussian[is_same_bin]
            before_gaussian = gaussian[: np.nonzero(is_same_bin)[0][0]]
            after_gaussian = gaussian[np.nonzero(is_same_bin)[0][-1] :]
            row = _ARM_FUNCS[bin_label](
                row, bin_labels, before_gaussian, after_gaussian
            )
        except KeyError:
            pass
        transition_matrix.append(row)

    transition_matrix = np.stack(transition_matrix, axis=1)
    is_track_interior = is_track_interior.ravel(order="F")
    transition_matrix[~is_track_interior] = 0.0
    transition_matrix[:, ~is_track_interior] = 0.0
    return _normalize_row_probability(transition_matrix)


def random_walk_on_track_graph(
    place_bin_centers,
    movement_mean,
    movement_var,
    place_bin_center_ind_to_node,
    distance_between_nodes,
):
    state_transition = np.zeros((place_bin_centers.size, place_bin_centers.size))
    gaussian = multivariate_normal(mean=movement_mean, cov=movement_var)

    for bin_ind1, node1 in enumerate(place_bin_center_ind_to_node):
        for bin_ind2, node2 in enumerate(place_bin_center_ind_to_node):
            try:
                state_transition[bin_ind1, bin_ind2] = gaussian.pdf(
                    distance_between_nodes[node1][node2]
                )
            except KeyError:
                # bins not on track interior will be -1 and not in distance
                # between nodes
                continue

    return state_transition
