import numpy as np

from pytest import mark
from replay_identification.multiunit_likelihood import (estimate_mean_rate,
                                                        poisson_mark_log_likelihood)


def test_estimate_mean_rate():
    test_multiunit = np.full((10, 4), np.nan)
    test_multiunit[[1, 5, 6], :] = 10
    test_position = np.ones((10,))
    test_position[5] = np.nan

    assert np.allclose(estimate_mean_rate(test_multiunit, test_position),
                       2 / 9)


@mark.parametrize('log_jmi, gpi, expected_likelihood', [
    (0, 10, -10),
    (10, 1, 9),
])
def test_poisson_mark_log_likelihood(log_jmi, gpi, expected_likelihood):
    log_likelihood = poisson_mark_log_likelihood(
        log_jmi, gpi, time_bin_size=1)

    assert np.allclose(log_likelihood, expected_likelihood)
