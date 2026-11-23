import itertools

import numpy as np
import pytest

from actrys.config import ObjectiveWeights
from actrys.optim.weights import wspsa_weight_matrix
from reference import prepare_weight_matrix_dense


@pytest.mark.parametrize("w", [w for w in itertools.product([0, 0.5], repeat=3) if any(w)])
def test_sparse_matches_dense(w):
    rng = np.random.default_rng(0)
    inc = rng.uniform(size=(7, 5)) * (rng.uniform(size=(7, 5)) > 0.6)
    ours = wspsa_weight_matrix(inc, ObjectiveWeights(*w))
    np.testing.assert_array_equal(ours.toarray(), prepare_weight_matrix_dense(inc, *w))
