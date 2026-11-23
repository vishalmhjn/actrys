"""W-SPSA weight matrices."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from actrys.config import ObjectiveWeights


def wspsa_weight_matrix(incidence: np.ndarray, weights: ObjectiveWeights) -> sp.csr_matrix:
    """Build the sparse W-SPSA weight matrix for the stacked deviation vector.

    The deviation vector is ``[counts | od | speeds]`` (blocks with zero weight
    omitted). Counts and speeds are influenced by the OD pairs that cross the
    detector (binary incidence), the OD-prior term only by the OD pair itself
    (identity). ``incidence`` has shape ``(n_od_intervals, n_detector_intervals)``.
    """
    binary = sp.csr_matrix((np.asarray(incidence) > 0).astype(np.int8))
    identity = sp.identity(binary.shape[0], dtype=np.int8, format="csr")
    blocks = []
    if weights.counts != 0:
        blocks.append(binary)
    if weights.od != 0:
        blocks.append(identity)
    if weights.speed != 0:
        blocks.append(binary)
    if not blocks:
        raise ValueError("All objective weights are zero")
    return sp.hstack(blocks, format="csr")
