import numpy as np
import pandas as pd
import pytest

from actrys.sumo.incidence import detector_incidence
from reference import detector_incidence_loop


def random_trips(rng, n=600, zones=4, edges=12, t0=0, t1=4 * 900):
    origins = rng.integers(0, zones, n).astype(str)
    dests = rng.integers(0, zones, n).astype(str)
    depart = rng.uniform(t0 - 100, t1, n).round(2)
    arrival = depart + rng.uniform(10, 1200, n).round(2)
    routes = [" ".join(f"e{j}" for j in rng.choice(edges, rng.integers(1, 6))) for _ in range(n)]
    return pd.DataFrame(
        {
            "from_taz": origins,
            "to_taz": dests,
            "depart": depart,
            "arrival": arrival,
            "route_edges": routes,
        }
    )


@pytest.mark.parametrize("threshold,binary", [(None, False), (0.1, False), (0.1, True)])
def test_matches_reference_loop(threshold, binary):
    rng = np.random.default_rng(3)
    trips = random_trips(rng)
    od_pairs = [f"{o}___{d}" for o in range(4) for d in range(4)][:14]  # some OD pairs not modelled
    detectors = ["e3", "e1", "e7", "e10", "nope", "e1"]
    kw = dict(t_start=1800, t_end=3600, t_warmup=900, interval=900)

    ours = detector_incidence(trips, od_pairs, detectors, threshold=threshold, binary=binary, **kw)
    ref = trips.rename(columns={"depart": "vehicle_depart", "arrival": "tripinfo_arrival"})
    ref["od_pair"] = ref["from_taz"] + "___" + ref["to_taz"]
    w_ref, a_ref = detector_incidence_loop(
        ref,
        od_pairs,
        detectors,
        kw["t_start"],
        kw["t_end"],
        kw["t_warmup"],
        kw["interval"],
        threshold=threshold is not None,
        threshold_value=threshold or 0,
        binary_rounding=binary,
    )

    assert ours.weight.shape == w_ref.shape == (14 * 3, 5 * 2)
    np.testing.assert_array_equal(ours.weight, w_ref)
    np.testing.assert_array_equal(ours.assignment, a_ref)
    assert ours.assignment.dtype == np.float32
    assert ours.assignment.any()


def test_empty_passages():
    trips = pd.DataFrame(
        {
            "from_taz": ["a"],
            "to_taz": ["b"],
            "depart": [10.0],
            "arrival": [20.0],
            "route_edges": ["x y"],
        }
    )
    m = detector_incidence(trips, ["a___b"], ["z"], t_start=0, t_end=60, t_warmup=0, interval=60)
    assert m.weight.shape == (1, 1) and not m.weight.any()
