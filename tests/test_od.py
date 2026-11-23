import numpy as np
import pytest

from actrys.sumo.od import ODDemand, ODPeriodFile, period_path, read_od_pairs

HOURS = [5.0, 6.0, 7.0, 8.0]


def test_period_path(tmp_path):
    assert period_path(tmp_path / "OD.txt", 7, 1.0).name == "OD_7.0_8.0.txt"
    assert (
        period_path(tmp_path / "OD_n_positive.txt", 7.25, 0.25).name == "OD_n_positive_7.25_7.5.txt"
    )


def test_read_munich(munich_demand):
    demand = ODDemand.read(munich_demand / "OD.txt", HOURS, 1.0)
    assert demand.n_intervals == 4
    vec = demand.vector()
    assert len(vec) == len(demand) == 4 * demand.n_pairs
    first = (munich_demand / "OD_5.0_6.0.txt").read_text().splitlines()[5].split()
    assert vec[0] == float(first[2])


def test_round_trip_is_byte_identical(munich_demand, tmp_path):
    demand = ODDemand.read(munich_demand / "OD.txt", HOURS, 1.0)
    paths = demand.write(tmp_path / "OD_x.txt", demand.vector(), HOURS, 1.0)
    for p, h in zip(paths, HOURS, strict=True):
        assert p.read_text() == (munich_demand / f"OD_{h}_{h + 1}.txt").read_text()


def test_write_scales_by_factor_and_truncates(tmp_path):
    src = tmp_path / "OD_7.0_8.0.txt"
    src.write_text("$OR;D2\n* From-Time  To-Time\n07.00 8.00\n* Factor\n2.00\n1 2 10\n3 4 5")
    period = ODPeriodFile.read(src)
    assert period.factor == 2.0
    out = tmp_path / "out.txt"
    period.write(out, np.array([7.0, 9.9]))
    assert out.read_text().splitlines()[-2:] == ["1 2 3", "3 4 4"]


def test_length_validation(munich_demand, tmp_path):
    demand = ODDemand.read(munich_demand / "OD.txt", HOURS, 1.0)
    with pytest.raises(ValueError, match="elements"):
        demand.write(tmp_path / "OD.txt", np.zeros(3), HOURS, 1.0)


def test_rejects_non_o_format(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("o d t\n1 2 3\n")
    with pytest.raises(ValueError):
        ODPeriodFile.read(p)


def test_read_od_pairs(munich_demand):
    pairs = read_od_pairs(munich_demand / "od_list.txt")
    demand = ODDemand.read(munich_demand / "OD.txt", HOURS, 1.0)
    assert len(pairs) == demand.n_pairs
    p0 = demand.periods[0]
    assert pairs[0] == f"{p0.origins[0]}___{p0.destinations[0]}"
