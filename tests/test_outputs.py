import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from actrys.sumo.outputs import (
    DetectorLayout,
    edge_counts,
    lane_to_edge,
    match_counts,
    match_speeds,
    read_detector_output,
    read_edge_data,
    read_tripinfo,
    read_vehroutes,
    synthetic_counts,
    synthetic_speeds,
    write_detector_variant,
)
from conftest import MUNICH

ADDITIONAL = """<additional>
  <edgeData id="3" file="edge_data_3600" freq="3600.0" />
  <e1Detector id="e1Detector_a_0" lane="a_0" pos="-20" freq="900" file="out.xml"/>
  <e1Detector id="e1Detector_a_1" lane="a_1" pos="-20" freq="900" file="out.xml"/>
  <e1Detector id="e1Detector_b#2_0" lane="b#2_0" pos="-20" freq="900" file="out.xml"/>
  <e1Detector id="e1Detector_x_y_0" lane="x_y_0" pos="-20" freq="900" file="out.xml"/>
</additional>"""

DET_OUT = """<detector>
  <interval begin="0.00" end="1800.00" id="e1Detector_a_0" entered="4"/>
  <interval begin="0.00" end="1800.00" id="e1Detector_a_1" entered="6"/>
  <interval begin="0.00" end="1800.00" id="e1Detector_b#2_0" entered="1"/>
  <interval begin="0.00" end="1800.00" id="e1Detector_x_y_0" entered="2"/>
  <interval begin="1800.00" end="3600.00" id="e1Detector_a_0" entered="1"/>
  <interval begin="1800.00" end="3600.00" id="e1Detector_a_1" entered="2"/>
  <interval begin="3600.00" end="5400.00" id="e1Detector_a_0" entered="7"/>
</detector>"""

EDGE_DATA = """<meandata>
  <interval begin="0.00" end="3600.00" id="3">
    <edge id="a" speed="10.5" sampledSeconds="3"/>
    <edge id="b#2" sampledSeconds="0"/>
    <edge id="zzz" speed="1"/>
  </interval>
  <interval begin="3600.00" end="7200.00" id="3">
    <edge id="a" speed="8.0"/>
  </interval>
</meandata>"""


@pytest.fixture
def files(tmp_path):
    for name, text in (("add.xml", ADDITIONAL), ("out.xml", DET_OUT), ("edge", EDGE_DATA)):
        (tmp_path / name).write_text(text)
    return tmp_path


def test_lane_to_edge():
    assert lane_to_edge("-1032#2_0") == "-1032#2"
    assert lane_to_edge("x_y_1") == "x_y"


def test_detector_layout(files):
    layout = DetectorLayout.read(files / "add.xml")
    assert layout.edges == ("a", "a", "b#2", "x_y")
    assert layout.unique_edges == ["a", "b#2", "x_y"]


def test_detector_layout_munich():
    layout = DetectorLayout.read(MUNICH / "additional.add.xml")
    assert len(layout.ids) == 757 and len(layout.unique_edges) == 389
    # edge ids derived from the lane equal id.split("_")[-2] for this file
    assert all(e == i.split("_")[-2] for i, e in zip(layout.ids, layout.edges, strict=True))


def test_edge_counts_meso_and_micro(files):
    layout = DetectorLayout.read(files / "add.xml")
    out = read_detector_output(files / "out.xml")
    meso = edge_counts(out, layout.edge_of, [0.0, 1.0], 1.0, "meso")
    # hour 0: a_0 = 4+1, a_1 = 6+2 -> mean 6.5
    assert meso.query("hour == 0 and det_id == 'a'")["count"].item() == 6.5
    assert meso[meso.hour == 0]["det_id"].tolist() == ["a", "b#2", "x_y"]  # sorted
    micro = edge_counts(out, layout.edge_of, [0.0], 1.0, "micro")
    assert micro.query("det_id == 'a'")["count"].item() == 13
    with pytest.raises(ValueError):
        edge_counts(out, layout.edge_of, [0.0], 1.0, "nano")


def test_synthetic_counts_truncate(files):
    layout = DetectorLayout.read(files / "add.xml")
    out = read_detector_output(files / "out.xml")
    df = synthetic_counts(out, layout.edge_of, [0.0], 1.0, 0, rng=np.random.default_rng(0))
    assert df.query("det_id == 'a'")["count"].item() == 6
    assert list(df.columns) == ["det_id", "count", "hour"]


def test_match_counts_keeps_observation_order(files):
    layout = DetectorLayout.read(files / "add.xml")
    out = read_detector_output(files / "out.xml")
    observed = pd.DataFrame(
        {"det_id": ["b#2", "a", "missing", "a"], "count": [3, 5, 1, 9], "hour": [0, 0, 0, 3600]}
    )
    obs, sim = match_counts(out, observed, layout.edge_of, [0.0, 1.0], 1.0)
    assert obs.tolist() == [3, 5, 9]
    assert sim.tolist() == [1, 6.5, 7]


def test_edge_data_and_speeds(files):
    ed = read_edge_data(files / "edge")
    assert np.isnan(ed.query("edge_id == 'b#2'")["speed"].item())
    syn = synthetic_speeds(ed, ["a", "b#2"], [0.0, 1.0], 1.0)
    assert syn["det_id"].tolist() == ["a", "b#2", "a"]
    obs, sim = match_speeds(ed, syn, [0.0, 1.0], 1.0)
    assert obs.tolist() == [10.5, 20, 8.0]  # NaN filled with 20
    assert sim.tolist() == [10.5, 20, 8.0]


def test_vehroutes_use_final_route(tmp_path):
    (tmp_path / "r.xml").write_text("""<routes>
      <vehicle id="1" depart="10.00" fromTaz="o" toTaz="d" arrival="50">
        <routeDistribution><route replacedOnEdge="e" edges="a b"/><route edges="a c"/></routeDistribution>
      </vehicle>
      <vehicle id="2" depart="20.00" fromTaz="o" toTaz="d"><route edges="x"/></vehicle>
    </routes>""")
    (tmp_path / "t.xml").write_text('<tripinfos><tripinfo id="1" arrival="50.00"/></tripinfos>')
    vr = read_vehroutes(tmp_path / "r.xml")
    assert vr["route_edges"].tolist() == ["a c", "x"]
    assert vr["depart"].tolist() == [10.0, 20.0]
    assert read_tripinfo(tmp_path / "t.xml")["arrival"].tolist() == [50.0]


def test_detector_variant_matches_shipped_munich_files(tmp_path):
    for suffix in ("positive", "negative"):
        out = tmp_path / f"add_{suffix}.xml"
        write_detector_variant(MUNICH / "additional.add.xml", out, suffix)
        ours = [(e.tag, dict(e.attrib)) for e in ET.parse(out).getroot().iter()]
        shipped = [
            (e.tag, dict(e.attrib))
            for e in ET.parse(MUNICH / f"additional.add_{suffix}.xml").getroot().iter()
        ]
        assert ours == shipped
