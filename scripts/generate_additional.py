"""Create a SUMO detector (additional) file from a detector -> SUMO edge mapping.

One ``e1Detector`` is placed on every lane of each mapped edge, ``--position``
metres from the edge start (negative: from the end).

Usage:
    python generate_additional.py mapping.csv network.net.xml --output additional.add.xml \
        --edge-col sumo_id --id-col munich_id --lat-col detector_lat --lon-col detector_lon
"""

import argparse
import xml.etree.ElementTree as ET

import pandas as pd
from sumo_network import edge_lengths


def create_detector_file(mapping: pd.DataFrame, lanes: pd.DataFrame, output: str, edge_col: str, id_col: str,
                         lat_col: str | None, lon_col: str | None, position: float = -20,
                         freq: float = 900, edge_data_freqs=(300, 900, 3600)) -> int:
    lanes = lanes.set_index("edge_id")
    root = ET.Element("additional")
    for i, f in enumerate(edge_data_freqs, 1):
        ET.SubElement(root, "edgeData", id=str(i), file=f"edge_data_{f}", freq=f"{float(f)}")
    seen_edges, seen_ids, n = set(), set(), 0
    for _, row in mapping.iterrows():
        edge, det = str(row[edge_col]), str(row[id_col])
        if edge in seen_edges or det in seen_ids or edge not in lanes.index:
            continue
        if lanes.loc[edge, "lane_length"] < abs(position):
            continue
        for k in range(int(lanes.loc[edge, "num_lanes"])):
            el = ET.SubElement(root, "e1Detector", id=f"e1Detector_{edge}_{k}", lane=f"{edge}_{k}",
                               pos=str(position), freq=str(freq), file="out.xml", name=det)
            if lat_col and lon_col:
                ET.SubElement(el, "param", key="C", value=f"{row[lat_col]}, {row[lon_col]}")
            n += 1
        seen_edges.add(edge)
        seen_ids.add(det)
    ET.ElementTree(root).write(output, encoding="UTF-8", xml_declaration=True)
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mapping", help="CSV mapping detectors to SUMO edges")
    p.add_argument("network", help="SUMO network (*.net.xml)")
    p.add_argument("--output", default="additional.add.xml")
    p.add_argument("--edge-col", default="sumo_id")
    p.add_argument("--id-col", default="munich_id")
    p.add_argument("--direction-col", default=None,
                   help="append this column to the detector id (e.g. 'direction' for BASt data)")
    p.add_argument("--lat-col", default=None)
    p.add_argument("--lon-col", default=None)
    p.add_argument("--position", type=float, default=-20)
    a = p.parse_args()
    mapping = pd.read_csv(a.mapping)
    if a.direction_col:
        mapping[a.id_col] = mapping[a.id_col].astype(str) + "_" + mapping[a.direction_col].astype(int).astype(str)
        mapping = mapping.drop_duplicates(subset=[a.id_col])
    n = create_detector_file(mapping, edge_lengths(a.network), a.output, a.edge_col, a.id_col,
                             a.lat_col, a.lon_col, a.position)
    print(f"Wrote {n} detectors to {a.output}")
