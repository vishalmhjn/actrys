"""SUMO network helpers: edge tables, GeoJSON export and TAZ creation from zone polygons.

Requires the optional ``scripts`` dependencies (``pip install actrys[scripts]``).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROAD_TYPES_FOR_TAZ = ("highway.secondary", "highway.tertiary", "highway.residential", "highway.trunk")


def read_plain_edges(path_edg_xml: str | Path) -> pd.DataFrame:
    """Edges of a plain ``*.edg.xml`` (from ``netconvert --plain-output-prefix``)."""
    rows = [dict(e.attrib) for e in ET.parse(path_edg_xml).getroot().iter("edge")]
    return pd.DataFrame(rows)


def edge_lengths(path_net_xml: str | Path) -> pd.DataFrame:
    """``edge_id``, ``num_lanes`` and (first lane) ``lane_length`` of a ``*.net.xml``."""
    rows = []
    for edge in ET.parse(path_net_xml).getroot().iter("edge"):
        if edge.get("function") == "internal":
            continue
        lanes = edge.findall("lane")
        if lanes:
            rows.append({"edge_id": edge.get("id"), "num_lanes": len(lanes),
                         "lane_length": float(lanes[0].get("length"))})
    return pd.DataFrame(rows)


def edges_to_geojson(path_edg_xml: str | Path, output_path: str | Path) -> None:
    """Write the edges of a plain edge file (geo-projected) as GeoJSON line strings."""
    df = read_plain_edges(path_edg_xml)
    if "shape" not in df:
        raise ValueError(f"{path_edg_xml} has no 'shape' attributes; use netconvert --proj.plain-geo")
    df = df.dropna(subset=["id", "shape", "type"])
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[float(c) for c in p.split(",")[:2]] for p in shape.split(" ")]},
            "properties": {"id": edge_id, "type": edge_type},
        }
        for edge_id, shape, edge_type in zip(df["id"], df["shape"], df["type"], strict=True)
    ]
    Path(output_path).write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=4))


def zones_to_tazs(network_geojson: str | Path, zones_geojson: str | Path, zone_id_attribute: str,
                  output_path: str | Path, road_types=ROAD_TYPES_FOR_TAZ, id_offset: int = 1) -> None:
    """Assign network edges to the zone polygon containing them and write a SUMO TAZ file.

    Numeric zone ids are shifted by ``id_offset`` (default 1, i.e. 0-based zone ids become 1-based).
    """
    import geopandas as gpd

    edges = gpd.read_file(network_geojson)
    zones = gpd.read_file(zones_geojson)[[zone_id_attribute, "geometry"]]
    taz = gpd.sjoin(edges, zones, how="inner", predicate="within")
    taz = taz[taz["type"].isin(road_types)]
    root = ET.Element("tazs")
    for zone_id, group in taz.groupby(zone_id_attribute):
        taz_id = zone_id + id_offset if isinstance(zone_id, int | float) else zone_id
        ET.SubElement(root, "taz", id=str(taz_id), edges=" ".join(group["id"].astype(str)))
    ET.ElementTree(root).write(output_path, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create SUMO TAZs from a network GeoJSON and zone polygons")
    parser.add_argument("network_geojson")
    parser.add_argument("zones_geojson")
    parser.add_argument("--zone-id", default="NO", help="zone id attribute in the zones file")
    parser.add_argument("--output", default="tazes.taz.xml")
    a = parser.parse_args()
    zones_to_tazs(a.network_geojson, a.zones_geojson, a.zone_id, a.output)
