"""Download the OSM road network (``highway=*``) inside a bounding box via the Overpass API.

Usage: python download_network.py --bbox S,W,N,E --output raw_network.osm
(``RAW_NETWORK`` and ``BBOX`` environment variables are used as defaults.)
"""

import argparse
import os

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(bbox: str) -> str:
    return f"""
    [out:xml][timeout:25];
    (
      node["highway"]({bbox});
      way["highway"]({bbox});
      relation["highway"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """


def download_raw_network_data(output: str, bbox: str, timeout: float = 300) -> None:
    response = requests.post(
        OVERPASS_URL,
        data={"data": overpass_query(bbox)},
        headers={"User-Agent": "actrys-network-download"},
        timeout=timeout,
    )
    response.raise_for_status()
    with open(output, "w") as f:
        f.write(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bbox", default=os.environ.get("BBOX"), help="south,west,north,east")
    parser.add_argument("--output", default=os.environ.get("RAW_NETWORK", "raw_network.osm"))
    args = parser.parse_args()
    if not args.bbox:
        parser.error("--bbox (or BBOX) is required")
    download_raw_network_data(args.output, args.bbox)
