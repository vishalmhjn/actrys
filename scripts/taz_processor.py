"""Keep only TAZs that contain at least one edge starting at a dead-end node (degree 1).

Usage: python taz_processor.py network.edg.csv raw_tazes.taz.csv new_tazes.csv
(inputs are ``xml2csv`` conversions; the environment variables ``input_edge_csv``,
``input_taz_csv`` and ``output_taz_csv`` are used as defaults.)
"""

import argparse
import os
from collections import Counter

import pandas as pd


def filter_taz_by_edges(input_edge_csv: str, input_taz_csv: str, output_taz_csv: str) -> None:
    network = pd.read_csv(input_edge_csv, sep=";")
    degree = Counter(network.edge_from)
    boundary_edges = set(network[network.edge_from.map(degree) == 1]["edge_id"])
    tazes = pd.read_csv(input_taz_csv, sep=";")
    keep = [taz_id for taz_id, edges in zip(tazes.taz_id, tazes.taz_edges, strict=True)
            if isinstance(edges, str) and boundary_edges.intersection(edges.split(" "))]
    tazes[tazes.taz_id.isin(keep)].to_csv(output_taz_csv, index=None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input_edge_csv", nargs="?", default=os.environ.get("input_edge_csv"))
    p.add_argument("input_taz_csv", nargs="?", default=os.environ.get("input_taz_csv"))
    p.add_argument("output_taz_csv", nargs="?", default=os.environ.get("output_taz_csv"))
    a = p.parse_args()
    filter_taz_by_edges(a.input_edge_csv, a.input_taz_csv, a.output_taz_csv)
