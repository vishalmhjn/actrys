"""Iteratively drop TAZ edges that cause unroutable trips.

Runs od2trips + duarouter, collects the source/destination edges of
"no connection" errors that occur more than ``--threshold`` times, removes them
from TAZs with more than three edges, and repeats.

Usage:
    python recursive_taz_filter.py --folder scenario/temp --network ../network.net.xml \
        --tazes ../tazes.taz.xml --od OD_5.0_6.0.txt,OD_6.0_7.0.txt --iterations 5
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

NO_CONNECTION = re.compile(r"No connection between edge '([^']*)' and edge '([^']*)' found\.")


def route(folder: Path, network: str, tazes: str, od_files: str, route2trips: str) -> None:
    subprocess.run(["od2trips", "-n", tazes, "-d", od_files, "-o", "trips.trips.xml"], cwd=folder, check=True)
    subprocess.run(["duarouter", "-n", network, "-r", "trips.trips.xml", "--ignore-errors", "-o", "routes.rou.xml",
                    "--error-log", "errors.txt"], cwd=folder, check=True)
    with (folder / "validated_trips.trips.xml").open("w") as out:
        subprocess.run([sys.executable, route2trips, "routes.rou.xml"], cwd=folder, stdout=out, check=True)


def disconnected_edges(error_file: Path, threshold: int) -> tuple[set[str], set[str]]:
    pairs = Counter(m.groups() for m in map(NO_CONNECTION.search, error_file.read_text().splitlines()) if m)
    frequent = [pair for pair, n in pairs.items() if n > threshold]
    return {s for s, _ in frequent}, {d for _, d in frequent}


def filter_tazes(taz_file: Path, remove: set[str], output: Path) -> None:
    tree = ET.parse(taz_file)
    for taz in tree.getroot().iter("taz"):
        edges = (taz.get("edges") or "").split()
        if len(edges) > 3:  # keep at least the small TAZs intact
            taz.set("edges", " ".join(e for e in edges if e not in remove))
    tree.write(output, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--folder", type=Path, required=True, help="working folder")
    p.add_argument("--network", required=True, help="network, relative to --folder")
    p.add_argument("--tazes", required=True, help="initial TAZ file, relative to --folder")
    p.add_argument("--od", required=True, help="comma-separated O-format files, relative to --folder")
    p.add_argument("--output", default="newtaZes.taz.xml")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--threshold", type=int, default=3)
    a = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from actrys.sumo.simulator import find_tool

    route2trips = str(find_tool("purgatory/route2trips.py", "route/route2trips.py"))
    tazes = a.tazes
    route(a.folder, a.network, tazes, a.od, route2trips)
    for _ in range(a.iterations):
        sources, destinations = disconnected_edges(a.folder / "errors.txt", a.threshold)
        filter_tazes(a.folder / tazes, sources | destinations, a.folder / a.output)
        tazes = a.output
        route(a.folder, a.network, tazes, a.od, route2trips)
    print("Done")
