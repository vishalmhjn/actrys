"""Count how many simulated vehicles use each edge and add it to a network GeoJSON.

Usage: python route_prevalence.py run_folder/routes.rou.xml network.geojson --output prevalence.geojson
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actrys.sumo.outputs import read_vehroutes  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("routes", type=Path, help="vehicle routes (vehroute output)")
    p.add_argument("network_geojson", type=Path, help="see sumo_to_geojson.py")
    p.add_argument("--output", type=Path, default=Path("network_edge_prevalence.geojson"))
    a = p.parse_args()
    usage = Counter(e for r in read_vehroutes(a.routes)["route_edges"].dropna() for e in r.split(" "))
    geo = json.loads(a.network_geojson.read_text())
    for feature in geo["features"]:
        feature["properties"]["edge_prevalence"] = usage.get(feature["properties"]["id"], 0)
    a.output.write_text(json.dumps(geo))
    print(f"Wrote {a.output}")
