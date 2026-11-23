"""Create SUMO TAZs from zone polygons (GeoJSON) for a SUMO network (needs netconvert).

Usage: python shp_to_taz.py network.net.xml zones.json --zone-id i --output tazes.taz.xml
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from sumo_network import edges_to_geojson, zones_to_tazs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("network", type=Path)
    parser.add_argument("zones", type=Path, help="zone polygons (GeoJSON)")
    parser.add_argument("--zone-id", default="i", help="zone id attribute")
    parser.add_argument("--output", type=Path, default=Path("tazes.taz.xml"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "network"
        subprocess.run(["netconvert", "--sumo-net-file", str(args.network), "--plain-output-prefix",
                        str(prefix), "--proj.plain-geo"], check=True)
        edges_to_geojson(f"{prefix}.edg.xml", Path(tmp) / "network.geojson")
        zones_to_tazs(Path(tmp) / "network.geojson", args.zones, args.zone_id, args.output)
    print(f"Wrote {args.output}")
