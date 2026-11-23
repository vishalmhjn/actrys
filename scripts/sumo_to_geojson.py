"""Export a SUMO network as GeoJSON (needs SUMO's netconvert).

Usage: python sumo_to_geojson.py network.net.xml --output network.geojson
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from sumo_network import edges_to_geojson

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("network", type=Path)
    parser.add_argument("--output", type=Path, default=Path("network.geojson"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "network"
        subprocess.run(["netconvert", "--sumo-net-file", str(args.network), "--plain-output-prefix",
                        str(prefix), "--proj.plain-geo"], check=True)
        edges_to_geojson(f"{prefix}.edg.xml", args.output)
    print(f"Wrote {args.output}")
