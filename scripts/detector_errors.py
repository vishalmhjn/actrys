"""Per-detector count errors (observed - simulated) with coordinates, e.g. for mapping in GIS.

Detector coordinates are read from ``<param key="C" value="lat, lon"/>`` children
in the additional file (as written by generate_additional.py).

Usage: python detector_errors.py additional.add.xml run/out.xml run/real_counts_complete.csv \
           --hours 7 8 --output errors.csv
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actrys.sumo.outputs import DetectorLayout, edge_counts, read_detector_output  # noqa: E402


def detector_coordinates(additional: Path) -> pd.DataFrame:
    rows = []
    for det in ET.parse(additional).getroot().iter():
        param = det.find("param[@key='C']")
        if det.get("lane") and param is not None:
            lat, lon = (float(v) for v in param.get("value").split(","))
            rows.append({"det_id": det.get("lane").rsplit("_", 1)[0], "lat": lat, "lon": lon})
    return pd.DataFrame(rows, columns=["det_id", "lat", "lon"]).groupby("det_id", as_index=False).mean()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("additional", type=Path)
    p.add_argument("detector_output", type=Path)
    p.add_argument("observed_counts", type=Path)
    p.add_argument("--hours", type=float, nargs="+", required=True, help="evaluation interval start hours")
    p.add_argument("--interval-hours", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=Path("errors.csv"))
    a = p.parse_args()
    layout = DetectorLayout.read(a.additional)
    sim = edge_counts(read_detector_output(a.detector_output), layout.edge_of, a.hours, a.interval_hours)
    obs = pd.read_csv(a.observed_counts).astype({"det_id": str})
    both = obs.merge(sim, on=["det_id", "hour"], suffixes=("_observed", "_simulated"))
    both["error"] = both["count_observed"] - both["count_simulated"]
    both.merge(detector_coordinates(a.additional), on="det_id", how="left").to_csv(a.output, index=False)
    print(f"Wrote {a.output}")
