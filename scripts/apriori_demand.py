"""Write time-dependent SUMO O-format demand from a static OD table and a temporal profile.

The static flows (columns ``from``, ``to``, ``trips``) are scaled per interval by
the daily profile (normalised to its peak), see :func:`write_od_periods`.

Usage: python apriori_demand.py static_od.csv --output demand/OD.txt --interval-hours 0.25
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actrys.synthetic.scenario import DAILY_PROFILE, fit_temporal_kernel  # noqa: E402


def write_od_periods(static_od: pd.DataFrame, hours: np.ndarray, profile: np.ndarray, base: Path,
                     interval_hours: float) -> int:
    """One ``<base>_<from>_<to>.txt`` per interval; returns the total number of trips."""
    total = 0
    for hour, weight in zip(hours, profile / profile.max(), strict=True):
        flows = (static_od["trips"].to_numpy() * weight).astype(int)
        total += int(flows.sum())
        rows = "\n".join(f"{o} {d} {f}" for o, d, f in zip(static_od["from"], static_od["to"], flows, strict=True))
        start, end = hour, hour + interval_hours
        header = f"$OR;D2\n* From-Time  To-Time\n{int(start):02d}.{int(round(start % 1 * 60)):02d} " \
                 f"{int(end):02d}.{int(round(end % 1 * 60)):02d}\n* Factor\n1.00\n"
        Path(base).with_name(f"{Path(base).stem}_{float(start)}_{float(end)}.txt").write_text(header + rows)
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("static_od", help="CSV with columns from, to, trips")
    p.add_argument("--output", type=Path, default=Path("OD.txt"), help="base name of the written files")
    p.add_argument("--interval-hours", type=float, default=1.0)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float, default=24)
    a = p.parse_args()
    hours = np.arange(a.start, a.end, a.interval_hours)
    model = fit_temporal_kernel(np.arange(24), DAILY_PROFILE)
    profile = model.predict(hours.reshape(-1, 1)).flatten()
    n = write_od_periods(pd.read_csv(a.static_od), hours, profile, a.output, a.interval_hours)
    print(f"Total trips: {n}")
