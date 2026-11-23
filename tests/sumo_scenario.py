"""Generate a tiny SUMO scenario (grid network) for integration tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from actrys.sumo.simulator import find_binary

ZONES = {
    "z1": ["A0A1", "A1A2"],
    "z2": ["D0D1", "D1D2"],
    "z3": ["B0C0", "C0D0"],
    "z4": ["B3C3", "A3B3"],
}
DETECTOR_EDGES = ["B0B1", "B1B2", "C1C2", "B1C1", "C2B2", "A1B1", "C2D2", "B2B3"]
HOURS = [5, 6, 7, 8]


def build(folder: Path, seed: int = 0) -> Path:
    """Write network, zones, demand and detectors to ``folder``; returns it."""
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            find_binary("netgenerate"),
            "--grid",
            "--grid.number",
            "4",
            "--grid.length",
            "300",
            "--default.lanenumber",
            "2",
            "--no-turnarounds",
            "true",
            "-o",
            str(folder / "network.net.xml"),
        ],
        check=True,
        capture_output=True,
    )
    tazes = "\n".join(f'    <taz id="{z}" edges="{" ".join(e)}"/>' for z, e in ZONES.items())
    (folder / "tazes.taz.xml").write_text(f"<additional>\n{tazes}\n</additional>\n")

    dets = "\n".join(
        f'    <e1Detector id="e1Detector_{e}_{lane}" lane="{e}_{lane}" pos="-20" freq="900" file="out.xml"/>'
        for e in DETECTOR_EDGES
        for lane in (0, 1)
    )
    edge_data = "\n".join(
        f'    <edgeData id="{i}" file="edge_data_{f}" freq="{f}"/>'
        for i, f in enumerate((300, 900, 3600), 1)
    )
    (folder / "additional.add.xml").write_text(
        f"<additional>\n{edge_data}\n{dets}\n</additional>\n"
    )

    rng = np.random.default_rng(seed)
    pairs = [(o, d) for o in ZONES for d in ZONES if o != d]
    base = rng.integers(20, 60, len(pairs))
    demand = folder / "demand"
    demand.mkdir(exist_ok=True)
    for h in HOURS:
        rows = "\n".join(
            f"{o} {d} {int(b * (0.6 + 0.2 * (h - 5)))}"
            for (o, d), b in zip(pairs, base, strict=True)
        )
        (demand / f"OD_{float(h)}_{float(h + 1)}.txt").write_text(
            f"$OR;D2\n* From-Time  To-Time\n{h:02d}.00 {h + 1}.00\n* Factor\n1.00\n{rows}"
        )
    (demand / "od_list.txt").write_text("o d t\n" + "\n".join(f"{o} {d} 1" for o, d in pairs))
    return folder
