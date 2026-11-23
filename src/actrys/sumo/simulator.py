"""Running the SUMO tool chain: od2trips -> duarouter -> route2trips -> sumo (mesoscopic)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from actrys.config import SupplyParams, TimeWindow

log = logging.getLogger(__name__)


class SumoNotFoundError(RuntimeError):
    pass


def sumo_home() -> Path | None:
    home = os.environ.get("SUMO_HOME")
    if home:
        return Path(home)
    try:  # the eclipse-sumo wheel
        import sumo  # type: ignore[import-not-found]

        return Path(sumo.SUMO_HOME)
    except (ImportError, AttributeError):
        return None


def find_binary(name: str) -> str:
    """Locate a SUMO binary on ``PATH`` or in ``$SUMO_HOME/bin``."""
    found = shutil.which(name)
    if found:
        return found
    home = sumo_home()
    if home is not None and (home / "bin" / name).exists():
        return str(home / "bin" / name)
    raise SumoNotFoundError(f"SUMO binary {name!r} not found; install SUMO and set SUMO_HOME")


def find_tool(*candidates: str) -> Path:
    """Locate a script under ``$SUMO_HOME/tools`` (first existing candidate)."""
    home = sumo_home()
    if home is None:
        raise SumoNotFoundError("SUMO_HOME is not set")
    for rel in candidates:
        path = home / "tools" / rel
        if path.exists():
            return path
    raise SumoNotFoundError(f"None of {candidates} found under {home / 'tools'}")


def _run(cmd: Sequence[str], **kwargs) -> None:
    log.debug("running %s", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


@dataclass(frozen=True)
class EvaluationOutputs:
    """Extra outputs written by evaluation runs (needed for the W-SPSA incidence matrix)."""

    summary: Path
    tripinfo: Path
    vehroutes: Path


class SumoSimulator:
    """Thin, stateless wrapper around the SUMO command-line tools of one scenario."""

    def __init__(self, network: Path, zones: Path, window: TimeWindow, routing_threads: int = 16):
        self.network = Path(network)
        self.zones = Path(zones)
        self.window = window
        self.routing_threads = routing_threads

    def generate_trips(self, od_files: Sequence[Path], trips: Path, routes: Path) -> None:
        """OD matrices -> trips -> routes, then keep only routable trips."""
        _run(
            [
                find_binary("od2trips"),
                "-n",
                self.zones,
                "-d",
                ",".join(map(str, od_files)),
                "-o",
                trips,
            ]
        )
        _run(
            [
                find_binary("duarouter"),
                "-n",
                self.network,
                "-r",
                trips,
                "--ignore-errors",
                "--no-warnings",
                "--no-step-log",
                "-o",
                routes,
            ]
        )
        route2trips = find_tool("purgatory/route2trips.py", "route/route2trips.py")
        with Path(trips).open("w") as out:
            _run([sys.executable, route2trips, routes], stdout=out)

    def command(
        self,
        trips: Path,
        additional: Path,
        supply: SupplyParams,
        outputs: EvaluationOutputs | None = None,
    ) -> list[str]:
        cmd = [
            find_binary("sumo"),
            "-n",
            self.network,
            "-r",
            trips,
            "--additional-files",
            additional,
            "-b",
            self.window.simulation_begin,
            "-e",
            self.window.simulation_end,
            "--no-step-log",
            "--device.rerouting.probability",
            supply.rerouting_prob,
            "--device.rerouting.threads",
            self.routing_threads,
            "--device.rerouting.period",
            int(supply.rerouting_period),
            "--device.rerouting.synchronize",
            "--device.rerouting.adaptation-interval",
            int(supply.rerouting_adaptation),
            "--device.rerouting.adaptation-steps",
            int(supply.rerouting_adaptation_steps),
            "--meso-tls-flow-penalty",
            supply.meso_tls_flow_penalty,
            "--meso-junction-control",
            "true",
            "--meso-tls-penalty",
            supply.tls_tt_penalty,
            "--meso-minor-penalty",
            int(supply.meso_minor_penalty),
            "--weights.priority-factor",
            supply.priority_factor,
        ]
        if outputs is not None:
            cmd += [
                "--summary",
                outputs.summary,
                "--tripinfo-output",
                outputs.tripinfo,
                "--vehroute-output",
                outputs.vehroutes,
            ]
        cmd += ["--mesosim", "--no-warnings"]
        return [str(c) for c in cmd]

    def simulate(
        self,
        trips: Path,
        additional: Path,
        supply: SupplyParams,
        outputs: EvaluationOutputs | None = None,
    ) -> None:
        """Run a mesoscopic simulation with dynamic rerouting."""
        _run(self.command(trips, additional, supply, outputs))
