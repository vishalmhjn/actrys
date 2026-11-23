import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MUNICH = REPO / "munich"


def sumo_available() -> bool:
    try:
        from actrys.sumo.simulator import find_binary, find_tool

        find_binary("sumo")
        find_binary("od2trips")
        find_binary("duarouter")
        find_tool("purgatory/route2trips.py", "route/route2trips.py")
        return True
    except Exception:
        return False


requires_sumo = pytest.mark.skipif(not sumo_available(), reason="SUMO not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def munich_demand(tmp_path):
    dst = tmp_path / "demand"
    shutil.copytree(MUNICH / "demand", dst)
    return dst
