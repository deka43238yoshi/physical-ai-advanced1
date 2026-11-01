from pathlib import Path

import pytest

from so101_dice_sim.settings import load_settings
from so101_dice_sim.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    return load_settings(ROOT / "config" / "default.json")


@pytest.fixture(scope="session")
def settled(settings):
    return run_simulation(settings)
